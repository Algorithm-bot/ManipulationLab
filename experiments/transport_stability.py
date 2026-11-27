
import csv
import json
import time
from pathlib import Path

import mujoco
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "simulation/ur5e_gripper.xml"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

TRIALS = 5
NOISE_STD_RAD = 0.02
RANDOM_SEED = 42
SPEED = 1.0

CONDITIONS = [
    "throughout",
    "pickup_only",
    "transport_only",
]

ARM_ACTUATORS = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
]

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint",
    "elbow_joint", "wrist_1_joint",
    "wrist_2_joint", "wrist_3_joint",
]

PICK_HIGH = np.array([
    -3.44408, -1.17748, 1.54195,
     1.20593,  1.57069, -3.44409
])

PICK_LOW = np.array([
    -3.44406, -1.04831, 1.67661,
     0.94215,  1.57071, -3.44406
])

LIFT = np.array([
    -3.44406, -1.12000, 1.59000,
     1.10500,  1.57071, -3.44406
])

PLACE_HIGH = np.array([
    -3.11503, -1.19326, 1.67689,
    -2.05453, -1.57077, 0.02656
])

PLACE_LOW = np.array([
    -3.11505, -1.09705, 1.75718,
    -2.23111, -1.57075, 0.02654
])

GRIPPER_COMMAND = 255

# ============================================================
# MODEL INITIALIZATION
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

rng = np.random.default_rng(RANDOM_SEED)

gripper_id = model.actuator("fingers_actuator").id
cube_id = model.body("grasp_cube").id
timestep = model.opt.timestep

NOISE_ENABLED = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def cube_position():
    mujoco.mj_forward(model, data)
    return data.xpos[cube_id].copy()


def set_arm_targets(targets):
    for actuator, target in zip(ARM_ACTUATORS, targets):
        actuator_id = model.actuator(actuator).id

        if NOISE_ENABLED:
            noisy_target = target + rng.normal(
                0, NOISE_STD_RAD
            )
        else:
            noisy_target = target

        lower, upper = model.actuator_ctrlrange[actuator_id]

        data.ctrl[actuator_id] = np.clip(
            noisy_target, lower, upper
        )


def move_to(targets, duration, gripper_command):
    start = np.array([
        data.qpos[model.joint(j).qposadr[0]]
        for j in ARM_JOINTS
    ])

    steps = max(1, int(duration / timestep))

    for i in range(steps):
        t = (i + 1) / steps
        smooth = 3 * t**2 - 2 * t**3

        intermediate = start + (targets - start) * smooth

        set_arm_targets(intermediate)
        data.ctrl[gripper_id] = gripper_command

        mujoco.mj_step(model, data)


def move_gripper(target, duration):
    start = data.ctrl[gripper_id]
    steps = max(1, int(duration / timestep))

    for i in range(steps):
        t = (i + 1) / steps
        smooth = 3 * t**2 - 2 * t**3

        data.ctrl[gripper_id] = start + (
            target - start
        ) * smooth

        mujoco.mj_step(model, data)


def simulate_steps(steps):
    for _ in range(steps):
        mujoco.mj_step(model, data)


# ============================================================
# EXPERIMENT
# ============================================================

def run_trial(trial, condition):
    global NOISE_ENABLED

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    start_time = time.perf_counter()

    # Pickup phase:
    # throughout and pickup_only receive noise.
    NOISE_ENABLED = condition in [
        "throughout", "pickup_only"
    ]

    move_to(PICK_HIGH, 12 * SPEED, 0)
    move_to(PICK_LOW, 8 * SPEED, 0)

    simulate_steps(int(1.0 / timestep))
    baseline = cube_position()

    move_gripper(GRIPPER_COMMAND, 2)
    simulate_steps(int(1.0 / timestep))

    # Lift
    move_to(LIFT, 5 * SPEED, GRIPPER_COMMAND)
    after_lift = cube_position()

    # Transport phase:
    # throughout and transport_only receive noise.
    NOISE_ENABLED = condition in [
        "throughout", "transport_only"
    ]

    move_to(PLACE_HIGH, 7 * SPEED, GRIPPER_COMMAND)
    after_transport = cube_position()

    # Placement phase: noise disabled for all conditions.
    NOISE_ENABLED = False

    move_to(PLACE_LOW, 5 * SPEED, GRIPPER_COMMAND)

    move_gripper(0, 2)
    simulate_steps(int(2.0 / timestep))

    after_release = cube_position()

    move_to(PLACE_HIGH, 5 * SPEED, 0)

    final_position = cube_position()

    # Metrics
    lift_height = float(
        after_lift[2] - baseline[2]
    )

    retention_height = float(
        after_transport[2] - baseline[2]
    )

    placement_error = float(np.linalg.norm(
        final_position[:2] - np.array([0.60, 0.15])
    ))

    pick_success = lift_height > 0.03
    retention_success = retention_height > 0.03

    placement_success = (
        placement_error < 0.03
        and abs(final_position[2] - 0.02) < 0.015
    )

    overall_success = (
        pick_success
        and retention_success
        and placement_success
    )

    result = {
        "condition": condition,
        "trial": trial,
        "noise_std_rad": NOISE_STD_RAD,
        "pick_success": bool(pick_success),
        "retention_success": bool(retention_success),
        "placement_success": bool(placement_success),
        "overall_success": bool(overall_success),
        "lift_height_m": lift_height,
        "retention_height_m": retention_height,
        "placement_error_m": placement_error,
        "final_x_m": float(final_position[0]),
        "final_y_m": float(final_position[1]),
        "final_z_m": float(final_position[2]),
        "execution_time_sec": (
            time.perf_counter() - start_time
        ),
    }

    print(
        f"{condition:15s} | Trial {trial}/{TRIALS} | "
        f"{'SUCCESS' if overall_success else 'FAIL'} | "
        f"Lift: {lift_height:.4f} m | "
        f"Retention: {retention_height:.4f} m | "
        f"Placement error: {placement_error:.4f} m"
    )

    return result


# ============================================================
# RUN ALL CONDITIONS
# ============================================================

print("=" * 75)
print("PICK-AND-PLACE NOISE PHASE EXPERIMENT")
print("=" * 75)
print(f"Noise standard deviation: {NOISE_STD_RAD:.3f} rad")
print(f"Trials per condition: {TRIALS}")
print(f"Movement speed: {SPEED:.1f}x")
print()

results = []

for condition in CONDITIONS:
    print(f"\nRunning condition: {condition}")

    for trial in range(1, TRIALS + 1):
        results.append(run_trial(trial, condition))


# ============================================================
# SUMMARY
# ============================================================

summary = []

for condition in CONDITIONS:
    group = [
        r for r in results
        if r["condition"] == condition
    ]

    summary.append({
        "condition": condition,
        "trials": len(group),
        "successful_trials": sum(
            r["overall_success"] for r in group
        ),
        "pick_success_rate": float(np.mean([
            r["pick_success"] for r in group
        ])),
        "retention_success_rate": float(np.mean([
            r["retention_success"] for r in group
        ])),
        "placement_success_rate": float(np.mean([
            r["placement_success"] for r in group
        ])),
        "overall_success_rate": float(np.mean([
            r["overall_success"] for r in group
        ])),
        "mean_placement_error_m": float(np.mean([
            r["placement_error_m"] for r in group
        ])),
    })


print("\n" + "=" * 75)
print("EXPERIMENT SUMMARY")
print("=" * 75)

for item in summary:
    print(
        f"{item['condition']:15s} | "
        f"Success: {item['successful_trials']}/{item['trials']} | "
        f"Pickup: {item['pick_success_rate']:.0%} | "
        f"Retention: {item['retention_success_rate']:.0%} | "
        f"Placement: {item['placement_success_rate']:.0%} | "
        f"Mean error: {item['mean_placement_error_m']:.5f} m"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

csv_path = OUTPUT_DIR / "transport_noise_phase.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)


report = {
    "experiment": "noise_phase_isolation",
    "noise_std_rad": NOISE_STD_RAD,
    "random_seed": RANDOM_SEED,
    "trials_per_condition": TRIALS,
    "movement_speed": SPEED,
    "summary": summary,
    "results": results,
}

json_path = OUTPUT_DIR / "transport_noise_phase_report.json"

with open(json_path, "w") as f:
    json.dump(report, f, indent=4)


print("\nGenerated files:")
print(csv_path)
print(json_path)
