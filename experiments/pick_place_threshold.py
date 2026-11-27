
import csv
import json
import time
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = "simulation/ur5e_gripper.xml"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

TRIALS_PER_LEVEL = 5
NOISE_LEVELS = [0.015, 0.020, 0.025]

ARM_ACTUATORS = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
]

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint",
    "elbow_joint", "wrist_1_joint",
    "wrist_2_joint", "wrist_3_joint",
]

GRIPPER_COMMAND = 255

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

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

gripper_id = model.actuator("fingers_actuator").id
cube_id = model.body("grasp_cube").id
timestep = model.opt.timestep

rng = np.random.default_rng(42)


def cube_position():
    mujoco.mj_forward(model, data)
    return data.xpos[cube_id].copy()


def set_arm_targets(targets, noise_std):
    for actuator, target in zip(ARM_ACTUATORS, targets):
        actuator_id = model.actuator(actuator).id
        noisy_target = target + rng.normal(0, noise_std)

        lower, upper = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = np.clip(noisy_target, lower, upper)


def simulate_steps(steps):
    for _ in range(steps):
        mujoco.mj_step(model, data)


def move_to(targets, duration, gripper_command, noise_std):
    start = np.array([
        data.qpos[model.joint(j).qposadr[0]]
        for j in ARM_JOINTS
    ])

    steps = int(duration / timestep)

    for i in range(steps):
        t = (i + 1) / steps
        smooth = 3 * t**2 - 2 * t**3
        intermediate = start + (targets - start) * smooth

        set_arm_targets(intermediate, noise_std)
        data.ctrl[gripper_id] = gripper_command
        mujoco.mj_step(model, data)


def move_gripper(target, duration):
    start = data.ctrl[gripper_id]
    steps = int(duration / timestep)

    for i in range(steps):
        t = (i + 1) / steps
        smooth = 3 * t**2 - 2 * t**3
        data.ctrl[gripper_id] = start + (target - start) * smooth
        mujoco.mj_step(model, data)


def check_joint_limits():
    violations = []

    for joint_name in ARM_JOINTS:
        joint_id = model.joint(joint_name).id

        if model.jnt_limited[joint_id]:
            address = model.jnt_qposadr[joint_id]
            position = data.qpos[address]
            lower, upper = model.jnt_range[joint_id]

            if position < lower - 1e-5 or position > upper + 1e-5:
                violations.append(joint_name)

    return violations


def run_trial(trial_number, noise_level):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    start_time = time.perf_counter()

    move_to(PICK_HIGH, 12, 0, noise_level)
    move_to(PICK_LOW, 8, 0, noise_level)

    simulate_steps(int(1.0 / timestep))
    baseline = cube_position()

    move_gripper(GRIPPER_COMMAND, 2)
    simulate_steps(int(1.0 / timestep))

    move_to(LIFT, 5, GRIPPER_COMMAND, noise_level)
    after_lift = cube_position()

    move_to(PLACE_HIGH, 7, GRIPPER_COMMAND, noise_level)
    after_transport = cube_position()

    move_to(PLACE_LOW, 5, GRIPPER_COMMAND, noise_level)

    move_gripper(0, 2)
    simulate_steps(int(2.0 / timestep))

    move_to(PLACE_HIGH, 5, 0, noise_level)
    final_position = cube_position()

    violations = check_joint_limits()

    lift_height = float(after_lift[2] - baseline[2])
    retention_height = float(after_transport[2] - baseline[2])

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
        and len(violations) == 0
    )

    elapsed = time.perf_counter() - start_time

    result = {
        "trial": trial_number,
        "noise_level": noise_level,
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
        "joint_limit_violations": len(violations),
        "execution_time_sec": elapsed,
    }

    print(
        f"Noise {noise_level:.1%} | "
        f"Trial {trial_number}/{TRIALS_PER_LEVEL}: "
        f"{'SUCCESS' if overall_success else 'FAIL'} | "
        f"Lift: {lift_height:.4f} m | "
        f"Placement error: {placement_error:.4f} m"
    )

    return result


print("=" * 65)
print("PICK-AND-PLACE THRESHOLD EXPERIMENT")
print("=" * 65)

results = []

for noise_level in NOISE_LEVELS:
    for trial in range(1, TRIALS_PER_LEVEL + 1):
        results.append(run_trial(trial, noise_level))

summary = []

for noise_level in NOISE_LEVELS:
    group = [r for r in results if r["noise_level"] == noise_level]
    successful = sum(r["overall_success"] for r in group)

    summary.append({
        "noise_level": noise_level,
        "trials": len(group),
        "successful_trials": int(successful),
        "success_rate": float(successful / len(group)),
        "mean_lift_height_m": float(np.mean(
            [r["lift_height_m"] for r in group]
        )),
        "mean_placement_error_m": float(np.mean(
            [r["placement_error_m"] for r in group]
        )),
        "joint_limit_violations": int(sum(
            r["joint_limit_violations"] for r in group
        )),
    })

print("\n" + "=" * 65)
print("THRESHOLD SUMMARY")
print("=" * 65)

for item in summary:
    print(
        f"Noise: {item['noise_level']:.1%} | "
        f"Success: {item['successful_trials']}/{item['trials']} | "
        f"Rate: {item['success_rate']:.0%} | "
        f"Mean placement error: "
        f"{item['mean_placement_error_m']:.5f} m | "
        f"Joint violations: {item['joint_limit_violations']}"
    )

csv_path = OUTPUT_DIR / "pick_place_threshold.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

report = {
    "experiment": "actuator_command_noise_threshold",
    "noise_type": "independent Gaussian noise on arm actuator position targets",
    "noise_levels": NOISE_LEVELS,
    "trials_per_level": TRIALS_PER_LEVEL,
    "random_seed": 42,
    "summary": summary,
    "results": results,
}

json_path = OUTPUT_DIR / "pick_place_threshold_report.json"

with open(json_path, "w") as f:
    json.dump(report, f, indent=4)

print("\nGenerated files:")
print(csv_path)
print(json_path)
