
import csv
import time
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = "simulation/ur5e_gripper.xml"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

NOISE_LEVELS = [0.015, 0.020, 0.025]
TRIALS_PER_LEVEL = 3
RANDOM_SEED = 42

ARM_ACTUATORS = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
]

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint",
    "elbow_joint", "wrist_1_joint",
    "wrist_2_joint", "wrist_3_joint",
]

PAD_NAMES = {
    "right_pad1", "right_pad2",
    "left_pad1", "left_pad2",
}

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

cube_id = model.body("grasp_cube").id
cube_geom_id = model.geom("cube_geom").id
gripper_id = model.actuator("fingers_actuator").id

timestep = model.opt.timestep
rng = np.random.default_rng(RANDOM_SEED)


def cube_position():
    mujoco.mj_forward(model, data)
    return data.xpos[cube_id].copy()


def get_pad_contacts():
    contacts = []

    for i in range(data.ncon):
        contact = data.contact[i]

        if cube_geom_id not in (contact.geom1, contact.geom2):
            continue

        other_id = (
            contact.geom2 if contact.geom1 == cube_geom_id
            else contact.geom1
        )

        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, other_id
        )

        if name in PAD_NAMES:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)

            contacts.append({
                "pad": name,
                "distance_m": float(contact.dist),
                "normal_force_N": float(abs(force[0])),
            })

    return contacts


def contact_metrics():
    contacts = get_pad_contacts()

    return {
        "contact_count": len(contacts),
        "contact_pads": ";".join(
            sorted(set(c["pad"] for c in contacts))
        ),
        "total_normal_force_N": sum(
            c["normal_force_N"] for c in contacts
        ),
        "minimum_contact_distance_m": min(
            [c["distance_m"] for c in contacts],
            default=float("nan"),
        ),
    }


def set_arm_targets(targets, noise_std):
    for actuator, target in zip(ARM_ACTUATORS, targets):
        actuator_id = model.actuator(actuator).id

        noisy_target = target + rng.normal(0, noise_std)

        lower, upper = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = np.clip(
            noisy_target, lower, upper
        )


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

        data.ctrl[gripper_id] = start + (
            target - start
        ) * smooth

        mujoco.mj_step(model, data)


def record_stage(rows, noise, trial, stage):
    position = cube_position()
    metrics = contact_metrics()

    row = {
        "noise_level": noise,
        "trial": trial,
        "stage": stage,
        "cube_x_m": float(position[0]),
        "cube_y_m": float(position[1]),
        "cube_z_m": float(position[2]),
        **metrics,
    }

    rows.append(row)

    print(
        f"{noise:.1%} | Trial {trial} | {stage:15s} | "
        f"Contacts: {metrics['contact_count']} | "
        f"Pads: {metrics['contact_pads']} | "
        f"Force: {metrics['total_normal_force_N']:.3f} N | "
        f"Cube Z: {position[2]:.4f} m"
    )


def run_trial(noise, trial, rows):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    move_to(PICK_HIGH, 12, 0, noise)
    move_to(PICK_LOW, 8, 0, noise)

    mujoco.mj_step(model, data)

    record_stage(rows, noise, trial, "pre_grasp")

    move_gripper(GRIPPER_COMMAND, 2)
    for _ in range(int(1.0 / timestep)):
        mujoco.mj_step(model, data)

    record_stage(rows, noise, trial, "closed_gripper")

    move_to(LIFT, 5, GRIPPER_COMMAND, noise)
    record_stage(rows, noise, trial, "after_lift")

    move_to(PLACE_HIGH, 7, GRIPPER_COMMAND, noise)
    record_stage(rows, noise, trial, "after_transport")

    move_to(PLACE_LOW, 5, GRIPPER_COMMAND, noise)
    record_stage(rows, noise, trial, "at_placement")

    move_gripper(0, 2)
    for _ in range(int(2.0 / timestep)):
        mujoco.mj_step(model, data)

    record_stage(rows, noise, trial, "after_release")

    move_to(PLACE_HIGH, 5, 0, noise)
    record_stage(rows, noise, trial, "final")


print("=" * 75)
print("GRASP FAILURE DIAGNOSTIC")
print("=" * 75)

rows = []

for noise in NOISE_LEVELS:
    for trial in range(1, TRIALS_PER_LEVEL + 1):
        run_trial(noise, trial, rows)

csv_path = OUTPUT_DIR / "grasp_failure_diagnostic.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nDiagnostic saved:", csv_path)
