
import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "simulation/ur5e_gripper.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

ARM_ACTUATORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

HIGH = np.array([
    -3.44408, -1.17748, 1.54195,
     1.20593,  1.57069, -3.44409
])

LOW = np.array([
    -3.44406, -1.04831, 1.67661,
     0.94215,  1.57071, -3.44406
])

# Raise pinch site by approximately 5 cm.
LIFT = np.array([
    -3.44406, -1.12000, 1.59000,
     1.10500,  1.57071, -3.44406
])

TIMESTEP = model.opt.timestep


def set_arm_targets(targets):
    for actuator, target in zip(ARM_ACTUATORS, targets):
        data.ctrl[model.actuator(actuator).id] = target


def move_to(targets, duration, viewer, gripper_command):
    start = np.array([
        data.qpos[model.joint(j).qposadr[0]]
        for j in ARM_JOINTS
    ])

    steps = int(duration / TIMESTEP)
    wall_start = time.perf_counter()

    for i in range(steps):
        t = (i + 1) / steps
        smooth = 3 * t**2 - 2 * t**3
        intermediate = start + (targets - start) * smooth

        set_arm_targets(intermediate)
        data.ctrl[model.actuator("fingers_actuator").id] = gripper_command

        mujoco.mj_step(model, data)
        viewer.sync()

        remaining = wall_start + (i + 1) * TIMESTEP - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)


def get_cube_height():
    return float(data.xpos[model.body("grasp_cube").id][2])


mujoco.mj_resetData(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:

    print("Moving to pre-grasp...")
    move_to(HIGH, 6, viewer, 0)

    print("Descending to cube...")
    move_to(LOW, 6, viewer, 0)

    print("Closing gripper...")
    move_to(LOW, 2, viewer, 255)

    cube_before = get_cube_height()

    print("\nCube height before lift:", cube_before)

    print("Lifting arm...")
    move_to(LIFT, 5, viewer, 255)

    cube_after = get_cube_height()

    print("\nLIFT TEST RESULTS")
    print("------------------------------")
    print("Cube height before:", cube_before)
    print("Cube height after: ", cube_after)
    print("Cube vertical movement:", cube_after - cube_before)

    if cube_after - cube_before > 0.03:
        print("RESULT: Cube lifted successfully.")
    else:
        print("RESULT: Stable lift not yet demonstrated.")

    print("\nInspect the result in the viewer.")

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
