
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

TIMESTEP = model.opt.timestep


def set_arm_targets(targets):
    for actuator, target in zip(ARM_ACTUATORS, targets):
        data.ctrl[model.actuator(actuator).id] = target

    # Keep gripper open
    data.ctrl[model.actuator("fingers_actuator").id] = 0


def move_to(targets, duration, viewer):
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

        mujoco.mj_step(model, data)
        viewer.sync()

        # Pace simulation to real time
        target_wall_time = wall_start + (i + 1) * TIMESTEP
        remaining = target_wall_time - time.perf_counter()

        if remaining > 0:
            time.sleep(remaining)


mujoco.mj_resetData(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:

    print("Moving to pre-grasp position over 12 seconds...")
    move_to(HIGH, 8, viewer)

    print("Descending to grasp position over 8 seconds...")
    move_to(LOW, 8, viewer)

    mujoco.mj_forward(model, data)

    pinch_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "gripper_pinch"
    )

    cube_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "grasp_cube"
    )

    print("\nFINAL POSITION")
    print("Pinch:", data.site_xpos[pinch_id])
    print("Cube:", data.xpos[cube_id])
    print("Gripper: OPEN")

    print("\nRobot is holding its final position.")
    print("Close the viewer when finished.")

    while viewer.is_running():
        data.ctrl[model.actuator("fingers_actuator").id] = 0
        viewer.sync()
        time.sleep(0.01)
