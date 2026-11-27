
import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "simulation/ur5e_gripper.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

ARM_ACTUATORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

GRIPPER = model.actuator("fingers_actuator").id
CUBE = model.body("grasp_cube").id

TIMESTEP = model.opt.timestep

# Original cube pickup configuration
PICK_HIGH = np.array([
    -3.44408, -1.17748, 1.54195,
     1.20593,  1.57069, -3.44409
])

PICK_LOW = np.array([
    -3.44406, -1.04831, 1.67661,
     0.94215,  1.57071, -3.44406
])

# Previously tested lifting configuration
LIFT = np.array([
    -3.44406, -1.12000, 1.59000,
     1.10500,  1.57071, -3.44406
])

# Destination approach: [0.60, 0.15, 0.12]
PLACE_HIGH = np.array([
    -3.11503, -1.19326, 1.67689,
    -2.05453, -1.57077, 0.02656
])

# Destination placement: [0.60, 0.15, 0.045]
PLACE_LOW = np.array([
    -3.11505, -1.09705, 1.75718,
    -2.23111, -1.57075, 0.02654
])


def set_arm(targets):
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

        set_arm(intermediate)
        data.ctrl[GRIPPER] = gripper_command

        mujoco.mj_step(model, data)
        viewer.sync()

        remaining = (
            wall_start + (i + 1) * TIMESTEP
            - time.perf_counter()
        )

        if remaining > 0:
            time.sleep(remaining)


def move_gripper(target, duration, viewer):
    start = data.ctrl[GRIPPER]
    steps = int(duration / TIMESTEP)
    wall_start = time.perf_counter()

    for i in range(steps):
        t = (i + 1) / steps
        smooth = 3 * t**2 - 2 * t**3

        data.ctrl[GRIPPER] = start + (target - start) * smooth

        mujoco.mj_step(model, data)
        viewer.sync()

        remaining = (
            wall_start + (i + 1) * TIMESTEP
            - time.perf_counter()
        )

        if remaining > 0:
            time.sleep(remaining)


def hold(duration, viewer):
    steps = int(duration / TIMESTEP)
    wall_start = time.perf_counter()

    for i in range(steps):
        mujoco.mj_step(model, data)
        viewer.sync()

        remaining = (
            wall_start + (i + 1) * TIMESTEP
            - time.perf_counter()
        )

        if remaining > 0:
            time.sleep(remaining)


def cube_position():
    mujoco.mj_forward(model, data)
    return data.xpos[CUBE].copy()


mujoco.mj_resetData(model, data)

initial_cube_position = cube_position()

with mujoco.viewer.launch_passive(model, data) as viewer:

    print("\n========== PICK AND PLACE ==========\n")

    print("1. Approaching cube...")
    move_to(PICK_HIGH, 12, viewer, 0)

    print("2. Descending to cube...")
    move_to(PICK_LOW, 8, viewer, 0)

    print("3. Closing gripper...")
    move_gripper(255, 2, viewer)

    hold(1, viewer)

    print("4. Lifting cube...")
    move_to(LIFT, 5, viewer, 255)

    print("5. Transporting cube to destination...")
    move_to(PLACE_HIGH, 7, viewer, 255)

    print("6. Lowering cube...")
    move_to(PLACE_LOW, 5, viewer, 255)

    print("7. Releasing cube...")
    move_gripper(0, 2, viewer)

    hold(2, viewer)

    released_position = cube_position()

    print("8. Retreating...")
    move_to(PLACE_HIGH, 5, viewer, 0)

    final_position = cube_position()

    displacement = final_position - initial_cube_position

    print("\n========== EXPERIMENT RESULTS ==========")
    print("Initial cube position:", initial_cube_position)
    print("Position after release:", released_position)
    print("Final cube position:", final_position)
    print("Cube displacement:", displacement)

    print("\nExpected destination: [0.60, 0.15, approximately 0.02]")

    if (
        abs(final_position[0] - 0.60) < 0.03
        and abs(final_position[1] - 0.15) < 0.03
        and abs(final_position[2] - 0.02) < 0.015
    ):
        print("RESULT: PICK AND PLACE SUCCESS")
    else:
        print("RESULT: PLACEMENT NEEDS INVESTIGATION")

    print("\nViewer will remain open.")

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
