
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

PAD_NAMES = {
    "right_pad1", "right_pad2",
    "left_pad1", "left_pad2"
}

TIMESTEP = model.opt.timestep


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

        for actuator, target in zip(ARM_ACTUATORS, intermediate):
            data.ctrl[model.actuator(actuator).id] = target

        data.ctrl[model.actuator("fingers_actuator").id] = 0

        mujoco.mj_step(model, data)
        viewer.sync()

        remaining = wall_start + (i + 1) * TIMESTEP - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)


def get_pad_contacts():
    contacts = []

    cube_id = model.geom("cube_geom").id

    for i in range(data.ncon):
        contact = data.contact[i]

        if cube_id not in (contact.geom1, contact.geom2):
            continue

        other_id = (
            contact.geom2 if contact.geom1 == cube_id
            else contact.geom1
        )

        other_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, other_id
        )

        if other_name in PAD_NAMES:
            contacts.append((other_name, contact.dist))

    return contacts


mujoco.mj_resetData(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:

    print("Moving to pre-grasp position...")
    move_to(HIGH, 6, viewer)

    print("Descending to grasp position...")
    move_to(LOW, 6, viewer)

    print("Closing gripper gradually...")

    steps = int(3.0 / TIMESTEP)
    wall_start = time.perf_counter()
    max_contacts = 0

    for i in range(steps):
        t = (i + 1) / steps
        command = 255 * t

        data.ctrl[model.actuator("fingers_actuator").id] = command

        mujoco.mj_step(model, data)
        viewer.sync()

        contacts = get_pad_contacts()
        max_contacts = max(max_contacts, len(contacts))

        remaining = wall_start + (i + 1) * TIMESTEP - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    mujoco.mj_forward(model, data)

    print("\nGRASP CONTACT RESULTS")
    print("------------------------------")
    print("Maximum pad contacts:", max_contacts)

    final_contacts = get_pad_contacts()

    for name, distance in final_contacts:
        print(f"{name}: distance = {distance:.6f} m")

    print("Cube position:", data.xpos[model.body("grasp_cube").id])
    print("Gripper command:", data.ctrl[model.actuator("fingers_actuator").id])

    print("\nInspect the closed gripper.")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
