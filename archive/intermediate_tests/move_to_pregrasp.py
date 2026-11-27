
import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "simulation/ur5e_gripper.xml"

ARM_ACTUATORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3"
]

TARGET = np.array([
    -3.44408,
    -1.17748,
     1.54195,
     1.20593,
     1.57069,
    -3.44409
])

DURATION = 12.0

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

actuator_ids = [
    mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
    )
    for name in ARM_ACTUATORS
]

joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]

joint_ids = [
    mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name
    )
    for name in joint_names
]

if -1 in actuator_ids or -1 in joint_ids:
    raise RuntimeError("Missing arm actuator or joint")

qpos_ids = [
    model.jnt_qposadr[j] for j in joint_ids
]

initial = data.qpos[qpos_ids].copy()

# Keep gripper open during arm movement.
gripper_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR,
    "fingers_actuator"
)

data.ctrl[gripper_id] = 0

site_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE,
    "gripper_pinch"
)

print("Moving UR5e to pre-grasp pose...")
print("Initial joints:", initial)
print("Target joints:", TARGET)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    while viewer.is_running():
        elapsed = time.time() - start
        progress = min(elapsed / DURATION, 1.0)

        # Smoothstep interpolation.
        blend = 3 * progress**2 - 2 * progress**3

        command = initial + blend * (TARGET - initial)

        for actuator_id, value in zip(actuator_ids, command):
            data.ctrl[actuator_id] = value

        mujoco.mj_step(model, data)
        viewer.sync()

        if progress >= 1.0:
            break

        time.sleep(model.opt.timestep)

    # Allow the robot to settle.
    for _ in range(2000):
        if not viewer.is_running():
            break
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)

    mujoco.mj_forward(model, data)
    print("Gripper orientation:")
    print(data.site_xmat[site_id].reshape(3, 3))

    print("\nMOVEMENT RESULTS")
    print("================")
    print("Final joint positions:", data.qpos[qpos_ids])
    print("Gripper pinch position:", data.site_xpos[site_id])
    print("Target position: [0.45, 0.0, 0.16]")
    print("Movement sequence completed.")

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
