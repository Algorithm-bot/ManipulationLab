
import time
import mujoco
import mujoco.viewer

MODEL_PATH = "simulation/ur5e_gripper.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

actuator_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator"
)

if actuator_id == -1:
    raise RuntimeError("Could not find fingers_actuator")

print("GRIPPER TEST")
print("Actuator:", actuator_id)
print("Control range:", model.actuator_ctrlrange[actuator_id])

def run_for(viewer, command, duration):
    data.ctrl[actuator_id] = command
    start = time.time()

    while viewer.is_running() and time.time() - start < duration:
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("\n1. Opening gripper...")
    run_for(viewer, 0, 3)

    print("2. Closing gripper...")
    run_for(viewer, 255, 3)

    print("3. Opening gripper again...")
    run_for(viewer, 0, 3)

    print("\nTest sequence completed.")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
