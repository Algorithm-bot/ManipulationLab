
import time
import numpy as np
import mujoco
import mujoco.viewer

from simulation.inverse_kinematics import InverseKinematics


# Load robot
model = mujoco.MjModel.from_xml_path("simulation/robot.xml")
data = mujoco.MjData(model)

# Initialize inverse kinematics
ik = InverseKinematics()

# Target coordinates
target_x = 0.55
target_y = 0.25

# Calculate required joint angles
shoulder_angle, elbow_angle = ik.calculate(
    target_x, target_y
)

print("Target:", target_x, target_y)
print("Required shoulder angle:", round(shoulder_angle, 4))
print("Required elbow angle:", round(elbow_angle, 4))

# Find end-effector site
ee_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "end_effector"
)

start_time = time.time()
last_report = 0

print("\nStarting target-reaching simulation...")

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # Send target joint angles to actuators
        data.ctrl[0] = shoulder_angle
        data.ctrl[1] = elbow_angle

        mujoco.mj_step(model, data)
        viewer.sync()

        # Calculate actual end-effector position
        actual_position = data.site_xpos[ee_id].copy()

        actual_x = actual_position[0]
        actual_y = actual_position[1]

        # Calculate target position error
        error = np.sqrt(
            (target_x - actual_x)**2
            + (target_y - actual_y)**2
        )

        elapsed = time.time() - start_time

        # Report progress once per second
        if elapsed - last_report >= 1:
            print(
                f"Position: ({actual_x:.3f}, {actual_y:.3f}) | "
                f"Error: {error:.5f} m"
            )
            last_report = elapsed

        time.sleep(model.opt.timestep)

print("Simulation completed.")
