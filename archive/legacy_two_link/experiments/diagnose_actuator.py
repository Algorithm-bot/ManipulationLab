
import mujoco
import numpy as np

from simulation.inverse_kinematics import InverseKinematics

model = mujoco.MjModel.from_xml_path("simulation/robot.xml")
data = mujoco.MjData(model)

ik = InverseKinematics()

target_x, target_y = 0.55, 0.25
shoulder, elbow = ik.calculate(target_x, target_y)

mujoco.mj_resetData(model, data)

data.ctrl[0] = shoulder
data.ctrl[1] = elbow

for _ in range(int(10.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)

print("\nACTUATOR DIAGNOSTICS")
print("-" * 45)

print("Commanded angles:", data.ctrl)
print("Actual joint positions:", data.qpos)
print("Joint velocities:", data.qvel)
print("Actuator forces:", data.actuator_force)
print("Generalized actuator forces:", data.qfrc_actuator)
print("Passive forces:", data.qfrc_passive)
print("Bias forces:", data.qfrc_bias)

print("\nACTUATOR CONFIGURATION")
print("Control ranges:", model.actuator_ctrlrange)
print("Force ranges:", model.actuator_forcerange)
print("Gain parameters:", model.actuator_gainprm)
print("Bias parameters:", model.actuator_biasprm)
print("Joint damping:", model.dof_damping)
print("Joint armature:", model.dof_armature)

print("\nADDITIONAL CONTACT DIAGNOSTICS")
print("Number of contacts:", data.ncon)
print("Constraint forces:", data.qfrc_constraint)
print("Actuator force limiting:", model.actuator_forcelimited)
print("Joint positions:", data.qpos)
print("Joint accelerations:", data.qacc)

for i in range(data.ncon):
    contact = data.contact[i]
    print(
        f"Contact {i}:",
        "geom1 =", contact.geom1,
        "geom2 =", contact.geom2,
        "distance =", contact.dist
    )