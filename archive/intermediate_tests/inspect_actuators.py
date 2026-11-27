
import mujoco

model = mujoco.MjModel.from_xml_path(
    "simulation/ur5e_gripper.xml"
)

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("UR5e ACTUATOR INSPECTION")
print("=" * 55)

for i in range(model.nu):
    actuator = model.actuator(i)
    name = actuator.name

    print(f"\nActuator: {name}")
    print("Control range:", model.actuator_ctrlrange[i])
    print("Control limited:", model.actuator_ctrllimited[i])
    print("Gain parameters:", model.actuator_gainprm[i][:3])
    print("Bias parameters:", model.actuator_biasprm[i][:3])
    print("Transmission type:", model.actuator_trntype[i])

site_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, "gripper_pinch"
)

print("\nGRIPPER ORIENTATION")
print(data.site_xmat[site_id].reshape(3, 3))

print("\nINSPECTION COMPLETE")
