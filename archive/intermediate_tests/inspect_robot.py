

import mujoco

model = mujoco.MjModel.from_xml_path(
    "simulation/ur5e_gripper.xml"
)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

sites = ["attachment_site", "gripper_pinch"]

for name in sites:
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, name
    )

    if site_id == -1:
        print(f"{name}: NOT FOUND")
        continue

    print(f"{name} position:", data.site_xpos[site_id])

cube_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "grasp_cube"
)

print("Cube position:", data.xpos[cube_id])

print("\nJoint positions:")
for i in range(model.njnt):
    print(model.joint(i).name, data.qpos[model.jnt_qposadr[i]])
