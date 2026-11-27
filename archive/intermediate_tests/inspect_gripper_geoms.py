
import mujoco

model = mujoco.MjModel.from_xml_path("simulation/ur5e_gripper.xml")

print("ALL MODEL GEOMETRIES")
print("=" * 90)

for i in range(model.ngeom):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, i
    )

    geom = model.geom(i)

    print(
        f"ID: {i:3d} | "
        f"Name: {str(name):45s} | "
        f"Type: {geom.type} | "
        f"Size: {geom.size}"
    )
