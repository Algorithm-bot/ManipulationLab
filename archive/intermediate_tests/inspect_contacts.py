
import mujoco

model = mujoco.MjModel.from_xml_path(
    "simulation/ur5e_gripper.xml"
)
data = mujoco.MjData(model)

# Reproduce the final approach configuration.
joints = [
    -3.44405, -1.09412, 1.64373,
     1.02088, 1.57072, -3.44406
]

names = [
    "shoulder_pan_joint", "shoulder_lift_joint",
    "elbow_joint", "wrist_1_joint",
    "wrist_2_joint", "wrist_3_joint"
]

for name, value in zip(names, joints):
    jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name
    )
    data.qpos[model.jnt_qposadr[jid]] = value

mujoco.mj_forward(model, data)

print("ACTIVE CONTACTS")
print("================")

for i in range(data.ncon):
    contact = data.contact[i]

    geom1 = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
    )
    geom2 = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
    )

    print(f"{i+1}. {geom1} <-> {geom2}")
    print(f"   Distance: {contact.dist:.6f} m")
