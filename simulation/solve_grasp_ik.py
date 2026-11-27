
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

MODEL_PATH = "simulation/ur5e_gripper.xml"
TARGET = np.array([0.60, 0.15, 0.12])
# Desired gripper orientation: local Z points downward.
TARGET_ROTATION = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
], dtype=float)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

arm_names = [
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
    for name in arm_names
]

site_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, "gripper_pinch"
)

if site_id == -1 or -1 in joint_ids:
    raise RuntimeError("Required site or arm joint not found")

qpos_ids = [model.jnt_qposadr[j] for j in joint_ids]
dof_ids = [model.jnt_dofadr[j] for j in joint_ids]

lower = model.jnt_range[joint_ids, 0]
upper = model.jnt_range[joint_ids, 1]

jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

position_weight = 1.0
orientation_weight = 0.25
damping = 0.03

for iteration in range(4000):
    mujoco.mj_forward(model, data)

    current_position = data.site_xpos[site_id].copy()
    current_rotation = data.site_xmat[site_id].reshape(3, 3).copy()

    position_error = TARGET - current_position

    rotation_error = Rotation.from_matrix(
        TARGET_ROTATION @ current_rotation.T
    ).as_rotvec()

    if (
        np.linalg.norm(position_error) < 0.001
        and np.linalg.norm(rotation_error) < 0.02
    ):
        break

    mujoco.mj_jacSite(
        model, data, jacp, jacr, site_id
    )

    Jp = jacp[:, dof_ids]
    Jr = jacr[:, dof_ids]

    J = np.vstack([
        position_weight * Jp,
        orientation_weight * Jr
    ])

    error = np.concatenate([
        position_weight * position_error,
        orientation_weight * rotation_error
    ])

    dq = J.T @ np.linalg.solve(
        J @ J.T + damping**2 * np.eye(6),
        error
    )

    dq = np.clip(dq, -0.04, 0.04)

    data.qpos[qpos_ids] += dq
    data.qpos[qpos_ids] = np.clip(
        data.qpos[qpos_ids], lower, upper
    )

mujoco.mj_forward(model, data)

final_position = data.site_xpos[site_id].copy()
final_rotation = data.site_xmat[site_id].reshape(3, 3).copy()

position_error = np.linalg.norm(TARGET - final_position)

orientation_error = np.linalg.norm(
    Rotation.from_matrix(
        TARGET_ROTATION @ final_rotation.T
    ).as_rotvec()
)

print("\nFULL POSE IK RESULTS")
print("====================")
print("Target position:", TARGET)
print("Achieved position:", final_position)
print(f"Position error: {position_error:.6f} m")
print(f"Orientation error: {orientation_error:.6f} rad")
print("Achieved orientation:")
print(final_rotation)

print("\nJoint angles:")
for name, q in zip(arm_names, data.qpos[qpos_ids]):
    print(f"{name}: {q:.5f} rad")

if position_error < 0.001 and orientation_error < 0.02:
    print("\nSUCCESS: Full pose target reached numerically.")
else:
    print("\nWARNING: Full pose target not reached.")
