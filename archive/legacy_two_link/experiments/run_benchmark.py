
import csv
import numpy as np
import mujoco
from pathlib import Path

from simulation.inverse_kinematics import InverseKinematics


# Project paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "simulation" / "robot.xml"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# Load simulation
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

ik = InverseKinematics()

# Experiment configuration
TARGETS = [
    (0.55, 0.25),
    (0.65, 0.00),
    (0.45, 0.35),
    (0.45, -0.25),
    (0.30, 0.40),
]

DURATION = 10.0
SUCCESS_THRESHOLD = 0.01  # 10 mm
RAMP_DURATION = 2.0  # seconds to smoothly reach target joint angles

STEPS = int(DURATION / model.opt.timestep)

# Find end-effector
ee_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "end_effector"
)

telemetry_path = RESULTS_DIR / "telemetry.csv"
summary_path = RESULTS_DIR / "benchmark_summary.csv"

telemetry = []
summaries = []

print("\nMANIPULATIONLAB BENCHMARK")
print("=" * 55)

for trial_id, (target_x, target_y) in enumerate(TARGETS, start=1):

    # Calculate inverse kinematics
    shoulder_target, elbow_target = ik.calculate(target_x, target_y)

    # Check actuator limits
    if not (
        model.actuator_ctrlrange[0, 0]
        <= shoulder_target
        <= model.actuator_ctrlrange[0, 1]
    ):
        raise ValueError(f"Trial {trial_id}: shoulder angle exceeds limits")

    if not (
        model.actuator_ctrlrange[1, 0]
        <= elbow_target
        <= model.actuator_ctrlrange[1, 1]
    ):
        raise ValueError(f"Trial {trial_id}: elbow angle exceeds limits")

    # Reset simulation for this trial
    mujoco.mj_resetData(model, data)

    # Store initial joint positions
    start_positions = data.qpos[:2].copy()
    target_positions = np.array([shoulder_target, elbow_target])
    errors = []

    # Run experiment with smooth ramping
    for step in range(STEPS):
        elapsed = step * model.opt.timestep

        # Smoothly increase command from initial to target
        alpha = min(elapsed / RAMP_DURATION, 1.0)
        smooth_alpha = 3 * alpha**2 - 2 * alpha**3

        command = start_positions + smooth_alpha * (target_positions - start_positions)
        data.ctrl[0] = command[0]
        data.ctrl[1] = command[1]

        mujoco.mj_step(model, data)

        position = data.site_xpos[ee_id].copy()
        actual_x = float(position[0])
        actual_y = float(position[1])
        actual_z = float(position[2])

        error = float(np.sqrt(
            (target_x - actual_x) ** 2
            + (target_y - actual_y) ** 2
        ))
        errors.append(error)

        telemetry.append({
            "trial_id": trial_id,
            "simulation_time": float(data.time),
            "target_x": target_x,
            "target_y": target_y,
            "actual_x": actual_x,
            "actual_y": actual_y,
            "actual_z": actual_z,
            "shoulder_position": float(data.qpos[0]),
            "elbow_position": float(data.qpos[1]),
            "shoulder_velocity": float(data.qvel[0]),
            "elbow_velocity": float(data.qvel[1]),
            "shoulder_control": float(data.ctrl[0]),
            "elbow_control": float(data.ctrl[1]),
            "position_error": error
        })

    final_error = errors[-1]
    mean_error = float(np.mean(errors))
    success = final_error <= SUCCESS_THRESHOLD

    summaries.append({
        "trial_id": trial_id,
        "target_x": target_x,
        "target_y": target_y,
        "final_error_m": final_error,
        "mean_error_m": mean_error,
        "success": success
    })

    print(
        f"Trial {trial_id}: "
        f"Final error = {final_error:.5f} m | "
        f"Mean error = {mean_error:.5f} m | "
        f"Success = {success}"
    )


# Save telemetry dataset
with open(telemetry_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=telemetry[0].keys())
    writer.writeheader()
    writer.writerows(telemetry)

# Save benchmark summary
with open(summary_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
    writer.writeheader()
    writer.writerows(summaries)

print("\nBenchmark results saved successfully.")
print(f"Telemetry: {telemetry_path}")
print(f"Summary: {summary_path}")