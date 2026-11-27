
import json
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from simulation.inverse_kinematics import InverseKinematics
from experiments.system_identification import simulate_trajectory


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

# Load benchmark data
df = pd.read_csv(RESULTS / "telemetry.csv")
summary = pd.read_csv(RESULTS / "benchmark_summary.csv")

# Consistent plotting defaults
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3
})


def save_plot(filename):
    path = FIGURES / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# -------------------------------------------------
# 1. End-effector trajectories
# -------------------------------------------------

plt.figure()

for trial_id, group in df.groupby("trial_id"):
    plt.plot(
        group["actual_x"],
        group["actual_y"],
        label=f"Trial {trial_id}"
    )

    target_x = group["target_x"].iloc[0]
    target_y = group["target_y"].iloc[0]

    plt.scatter(target_x, target_y, marker="x", s=80)

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Robotic Arm End-Effector Trajectories")
plt.legend()
plt.axis("equal")

save_plot("end_effector_trajectories.png")


# -------------------------------------------------
# 2. Joint positions over time
# -------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for trial_id, group in df.groupby("trial_id"):
    axes[0].plot(
        group["simulation_time"],
        group["shoulder_position"],
        label=f"Trial {trial_id}"
    )

    axes[1].plot(
        group["simulation_time"],
        group["elbow_position"],
        label=f"Trial {trial_id}"
    )

axes[0].axhline(1.57, linestyle="--", label="Upper limit")
axes[0].axhline(-1.57, linestyle="--", label="Lower limit")
axes[0].set_ylabel("Shoulder Angle (rad)")
axes[0].set_title("Shoulder Joint Position")
axes[0].legend()

axes[1].axhline(2.0, linestyle="--", label="Upper limit")
axes[1].axhline(-2.0, linestyle="--", label="Lower limit")
axes[1].set_ylabel("Elbow Angle (rad)")
axes[1].set_xlabel("Simulation Time (s)")
axes[1].set_title("Elbow Joint Position")
axes[1].legend()

save_plot("joint_positions.png")


# -------------------------------------------------
# 3. Joint velocities over time
# -------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for trial_id, group in df.groupby("trial_id"):
    axes[0].plot(
        group["simulation_time"],
        group["shoulder_velocity"],
        label=f"Trial {trial_id}"
    )

    axes[1].plot(
        group["simulation_time"],
        group["elbow_velocity"],
        label=f"Trial {trial_id}"
    )

axes[0].set_ylabel("Shoulder Velocity (rad/s)")
axes[0].set_title("Shoulder Joint Velocity")
axes[0].legend()

axes[1].set_ylabel("Elbow Velocity (rad/s)")
axes[1].set_xlabel("Simulation Time (s)")
axes[1].set_title("Elbow Joint Velocity")
axes[1].legend()

save_plot("joint_velocities.png")


# -------------------------------------------------
# 4. Position error across trials
# -------------------------------------------------

plt.figure()

for trial_id, group in df.groupby("trial_id"):
    plt.plot(
        group["simulation_time"],
        group["position_error"] * 1000,
        label=f"Trial {trial_id}"
    )

plt.xlabel("Simulation Time (s)")
plt.ylabel("Position Error (mm)")
plt.title("End-Effector Position Error")
plt.legend()

save_plot("position_error.png")


# -------------------------------------------------
# 5. System identification comparison
# -------------------------------------------------

with open(RESULTS / "system_identification_report.json") as file:
    report = json.load(file)

reference = pd.read_csv(RESULTS / "reference_trajectory.csv")

estimated_kp = report["estimated_kp"]

times, predicted = simulate_trajectory(estimated_kp)

plt.figure()

plt.plot(
    reference["simulation_time"],
    reference["elbow_position_rad"],
    label="Reference trajectory",
    alpha=0.7
)

plt.plot(
    times,
    predicted,
    linestyle="--",
    label=f"Estimated model (kp={estimated_kp:.3f})"
)

plt.xlabel("Simulation Time (s)")
plt.ylabel("Elbow Position (rad)")
plt.title("System Identification: Reference vs Estimated Model")
plt.legend()

save_plot("system_identification.png")


# -------------------------------------------------
# Final report
# -------------------------------------------------

print("\nMANIPULATIONLAB VISUALIZATION")
print("=" * 45)
print(f"Trials visualized: {df['trial_id'].nunique()}")
print(f"Telemetry samples: {len(df)}")
print(f"Successful trials: {summary['success'].sum()}/{len(summary)}")
print(f"Estimated elbow gain: {estimated_kp:.4f}")
print(f"\nAll plots saved in: {FIGURES}")
