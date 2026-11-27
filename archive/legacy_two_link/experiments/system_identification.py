
import json
import re
import numpy as np
import pandas as pd
import mujoco

from pathlib import Path
from scipy.optimize import minimize_scalar

from simulation.inverse_kinematics import InverseKinematics


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "simulation" / "robot.xml"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# Experimental configuration
REFERENCE_KP = 14.0       # Known only when generating reference data
DURATION = 4.0
RAMP_DURATION = 2.0
NOISE_STD = 0.0005        # Simulated measurement noise in radians

TARGET_X = 0.55
TARGET_Y = 0.25

REFERENCE_FILE = RESULTS_DIR / "reference_trajectory.csv"
REPORT_FILE = RESULTS_DIR / "system_identification_report.json"


def build_model(elbow_kp):
    """Create a MuJoCo model with a specified elbow motor gain."""

    xml = MODEL_PATH.read_text(encoding="utf-8")

    pattern = (
        r'(<position\s+name="elbow_motor"'
        r'[^>]*\bkp=")[^"]+(")'
    )

    modified_xml, replacements = re.subn(
        pattern,
        lambda match: f"{match.group(1)}{elbow_kp}{match.group(2)}",
        xml
    )

    if replacements != 1:
        raise ValueError("Could not locate elbow motor kp in robot.xml")

    return mujoco.MjModel.from_xml_string(modified_xml)


def simulate_trajectory(elbow_kp):
    """Simulate the robot and return its elbow position over time."""

    model = build_model(elbow_kp)
    data = mujoco.MjData(model)

    ik = InverseKinematics()
    shoulder_target, elbow_target = ik.calculate(TARGET_X, TARGET_Y)

    start_positions = data.qpos[:2].copy()
    target_positions = np.array([shoulder_target, elbow_target])

    steps = int(DURATION / model.opt.timestep)

    times = []
    elbow_positions = []

    for step in range(steps):

        elapsed = step * model.opt.timestep

        alpha = min(elapsed / RAMP_DURATION, 1.0)
        smooth_alpha = 3 * alpha**2 - 2 * alpha**3

        command = start_positions + smooth_alpha * (
            target_positions - start_positions
        )

        data.ctrl[0] = command[0]
        data.ctrl[1] = command[1]

        mujoco.mj_step(model, data)

        times.append(data.time)
        elbow_positions.append(data.qpos[1])

    return np.array(times), np.array(elbow_positions)


def generate_reference():
    """Generate a reference dataset with a known motor gain."""

    times, positions = simulate_trajectory(REFERENCE_KP)

    rng = np.random.default_rng(42)
    noisy_positions = positions + rng.normal(
        0, NOISE_STD, size=len(positions)
    )

    df = pd.DataFrame({
        "simulation_time": times,
        "elbow_position_rad": noisy_positions
    })

    df.to_csv(REFERENCE_FILE, index=False)

    print(f"Reference trajectory generated: {len(df)} samples")
    print(f"Saved to: {REFERENCE_FILE}")


def identify_gain():
    """Estimate the unknown elbow motor gain from reference data."""

    reference = pd.read_csv(REFERENCE_FILE)

    observed = reference["elbow_position_rad"].to_numpy()

    def objective(candidate_kp):

        _, predicted = simulate_trajectory(candidate_kp)

        return float(np.mean((observed - predicted) ** 2))

    # Baseline error using the current XML gain
    baseline_error = objective(10.0)

    result = minimize_scalar(
        objective,
        bounds=(5.0, 25.0),
        method="bounded",
        options={"xatol": 0.001}
    )

    estimated_kp = float(result.x)
    final_error = float(result.fun)

    report = {
        "parameter": "elbow_motor_kp",
        "reference_kp": REFERENCE_KP,
        "estimated_kp": estimated_kp,
        "absolute_estimation_error": abs(REFERENCE_KP - estimated_kp),
        "baseline_kp": 10.0,
        "baseline_mse": baseline_error,
        "optimized_mse": final_error,
        "optimizer_success": bool(result.success),
        "reference_samples": len(reference)
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\nSYSTEM IDENTIFICATION RESULTS")
    print("=" * 45)
    print(f"Reference gain:       {REFERENCE_KP:.4f}")
    print(f"Estimated gain:       {estimated_kp:.4f}")
    print(f"Absolute error:       {abs(REFERENCE_KP - estimated_kp):.4f}")
    print(f"Baseline MSE (kp=10): {baseline_error:.8f}")
    print(f"Optimized MSE:        {final_error:.8f}")
    print(f"Optimizer successful: {result.success}")
    print(f"\nReport saved: {REPORT_FILE}")


if __name__ == "__main__":
    generate_reference()
    identify_gain()
