
import json
import re
import numpy as np
import pandas as pd
import mujoco

from pathlib import Path
from simulation.inverse_kinematics import InverseKinematics


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "simulation" / "robot.xml"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "robustness_results.csv"
OUTPUT_JSON = RESULTS_DIR / "robustness_report.json"

# Experiment configuration
TARGETS = [
    (0.55, 0.25),
    (0.65, 0.00),
    (0.45, 0.35),
    (0.45, -0.25),
    (0.30, 0.40),
]

ELBOW_GAINS = [8, 10, 12, 14, 16]
CONTROL_NOISE_LEVELS = [0.0, 0.005, 0.01]

REPEATS = 3
DURATION = 6.0
RAMP_DURATION = 2.0
SUCCESS_THRESHOLD = 0.01

rng = np.random.default_rng(2026)
ik = InverseKinematics()


def build_model(elbow_kp):
    """Build a MuJoCo model with a specified elbow motor gain."""

    xml = MODEL_PATH.read_text(encoding="utf-8")

    pattern = (
        r'(<position\s+name="elbow_motor"'
        r'[^>]*\bkp=")[^"]+(")'
    )

    modified_xml, count = re.subn(
        pattern,
        lambda match: f"{match.group(1)}{elbow_kp}{match.group(2)}",
        xml
    )

    if count != 1:
        raise ValueError("Could not update elbow motor gain")

    return mujoco.MjModel.from_xml_string(modified_xml)


def run_trial(model, target_x, target_y, noise_std):
    """Run one robot reaching experiment."""

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    shoulder_target, elbow_target = ik.calculate(target_x, target_y)

    start_positions = data.qpos[:2].copy()
    target_positions = np.array([shoulder_target, elbow_target])

    steps = int(DURATION / model.opt.timestep)

    errors = []
    max_shoulder_velocity = 0.0
    max_elbow_velocity = 0.0

    joint_limit_violations = 0

    for step in range(steps):

        elapsed = step * model.opt.timestep

        # Smooth trajectory generation
        alpha = min(elapsed / RAMP_DURATION, 1.0)
        smooth_alpha = 3 * alpha**2 - 2 * alpha**3

        command = start_positions + smooth_alpha * (
            target_positions - start_positions
        )

        # Inject Gaussian control disturbance
        noise = rng.normal(0, noise_std, size=2)
        noisy_command = command + noise

        # Respect actuator command limits
        noisy_command = np.clip(
            noisy_command,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1]
        )

        data.ctrl[:] = noisy_command

        mujoco.mj_step(model, data)

        # End-effector position error
        ee_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "end_effector"
        )

        position = data.site_xpos[ee_id]

        error = np.sqrt(
            (target_x - position[0])**2
            + (target_y - position[1])**2
        )

        errors.append(float(error))

        # Track peak joint velocities
        max_shoulder_velocity = max(
            max_shoulder_velocity,
            abs(float(data.qvel[0]))
        )

        max_elbow_velocity = max(
            max_elbow_velocity,
            abs(float(data.qvel[1]))
        )

        # Count joint position limit violations
        if not (-1.57 <= data.qpos[0] <= 1.57):
            joint_limit_violations += 1

        if not (-2.0 <= data.qpos[1] <= 2.0):
            joint_limit_violations += 1

    final_error = errors[-1]

    return {
        "final_error_m": final_error,
        "mean_error_m": float(np.mean(errors)),
        "success": bool(final_error <= SUCCESS_THRESHOLD),
        "max_shoulder_velocity": max_shoulder_velocity,
        "max_elbow_velocity": max_elbow_velocity,
        "joint_limit_violations": joint_limit_violations,
        "constraint_compliant": bool(joint_limit_violations == 0)
    }


def main():

    results = []

    total_experiments = (
        len(TARGETS)
        * len(ELBOW_GAINS)
        * len(CONTROL_NOISE_LEVELS)
        * REPEATS
    )

    experiment_number = 0

    print("\nMANIPULATIONLAB ROBUSTNESS TEST")
    print("=" * 55)
    print(f"Total experiments: {total_experiments}")

    for gain in ELBOW_GAINS:

        model = build_model(gain)

        for noise_std in CONTROL_NOISE_LEVELS:

            for target_x, target_y in TARGETS:

                for repeat in range(1, REPEATS + 1):

                    experiment_number += 1

                    metrics = run_trial(
                        model,
                        target_x,
                        target_y,
                        noise_std
                    )

                    record = {
                        "experiment_id": experiment_number,
                        "elbow_kp": gain,
                        "control_noise_std": noise_std,
                        "target_x": target_x,
                        "target_y": target_y,
                        "repeat": repeat,
                        **metrics
                    }

                    results.append(record)

                    print(
                        f"[{experiment_number}/{total_experiments}] "
                        f"kp={gain}, noise={noise_std:.3f}, "
                        f"target=({target_x}, {target_y}), "
                        f"error={metrics['final_error_m']:.5f} m, "
                        f"success={metrics['success']}"
                    )

    # Save individual experiment results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    # Aggregate by actuator gain and disturbance
    grouped = df.groupby(
        ["elbow_kp", "control_noise_std"]
    ).agg(
        experiments=("success", "count"),
        success_rate=("success", "mean"),
        mean_final_error=("final_error_m", "mean"),
        max_final_error=("final_error_m", "max"),
        mean_elbow_velocity=("max_elbow_velocity", "mean"),
        joint_limit_violations=("joint_limit_violations", "sum"),
        constraint_compliance_rate=("constraint_compliant", "mean")
    ).reset_index()

    report = {
        "total_experiments": len(df),
        "overall_success_rate": float(df["success"].mean()),
        "overall_constraint_compliance_rate": float(
            df["constraint_compliant"].mean()
        ),
        "overall_mean_final_error_m": float(
            df["final_error_m"].mean()
        ),
        "total_joint_limit_violations": int(
            df["joint_limit_violations"].sum()
        ),
        "configuration_results": grouped.to_dict(orient="records")
    }

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\nROBUSTNESS TEST SUMMARY")
    print("=" * 55)
    print(f"Experiments: {len(df)}")
    print(f"Overall success rate: {df['success'].mean() * 100:.2f}%")
    print(
        "Constraint compliance: "
        f"{df['constraint_compliant'].mean() * 100:.2f}%"
    )
    print(
        "Mean final error: "
        f"{df['final_error_m'].mean():.5f} m"
    )
    print(
        "Joint limit violations: "
        f"{df['joint_limit_violations'].sum()}"
    )

    print(f"\nCSV saved: {OUTPUT_CSV}")
    print(f"Report saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
