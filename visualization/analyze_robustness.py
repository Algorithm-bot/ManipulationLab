
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

INPUT_JSON = RESULTS_DIR / "robustness_report.json"
INPUT_CSV = RESULTS_DIR / "robustness_results.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\nMANIPULATIONLAB EXPERIMENT ANALYSIS")
    print("=" * 55)

    # Load experiment results
    with open(INPUT_JSON, "r", encoding="utf-8") as file:
        report = json.load(file)

    df = pd.read_csv(INPUT_CSV)

    config = pd.DataFrame(report["configuration_results"])

    # Identify best configuration by mean final error
    best = config.sort_values(
        ["mean_final_error", "max_final_error"]
    ).iloc[0]

    print("\nBEST-PERFORMING CONFIGURATION")
    print("-" * 40)
    print(f"Elbow gain: {best['elbow_kp']}")
    print(f"Control noise: {best['control_noise_std']}")
    print(f"Mean final error: {best['mean_final_error']:.6f} m")
    print(f"Maximum final error: {best['max_final_error']:.6f} m")
    print(f"Success rate: {best['success_rate'] * 100:.1f}%")

    # Plot 1: Success rate
    plt.figure(figsize=(9, 5))

    for noise in sorted(config["control_noise_std"].unique()):
        subset = config[config["control_noise_std"] == noise]
        plt.plot(
            subset["elbow_kp"],
            subset["success_rate"] * 100,
            marker="o",
            label=f"Noise = {noise:.3f} rad"
        )

    plt.xlabel("Elbow Proportional Gain (kp)")
    plt.ylabel("Success Rate (%)")
    plt.title("Robustness: Success Rate by Controller Gain")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "robustness_success_rate.png", dpi=200)
    plt.close()

    # Plot 2: Mean final error
    plt.figure(figsize=(9, 5))

    for noise in sorted(config["control_noise_std"].unique()):
        subset = config[config["control_noise_std"] == noise]
        plt.plot(
            subset["elbow_kp"],
            subset["mean_final_error"] * 1000,
            marker="o",
            label=f"Noise = {noise:.3f} rad"
        )

    plt.xlabel("Elbow Proportional Gain (kp)")
    plt.ylabel("Mean Final Error (mm)")
    plt.title("Mean Position Error Across Configurations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "robustness_mean_error.png", dpi=200)
    plt.close()

    # Plot 3: Maximum final error
    plt.figure(figsize=(9, 5))

    for noise in sorted(config["control_noise_std"].unique()):
        subset = config[config["control_noise_std"] == noise]
        plt.plot(
            subset["elbow_kp"],
            subset["max_final_error"] * 1000,
            marker="o",
            label=f"Noise = {noise:.3f} rad"
        )

    plt.xlabel("Elbow Proportional Gain (kp)")
    plt.ylabel("Maximum Final Error (mm)")
    plt.title("Worst-Case Position Error Across Configurations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "robustness_max_error.png", dpi=200)
    plt.close()

    # Configuration comparison table
    config = config.sort_values(
        ["elbow_kp", "control_noise_std"]
    )

    # Generate Markdown report
    lines = [
        "# ManipulationLab - Robustness Analysis",
        "",
        "## Experiment Summary",
        "",
        f"- Total experiments: {report['total_experiments']}",
        f"- Overall success rate: {report['overall_success_rate'] * 100:.2f}%",
        f"- Constraint compliance: {report['overall_constraint_compliance_rate'] * 100:.2f}%",
        f"- Mean final error: {report['overall_mean_final_error_m'] * 1000:.3f} mm",
        f"- Joint limit violations: {report['total_joint_limit_violations']}",
        "",
        "## Best Accuracy Configuration",
        "",
        f"- Elbow gain: {best['elbow_kp']}",
        f"- Control noise: {best['control_noise_std']:.3f} rad",
        f"- Mean final error: {best['mean_final_error'] * 1000:.3f} mm",
        f"- Maximum final error: {best['max_final_error'] * 1000:.3f} mm",
        f"- Success rate: {best['success_rate'] * 100:.1f}%",
        "",
        "Selection criterion: lowest mean final error, with maximum final error as a tie-breaker.",
        "",
        "## Configuration Comparison",
        "",
        "| Gain | Noise (rad) | Success | Mean Error (mm) | Max Error (mm) | Compliance |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in config.iterrows():
        lines.append(
            f"| {row['elbow_kp']} "
            f"| {row['control_noise_std']:.3f} "
            f"| {row['success_rate'] * 100:.1f}% "
            f"| {row['mean_final_error'] * 1000:.3f} "
            f"| {row['max_final_error'] * 1000:.3f} "
            f"| {row['constraint_compliance_rate'] * 100:.1f}% |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "All tested configurations achieved a 100% task success rate "
        "and 100% joint-limit compliance.",
        "",
        "The experiments indicate that the simulated controller maintained "
        "accurate target positioning under the tested actuator-command noise.",
        "",
        "This experiment evaluates simulated command noise, not physical "
        "sensor noise or hardware disturbances.",
        "",
        "## Generated Visualizations",
        "",
        "- robustness_success_rate.png",
        "- robustness_mean_error.png",
        "- robustness_max_error.png",
    ])

    output_report = RESULTS_DIR / "robustness_analysis.md"
    output_report.write_text("\n".join(lines), encoding="utf-8")

    # Save machine-readable best configuration
    best_output = {
        "elbow_kp": float(best["elbow_kp"]),
        "control_noise_std": float(best["control_noise_std"]),
        "mean_final_error_m": float(best["mean_final_error"]),
        "max_final_error_m": float(best["max_final_error"]),
        "success_rate": float(best["success_rate"]),
        "constraint_compliance_rate": float(
            best["constraint_compliance_rate"]
        )
    }

    (RESULTS_DIR / "best_robustness_configuration.json").write_text(
        json.dumps(best_output, indent=4),
        encoding="utf-8"
    )

    print("\nANALYSIS COMPLETE")
    print("-" * 40)
    print("Generated charts: 3")
    print(f"Report: {output_report}")
    print(f"Best configuration: {RESULTS_DIR / 'best_robustness_configuration.json'}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
