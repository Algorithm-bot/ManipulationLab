
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "results" / "telemetry.csv"
CLEAN_FILE = ROOT / "results" / "validated_telemetry.csv"
REPORT_FILE = ROOT / "results" / "validation_report.json"

# Robot configuration
JOINT_LIMITS = {
    "shoulder_position": (-1.57, 1.57),
    "elbow_position": (-2.0, 2.0),
}

VELOCITY_LIMITS = {
    "shoulder_velocity": 10.0,
    "elbow_velocity": 10.0,
}

REQUIRED_COLUMNS = [
    "trial_id",
    "simulation_time",
    "target_x",
    "target_y",
    "actual_x",
    "actual_y",
    "actual_z",
    "shoulder_position",
    "elbow_position",
    "shoulder_velocity",
    "elbow_velocity",
    "shoulder_control",
    "elbow_control",
    "position_error",
]


def validate_telemetry():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    issues = []
    initial_rows = len(df)

    # 1. Required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 2. Convert measurement fields to numeric
    numeric_columns = [
        col for col in REQUIRED_COLUMNS
        if col != "trial_id"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Detect missing and non-finite measurements
    missing_counts = df[REQUIRED_COLUMNS].isna().sum()

    for col, count in missing_counts.items():
        if count > 0:
            issues.append({
                "type": "missing_values",
                "column": col,
                "count": int(count)
            })

    numeric_values = df[numeric_columns].to_numpy(dtype=float)
    non_finite_mask = ~np.isfinite(numeric_values)

    non_finite_count = int(non_finite_mask.sum())

    if non_finite_count:
        issues.append({
            "type": "non_finite_values",
            "count": non_finite_count
        })

    # 4. Remove rows with invalid required values
    valid_mask = (
        df[REQUIRED_COLUMNS].notna().all(axis=1)
        & np.isfinite(df[numeric_columns]).all(axis=1)
    )

    clean = df.loc[valid_mask].copy()

    # 5. Detect duplicate timestamps within trials
    duplicates = clean.duplicated(
        subset=["trial_id", "simulation_time"],
        keep="first"
    )

    duplicate_count = int(duplicates.sum())

    if duplicate_count:
        issues.append({
            "type": "duplicate_timestamps",
            "count": duplicate_count
        })

    clean = clean.loc[~duplicates].copy()

    # 6. Joint position limits
    for col, (lower, upper) in JOINT_LIMITS.items():

        invalid = (
            (clean[col] < lower)
            | (clean[col] > upper)
        )

        count = int(invalid.sum())

        if count:
            issues.append({
                "type": "joint_limit_violation",
                "column": col,
                "count": count
            })

    # 7. Joint velocity limits
    for col, limit in VELOCITY_LIMITS.items():

        invalid = clean[col].abs() > limit
        count = int(invalid.sum())

        if count:
            issues.append({
                "type": "velocity_limit_violation",
                "column": col,
                "count": count
            })

    # 8. Sampling interval validation
    sampling_results = []

    for trial_id, group in clean.groupby("trial_id"):

        group = group.sort_values("simulation_time")

        intervals = group["simulation_time"].diff().dropna()

        if len(intervals) == 0:
            continue

        irregular = ~np.isclose(
            intervals,
            0.002,
            atol=1e-7,
            rtol=0
        )

        count = int(irregular.sum())

        if count:
            issues.append({
                "type": "irregular_sampling",
                "trial_id": int(trial_id),
                "count": count
            })

        sampling_results.append({
            "trial_id": int(trial_id),
            "mean_interval_seconds": float(intervals.mean()),
            "min_interval_seconds": float(intervals.min()),
            "max_interval_seconds": float(intervals.max()),
            "irregular_intervals": count
        })

    # 9. Save validated data
    clean = clean.sort_values(
        ["trial_id", "simulation_time"]
    )

    clean.to_csv(CLEAN_FILE, index=False)

    # 10. Generate report
    report = {
        "input_file": str(INPUT_FILE.name),
        "total_input_rows": initial_rows,
        "valid_output_rows": len(clean),
        "removed_rows": initial_rows - len(clean),
        "joint_limit_violations": sum(
            i["count"] for i in issues
            if i["type"] == "joint_limit_violation"
        ),
        "velocity_limit_violations": sum(
            i["count"] for i in issues
            if i["type"] == "velocity_limit_violation"
        ),
        "duplicate_timestamps": duplicate_count,
        "non_finite_values": non_finite_count,
        "sampling_analysis": sampling_results,
        "issues": issues,
        "validation_passed": len(issues) == 0,
    }

    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    print("\nMANIPULATIONLAB DATA VALIDATION")
    print("=" * 45)

    print("Input rows:", initial_rows)
    print("Valid output rows:", len(clean))
    print("Removed rows:", initial_rows - len(clean))
    print("Duplicate timestamps:", duplicate_count)
    print("Non-finite values:", non_finite_count)
    print("Issues detected:", len(issues))
    print("Validation passed:", report["validation_passed"])

    print("\nGenerated files:")
    print(CLEAN_FILE)
    print(REPORT_FILE)


if __name__ == "__main__":
    validate_telemetry()
