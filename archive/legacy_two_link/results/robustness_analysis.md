# ManipulationLab - Robustness Analysis

## Experiment Summary

- Total experiments: 225
- Overall success rate: 100.00%
- Constraint compliance: 100.00%
- Mean final error: 0.506 mm
- Joint limit violations: 0

## Best Accuracy Configuration

- Elbow gain: 8.0
- Control noise: 0.000 rad
- Mean final error: 0.352 mm
- Maximum final error: 0.794 mm
- Success rate: 100.0%

Selection criterion: lowest mean final error, with maximum final error as a tie-breaker.

## Configuration Comparison

| Gain | Noise (rad) | Success | Mean Error (mm) | Max Error (mm) | Compliance |
|---:|---:|---:|---:|---:|---:|
| 8.0 | 0.000 | 100.0% | 0.352 | 0.794 | 100.0% |
| 8.0 | 0.005 | 100.0% | 0.525 | 1.397 | 100.0% |
| 8.0 | 0.010 | 100.0% | 0.769 | 2.438 | 100.0% |
| 10.0 | 0.000 | 100.0% | 0.367 | 0.794 | 100.0% |
| 10.0 | 0.005 | 100.0% | 0.501 | 1.561 | 100.0% |
| 10.0 | 0.010 | 100.0% | 0.680 | 1.483 | 100.0% |
| 12.0 | 0.000 | 100.0% | 0.375 | 0.799 | 100.0% |
| 12.0 | 0.005 | 100.0% | 0.544 | 1.328 | 100.0% |
| 12.0 | 0.010 | 100.0% | 0.493 | 0.904 | 100.0% |
| 14.0 | 0.000 | 100.0% | 0.379 | 0.804 | 100.0% |
| 14.0 | 0.005 | 100.0% | 0.413 | 0.810 | 100.0% |
| 14.0 | 0.010 | 100.0% | 0.577 | 1.017 | 100.0% |
| 16.0 | 0.000 | 100.0% | 0.382 | 0.808 | 100.0% |
| 16.0 | 0.005 | 100.0% | 0.467 | 1.003 | 100.0% |
| 16.0 | 0.010 | 100.0% | 0.766 | 2.381 | 100.0% |

## Interpretation

All tested configurations achieved a 100% task success rate and 100% joint-limit compliance.

The experiments indicate that the simulated controller maintained accurate target positioning under the tested actuator-command noise.

This experiment evaluates simulated command noise, not physical sensor noise or hardware disturbances.

## Generated Visualizations

- robustness_success_rate.png
- robustness_mean_error.png
- robustness_max_error.png