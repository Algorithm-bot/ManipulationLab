# ManipulationLab
### Robotic Manipulation, Simulation, Benchmarking & Robustness Evaluation

ManipulationLab is a physics-based robotic manipulation research and evaluation project built using **MuJoCo and Python**. It simulates a UR5e industrial robotic arm equipped with a Robotiq 2F-85 parallel gripper and evaluates its ability to perform object grasping, lifting, transportation, and placement.

The project focuses on robotic manipulation performance, inverse kinematics, contact-based grasp evaluation, repeatability benchmarking, and robustness under simulated actuator-command noise.

## Key Features

- **Realistic robot simulation:** UR5e robotic manipulator with a Robotiq 2F-85 gripper using MuJoCo Menagerie models.
- **Inverse kinematics:** Full-pose inverse kinematics with position and orientation error minimization using damped least squares.
- **Automated pick-and-place:** Executes a complete manipulation sequence involving approach, grasp, lift, transport, placement, and release.
- **Grasp diagnostics:** Evaluates gripper-pad contact points and estimates contact forces using MuJoCo contact data.
- **Automated benchmarking:** Repeats pick-and-place experiments and measures pickup success, object retention, placement accuracy, and execution time.
- **Robustness evaluation:** Introduces Gaussian actuator-command noise to investigate manipulation reliability.
- **Failure analysis:** Examines grasp and transport failures using contact measurements and object-position telemetry.
- **Phase-specific evaluation:** Compares the effects of applying actuator-command noise during different stages of manipulation.
- **Automated reporting:** Exports experiment results to CSV and JSON for reproducibility and analysis.

## System Architecture

```text
ManipulationLab/
│
├── simulation/
│   ├── build_ur5e_gripper.py
│   ├── ur5e_gripper.xml
│   ├── solve_grasp_ik.py
│   └── pick_and_place.py
│
├── experiments/
│   ├── pick_place_benchmark.py
│   ├── pick_place_robustness.py
│   ├── pick_place_threshold.py
│   ├── grasp_failure_diagnostic.py
│   └── transport_stability.py
│
├── visualization/
│   ├── create_plots.py
│   └── analyze_robustness.py
│
├── pipeline/
│   └── validate_telemetry.py
│
├── external/
│   └── mujoco_menagerie/
│       ├── universal_robots_ur5e/
│       └── robotiq_2f85/
│
├── results/
│   ├── pick_place_benchmark.csv
│   ├── pick_place_report.json
│   ├── pick_place_robustness.csv
│   ├── pick_place_threshold.csv
│   ├── grasp_failure_diagnostic.csv
│   ├── transport_noise_phase.csv
│   └── transport_noise_phase_report.json
│
├── archive/
│   ├── legacy_two_link/
│   └── intermediate_tests/
│
├── main.py
├── requirements.txt
└── README.md
```

*Note: The results directory also contains additional JSON reports and experiment outputs. The archive preserves the original two-link arm implementation and intermediate development scripts.*

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Simulation orchestration and experiment automation |
| MuJoCo | Physics-based robotics simulation |
| MuJoCo Menagerie | UR5e and Robotiq robot models |
| NumPy | Numerical computation and joint-space operations |
| SciPy | Inverse kinematics and numerical optimization |
| Pandas | Experimental data processing |
| Matplotlib | Performance visualization |

## Installation

Clone the repository:

```bash
git clone https://github.com/Algorithm-bot/ManipulationLab.git
cd ManipulationLab
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project uses robot models from MuJoCo Menagerie, included under `external/mujoco_menagerie/`.

## Running the Simulation

Run commands from the project root.

**1. Build the combined UR5e and gripper model**

```bash
python simulation/build_ur5e_gripper.py
```

**2. Launch the MuJoCo viewer**

```bash
python -m mujoco.viewer --mjcf simulation/ur5e_gripper.xml
```

**3. Verify inverse kinematics**

```bash
python simulation/solve_grasp_ik.py
```

**4. Run automated pick-and-place**

```bash
python simulation/pick_and_place.py
```

## Experimental Evaluation

### 1. Automated Pick-and-Place Benchmark

Five repeated pick-and-place trials were performed under identical initial conditions.

| Metric | Result |
|---|---:|
| Total trials | 5 |
| Successful trials | 5/5 |
| Overall success rate | 100% |
| Pickup success rate | 100% |
| Object retention rate | 100% |
| Placement success rate | 100% |
| Mean lift height | 68.68 mm |
| Mean placement error | 2.19 mm |
| Maximum placement error | 2.19 mm |
| Joint-limit violations | 0 |

The experiment evaluates repeatability under identical simulation conditions, rather than randomized environmental robustness.

### 2. Actuator-Command Noise Robustness

Gaussian noise was added to arm position actuator targets during simulation.

The experiment investigated how increasing actuator-command noise affects manipulation reliability.

| Noise standard deviation | Successful trials | Success rate |
|---|---:|---:|
| 0.00 rad | 5/5 | 100% |
| 0.01 rad | 5/5 | 100% |
| 0.03 rad | 0/5 | 0% |

A further threshold experiment investigated intermediate noise levels.

| Noise standard deviation | Successful trials | Success rate |
|---|---:|---:|
| 0.015 rad | 5/5 | 100% |
| 0.020 rad | 1/5 | 20% |
| 0.025 rad | 0/5 | 0% |

These results demonstrate sensitivity to actuator-command perturbations under the tested simulation conditions.

### 3. Grasp Failure Diagnostics

Contact-based diagnostics were implemented to investigate grasp stability.

The diagnostic system records:

- Number of gripper-pad contact points.
- Contacting gripper-pad geometries.
- Estimated contact-frame normal forces.
- Object position at different manipulation stages.
- Changes in object position during lifting and transportation.

This makes it possible to investigate whether failures occur during initial grasping, lifting, or transportation.

*Contact forces are MuJoCo solver estimates, not calibrated physical gripper-force measurements.*

### 4. Phase-Specific Noise Evaluation

An additional experiment investigated how the stage at which actuator-command noise is introduced affects manipulation performance.

Five trials were conducted under each condition, using a noise standard deviation of 0.02 rad.

| Noise condition | Successful trials | Success rate | Mean placement error |
|---|---:|---:|---:|
| Approach, lift and transport | 3/5 | 60% | 84.35 mm |
| Approach and lift only | 3/5 | 60% | 97.29 mm |
| Transport only | 5/5 | 100% | 2.35 mm |

The results suggest that clean approach and lifting commands are important for reliable grasp acquisition and object retention.

This experiment provides evidence of sensitivity to noise during pickup and lifting, although it does not independently isolate each of those stages.

## Visualizations and Reports

The project maintains structured experimental outputs for analysis and comparison.

Generated outputs include:

- Pick-and-place benchmark reports.
- Placement error measurements.
- Grasp contact diagnostics.
- Actuator-command noise robustness results.
- Transport stability comparisons.
- CSV datasets and machine-readable JSON reports.

## Limitations

- All experiments are conducted in MuJoCo; no physical robot hardware has been tested.
- Results depend on the simulated robot model, contact parameters, actuator configuration, and controller behavior.
- Actuator-command noise experiments do not represent physical sensor noise or external environmental disturbances.
- Contact-force measurements are simulation estimates and have not been calibrated against physical measurements.
- Current manipulation experiments focus on a single cube and a fixed pick-and-place task.
- Simulation results do not establish sim-to-real transfer performance.

## Future Work

- Integrate ROS 2 for robotics middleware and communication.
- Introduce randomized object positions and orientations.
- Evaluate sensor noise and external force disturbances.
- Develop more advanced grasp planning and contact-aware control.
- Explore trajectory optimization and motion planning.
- Investigate sim-to-real transfer using physical robot measurements.
- Extend benchmarking to multiple objects and manipulation tasks.

## Author

**Sahil Sawant**

GitHub: [Algorithm-bot](https://github.com/Algorithm-bot)