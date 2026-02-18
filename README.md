# Flying Robot Control

A comprehensive robotics and control project focused on **aerial robotics** with advanced control algorithms for quadrotor UAVs. This project implements multiple control strategies including PID, LQR, MPC, and geometric control for SE(3) dynamics.

## Project Overview

This project provides a complete framework for:
- **Control Systems**: PID, LQR, MPC, geometric control, robust control
- **Simulation**: PyBullet physics simulation with realistic quadrotor dynamics
- **State Estimation**: EKF/UKF for position and attitude estimation
- **Planning**: Trajectory generation and path following
- **Evaluation**: Comprehensive metrics and benchmarking

## Safety Warning

⚠️ **SIMULATION ONLY - DO NOT USE ON REAL HARDWARE** ⚠️

This project is for **RESEARCH AND EDUCATION PURPOSES ONLY**. See [DISCLAIMER.md](DISCLAIMER.md) for critical safety information.

## Quick Start

### Prerequisites

- Python 3.10+
- ROS 2 Humble (optional, for advanced features)
- CUDA-capable GPU (optional, for faster simulation)

### Installation

```bash
# Clone and setup
git clone https://github.com/kryptologyst/Flying-Robot-Control.git
cd Flying-Robot-Control

# Install dependencies
pip install -r requirements.txt

# For ROS 2 features (optional)
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
```

### Basic Usage

```bash
# Run basic PID control simulation
python scripts/run_simulation.py --controller pid --target 5,5,5

# Run advanced MPC control
python scripts/run_simulation.py --controller mpc --target 10,10,5

# Interactive demo
streamlit run demo/app.py

# ROS 2 launch (if ROS 2 is installed)
ros2 launch flying_robot_control simulation.launch.py
```

## Project Structure

```
├── src/                          # Core source code
│   ├── controllers/              # Control algorithms
│   │   ├── pid.py               # PID controller
│   │   ├── lqr.py               # LQR controller
│   │   ├── mpc.py               # Model Predictive Control
│   │   └── geometric.py         # Geometric control (SE3)
│   ├── dynamics/                # Quadrotor dynamics models
│   │   ├── quadrotor.py         # Main dynamics class
│   │   └── parameters.py       # Physical parameters
│   ├── estimation/              # State estimation
│   │   ├── ekf.py              # Extended Kalman Filter
│   │   └── ukf.py              # Unscented Kalman Filter
│   ├── planning/                # Trajectory planning
│   │   ├── trajectory.py       # Trajectory generation
│   │   └── path_following.py   # Path following algorithms
│   └── utils/                   # Utilities
│       ├── math_utils.py       # Mathematical utilities
│       └── visualization.py    # Plotting and visualization
├── robots/                      # Robot descriptions
│   ├── urdf/                   # URDF files
│   └── meshes/                 # 3D meshes
├── config/                     # Configuration files
│   ├── controllers.yaml        # Controller parameters
│   ├── simulation.yaml         # Simulation settings
│   └── quadrotor.yaml          # Quadrotor parameters
├── launch/                     # ROS 2 launch files
├── scripts/                    # Executable scripts
├── tests/                      # Unit tests
├── demo/                       # Interactive demos
├── assets/                     # Generated artifacts
└── data/                       # Datasets and logs
```

## Control Algorithms

### 1. PID Control
- Position control for x, y, z coordinates
- Attitude control for roll, pitch, yaw
- Configurable gains and integral windup protection

### 2. LQR Control
- Linear Quadratic Regulator for optimal control
- State feedback with optimal gain matrix
- Handles linearized dynamics around hover condition

### 3. Model Predictive Control (MPC)
- Nonlinear MPC with constraints
- Real-time optimization using CasADi
- Handles actuator limits and safety constraints

### 4. Geometric Control
- SE(3) geometric controller for quadrotors
- Direct control of position and attitude
- Robust to model uncertainties

## Simulation Features

- **Physics Engine**: PyBullet for realistic dynamics
- **Sensors**: IMU, camera, GPS simulation
- **Environment**: 3D world with obstacles
- **Visualization**: Real-time 3D visualization
- **Logging**: Comprehensive data logging

## Evaluation Metrics

### Control Performance
- **Position Tracking**: RMSE, overshoot, settling time
- **Attitude Control**: Angular error, stability margins
- **Control Effort**: Energy consumption, actuator usage
- **Robustness**: Performance under disturbances

### Planning Performance
- **Path Following**: Cross-track error, along-track error
- **Trajectory Smoothness**: Jerk, acceleration limits
- **Computational**: Planning time, convergence rate

## Configuration

All parameters are configurable via YAML files:

```yaml
# config/controllers.yaml
pid:
  position:
    kp: [1.0, 1.0, 1.0]
    ki: [0.1, 0.1, 0.1]
    kd: [0.5, 0.5, 0.5]
  attitude:
    kp: [2.0, 2.0, 1.0]
    ki: [0.05, 0.05, 0.05]
    kd: [0.3, 0.3, 0.3]

lqr:
  Q: [10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
  R: [0.1, 0.1, 0.1, 0.1]

mpc:
  horizon: 10
  dt: 0.1
  max_thrust: 20.0
  max_torque: 2.0
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_controllers.py
pytest tests/test_dynamics.py
pytest tests/test_estimation.py
```

## Contributing

1. Follow the code style (black + ruff)
2. Add type hints and docstrings
3. Write tests for new features
4. Update documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on research in geometric control for quadrotors
- Inspired by modern robotics control frameworks
- Educational focus on control theory applications
# Flying-Robot-Control
