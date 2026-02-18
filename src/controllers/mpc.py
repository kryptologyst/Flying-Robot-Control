"""Model Predictive Control (MPC) for quadrotor control.

This module implements an MPC controller for quadrotor UAV control using CasADi
for optimization. The controller handles constraints and provides optimal control
over a prediction horizon.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from numpy.typing import NDArray
try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False
    ca = None

from dynamics.quadrotor import QuadrotorDynamics
from dynamics.parameters import QuadrotorParameters


@dataclass
class MPCConfig:
    """MPC controller configuration.
    
    Attributes:
        horizon: Prediction horizon (number of steps)
        dt: Time step in seconds
        max_thrust: Maximum thrust per motor in N
        min_thrust: Minimum thrust per motor in N
        max_torque: Maximum torque per motor in N⋅m
        position_weight: Weight for position tracking
        velocity_weight: Weight for velocity tracking
        attitude_weight: Weight for attitude tracking
        angular_velocity_weight: Weight for angular velocity tracking
        control_weight: Weight for control effort
        terminal_weight: Weight for terminal state
    """
    horizon: int = 10
    dt: float = 0.1
    max_thrust: float = 20.0
    min_thrust: float = 0.0
    max_torque: float = 2.0
    position_weight: float = 10.0
    velocity_weight: float = 1.0
    attitude_weight: float = 1.0
    angular_velocity_weight: float = 0.1
    control_weight: float = 0.1
    terminal_weight: float = 100.0


class MPCController:
    """Model Predictive Controller for quadrotor control.
    
    This controller implements MPC using CasADi for optimization. It solves
    a constrained optimization problem over a prediction horizon to compute
    optimal control inputs.
    """
    
    def __init__(
        self,
        quadrotor_params: QuadrotorParameters,
        config: Optional[MPCConfig] = None
    ) -> None:
        """Initialize MPC controller.
        
        Args:
            quadrotor_params: Quadrotor physical parameters
            config: MPC configuration parameters
        """
        if not CASADI_AVAILABLE:
            raise ImportError("CasADi is required for MPC controller. Install with: pip install casadi")
            
        self.params = quadrotor_params
        self.config = config or MPCConfig()
        
        # State and control dimensions
        self.nx = 13  # State dimension [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        self.nu = 4   # Control dimension [T1, T2, T3, T4]
        
        # Optimization variables
        self.opti = None
        self.X = None  # State trajectory
        self.U = None  # Control trajectory
        self.x0 = None  # Initial state parameter
        
        # Reference trajectory
        self.reference_trajectory = None
        
        # Solver
        self.solver = None
        
        # Initialize optimization problem
        self._setup_optimization()
        
    def _setup_optimization(self) -> None:
        """Set up the MPC optimization problem."""
        # Create optimization problem
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(self.nx, self.config.horizon + 1)  # State trajectory
        self.U = self.opti.variable(self.nu, self.config.horizon)      # Control trajectory
        
        # Parameters
        self.x0 = self.opti.parameter(self.nx)  # Initial state
        
        # Cost function
        cost = 0
        
        # Stage cost
        for k in range(self.config.horizon):
            # State tracking cost
            if self.reference_trajectory is not None:
                x_ref = self.reference_trajectory[:, k]
                cost += self.config.position_weight * ca.sumsqr(self.X[0:3, k] - x_ref[0:3])
                cost += self.config.velocity_weight * ca.sumsqr(self.X[3:6, k] - x_ref[3:6])
                cost += self.config.attitude_weight * ca.sumsqr(self.X[6:10, k] - x_ref[6:10])
                cost += self.config.angular_velocity_weight * ca.sumsqr(self.X[10:13, k] - x_ref[10:13])
            
            # Control effort cost
            cost += self.config.control_weight * ca.sumsqr(self.U[:, k])
        
        # Terminal cost
        if self.reference_trajectory is not None:
            x_ref_terminal = self.reference_trajectory[:, -1]
            cost += self.config.terminal_weight * ca.sumsqr(self.X[:, -1] - x_ref_terminal)
        
        self.opti.minimize(cost)
        
        # Constraints
        # Initial state constraint
        self.opti.subject_to(self.X[:, 0] == self.x0)
        
        # Dynamics constraints
        for k in range(self.config.horizon):
            # State transition using simplified dynamics
            x_k = self.X[:, k]
            u_k = self.U[:, k]
            x_next = self._dynamics_casadi(x_k, u_k)
            self.opti.subject_to(self.X[:, k+1] == x_next)
        
        # Control constraints
        for k in range(self.config.horizon):
            # Thrust limits
            self.opti.subject_to(self.U[:, k] >= self.config.min_thrust)
            self.opti.subject_to(self.U[:, k] <= self.config.max_thrust)
        
        # State constraints (optional)
        for k in range(self.config.horizon + 1):
            # Position bounds (optional)
            # self.opti.subject_to(self.X[0:3, k] >= -10)  # Lower bounds
            # self.opti.subject_to(self.X[0:3, k] <= 10)   # Upper bounds
            
            # Velocity bounds
            max_vel = self.params.max_velocity
            self.opti.subject_to(self.X[3:6, k] >= -max_vel)
            self.opti.subject_to(self.X[3:6, k] <= max_vel)
            
            # Angular velocity bounds
            max_ang_vel = self.params.max_angular_velocity
            self.opti.subject_to(self.X[10:13, k] >= -max_ang_vel)
            self.opti.subject_to(self.X[10:13, k] <= max_ang_vel)
        
        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-6,
        }
        
        self.opti.solver('ipopt', opts)
        
    def _dynamics_casadi(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """CasADi dynamics function for optimization.
        
        Args:
            x: Current state vector
            u: Control input vector
            
        Returns:
            Next state vector
        """
        # Extract state components
        position = x[0:3]
        velocity = x[3:6]
        quaternion = x[6:10]
        angular_velocity = x[10:13]
        
        # Normalize quaternion
        quaternion = quaternion / ca.norm_2(quaternion)
        
        # Convert control to forces and torques
        total_thrust = ca.sum1(u)
        
        # Rotation matrix from quaternion
        qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        R = ca.vertcat(
            ca.horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
            ca.horzcat(2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)),
            ca.horzcat(2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2))
        )
        
        # Thrust force in world frame
        thrust_body = ca.vertcat(0, 0, total_thrust)
        thrust_world = R @ thrust_body
        
        # Gravity force
        gravity_force = ca.vertcat(0, 0, -self.params.mass * self.params.gravity)
        
        # Drag force
        drag_force = -self.params.drag_coefficient * velocity
        
        # Total force
        total_force = thrust_world + gravity_force + drag_force
        
        # Position dynamics
        position_dot = velocity
        velocity_dot = total_force / self.params.mass
        
        # Quaternion dynamics
        omega_quat = ca.vertcat(0, angular_velocity[0], angular_velocity[1], angular_velocity[2])
        quaternion_dot = 0.5 * self._quaternion_multiply_casadi(quaternion, omega_quat)
        
        # Angular velocity dynamics (simplified)
        L = self.params.arm_length
        torque_body = ca.vertcat(
            L * (u[1] - u[3]),  # Roll torque
            L * (u[0] - u[2]),  # Pitch torque
            self.params.max_torque * (u[0] - u[1] + u[2] - u[3])  # Yaw torque
        )
        
        angular_velocity_dot = ca.solve(self.params.inertia, torque_body)
        
        # State derivative
        x_dot = ca.vertcat(
            position_dot,
            velocity_dot,
            quaternion_dot,
            angular_velocity_dot
        )
        
        # Integration
        x_next = x + x_dot * self.config.dt
        
        # Normalize quaternion
        x_next[6:10] = x_next[6:10] / ca.norm_2(x_next[6:10])
        
        return x_next
    
    def _quaternion_multiply_casadi(self, q1: ca.MX, q2: ca.MX) -> ca.MX:
        """Multiply two quaternions using CasADi.
        
        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]
            
        Returns:
            Product quaternion [w, x, y, z]
        """
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return ca.vertcat(w, x, y, z)
    
    def set_reference_trajectory(self, trajectory: NDArray[np.float64]) -> None:
        """Set reference trajectory for tracking.
        
        Args:
            trajectory: Reference trajectory (13 x horizon+1)
        """
        assert trajectory.shape[0] == self.nx, f"Expected trajectory shape ({self.nx}, horizon+1), got {trajectory.shape}"
        assert trajectory.shape[1] == self.config.horizon + 1, f"Expected horizon+1={self.config.horizon+1}, got {trajectory.shape[1]}"
        
        self.reference_trajectory = trajectory.copy()
        
        # Rebuild optimization problem with new reference
        self._setup_optimization()
    
    def compute_control(
        self,
        quadrotor: QuadrotorDynamics,
        target_position: NDArray[np.float64],
        target_velocity: Optional[NDArray[np.float64]] = None,
        target_attitude: Optional[NDArray[np.float64]] = None,
        target_angular_velocity: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """Compute MPC control signal.
        
        Args:
            quadrotor: Quadrotor dynamics instance
            target_position: Target position [x, y, z]
            target_velocity: Target velocity [vx, vy, vz]. If None, assumed to be zero.
            target_attitude: Target attitude [roll, pitch, yaw]. If None, computed from position control.
            target_angular_velocity: Target angular velocity [wx, wy, wz]. If None, assumed to be zero.
            
        Returns:
            Motor thrusts [T1, T2, T3, T4]
        """
        # Get current state
        current_state = quadrotor.get_state()
        
        # Set defaults
        if target_velocity is None:
            target_velocity = np.zeros(3)
        if target_attitude is None:
            target_attitude = np.zeros(3)
        if target_angular_velocity is None:
            target_angular_velocity = np.zeros(3)
        
        # Generate reference trajectory
        reference_trajectory = self._generate_reference_trajectory(
            target_position, target_velocity, target_attitude, target_angular_velocity
        )
        
        # Set reference trajectory
        self.set_reference_trajectory(reference_trajectory)
        
        # Set initial state
        self.opti.set_value(self.x0, current_state)
        
        try:
            # Solve optimization problem
            sol = self.opti.solve()
            
            # Extract control input
            control_trajectory = sol.value(self.U)
            thrusts = control_trajectory[:, 0]  # First control input
            
            # Apply motor limits
            thrusts = np.clip(thrusts, self.config.min_thrust, self.config.max_thrust)
            
            return thrusts
            
        except Exception as e:
            print(f"MPC optimization failed: {e}")
            # Fallback to hover thrust
            hover_thrust = self.params.mass * self.params.gravity / 4.0
            return np.array([hover_thrust, hover_thrust, hover_thrust, hover_thrust])
    
    def _generate_reference_trajectory(
        self,
        target_position: NDArray[np.float64],
        target_velocity: NDArray[np.float64],
        target_attitude: NDArray[np.float64],
        target_angular_velocity: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Generate reference trajectory for MPC.
        
        Args:
            target_position: Target position [x, y, z]
            target_velocity: Target velocity [vx, vy, vz]
            target_attitude: Target attitude [roll, pitch, yaw]
            target_angular_velocity: Target angular velocity [wx, wy, wz]
            
        Returns:
            Reference trajectory (13 x horizon+1)
        """
        # Convert Euler angles to quaternion
        roll, pitch, yaw = target_attitude
        
        # Quaternion from Euler angles (ZYX convention)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        target_quaternion = np.array([qw, qx, qy, qz])
        
        # Generate reference state
        reference_state = np.concatenate([
            target_position,
            target_velocity,
            target_quaternion,
            target_angular_velocity
        ])
        
        # Create trajectory (constant reference for now)
        trajectory = np.tile(reference_state, (self.config.horizon + 1, 1)).T
        
        return trajectory
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status information.
        
        Returns:
            Dictionary containing controller status
        """
        return {
            'horizon': self.config.horizon,
            'dt': self.config.dt,
            'max_thrust': self.config.max_thrust,
            'min_thrust': self.config.min_thrust,
            'position_weight': self.config.position_weight,
            'velocity_weight': self.config.velocity_weight,
            'attitude_weight': self.config.attitude_weight,
            'angular_velocity_weight': self.config.angular_velocity_weight,
            'control_weight': self.config.control_weight,
            'terminal_weight': self.config.terminal_weight,
        }
