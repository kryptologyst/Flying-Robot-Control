"""LQR (Linear Quadratic Regulator) controller for quadrotor control.

This module implements an LQR controller for quadrotor UAV control, providing
optimal state feedback control based on linearized dynamics around hover condition.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_are

from dynamics.quadrotor import QuadrotorDynamics
from dynamics.parameters import QuadrotorParameters


@dataclass
class LQRWeights:
    """LQR cost function weights.
    
    Attributes:
        Q: State weight matrix (12x12) - penalizes deviations from reference
        R: Control weight matrix (4x4) - penalizes control effort
        Qf: Terminal state weight matrix (12x12) - for finite horizon problems
    """
    Q: NDArray[np.float64] = None
    R: NDArray[np.float64] = None
    Qf: Optional[NDArray[np.float64]] = None
    
    def __post_init__(self) -> None:
        """Initialize default weights if not provided."""
        if self.Q is None:
            # Default state weights: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
            self.Q = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1])
        
        if self.R is None:
            # Default control weights: [T1, T2, T3, T4]
            self.R = np.diag([0.1, 0.1, 0.1, 0.1])


class LQRController:
    """Linear Quadratic Regulator for quadrotor control.
    
    This controller implements LQR control based on linearized quadrotor dynamics
    around the hover condition. The controller provides optimal state feedback
    control that minimizes a quadratic cost function.
    """
    
    def __init__(
        self,
        quadrotor_params: QuadrotorParameters,
        weights: Optional[LQRWeights] = None,
        dt: float = 0.01
    ) -> None:
        """Initialize LQR controller.
        
        Args:
            quadrotor_params: Quadrotor physical parameters
            weights: LQR cost function weights
            dt: Control loop time step in seconds
        """
        self.params = quadrotor_params
        self.weights = weights or LQRWeights()
        self.dt = dt
        
        # Linearized dynamics matrices
        self.A = None  # State transition matrix (12x12)
        self.B = None  # Control input matrix (12x4)
        
        # LQR gain matrix
        self.K = None  # Feedback gain matrix (4x12)
        
        # Reference state (hover condition)
        self.reference_state = np.zeros(12)
        
        # Initialize linearized dynamics
        self._compute_linearized_dynamics()
        self._compute_lqr_gain()
        
    def _compute_linearized_dynamics(self) -> None:
        """Compute linearized dynamics matrices around hover condition.
        
        The linearized dynamics are computed around the hover condition where:
        - Position: arbitrary (linearized around current position)
        - Velocity: zero
        - Attitude: zero (level flight)
        - Angular velocity: zero
        - Thrust: mg/4 per motor (hover thrust)
        """
        # State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        # Control vector: [T1, T2, T3, T4]
        
        # Hover thrust per motor
        hover_thrust = self.params.mass * self.params.gravity / 4.0
        
        # State transition matrix A (12x12)
        self.A = np.zeros((12, 12))
        
        # Position dynamics: d/dt [x, y, z] = [vx, vy, vz]
        self.A[0:3, 3:6] = np.eye(3)
        
        # Velocity dynamics: d/dt [vx, vy, vz] = [0, 0, g] + R * [0, 0, T_total/m]
        # Linearized around hover: R ≈ I + [roll, pitch, 0] × (small angle approximation)
        # So: dvx/dt ≈ g * pitch, dvy/dt ≈ -g * roll, dvz/dt ≈ T_total/m - g
        
        # Attitude dynamics: d/dt [roll, pitch, yaw] = [wx, wy, wz] (small angle approximation)
        self.A[6:9, 9:12] = np.eye(3)
        
        # Angular velocity dynamics: d/dt [wx, wy, wz] = I^-1 * tau
        # Linearized around hover
        
        # Control input matrix B (12x4)
        self.B = np.zeros((12, 4))
        
        # Position control through attitude
        # Total thrust affects z-velocity
        self.B[5, :] = 1.0 / self.params.mass  # All motors contribute to z-acceleration
        
        # Attitude control through differential thrust
        L = self.params.arm_length
        
        # Roll control: T2 - T4
        self.B[9, 1] = L / self.params.inertia[0, 0]   # Right motor
        self.B[9, 3] = -L / self.params.inertia[0, 0]  # Left motor
        
        # Pitch control: T1 - T3
        self.B[10, 0] = L / self.params.inertia[1, 1]  # Front motor
        self.B[10, 2] = -L / self.params.inertia[1, 1] # Back motor
        
        # Yaw control: T1 - T2 + T3 - T4
        yaw_gain = self.params.max_torque / self.params.inertia[2, 2]
        self.B[11, 0] = yaw_gain   # Front motor
        self.B[11, 1] = -yaw_gain  # Right motor
        self.B[11, 2] = yaw_gain   # Back motor
        self.B[11, 3] = -yaw_gain  # Left motor
        
        # Convert to discrete time
        self.A = np.eye(12) + self.A * self.dt
        self.B = self.B * self.dt
        
    def _compute_lqr_gain(self) -> None:
        """Compute LQR feedback gain matrix using algebraic Riccati equation."""
        try:
            # Solve continuous-time algebraic Riccati equation
            P = solve_continuous_are(self.A, self.B, self.weights.Q, self.weights.R)
            
            # Compute feedback gain matrix
            self.K = np.linalg.solve(self.weights.R, self.B.T @ P)
            
        except np.linalg.LinAlgError:
            print("Warning: Could not solve algebraic Riccati equation. Using manual gain computation.")
            # Fallback: simple manual gain computation
            self._compute_manual_gains()
    
    def _compute_manual_gains(self) -> None:
        """Compute manual gains as fallback when Riccati equation fails."""
        # Simple proportional gains based on system characteristics
        self.K = np.zeros((4, 12))
        
        # Position control gains
        self.K[:, 0:3] = np.array([
            [0.5, 0.0, 0.0],   # Front motor
            [0.0, 0.5, 0.0],   # Right motor
            [-0.5, 0.0, 0.0],  # Back motor
            [0.0, -0.5, 0.0]   # Left motor
        ])
        
        # Velocity control gains
        self.K[:, 3:6] = np.array([
            [0.1, 0.0, 0.0],   # Front motor
            [0.0, 0.1, 0.0],   # Right motor
            [-0.1, 0.0, 0.0],  # Back motor
            [0.0, -0.1, 0.0]   # Left motor
        ])
        
        # Attitude control gains
        self.K[:, 6:9] = np.array([
            [0.0, 1.0, 0.0],   # Front motor
            [1.0, 0.0, 0.0],   # Right motor
            [0.0, -1.0, 0.0],  # Back motor
            [-1.0, 0.0, 0.0]   # Left motor
        ])
        
        # Angular velocity control gains
        self.K[:, 9:12] = np.array([
            [0.0, 0.2, 0.0],   # Front motor
            [0.2, 0.0, 0.0],   # Right motor
            [0.0, -0.2, 0.0],  # Back motor
            [-0.2, 0.0, 0.0]   # Left motor
        ])
        
        # Add hover thrust offset
        hover_thrust = self.params.mass * self.params.gravity / 4.0
        self.K[:, 2] += hover_thrust  # Add hover thrust to z-control
    
    def set_reference(self, reference_state: NDArray[np.float64]) -> None:
        """Set reference state for tracking.
        
        Args:
            reference_state: Reference state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        """
        assert reference_state.shape == (12,), f"Expected reference state shape (12,), got {reference_state.shape}"
        self.reference_state = reference_state.copy()
    
    def compute_control(
        self,
        quadrotor: QuadrotorDynamics,
        target_position: NDArray[np.float64],
        target_velocity: Optional[NDArray[np.float64]] = None,
        target_attitude: Optional[NDArray[np.float64]] = None,
        target_angular_velocity: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """Compute LQR control signal.
        
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
        position = quadrotor.get_position()
        velocity = quadrotor.get_velocity()
        attitude = quadrotor.get_euler_angles()
        angular_velocity = quadrotor.get_angular_velocity()
        
        # Set defaults
        if target_velocity is None:
            target_velocity = np.zeros(3)
        if target_attitude is None:
            target_attitude = np.zeros(3)
        if target_angular_velocity is None:
            target_angular_velocity = np.zeros(3)
        
        # Construct current state vector
        current_state = np.concatenate([
            position,
            velocity,
            attitude,
            angular_velocity
        ])
        
        # Construct reference state vector
        reference_state = np.concatenate([
            target_position,
            target_velocity,
            target_attitude,
            target_angular_velocity
        ])
        
        # Compute state error
        state_error = current_state - reference_state
        
        # Apply LQR control law: u = -K * (x - x_ref)
        control_input = -self.K @ state_error
        
        # Convert control input to motor thrusts
        # Control input represents desired thrusts
        thrusts = control_input
        
        # Apply motor limits
        thrusts = np.clip(thrusts, 0.0, self.params.max_thrust)
        
        return thrusts
    
    def get_gain_matrix(self) -> NDArray[np.float64]:
        """Get LQR feedback gain matrix.
        
        Returns:
            Feedback gain matrix K (4x12)
        """
        return self.K.copy()
    
    def get_linearized_matrices(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get linearized dynamics matrices.
        
        Returns:
            Tuple of (A, B) matrices
        """
        return self.A.copy(), self.B.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status information.
        
        Returns:
            Dictionary containing controller status
        """
        return {
            'gain_matrix': self.K.tolist(),
            'reference_state': self.reference_state.tolist(),
            'weights_Q': self.weights.Q.tolist(),
            'weights_R': self.weights.R.tolist(),
        }
