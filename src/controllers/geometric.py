"""Geometric controller for quadrotor control on SE(3).

This module implements a geometric controller for quadrotor UAV control based on
the SE(3) manifold. This controller provides robust control for position and
attitude tracking with geometric guarantees.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from numpy.typing import NDArray

from dynamics.quadrotor import QuadrotorDynamics
from dynamics.parameters import QuadrotorParameters


@dataclass
class GeometricControllerGains:
    """Gains for geometric controller.
    
    Attributes:
        kx: Position control gain
        kv: Velocity control gain
        kr: Attitude control gain
        kw: Angular velocity control gain
        kR: Rotation control gain
        kw_omega: Angular velocity control gain
    """
    kx: float = 1.0      # Position control gain
    kv: float = 1.0       # Velocity control gain
    kr: float = 1.0       # Attitude control gain
    kw: float = 1.0       # Angular velocity control gain
    kR: float = 1.0       # Rotation control gain
    kw_omega: float = 1.0  # Angular velocity control gain


class GeometricController:
    """Geometric controller for quadrotor control on SE(3).
    
    This controller implements geometric control on the SE(3) manifold for
    quadrotor UAV control. It provides robust position and attitude tracking
    with geometric guarantees and is based on the work of Lee et al.
    """
    
    def __init__(
        self,
        quadrotor_params: QuadrotorParameters,
        gains: Optional[GeometricControllerGains] = None
    ) -> None:
        """Initialize geometric controller.
        
        Args:
            quadrotor_params: Quadrotor physical parameters
            gains: Controller gains
        """
        self.params = quadrotor_params
        self.gains = gains or GeometricControllerGains()
        
        # Reference trajectory
        self.reference_position = np.zeros(3)
        self.reference_velocity = np.zeros(3)
        self.reference_acceleration = np.zeros(3)
        self.reference_jerk = np.zeros(3)
        self.reference_yaw = 0.0
        self.reference_yaw_rate = 0.0
        
    def set_reference(
        self,
        position: NDArray[np.float64],
        velocity: Optional[NDArray[np.float64]] = None,
        acceleration: Optional[NDArray[np.float64]] = None,
        jerk: Optional[NDArray[np.float64]] = None,
        yaw: float = 0.0,
        yaw_rate: float = 0.0
    ) -> None:
        """Set reference trajectory.
        
        Args:
            position: Reference position [x, y, z]
            velocity: Reference velocity [vx, vy, vz]. If None, assumed to be zero.
            acceleration: Reference acceleration [ax, ay, az]. If None, assumed to be zero.
            jerk: Reference jerk [jx, jy, jz]. If None, assumed to be zero.
            yaw: Reference yaw angle in radians
            yaw_rate: Reference yaw rate in rad/s
        """
        self.reference_position = position.copy()
        self.reference_velocity = velocity.copy() if velocity is not None else np.zeros(3)
        self.reference_acceleration = acceleration.copy() if acceleration is not None else np.zeros(3)
        self.reference_jerk = jerk.copy() if jerk is not None else np.zeros(3)
        self.reference_yaw = yaw
        self.reference_yaw_rate = yaw_rate
    
    def _compute_desired_attitude(
        self,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        desired_acceleration: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute desired attitude from position control.
        
        Args:
            position: Current position [x, y, z]
            velocity: Current velocity [vx, vy, vz]
            desired_acceleration: Desired acceleration [ax, ay, az]
            
        Returns:
            Tuple of (desired_rotation_matrix, desired_angular_velocity)
        """
        # Desired thrust direction
        gravity_vector = np.array([0, 0, self.params.gravity])
        thrust_direction = (desired_acceleration + gravity_vector) / np.linalg.norm(desired_acceleration + gravity_vector)
        
        # Desired yaw direction (projected onto horizontal plane)
        yaw_direction = np.array([np.cos(self.reference_yaw), np.sin(self.reference_yaw), 0])
        
        # Desired rotation matrix
        z_body = thrust_direction
        y_body = np.cross(z_body, yaw_direction)
        y_body = y_body / np.linalg.norm(y_body)
        x_body = np.cross(y_body, z_body)
        
        desired_rotation = np.column_stack([x_body, y_body, z_body])
        
        # Desired angular velocity (simplified computation)
        desired_angular_velocity = np.zeros(3)
        
        return desired_rotation, desired_angular_velocity
    
    def _rotation_error(self, R_desired: NDArray[np.float64], R_current: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute rotation error.
        
        Args:
            R_desired: Desired rotation matrix
            R_current: Current rotation matrix
            
        Returns:
            Rotation error vector
        """
        # Rotation error: e_R = 0.5 * trace(R_desired^T * R_current) * skew(R_desired^T * R_current)
        R_error = R_desired.T @ R_current
        trace_error = np.trace(R_error)
        
        # Skew-symmetric part
        skew_error = R_error - R_error.T
        
        # Extract rotation error vector
        error_vector = np.array([skew_error[2, 1], skew_error[0, 2], skew_error[1, 0]])
        
        return error_vector
    
    def _angular_velocity_error(
        self,
        omega_desired: NDArray[np.float64],
        omega_current: NDArray[np.float64],
        R_desired: NDArray[np.float64],
        R_current: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute angular velocity error.
        
        Args:
            omega_desired: Desired angular velocity
            omega_current: Current angular velocity
            R_desired: Desired rotation matrix
            R_current: Current rotation matrix
            
        Returns:
            Angular velocity error vector
        """
        # Transform desired angular velocity to current frame
        omega_desired_current = R_current.T @ R_desired @ omega_desired
        
        # Angular velocity error
        error = omega_current - omega_desired_current
        
        return error
    
    def compute_control(
        self,
        quadrotor: QuadrotorDynamics,
        target_position: NDArray[np.float64],
        target_velocity: Optional[NDArray[np.float64]] = None,
        target_acceleration: Optional[NDArray[np.float64]] = None,
        target_yaw: float = 0.0
    ) -> NDArray[np.float64]:
        """Compute geometric control signal.
        
        Args:
            quadrotor: Quadrotor dynamics instance
            target_position: Target position [x, y, z]
            target_velocity: Target velocity [vx, vy, vz]. If None, assumed to be zero.
            target_acceleration: Target acceleration [ax, ay, az]. If None, assumed to be zero.
            target_yaw: Target yaw angle in radians
            
        Returns:
            Motor thrusts [T1, T2, T3, T4]
        """
        # Set defaults
        if target_velocity is None:
            target_velocity = np.zeros(3)
        if target_acceleration is None:
            target_acceleration = np.zeros(3)
        
        # Set reference
        self.set_reference(target_position, target_velocity, target_acceleration, yaw=target_yaw)
        
        # Get current state
        position = quadrotor.get_position()
        velocity = quadrotor.get_velocity()
        R_current = quadrotor.get_rotation_matrix()
        omega_current = quadrotor.get_angular_velocity()
        
        # Position control
        position_error = position - self.reference_position
        velocity_error = velocity - self.reference_velocity
        
        # Desired acceleration from position control
        desired_acceleration = (
            -self.gains.kx * position_error
            - self.gains.kv * velocity_error
            + self.reference_acceleration
        )
        
        # Compute desired attitude
        R_desired, omega_desired = self._compute_desired_attitude(
            position, velocity, desired_acceleration
        )
        
        # Attitude control
        rotation_error = self._rotation_error(R_desired, R_current)
        angular_velocity_error = self._angular_velocity_error(
            omega_desired, omega_current, R_desired, R_current
        )
        
        # Desired angular acceleration
        desired_angular_acceleration = (
            -self.gains.kR * rotation_error
            - self.gains.kw_omega * angular_velocity_error
        )
        
        # Compute desired thrust
        gravity_vector = np.array([0, 0, self.params.gravity])
        total_acceleration = desired_acceleration + gravity_vector
        desired_thrust = self.params.mass * np.linalg.norm(total_acceleration)
        
        # Compute desired torques
        desired_torques = self.params.inertia @ desired_angular_acceleration
        
        # Convert to motor thrusts
        thrusts = self._thrust_allocation(desired_thrust, desired_torques)
        
        # Apply motor limits
        thrusts = np.clip(thrusts, 0.0, self.params.max_thrust)
        
        return thrusts
    
    def _thrust_allocation(
        self,
        desired_thrust: float,
        desired_torques: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Convert desired thrust and torques to motor thrusts.
        
        Args:
            desired_thrust: Desired total thrust in N
            desired_torques: Desired torques [tau_x, tau_y, tau_z] in N⋅m
            
        Returns:
            Motor thrusts [T1, T2, T3, T4] in N
        """
        L = self.params.arm_length
        
        # Allocation matrix
        # [T1]   [1  1  1  1] [T_total]
        # [T2] = [L -L  L -L] [tau_x  ]
        # [T3]   [L  L -L -L] [tau_y  ]
        # [T4]   [c -c  c -c] [tau_z  ]
        
        # where c is the yaw torque coefficient
        c = self.params.max_torque
        
        # Allocation matrix
        A = np.array([
            [1, 1, 1, 1],
            [L, -L, L, -L],
            [L, L, -L, -L],
            [c, -c, c, -c]
        ])
        
        # Desired outputs
        desired_outputs = np.array([desired_thrust, desired_torques[0], desired_torques[1], desired_torques[2]])
        
        # Solve for motor thrusts
        try:
            thrusts = np.linalg.solve(A, desired_outputs)
        except np.linalg.LinAlgError:
            # Fallback: simple allocation
            base_thrust = desired_thrust / 4.0
            thrusts = np.array([
                base_thrust + desired_torques[1] / (4 * L) + desired_torques[2] / 4,
                base_thrust - desired_torques[0] / (4 * L) - desired_torques[2] / 4,
                base_thrust - desired_torques[1] / (4 * L) + desired_torques[2] / 4,
                base_thrust + desired_torques[0] / (4 * L) - desired_torques[2] / 4
            ])
        
        return thrusts
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status information.
        
        Returns:
            Dictionary containing controller status
        """
        return {
            'kx': self.gains.kx,
            'kv': self.gains.kv,
            'kr': self.gains.kr,
            'kw': self.gains.kw,
            'kR': self.gains.kR,
            'kw_omega': self.gains.kw_omega,
            'reference_position': self.reference_position.tolist(),
            'reference_velocity': self.reference_velocity.tolist(),
            'reference_acceleration': self.reference_acceleration.tolist(),
            'reference_yaw': self.reference_yaw,
            'reference_yaw_rate': self.reference_yaw_rate,
        }
