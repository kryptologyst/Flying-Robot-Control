"""Quadrotor dynamics and physical parameters.

This module implements the mathematical model of a quadrotor UAV including
dynamics, kinematics, and physical parameters.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class QuadrotorParameters:
    """Physical parameters of a quadrotor UAV.
    
    Attributes:
        mass: Mass of the quadrotor in kg
        arm_length: Distance from center to motor in m
        inertia: Moment of inertia matrix (3x3) in kg⋅m²
        max_thrust: Maximum thrust per motor in N
        max_torque: Maximum torque per motor in N⋅m
        gravity: Gravitational acceleration in m/s²
        drag_coefficient: Air drag coefficient
    """
    mass: float = 1.0  # kg
    arm_length: float = 0.25  # m
    inertia: NDArray[np.float64] = None
    max_thrust: float = 20.0  # N
    max_torque: float = 2.0  # N⋅m
    gravity: float = 9.81  # m/s²
    drag_coefficient: float = 0.1
    
    def __post_init__(self) -> None:
        """Initialize default inertia matrix if not provided."""
        if self.inertia is None:
            # Default inertia matrix for a symmetric quadrotor
            self.inertia = np.array([
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.2]
            ])


class QuadrotorDynamics:
    """Quadrotor dynamics model with 6-DOF rigid body dynamics.
    
    This class implements the mathematical model of a quadrotor UAV including:
    - Position dynamics (x, y, z)
    - Attitude dynamics (roll, pitch, yaw)
    - Motor dynamics and thrust allocation
    - Environmental forces (gravity, drag)
    """
    
    def __init__(self, params: Optional[QuadrotorParameters] = None) -> None:
        """Initialize quadrotor dynamics.
        
        Args:
            params: Quadrotor physical parameters. If None, uses default values.
        """
        self.params = params or QuadrotorParameters()
        
        # State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        # Position (3), velocity (3), quaternion (4), angular velocity (3)
        self.state_dim = 13
        self.control_dim = 4  # Motor thrusts
        
        # Initialize state
        self.state = np.zeros(self.state_dim)
        self.state[6] = 1.0  # Initialize quaternion w component to 1
        
    def get_state(self) -> NDArray[np.float64]:
        """Get current state vector.
        
        Returns:
            State vector [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        """
        return self.state.copy()
    
    def set_state(self, state: NDArray[np.float64]) -> None:
        """Set quadrotor state.
        
        Args:
            state: State vector [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        """
        assert state.shape == (self.state_dim,), f"Expected state shape ({self.state_dim},), got {state.shape}"
        self.state = state.copy()
    
    def get_position(self) -> NDArray[np.float64]:
        """Get current position.
        
        Returns:
            Position vector [x, y, z] in meters
        """
        return self.state[:3]
    
    def get_velocity(self) -> NDArray[np.float64]:
        """Get current velocity.
        
        Returns:
            Velocity vector [vx, vy, vz] in m/s
        """
        return self.state[3:6]
    
    def get_quaternion(self) -> NDArray[np.float64]:
        """Get current attitude as quaternion.
        
        Returns:
            Quaternion [qw, qx, qy, qz] (unit quaternion)
        """
        return self.state[6:10]
    
    def get_angular_velocity(self) -> NDArray[np.float64]:
        """Get current angular velocity.
        
        Returns:
            Angular velocity [wx, wy, wz] in rad/s
        """
        return self.state[10:13]
    
    def get_euler_angles(self) -> NDArray[np.float64]:
        """Get current attitude as Euler angles (ZYX convention).
        
        Returns:
            Euler angles [roll, pitch, yaw] in radians
        """
        q = self.get_quaternion()
        qw, qx, qy, qz = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def get_rotation_matrix(self) -> NDArray[np.float64]:
        """Get rotation matrix from body to world frame.
        
        Returns:
            3x3 rotation matrix
        """
        q = self.get_quaternion()
        qw, qx, qy, qz = q
        
        # Convert quaternion to rotation matrix
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        
        return R
    
    def thrust_to_forces(self, thrusts: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert motor thrusts to total force and torque.
        
        Args:
            thrusts: Motor thrusts [T1, T2, T3, T4] in N
            
        Returns:
            Tuple of (total_force, total_torque) in world frame
        """
        # Clamp thrusts to limits
        thrusts = np.clip(thrusts, 0.0, self.params.max_thrust)
        
        # Total thrust force in body frame (z-axis)
        total_thrust = np.sum(thrusts)
        force_body = np.array([0.0, 0.0, total_thrust])
        
        # Convert to world frame
        R = self.get_rotation_matrix()
        force_world = R @ force_body
        
        # Torques in body frame
        # Motor layout: 1-front, 2-right, 3-back, 4-left (clockwise from front)
        L = self.params.arm_length
        torque_body = np.array([
            L * (thrusts[1] - thrusts[3]),  # Roll torque
            L * (thrusts[0] - thrusts[2]),  # Pitch torque
            self.params.max_torque * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3])  # Yaw torque
        ])
        
        return force_world, torque_body
    
    def dynamics(self, state: NDArray[np.float64], control: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute state derivatives.
        
        Args:
            state: Current state vector
            control: Motor thrusts [T1, T2, T3, T4]
            
        Returns:
            State derivatives
        """
        # Extract state components
        position = state[:3]
        velocity = state[3:6]
        quaternion = state[6:10]
        angular_velocity = state[10:13]
        
        # Normalize quaternion
        quaternion = quaternion / np.linalg.norm(quaternion)
        
        # Convert control to forces and torques
        force_world, torque_body = self.thrust_to_forces(control)
        
        # Position dynamics: d/dt [x, y, z] = [vx, vy, vz]
        position_dot = velocity
        
        # Velocity dynamics: m * dv/dt = F_total - m*g + drag
        gravity_force = np.array([0.0, 0.0, -self.params.mass * self.params.gravity])
        drag_force = -self.params.drag_coefficient * velocity
        velocity_dot = (force_world + gravity_force + drag_force) / self.params.mass
        
        # Quaternion dynamics: d/dt q = 0.5 * q * [0, wx, wy, wz]
        omega_quat = np.array([0.0, angular_velocity[0], angular_velocity[1], angular_velocity[2]])
        quaternion_dot = 0.5 * self._quaternion_multiply(quaternion, omega_quat)
        
        # Angular velocity dynamics: I * dw/dt = tau - w × (I * w)
        Iw = self.params.inertia @ angular_velocity
        angular_velocity_dot = np.linalg.solve(
            self.params.inertia,
            torque_body - np.cross(angular_velocity, Iw)
        )
        
        # Combine derivatives
        state_dot = np.concatenate([
            position_dot,
            velocity_dot,
            quaternion_dot,
            angular_velocity_dot
        ])
        
        return state_dot
    
    def _quaternion_multiply(self, q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
        """Multiply two quaternions.
        
        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]
            
        Returns:
            Product quaternion [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def step(self, control: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        """Integrate dynamics for one time step.
        
        Args:
            control: Motor thrusts [T1, T2, T3, T4]
            dt: Time step in seconds
            
        Returns:
            New state vector
        """
        # Use Runge-Kutta 4th order integration
        k1 = self.dynamics(self.state, control)
        k2 = self.dynamics(self.state + 0.5*dt*k1, control)
        k3 = self.dynamics(self.state + 0.5*dt*k2, control)
        k4 = self.dynamics(self.state + dt*k3, control)
        
        self.state += (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize quaternion to prevent drift
        self.state[6:10] = self.state[6:10] / np.linalg.norm(self.state[6:10])
        
        return self.state.copy()
    
    def reset(self, initial_state: Optional[NDArray[np.float64]] = None) -> None:
        """Reset quadrotor to initial state.
        
        Args:
            initial_state: Initial state vector. If None, resets to origin with zero velocity.
        """
        if initial_state is None:
            self.state = np.zeros(self.state_dim)
            self.state[6] = 1.0  # Initialize quaternion w component to 1
        else:
            self.set_state(initial_state)
