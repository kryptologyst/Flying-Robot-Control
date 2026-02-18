"""PID controller implementation for quadrotor control.

This module implements a modern PID controller with integral windup protection,
derivative kick prevention, and safety limits for quadrotor UAV control.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from numpy.typing import NDArray

from dynamics.quadrotor import QuadrotorDynamics
from dynamics.parameters import QuadrotorParameters


@dataclass
class PIDGains:
    """PID controller gains for position and attitude control.
    
    Attributes:
        position_kp: Proportional gains for position [x, y, z]
        position_ki: Integral gains for position [x, y, z]
        position_kd: Derivative gains for position [x, y, z]
        attitude_kp: Proportional gains for attitude [roll, pitch, yaw]
        attitude_ki: Integral gains for attitude [roll, pitch, yaw]
        attitude_kd: Derivative gains for attitude [roll, pitch, yaw]
    """
    position_kp: NDArray[np.float64] = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    position_ki: NDArray[np.float64] = field(default_factory=lambda: np.array([0.1, 0.1, 0.1]))
    position_kd: NDArray[np.float64] = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))
    
    attitude_kp: NDArray[np.float64] = field(default_factory=lambda: np.array([2.0, 2.0, 1.0]))
    attitude_ki: NDArray[np.float64] = field(default_factory=lambda: np.array([0.05, 0.05, 0.05]))
    attitude_kd: NDArray[np.float64] = field(default_factory=lambda: np.array([0.3, 0.3, 0.3]))


@dataclass
class PIDLimits:
    """Safety limits for PID controller.
    
    Attributes:
        max_position_error: Maximum position error before emergency stop
        max_velocity_error: Maximum velocity error before emergency stop
        max_integral_windup: Maximum integral windup for anti-windup
        max_control_output: Maximum control output
        min_control_output: Minimum control output
    """
    max_position_error: float = 10.0  # m
    max_velocity_error: float = 5.0  # m/s
    max_integral_windup: float = 2.0  # Integral windup limit
    max_control_output: float = 20.0  # N
    min_control_output: float = 0.0  # N


class PIDController:
    """PID controller for quadrotor position and attitude control.
    
    This controller implements separate PID loops for position and attitude control
    with safety features including integral windup protection, derivative kick
    prevention, and emergency stop capabilities.
    """
    
    def __init__(
        self,
        gains: Optional[PIDGains] = None,
        limits: Optional[PIDLimits] = None,
        dt: float = 0.01
    ) -> None:
        """Initialize PID controller.
        
        Args:
            gains: PID controller gains
            limits: Safety limits for the controller
            dt: Control loop time step in seconds
        """
        self.gains = gains or PIDGains()
        self.limits = limits or PIDLimits()
        self.dt = dt
        
        # Controller state
        self.position_integral = np.zeros(3)
        self.position_previous_error = np.zeros(3)
        self.position_previous_derivative = np.zeros(3)
        
        self.attitude_integral = np.zeros(3)
        self.attitude_previous_error = np.zeros(3)
        self.attitude_previous_derivative = np.zeros(3)
        
        # Safety flags
        self.emergency_stop = False
        self.last_control_time = 0.0
        
    def reset(self) -> None:
        """Reset controller state."""
        self.position_integral.fill(0.0)
        self.position_previous_error.fill(0.0)
        self.position_previous_derivative.fill(0.0)
        
        self.attitude_integral.fill(0.0)
        self.attitude_previous_error.fill(0.0)
        self.attitude_previous_derivative.fill(0.0)
        
        self.emergency_stop = False
        
    def _check_safety_limits(self, position_error: NDArray[np.float64], velocity_error: NDArray[np.float64]) -> bool:
        """Check if safety limits are exceeded.
        
        Args:
            position_error: Current position error
            velocity_error: Current velocity error
            
        Returns:
            True if safety limits are exceeded
        """
        max_pos_error = np.max(np.abs(position_error))
        max_vel_error = np.max(np.abs(velocity_error))
        
        if max_pos_error > self.limits.max_position_error:
            print(f"WARNING: Position error {max_pos_error:.2f} exceeds limit {self.limits.max_position_error}")
            return True
            
        if max_vel_error > self.limits.max_velocity_error:
            print(f"WARNING: Velocity error {max_vel_error:.2f} exceeds limit {self.limits.max_velocity_error}")
            return True
            
        return False
    
    def _compute_pid(
        self,
        error: NDArray[np.float64],
        integral: NDArray[np.float64],
        previous_error: NDArray[np.float64],
        previous_derivative: NDArray[np.float64],
        kp: NDArray[np.float64],
        ki: NDArray[np.float64],
        kd: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Compute PID control signal with anti-windup and derivative kick prevention.
        
        Args:
            error: Current error
            integral: Current integral term
            previous_error: Previous error for derivative calculation
            previous_derivative: Previous derivative for smoothing
            kp: Proportional gains
            ki: Integral gains
            kd: Derivative gains
            
        Returns:
            Tuple of (control_signal, updated_integral, updated_derivative)
        """
        # Proportional term
        proportional = kp * error
        
        # Integral term with anti-windup
        integral += error * self.dt
        integral = np.clip(integral, -self.limits.max_integral_windup, self.limits.max_integral_windup)
        integral_term = ki * integral
        
        # Derivative term with derivative kick prevention
        # Use derivative of measurement instead of derivative of error
        derivative = (error - previous_error) / self.dt
        # Apply low-pass filter to derivative to reduce noise
        alpha = 0.1  # Filter coefficient
        derivative = alpha * derivative + (1 - alpha) * previous_derivative
        derivative_term = kd * derivative
        
        # Total control signal
        control_signal = proportional + integral_term - derivative_term
        
        return control_signal, integral, derivative
    
    def compute_position_control(
        self,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        target_position: NDArray[np.float64],
        target_velocity: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """Compute position control signal.
        
        Args:
            position: Current position [x, y, z]
            velocity: Current velocity [vx, vy, vz]
            target_position: Target position [x, y, z]
            target_velocity: Target velocity [vx, vy, vz]. If None, assumed to be zero.
            
        Returns:
            Desired acceleration [ax, ay, az] in world frame
        """
        if target_velocity is None:
            target_velocity = np.zeros(3)
            
        # Compute errors
        position_error = target_position - position
        velocity_error = target_velocity - velocity
        
        # Check safety limits
        if self._check_safety_limits(position_error, velocity_error):
            self.emergency_stop = True
            return np.zeros(3)
        
        # Compute PID control
        desired_acceleration, self.position_integral, self.position_previous_derivative = self._compute_pid(
            position_error,
            self.position_integral,
            self.position_previous_error,
            self.position_previous_derivative,
            self.gains.position_kp,
            self.gains.position_ki,
            self.gains.position_kd
        )
        
        # Update previous error
        self.position_previous_error = position_error.copy()
        
        # Apply limits
        desired_acceleration = np.clip(
            desired_acceleration,
            -self.limits.max_control_output,
            self.limits.max_control_output
        )
        
        return desired_acceleration
    
    def compute_attitude_control(
        self,
        attitude: NDArray[np.float64],
        angular_velocity: NDArray[np.float64],
        target_attitude: NDArray[np.float64],
        target_angular_velocity: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """Compute attitude control signal.
        
        Args:
            attitude: Current attitude as Euler angles [roll, pitch, yaw]
            angular_velocity: Current angular velocity [wx, wy, wz]
            target_attitude: Target attitude as Euler angles [roll, pitch, yaw]
            target_angular_velocity: Target angular velocity [wx, wy, wz]. If None, assumed to be zero.
            
        Returns:
            Desired angular acceleration [wax, way, waz] in body frame
        """
        if target_angular_velocity is None:
            target_angular_velocity = np.zeros(3)
            
        # Compute attitude error (handle angle wrapping)
        attitude_error = target_attitude - attitude
        attitude_error = np.arctan2(np.sin(attitude_error), np.cos(attitude_error))
        
        angular_velocity_error = target_angular_velocity - angular_velocity
        
        # Check safety limits
        if self._check_safety_limits(attitude_error, angular_velocity_error):
            self.emergency_stop = True
            return np.zeros(3)
        
        # Compute PID control
        desired_angular_acceleration, self.attitude_integral, self.attitude_previous_derivative = self._compute_pid(
            attitude_error,
            self.attitude_integral,
            self.attitude_previous_error,
            self.attitude_previous_derivative,
            self.gains.attitude_kp,
            self.gains.attitude_ki,
            self.gains.attitude_kd
        )
        
        # Update previous error
        self.attitude_previous_error = attitude_error.copy()
        
        # Apply limits
        desired_angular_acceleration = np.clip(
            desired_angular_acceleration,
            -self.limits.max_control_output,
            self.limits.max_control_output
        )
        
        return desired_angular_acceleration
    
    def compute_control(
        self,
        quadrotor: QuadrotorDynamics,
        target_position: NDArray[np.float64],
        target_attitude: Optional[NDArray[np.float64]] = None,
        target_velocity: Optional[NDArray[np.float64]] = None,
        target_angular_velocity: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """Compute complete control signal for quadrotor.
        
        Args:
            quadrotor: Quadrotor dynamics instance
            target_position: Target position [x, y, z]
            target_attitude: Target attitude [roll, pitch, yaw]. If None, computed from position control.
            target_velocity: Target velocity [vx, vy, vz]. If None, assumed to be zero.
            target_angular_velocity: Target angular velocity [wx, wy, wz]. If None, assumed to be zero.
            
        Returns:
            Motor thrusts [T1, T2, T3, T4]
        """
        if self.emergency_stop:
            return np.zeros(4)
        
        # Get current state
        position = quadrotor.get_position()
        velocity = quadrotor.get_velocity()
        attitude = quadrotor.get_euler_angles()
        angular_velocity = quadrotor.get_angular_velocity()
        
        # Compute position control (desired acceleration)
        desired_acceleration = self.compute_position_control(
            position, velocity, target_position, target_velocity
        )
        
        # Compute desired attitude from position control
        if target_attitude is None:
            # Compute desired attitude from desired acceleration
            # This is a simplified approach - in practice, you'd use more sophisticated methods
            gravity = quadrotor.params.gravity
            total_acceleration = np.linalg.norm(desired_acceleration + np.array([0, 0, gravity]))
            
            # Desired thrust direction
            thrust_direction = (desired_acceleration + np.array([0, 0, gravity])) / total_acceleration
            
            # Convert to Euler angles (simplified)
            target_roll = np.arctan2(thrust_direction[1], thrust_direction[2])
            target_pitch = -np.arcsin(thrust_direction[0])
            target_yaw = 0.0  # Keep yaw at zero for now
            
            target_attitude = np.array([target_roll, target_pitch, target_yaw])
        else:
            # Use provided target attitude
            gravity = quadrotor.params.gravity
            total_acceleration = np.linalg.norm(desired_acceleration + np.array([0, 0, gravity]))
        
        # Compute attitude control
        desired_angular_acceleration = self.compute_attitude_control(
            attitude, angular_velocity, target_attitude, target_angular_velocity
        )
        
        # Convert to motor thrusts (simplified allocation)
        # This is a basic allocation - in practice, you'd use more sophisticated methods
        total_thrust = quadrotor.params.mass * total_acceleration
        
        # Basic motor allocation (this is simplified)
        base_thrust = total_thrust / 4.0
        
        # Add attitude control torques
        L = quadrotor.params.arm_length
        roll_torque = desired_angular_acceleration[0] * quadrotor.params.inertia[0, 0]
        pitch_torque = desired_angular_acceleration[1] * quadrotor.params.inertia[1, 1]
        yaw_torque = desired_angular_acceleration[2] * quadrotor.params.inertia[2, 2]
        
        # Motor thrusts (simplified allocation)
        thrusts = np.array([
            base_thrust + pitch_torque / (4 * L) + yaw_torque / 4,  # Front
            base_thrust + roll_torque / (4 * L) - yaw_torque / 4,   # Right
            base_thrust - pitch_torque / (4 * L) + yaw_torque / 4,  # Back
            base_thrust - roll_torque / (4 * L) - yaw_torque / 4    # Left
        ])
        
        # Apply motor limits
        thrusts = np.clip(thrusts, 0.0, quadrotor.params.max_thrust)
        
        return thrusts
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status information.
        
        Returns:
            Dictionary containing controller status
        """
        return {
            'emergency_stop': self.emergency_stop,
            'position_integral': self.position_integral.tolist(),
            'attitude_integral': self.attitude_integral.tolist(),
            'position_previous_error': self.position_previous_error.tolist(),
            'attitude_previous_error': self.attitude_previous_error.tolist(),
        }
