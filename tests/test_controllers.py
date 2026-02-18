"""Unit tests for PID controller."""

import pytest
import numpy as np
from src.controllers.pid import PIDController, PIDGains, PIDLimits
from src.dynamics.quadrotor import QuadrotorDynamics
from src.dynamics.parameters import QuadrotorParameters


class TestPIDController:
    """Test cases for PIDController class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = QuadrotorParameters()
        self.quadrotor = QuadrotorDynamics(self.params)
        
        # Set up PID controller
        gains = PIDGains()
        limits = PIDLimits()
        self.controller = PIDController(gains, limits, dt=0.01)
    
    def test_initialization(self):
        """Test PID controller initialization."""
        assert self.controller.dt == 0.01
        assert not self.controller.emergency_stop
        assert np.allclose(self.controller.position_integral, np.zeros(3))
        assert np.allclose(self.controller.position_previous_error, np.zeros(3))
    
    def test_reset_functionality(self):
        """Test controller reset."""
        # Modify controller state
        self.controller.position_integral[0] = 1.0
        self.controller.emergency_stop = True
        
        # Reset
        self.controller.reset()
        
        # Should be back to initial state
        assert np.allclose(self.controller.position_integral, np.zeros(3))
        assert not self.controller.emergency_stop
    
    def test_position_control(self):
        """Test position control computation."""
        # Set up test scenario
        position = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        target_position = np.array([1.0, 2.0, 3.0])
        target_velocity = np.array([0.0, 0.0, 0.0])
        
        # Compute control
        desired_acceleration = self.controller.compute_position_control(
            position, velocity, target_position, target_velocity
        )
        
        # Should have expected structure
        assert desired_acceleration.shape == (3,)
        
        # Should be non-zero for non-zero error
        assert not np.allclose(desired_acceleration, np.zeros(3))
    
    def test_attitude_control(self):
        """Test attitude control computation."""
        # Set up test scenario
        attitude = np.array([0.0, 0.0, 0.0])
        angular_velocity = np.array([0.0, 0.0, 0.0])
        target_attitude = np.array([0.1, 0.2, 0.3])
        target_angular_velocity = np.array([0.0, 0.0, 0.0])
        
        # Compute control
        desired_angular_acceleration = self.controller.compute_attitude_control(
            attitude, angular_velocity, target_attitude, target_angular_velocity
        )
        
        # Should have expected structure
        assert desired_angular_acceleration.shape == (3,)
        
        # Should be non-zero for non-zero error
        assert not np.allclose(desired_angular_acceleration, np.zeros(3))
    
    def test_safety_limits(self):
        """Test safety limit checking."""
        # Test with large position error
        large_position_error = np.array([20.0, 0.0, 0.0])  # Exceeds limit
        small_velocity_error = np.array([0.1, 0.1, 0.1])
        
        # Should trigger safety check
        is_unsafe = self.controller._check_safety_limits(large_position_error, small_velocity_error)
        assert is_unsafe
        
        # Test with normal errors
        normal_position_error = np.array([1.0, 1.0, 1.0])
        normal_velocity_error = np.array([0.1, 0.1, 0.1])
        
        is_unsafe = self.controller._check_safety_limits(normal_position_error, normal_velocity_error)
        assert not is_unsafe
    
    def test_integral_windup_protection(self):
        """Test integral windup protection."""
        # Set up scenario with large error
        error = np.array([10.0, 10.0, 10.0])
        integral = np.array([0.0, 0.0, 0.0])
        previous_error = np.array([0.0, 0.0, 0.0])
        previous_derivative = np.array([0.0, 0.0, 0.0])
        
        # Compute PID with windup protection
        control_signal, updated_integral, updated_derivative = self.controller._compute_pid(
            error, integral, previous_error, previous_derivative,
            self.controller.gains.position_kp,
            self.controller.gains.position_ki,
            self.controller.gains.position_kd
        )
        
        # Integral should be clamped
        assert np.all(updated_integral <= self.controller.limits.max_integral_windup)
        assert np.all(updated_integral >= -self.controller.limits.max_integral_windup)
    
    def test_control_limits(self):
        """Test control output limits."""
        # Set up scenario that would produce large control signal
        position = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        target_position = np.array([100.0, 100.0, 100.0])  # Large target
        
        desired_acceleration = self.controller.compute_position_control(
            position, velocity, target_position
        )
        
        # Should be clamped to limits
        assert np.all(desired_acceleration <= self.controller.limits.max_control_output)
        assert np.all(desired_acceleration >= -self.controller.limits.max_control_output)
    
    def test_complete_control_loop(self):
        """Test complete control loop."""
        # Set up quadrotor at origin
        self.quadrotor.reset()
        
        # Set target position
        target_position = np.array([1.0, 1.0, 1.0])
        
        # Compute control
        thrusts = self.controller.compute_control(
            self.quadrotor,
            target_position
        )
        
        # Should have expected structure
        assert thrusts.shape == (4,)
        
        # Should be within motor limits
        assert np.all(thrusts >= 0.0)
        assert np.all(thrusts <= self.params.max_thrust)
    
    def test_emergency_stop(self):
        """Test emergency stop functionality."""
        # Trigger emergency stop
        self.controller.emergency_stop = True
        
        # Control should return zeros
        thrusts = self.controller.compute_control(
            self.quadrotor,
            np.array([1.0, 1.0, 1.0])
        )
        
        assert np.allclose(thrusts, np.zeros(4))
    
    def test_status_reporting(self):
        """Test status reporting."""
        status = self.controller.get_status()
        
        # Should contain expected keys
        expected_keys = ['emergency_stop', 'position_integral', 'attitude_integral',
                        'position_previous_error', 'attitude_previous_error']
        
        for key in expected_keys:
            assert key in status


if __name__ == "__main__":
    pytest.main([__file__])
