"""Unit tests for quadrotor dynamics."""

import pytest
import numpy as np
from src.dynamics.quadrotor import QuadrotorDynamics
from src.dynamics.parameters import QuadrotorParameters


class TestQuadrotorDynamics:
    """Test cases for QuadrotorDynamics class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = QuadrotorParameters()
        self.quadrotor = QuadrotorDynamics(self.params)
    
    def test_initialization(self):
        """Test quadrotor initialization."""
        assert self.quadrotor.params.mass == 1.0
        assert self.quadrotor.params.arm_length == 0.25
        assert self.quadrotor.state_dim == 13
        assert self.quadrotor.control_dim == 4
    
    def test_state_getters(self):
        """Test state getter methods."""
        # Set a known state
        test_state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.03])
        self.quadrotor.set_state(test_state)
        
        # Test getters
        position = self.quadrotor.get_position()
        velocity = self.quadrotor.get_velocity()
        quaternion = self.quadrotor.get_quaternion()
        angular_velocity = self.quadrotor.get_angular_velocity()
        
        np.testing.assert_array_equal(position, test_state[:3])
        np.testing.assert_array_equal(velocity, test_state[3:6])
        np.testing.assert_array_equal(quaternion, test_state[6:10])
        np.testing.assert_array_equal(angular_velocity, test_state[10:13])
    
    def test_euler_angles_conversion(self):
        """Test Euler angles conversion."""
        # Test identity quaternion (no rotation)
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.quadrotor.state[6:10] = identity_quat
        euler = self.quadrotor.get_euler_angles()
        
        np.testing.assert_array_almost_equal(euler, np.zeros(3), decimal=5)
    
    def test_rotation_matrix(self):
        """Test rotation matrix computation."""
        # Test identity quaternion
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.quadrotor.state[6:10] = identity_quat
        R = self.quadrotor.get_rotation_matrix()
        
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=5)
    
    def test_thrust_to_forces(self):
        """Test thrust to forces conversion."""
        # Test hover condition
        hover_thrust = self.params.mass * self.params.gravity / 4.0
        thrusts = np.array([hover_thrust, hover_thrust, hover_thrust, hover_thrust])
        
        force_world, torque_body = self.quadrotor.thrust_to_forces(thrusts)
        
        # Total thrust should equal weight
        expected_total_thrust = self.params.mass * self.params.gravity
        np.testing.assert_almost_equal(np.linalg.norm(force_world), expected_total_thrust, decimal=5)
        
        # Torques should be zero for symmetric thrusts
        np.testing.assert_array_almost_equal(torque_body, np.zeros(3), decimal=5)
    
    def test_dynamics_consistency(self):
        """Test dynamics consistency."""
        # Test with zero control input
        zero_control = np.zeros(4)
        state_dot = self.quadrotor.dynamics(self.quadrotor.state, zero_control)
        
        # Should have expected structure
        assert state_dot.shape == (13,)
        
        # Position derivative should equal velocity
        np.testing.assert_array_equal(state_dot[:3], self.quadrotor.state[3:6])
    
    def test_step_integration(self):
        """Test step integration."""
        initial_position = self.quadrotor.get_position().copy()
        
        # Apply small control input
        small_thrust = np.array([0.1, 0.1, 0.1, 0.1])
        dt = 0.01
        
        self.quadrotor.step(small_thrust, dt)
        
        # Position should change
        new_position = self.quadrotor.get_position()
        assert not np.array_equal(initial_position, new_position)
    
    def test_quaternion_normalization(self):
        """Test quaternion normalization after step."""
        # Set unnormalized quaternion
        unnormalized_quat = np.array([2.0, 1.0, 0.5, 0.25])
        self.quadrotor.state[6:10] = unnormalized_quat
        
        # Take a step
        self.quadrotor.step(np.zeros(4), 0.01)
        
        # Quaternion should be normalized
        quat_norm = np.linalg.norm(self.quadrotor.get_quaternion())
        np.testing.assert_almost_equal(quat_norm, 1.0, decimal=5)
    
    def test_reset_functionality(self):
        """Test reset functionality."""
        # Modify state
        self.quadrotor.state[0] = 10.0
        
        # Reset
        self.quadrotor.reset()
        
        # Should be back to initial state
        assert self.quadrotor.state[0] == 0.0
        assert self.quadrotor.state[6] == 1.0  # Identity quaternion


if __name__ == "__main__":
    pytest.main([__file__])
