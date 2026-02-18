"""Unit tests for mathematical utilities."""

import pytest
import numpy as np
from src.utils.math_utils import (
    euler_to_quaternion, quaternion_to_euler,
    quaternion_to_rotation_matrix, rotation_matrix_to_quaternion,
    quaternion_multiply, quaternion_conjugate, quaternion_norm,
    quaternion_normalize, skew_symmetric, rotation_error,
    wrap_angle, wrap_angles, saturate, deadzone,
    low_pass_filter, derivative, integrate,
    normalize_vector, angle_between_vectors,
    project_vector, reject_vector
)


class TestQuaternionOperations:
    """Test cases for quaternion operations."""
    
    def test_euler_to_quaternion_identity(self):
        """Test Euler to quaternion conversion for identity rotation."""
        quat = euler_to_quaternion(0.0, 0.0, 0.0)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(quat, expected)
    
    def test_quaternion_to_euler_identity(self):
        """Test quaternion to Euler conversion for identity quaternion."""
        euler = quaternion_to_euler(np.array([1.0, 0.0, 0.0, 0.0]))
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(euler, expected)
    
    def test_euler_quaternion_roundtrip(self):
        """Test roundtrip conversion: Euler -> quaternion -> Euler."""
        original_euler = np.array([0.1, 0.2, 0.3])
        quat = euler_to_quaternion(*original_euler)
        recovered_euler = quaternion_to_euler(quat)
        
        # Should recover original angles (within numerical precision)
        np.testing.assert_array_almost_equal(original_euler, recovered_euler, decimal=5)
    
    def test_quaternion_to_rotation_matrix_identity(self):
        """Test quaternion to rotation matrix for identity quaternion."""
        R = quaternion_to_rotation_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(R, expected)
    
    def test_rotation_matrix_to_quaternion_identity(self):
        """Test rotation matrix to quaternion for identity matrix."""
        quat = rotation_matrix_to_quaternion(np.eye(3))
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(quat, expected)
    
    def test_quaternion_multiply_identity(self):
        """Test quaternion multiplication with identity."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.5, 0.5, 0.5, 0.5])
        
        result = quaternion_multiply(q1, q2)
        np.testing.assert_array_almost_equal(result, q2)
        
        result = quaternion_multiply(q2, q1)
        np.testing.assert_array_almost_equal(result, q2)
    
    def test_quaternion_conjugate(self):
        """Test quaternion conjugate."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        conj = quaternion_conjugate(q)
        expected = np.array([0.5, -0.5, -0.5, -0.5])
        np.testing.assert_array_almost_equal(conj, expected)
    
    def test_quaternion_norm(self):
        """Test quaternion norm computation."""
        q = np.array([3.0, 4.0, 0.0, 0.0])
        norm = quaternion_norm(q)
        expected = 5.0
        np.testing.assert_almost_equal(norm, expected)
    
    def test_quaternion_normalize(self):
        """Test quaternion normalization."""
        q = np.array([3.0, 4.0, 0.0, 0.0])
        normalized = quaternion_normalize(q)
        norm = quaternion_norm(normalized)
        np.testing.assert_almost_equal(norm, 1.0)
    
    def test_quaternion_normalize_zero(self):
        """Test quaternion normalization with zero quaternion."""
        q = np.array([0.0, 0.0, 0.0, 0.0])
        normalized = quaternion_normalize(q)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(normalized, expected)


class TestMatrixOperations:
    """Test cases for matrix operations."""
    
    def test_skew_symmetric(self):
        """Test skew-symmetric matrix computation."""
        v = np.array([1.0, 2.0, 3.0])
        S = skew_symmetric(v)
        
        # Should be skew-symmetric
        assert np.allclose(S, -S.T)
        
        # Should have correct structure
        expected = np.array([
            [0, -3, 2],
            [3, 0, -1],
            [-2, 1, 0]
        ])
        np.testing.assert_array_almost_equal(S, expected)
    
    def test_rotation_error_identity(self):
        """Test rotation error computation for identical rotations."""
        R = np.eye(3)
        error = rotation_error(R, R)
        expected = np.zeros(3)
        np.testing.assert_array_almost_equal(error, expected)


class TestAngleOperations:
    """Test cases for angle operations."""
    
    def test_wrap_angle(self):
        """Test angle wrapping."""
        # Test normal angles
        assert wrap_angle(0.5) == 0.5
        assert wrap_angle(-0.5) == -0.5
        
        # Test wrapping
        assert abs(wrap_angle(3.5) - (3.5 - 2*np.pi)) < 1e-10
        assert abs(wrap_angle(-3.5) - (-3.5 + 2*np.pi)) < 1e-10
    
    def test_wrap_angles(self):
        """Test angle wrapping for arrays."""
        angles = np.array([0.5, -0.5, 3.5, -3.5])
        wrapped = wrap_angles(angles)
        
        # All angles should be in [-π, π]
        assert np.all(wrapped >= -np.pi)
        assert np.all(wrapped <= np.pi)


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_saturate(self):
        """Test saturation function."""
        assert saturate(5.0, 0.0, 10.0) == 5.0
        assert saturate(-5.0, 0.0, 10.0) == 0.0
        assert saturate(15.0, 0.0, 10.0) == 10.0
    
    def test_deadzone(self):
        """Test deadzone function."""
        assert deadzone(0.1, 0.2) == 0.0
        assert deadzone(0.3, 0.2) == 0.3
        assert deadzone(-0.1, 0.2) == 0.0
        assert deadzone(-0.3, 0.2) == -0.3
    
    def test_low_pass_filter(self):
        """Test low-pass filter."""
        # Test with constant input
        result = low_pass_filter(1.0, 0.0, 0.5)
        assert result == 0.5
        
        # Test with previous value
        result = low_pass_filter(0.0, 1.0, 0.5)
        assert result == 0.5
    
    def test_derivative(self):
        """Test numerical derivative."""
        result = derivative(1.0, 0.0, 0.1)
        assert result == 10.0
    
    def test_integrate(self):
        """Test numerical integration."""
        result = integrate(1.0, 0.0, 0.1)
        assert result == 0.1
    
    def test_normalize_vector(self):
        """Test vector normalization."""
        v = np.array([3.0, 4.0, 0.0])
        normalized = normalize_vector(v)
        norm = np.linalg.norm(normalized)
        np.testing.assert_almost_equal(norm, 1.0)
    
    def test_normalize_vector_zero(self):
        """Test vector normalization with zero vector."""
        v = np.array([0.0, 0.0, 0.0])
        normalized = normalize_vector(v)
        np.testing.assert_array_almost_equal(normalized, v)
    
    def test_angle_between_vectors(self):
        """Test angle between vectors."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        angle = angle_between_vectors(v1, v2)
        np.testing.assert_almost_equal(angle, np.pi/2)
        
        # Test parallel vectors
        v3 = np.array([2.0, 0.0, 0.0])
        angle = angle_between_vectors(v1, v3)
        np.testing.assert_almost_equal(angle, 0.0)
    
    def test_project_vector(self):
        """Test vector projection."""
        v = np.array([1.0, 1.0, 0.0])
        n = np.array([1.0, 0.0, 0.0])
        projection = project_vector(v, n)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(projection, expected)
    
    def test_reject_vector(self):
        """Test vector rejection."""
        v = np.array([1.0, 1.0, 0.0])
        n = np.array([1.0, 0.0, 0.0])
        rejection = reject_vector(v, n)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(rejection, expected)


if __name__ == "__main__":
    pytest.main([__file__])
