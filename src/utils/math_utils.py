"""Mathematical utilities for quadrotor control.

This module provides mathematical utilities including coordinate transformations,
quaternion operations, and other mathematical functions used in quadrotor control.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """Convert Euler angles to quaternion.
    
    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians
        
    Returns:
        Quaternion [w, x, y, z]
    """
    # ZYX convention (yaw-pitch-roll)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def quaternion_to_euler(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert quaternion to Euler angles.
    
    Args:
        quaternion: Quaternion [w, x, y, z]
        
    Returns:
        Euler angles [roll, pitch, yaw] in radians
    """
    qw, qx, qy, qz = quaternion
    
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


def quaternion_to_rotation_matrix(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert quaternion to rotation matrix.
    
    Args:
        quaternion: Quaternion [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    qw, qx, qy, qz = quaternion
    
    # Normalize quaternion
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R


def rotation_matrix_to_quaternion(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert rotation matrix to quaternion.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion [w, x, y, z]
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def quaternion_multiply(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
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


def quaternion_conjugate(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute quaternion conjugate.
    
    Args:
        quaternion: Quaternion [w, x, y, z]
        
    Returns:
        Conjugate quaternion [w, -x, -y, -z]
    """
    return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])


def quaternion_norm(quaternion: NDArray[np.float64]) -> float:
    """Compute quaternion norm.
    
    Args:
        quaternion: Quaternion [w, x, y, z]
        
    Returns:
        Quaternion norm
    """
    return np.sqrt(np.sum(quaternion**2))


def quaternion_normalize(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize quaternion to unit quaternion.
    
    Args:
        quaternion: Quaternion [w, x, y, z]
        
    Returns:
        Normalized quaternion [w, x, y, z]
    """
    norm = quaternion_norm(quaternion)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])  # Default to identity quaternion
    return quaternion / norm


def skew_symmetric(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute skew-symmetric matrix from vector.
    
    Args:
        vector: 3D vector [x, y, z]
        
    Returns:
        3x3 skew-symmetric matrix
    """
    x, y, z = vector
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])


def rotation_error(R_desired: NDArray[np.float64], R_current: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute rotation error between two rotation matrices.
    
    Args:
        R_desired: Desired rotation matrix
        R_current: Current rotation matrix
        
    Returns:
        Rotation error vector
    """
    R_error = R_desired.T @ R_current
    trace_error = np.trace(R_error)
    
    # Skew-symmetric part
    skew_error = R_error - R_error.T
    
    # Extract rotation error vector
    error_vector = np.array([skew_error[2, 1], skew_error[0, 2], skew_error[1, 0]])
    
    return error_vector


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π].
    
    Args:
        angle: Angle in radians
        
    Returns:
        Wrapped angle in radians
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def wrap_angles(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap angles to [-π, π].
    
    Args:
        angles: Array of angles in radians
        
    Returns:
        Array of wrapped angles in radians
    """
    return np.arctan2(np.sin(angles), np.cos(angles))


def saturate(value: float, min_val: float, max_val: float) -> float:
    """Saturate value between min and max.
    
    Args:
        value: Value to saturate
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Saturated value
    """
    return np.clip(value, min_val, max_val)


def deadzone(value: float, threshold: float) -> float:
    """Apply deadzone to value.
    
    Args:
        value: Value to apply deadzone to
        threshold: Deadzone threshold
        
    Returns:
        Value with deadzone applied
    """
    if abs(value) < threshold:
        return 0.0
    return value


def low_pass_filter(
    current_value: float,
    previous_value: float,
    alpha: float
) -> float:
    """Apply low-pass filter.
    
    Args:
        current_value: Current input value
        previous_value: Previous output value
        alpha: Filter coefficient (0 < alpha < 1)
        
    Returns:
        Filtered value
    """
    return alpha * current_value + (1 - alpha) * previous_value


def derivative(
    current_value: float,
    previous_value: float,
    dt: float
) -> float:
    """Compute numerical derivative.
    
    Args:
        current_value: Current value
        previous_value: Previous value
        dt: Time step
        
    Returns:
        Derivative value
    """
    return (current_value - previous_value) / dt


def integrate(
    current_value: float,
    previous_integral: float,
    dt: float
) -> float:
    """Compute numerical integral using trapezoidal rule.
    
    Args:
        current_value: Current value
        previous_integral: Previous integral value
        dt: Time step
        
    Returns:
        Integral value
    """
    return previous_integral + current_value * dt


def normalize_vector(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize vector to unit length.
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return np.zeros_like(vector)
    return vector / norm


def angle_between_vectors(v1: NDArray[np.float64], v2: NDArray[np.float64]) -> float:
    """Compute angle between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Angle in radians
    """
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(dot_product)


def project_vector(v: NDArray[np.float64], n: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project vector v onto vector n.
    
    Args:
        v: Vector to project
        n: Vector to project onto
        
    Returns:
        Projected vector
    """
    n_norm = normalize_vector(n)
    return np.dot(v, n_norm) * n_norm


def reject_vector(v: NDArray[np.float64], n: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reject vector v from vector n (orthogonal component).
    
    Args:
        v: Vector to reject
        n: Vector to reject from
        
    Returns:
        Rejected vector
    """
    return v - project_vector(v, n)
