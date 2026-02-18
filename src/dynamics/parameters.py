"""Quadrotor physical parameters and configurations."""

from __future__ import annotations

import numpy as np
from typing import Dict, Any
from dataclasses import dataclass, field
from numpy.typing import NDArray


@dataclass
class QuadrotorParameters:
    """Physical parameters of a quadrotor UAV.
    
    This class contains all the physical parameters needed to model
    a quadrotor UAV including mass, inertia, motor characteristics,
    and environmental parameters.
    """
    
    # Basic physical properties
    mass: float = 1.0  # kg
    arm_length: float = 0.25  # m
    gravity: float = 9.81  # m/s²
    
    # Inertia matrix (3x3) in kg⋅m²
    inertia: NDArray[np.float64] = field(default_factory=lambda: np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.2]
    ]))
    
    # Motor characteristics
    max_thrust: float = 20.0  # N per motor
    max_torque: float = 2.0  # N⋅m per motor
    motor_time_constant: float = 0.02  # s
    
    # Aerodynamic parameters
    drag_coefficient: float = 0.1  # Linear drag coefficient
    drag_coefficient_quad: float = 0.01  # Quadratic drag coefficient
    
    # Sensor noise parameters (for simulation)
    position_noise_std: float = 0.01  # m
    velocity_noise_std: float = 0.05  # m/s
    attitude_noise_std: float = 0.01  # rad
    angular_velocity_noise_std: float = 0.05  # rad/s
    
    # Control limits
    max_velocity: float = 10.0  # m/s
    max_angular_velocity: float = 5.0  # rad/s
    max_acceleration: float = 20.0  # m/s²
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary.
        
        Returns:
            Dictionary representation of parameters
        """
        return {
            'mass': self.mass,
            'arm_length': self.arm_length,
            'gravity': self.gravity,
            'inertia': self.inertia.tolist(),
            'max_thrust': self.max_thrust,
            'max_torque': self.max_torque,
            'motor_time_constant': self.motor_time_constant,
            'drag_coefficient': self.drag_coefficient,
            'drag_coefficient_quad': self.drag_coefficient_quad,
            'position_noise_std': self.position_noise_std,
            'velocity_noise_std': self.velocity_noise_std,
            'attitude_noise_std': self.attitude_noise_std,
            'angular_velocity_noise_std': self.angular_velocity_noise_std,
            'max_velocity': self.max_velocity,
            'max_angular_velocity': self.max_angular_velocity,
            'max_acceleration': self.max_acceleration,
        }
    
    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> QuadrotorParameters:
        """Create parameters from dictionary.
        
        Args:
            params_dict: Dictionary containing parameter values
            
        Returns:
            QuadrotorParameters instance
        """
        # Convert inertia list back to numpy array
        if 'inertia' in params_dict and isinstance(params_dict['inertia'], list):
            params_dict['inertia'] = np.array(params_dict['inertia'])
        
        return cls(**params_dict)


# Predefined quadrotor configurations
QUADROTOR_CONFIGS = {
    'small': QuadrotorParameters(
        mass=0.5,
        arm_length=0.15,
        inertia=np.array([
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, 0.0, 0.1]
        ]),
        max_thrust=10.0,
        max_torque=1.0,
    ),
    
    'medium': QuadrotorParameters(
        mass=1.0,
        arm_length=0.25,
        inertia=np.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.2]
        ]),
        max_thrust=20.0,
        max_torque=2.0,
    ),
    
    'large': QuadrotorParameters(
        mass=2.0,
        arm_length=0.35,
        inertia=np.array([
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.4]
        ]),
        max_thrust=40.0,
        max_torque=4.0,
    ),
    
    'racing': QuadrotorParameters(
        mass=0.8,
        arm_length=0.2,
        inertia=np.array([
            [0.08, 0.0, 0.0],
            [0.0, 0.08, 0.0],
            [0.0, 0.0, 0.16]
        ]),
        max_thrust=30.0,
        max_torque=3.0,
        max_velocity=15.0,
        max_angular_velocity=8.0,
    ),
}


def get_quadrotor_config(config_name: str) -> QuadrotorParameters:
    """Get predefined quadrotor configuration.
    
    Args:
        config_name: Name of the configuration ('small', 'medium', 'large', 'racing')
        
    Returns:
        QuadrotorParameters instance
        
    Raises:
        ValueError: If config_name is not recognized
    """
    if config_name not in QUADROTOR_CONFIGS:
        available_configs = list(QUADROTOR_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available configs: {available_configs}")
    
    return QUADROTOR_CONFIGS[config_name]
