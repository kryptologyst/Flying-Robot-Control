"""Flying Robot Control Package."""

__version__ = "1.0.0"
__author__ = "Robotics Team"
__email__ = "team@example.com"

from .dynamics.quadrotor import QuadrotorDynamics
from .dynamics.parameters import QuadrotorParameters, get_quadrotor_config
from .controllers.pid import PIDController, PIDGains, PIDLimits
from .controllers.lqr import LQRController, LQRWeights
from .controllers.mpc import MPCController, MPCConfig
from .controllers.geometric import GeometricController, GeometricControllerGains
from .utils.math_utils import *
from .utils.visualization import *

__all__ = [
    # Dynamics
    "QuadrotorDynamics",
    "QuadrotorParameters",
    "get_quadrotor_config",
    
    # Controllers
    "PIDController",
    "PIDGains", 
    "PIDLimits",
    "LQRController",
    "LQRWeights",
    "MPCController",
    "MPCConfig",
    "GeometricController",
    "GeometricControllerGains",
]
