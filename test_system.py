#!/usr/bin/env python3
"""Quick test script to verify the quadrotor control system works."""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.dynamics.quadrotor import QuadrotorDynamics
        from src.dynamics.parameters import QuadrotorParameters
        from src.controllers.pid import PIDController, PIDGains, PIDLimits
        from src.controllers.lqr import LQRController, LQRWeights
        from src.controllers.mpc import MPCController, MPCConfig
        from src.controllers.geometric import GeometricController, GeometricControllerGains
        from src.utils.math_utils import euler_to_quaternion, quaternion_to_euler
        from src.utils.visualization import plot_trajectory_3d
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        # Test quadrotor dynamics
        from src.dynamics.quadrotor import QuadrotorDynamics
        from src.dynamics.parameters import QuadrotorParameters
        
        params = QuadrotorParameters()
        quadrotor = QuadrotorDynamics(params)
        
        # Test state getters
        position = quadrotor.get_position()
        velocity = quadrotor.get_velocity()
        attitude = quadrotor.get_euler_angles()
        
        assert position.shape == (3,)
        assert velocity.shape == (3,)
        assert attitude.shape == (3,)
        
        print("✓ Quadrotor dynamics working")
        
        # Test PID controller
        from src.controllers.pid import PIDController, PIDGains, PIDLimits
        
        gains = PIDGains()
        limits = PIDLimits()
        controller = PIDController(gains, limits, dt=0.01)
        
        # Test control computation
        target_position = np.array([1.0, 1.0, 1.0])
        thrusts = controller.compute_control(quadrotor, target_position)
        
        assert thrusts.shape == (4,)
        assert np.all(thrusts >= 0.0)
        assert np.all(thrusts <= params.max_thrust)
        
        print("✓ PID controller working")
        
        # Test LQR controller
        from src.controllers.lqr import LQRController, LQRWeights
        
        lqr_weights = LQRWeights()
        lqr_controller = LQRController(params, lqr_weights, dt=0.01)
        
        lqr_thrusts = lqr_controller.compute_control(quadrotor, target_position)
        
        assert lqr_thrusts.shape == (4,)
        assert np.all(lqr_thrusts >= 0.0)
        assert np.all(lqr_thrusts <= params.max_thrust)
        
        print("✓ LQR controller working")
        
        # Test Geometric controller
        from src.controllers.geometric import GeometricController, GeometricControllerGains
        
        geo_gains = GeometricControllerGains()
        geo_controller = GeometricController(params, geo_gains)
        
        geo_thrusts = geo_controller.compute_control(quadrotor, target_position)
        
        assert geo_thrusts.shape == (4,)
        assert np.all(geo_thrusts >= 0.0)
        assert np.all(geo_thrusts <= params.max_thrust)
        
        print("✓ Geometric controller working")
        
        # Test math utilities
        from src.utils.math_utils import euler_to_quaternion, quaternion_to_euler
        
        euler_angles = np.array([0.1, 0.2, 0.3])
        quat = euler_to_quaternion(*euler_angles)
        recovered_euler = quaternion_to_euler(quat)
        
        np.testing.assert_array_almost_equal(euler_angles, recovered_euler, decimal=5)
        
        print("✓ Math utilities working")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def test_simulation():
    """Test a simple simulation."""
    print("\nTesting simple simulation...")
    
    try:
        from src.dynamics.quadrotor import QuadrotorDynamics
        from src.dynamics.parameters import QuadrotorParameters
        from src.controllers.pid import PIDController, PIDGains, PIDLimits
        
        # Set up simulation
        params = QuadrotorParameters()
        quadrotor = QuadrotorDynamics(params)
        
        gains = PIDGains()
        limits = PIDLimits()
        controller = PIDController(gains, limits, dt=0.01)
        
        # Run short simulation
        target_position = np.array([1.0, 1.0, 1.0])
        n_steps = 100
        
        positions = []
        for step in range(n_steps):
            position = quadrotor.get_position()
            positions.append(position.copy())
            
            thrusts = controller.compute_control(quadrotor, target_position)
            quadrotor.step(thrusts, 0.01)
        
        positions = np.array(positions)
        
        # Check that quadrotor moved
        initial_position = positions[0]
        final_position = positions[-1]
        
        distance_moved = np.linalg.norm(final_position - initial_position)
        assert distance_moved > 0.1, "Quadrotor should have moved significantly"
        
        print(f"✓ Simulation completed successfully")
        print(f"  Initial position: {initial_position}")
        print(f"  Final position: {final_position}")
        print(f"  Distance moved: {distance_moved:.3f} m")
        
        return True
        
    except Exception as e:
        print(f"✗ Simulation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚁 Flying Robot Control - Quick Test")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run tests
    tests = [
        test_imports,
        test_basic_functionality,
        test_simulation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
