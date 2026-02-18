"""Main simulation script for quadrotor control.

This script runs simulations with different controllers and generates
comprehensive results and visualizations.
"""

from __future__ import annotations

import argparse
import numpy as np
import yaml
import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set up deterministic seeding
np.random.seed(42)

# Import our modules
from dynamics.quadrotor import QuadrotorDynamics
from dynamics.parameters import QuadrotorParameters, get_quadrotor_config
from controllers.pid import PIDController, PIDGains, PIDLimits
from controllers.lqr import LQRController, LQRWeights
from controllers.mpc import MPCController, MPCConfig
from controllers.geometric import GeometricController, GeometricControllerGains
from utils.visualization import (
    plot_trajectory_3d, plot_position_error, plot_attitude,
    plot_control_signals, plot_velocity, plot_angular_velocity,
    create_summary_plot
)


class QuadrotorSimulation:
    """Main simulation class for quadrotor control."""
    
    def __init__(self, config_path: str = "config/quadrotor.yaml"):
        """Initialize simulation with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.setup_directories()
        
        # Simulation parameters
        self.dt = self.config['simulation']['dt']
        self.duration = self.config['simulation']['duration']
        self.n_steps = int(self.duration / self.dt)
        
        # Initialize quadrotor
        self.quadrotor_params = QuadrotorParameters(**self.config['quadrotor'])
        self.quadrotor = QuadrotorDynamics(self.quadrotor_params)
        
        # Initialize controllers
        self.controllers = self._initialize_controllers()
        
        # Initialize state storage
        self.reset_simulation()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_directories(self) -> None:
        """Create necessary directories for outputs."""
        directories = ['assets', 'data', 'assets/plots', 'assets/trajectories']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _initialize_controllers(self) -> Dict[str, Any]:
        """Initialize all controllers.
        
        Returns:
            Dictionary of initialized controllers
        """
        controllers = {}
        
        # PID Controller
        pid_gains = PIDGains(
            position_kp=np.array(self.config['pid']['position']['kp']),
            position_ki=np.array(self.config['pid']['position']['ki']),
            position_kd=np.array(self.config['pid']['position']['kd']),
            attitude_kp=np.array(self.config['pid']['attitude']['kp']),
            attitude_ki=np.array(self.config['pid']['attitude']['ki']),
            attitude_kd=np.array(self.config['pid']['attitude']['kd'])
        )
        pid_limits = PIDLimits(**self.config['pid']['limits'])
        controllers['pid'] = PIDController(pid_gains, pid_limits, self.dt)
        
        # LQR Controller
        lqr_weights = LQRWeights(
            Q=np.diag(self.config['lqr']['Q']),
            R=np.diag(self.config['lqr']['R'])
        )
        controllers['lqr'] = LQRController(self.quadrotor_params, lqr_weights, self.dt)
        
        # MPC Controller (only if CasADi is available)
        try:
            mpc_config = MPCConfig(**self.config['mpc'])
            controllers['mpc'] = MPCController(self.quadrotor_params, mpc_config)
        except ImportError:
            print("Warning: MPC controller not available (CasADi not installed)")
        
        # Geometric Controller
        geometric_gains = GeometricControllerGains(**self.config['geometric'])
        controllers['geometric'] = GeometricController(self.quadrotor_params, geometric_gains)
        
        return controllers
    
    def reset_simulation(self) -> None:
        """Reset simulation to initial conditions."""
        # Set initial state
        initial_state = np.zeros(13)
        initial_state[:3] = np.array(self.config['simulation']['initial_position'])
        initial_state[3:6] = np.array(self.config['simulation']['initial_velocity'])
        initial_state[6:10] = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        initial_state[10:13] = np.array(self.config['simulation']['initial_angular_velocity'])
        
        self.quadrotor.set_state(initial_state)
        
        # Initialize data storage
        self.time = np.zeros(self.n_steps)
        self.positions = np.zeros((self.n_steps, 3))
        self.velocities = np.zeros((self.n_steps, 3))
        self.attitudes = np.zeros((self.n_steps, 3))
        self.angular_velocities = np.zeros((self.n_steps, 3))
        self.control_signals = np.zeros((self.n_steps, 4))
        self.position_errors = np.zeros((self.n_steps, 3))
        self.velocity_errors = np.zeros((self.n_steps, 3))
        self.attitude_errors = np.zeros((self.n_steps, 3))
        
        # Target states
        self.target_position = np.array(self.config['simulation']['target_position'])
        self.target_velocity = np.array(self.config['simulation']['target_velocity'])
        self.target_attitude = np.array(self.config['simulation']['target_attitude'])
        self.target_angular_velocity = np.array(self.config['simulation']['target_angular_velocity'])
        
        # Reset controllers
        for controller in self.controllers.values():
            if hasattr(controller, 'reset'):
                controller.reset()
    
    def run_simulation(self, controller_name: str = 'pid') -> Dict[str, Any]:
        """Run simulation with specified controller.
        
        Args:
            controller_name: Name of controller to use ('pid', 'lqr', 'mpc', 'geometric')
            
        Returns:
            Dictionary containing simulation results
        """
        if controller_name not in self.controllers:
            raise ValueError(f"Unknown controller: {controller_name}")
        
        controller = self.controllers[controller_name]
        self.reset_simulation()
        
        print(f"Running simulation with {controller_name.upper()} controller...")
        print(f"Target position: {self.target_position}")
        print(f"Simulation duration: {self.duration}s")
        print(f"Time step: {self.dt}s")
        print(f"Number of steps: {self.n_steps}")
        
        # Run simulation
        for step in range(self.n_steps):
            current_time = step * self.dt
            self.time[step] = current_time
            
            # Get current state
            position = self.quadrotor.get_position()
            velocity = self.quadrotor.get_velocity()
            attitude = self.quadrotor.get_euler_angles()
            angular_velocity = self.quadrotor.get_angular_velocity()
            
            # Store state
            self.positions[step] = position
            self.velocities[step] = velocity
            self.attitudes[step] = attitude
            self.angular_velocities[step] = angular_velocity
            
            # Compute errors
            self.position_errors[step] = position - self.target_position
            self.velocity_errors[step] = velocity - self.target_velocity
            self.attitude_errors[step] = attitude - self.target_attitude
            
            # Compute control signal
            control_signal = controller.compute_control(
                self.quadrotor,
                self.target_position,
                self.target_velocity,
                self.target_attitude,
                self.target_angular_velocity
            )
            
            # Store control signal
            self.control_signals[step] = control_signal
            
            # Update quadrotor dynamics
            self.quadrotor.step(control_signal, self.dt)
            
            # Print progress
            if step % (self.n_steps // 10) == 0:
                progress = (step / self.n_steps) * 100
                print(f"Progress: {progress:.1f}%")
        
        print("Simulation completed!")
        
        # Compute performance metrics
        metrics = self._compute_metrics()
        
        # Generate visualizations
        if self.config['visualization']['plot_trajectory']:
            self._generate_plots(controller_name)
        
        return {
            'controller': controller_name,
            'time': self.time,
            'positions': self.positions,
            'velocities': self.velocities,
            'attitudes': self.attitudes,
            'angular_velocities': self.angular_velocities,
            'control_signals': self.control_signals,
            'position_errors': self.position_errors,
            'velocity_errors': self.velocity_errors,
            'attitude_errors': self.attitude_errors,
            'target_position': self.target_position,
            'target_velocity': self.target_velocity,
            'target_attitude': self.target_attitude,
            'target_angular_velocity': self.target_angular_velocity,
            'metrics': metrics
        }
    
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        # Position tracking metrics
        position_rmse = np.sqrt(np.mean(np.sum(self.position_errors**2, axis=1)))
        position_mae = np.mean(np.linalg.norm(self.position_errors, axis=1))
        position_max_error = np.max(np.linalg.norm(self.position_errors, axis=1))
        
        # Velocity tracking metrics
        velocity_rmse = np.sqrt(np.mean(np.sum(self.velocity_errors**2, axis=1)))
        velocity_mae = np.mean(np.linalg.norm(self.velocity_errors, axis=1))
        velocity_max_error = np.max(np.linalg.norm(self.velocity_errors, axis=1))
        
        # Attitude tracking metrics
        attitude_rmse = np.sqrt(np.mean(np.sum(self.attitude_errors**2, axis=1)))
        attitude_mae = np.mean(np.linalg.norm(self.attitude_errors, axis=1))
        attitude_max_error = np.max(np.linalg.norm(self.attitude_errors, axis=1))
        
        # Control effort metrics
        control_effort = np.sum(np.sum(self.control_signals**2, axis=1)) * self.dt
        control_effort_per_motor = np.sum(self.control_signals**2, axis=0) * self.dt
        
        # Settling time (time to reach within 5% of target)
        position_magnitude = np.linalg.norm(self.position_errors, axis=1)
        target_magnitude = np.linalg.norm(self.target_position)
        settling_threshold = 0.05 * target_magnitude
        
        settling_time = self.duration  # Default to full duration
        for i, error in enumerate(position_magnitude):
            if error < settling_threshold:
                settling_time = i * self.dt
                break
        
        # Overshoot (maximum overshoot percentage)
        if target_magnitude > 0:
            overshoot = (np.max(position_magnitude) - target_magnitude) / target_magnitude * 100
        else:
            overshoot = 0.0
        
        return {
            'position_rmse': position_rmse,
            'position_mae': position_mae,
            'position_max_error': position_max_error,
            'velocity_rmse': velocity_rmse,
            'velocity_mae': velocity_mae,
            'velocity_max_error': velocity_max_error,
            'attitude_rmse': attitude_rmse,
            'attitude_mae': attitude_mae,
            'attitude_max_error': attitude_max_error,
            'control_effort': control_effort,
            'control_effort_per_motor': control_effort_per_motor.tolist(),
            'settling_time': settling_time,
            'overshoot_percent': overshoot,
            'final_position_error': np.linalg.norm(self.position_errors[-1]),
            'final_velocity_error': np.linalg.norm(self.velocity_errors[-1]),
            'final_attitude_error': np.linalg.norm(self.attitude_errors[-1])
        }
    
    def _generate_plots(self, controller_name: str) -> None:
        """Generate visualization plots.
        
        Args:
            controller_name: Name of controller used
        """
        print("Generating plots...")
        
        # Create plots directory
        plots_dir = Path(f"assets/plots/{controller_name}")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 3D trajectory plot
        plot_trajectory_3d(
            self.positions,
            self.target_position,
            f"{controller_name.upper()} Controller - 3D Trajectory",
            str(plots_dir / "trajectory_3d.png")
        )
        
        # Position error plot
        plot_position_error(
            self.time,
            self.position_errors,
            f"{controller_name.upper()} Controller - Position Error",
            str(plots_dir / "position_error.png")
        )
        
        # Attitude plot
        plot_attitude(
            self.time,
            self.attitudes,
            None,  # No target attitude for now
            f"{controller_name.upper()} Controller - Attitude",
            str(plots_dir / "attitude.png")
        )
        
        # Velocity plot
        plot_velocity(
            self.time,
            self.velocities,
            None,  # No target velocity for now
            f"{controller_name.upper()} Controller - Velocity",
            str(plots_dir / "velocity.png")
        )
        
        # Angular velocity plot
        plot_angular_velocity(
            self.time,
            self.angular_velocities,
            None,  # No target angular velocity for now
            f"{controller_name.upper()} Controller - Angular Velocity",
            str(plots_dir / "angular_velocity.png")
        )
        
        # Control signals plot
        plot_control_signals(
            self.time,
            self.control_signals,
            f"{controller_name.upper()} Controller - Control Signals",
            str(plots_dir / "control_signals.png")
        )
        
        # Summary plot
        create_summary_plot(
            self.time,
            self.positions,
            self.velocities,
            self.attitudes,
            self.control_signals,
            self.target_position,
            str(plots_dir / "summary.png")
        )
        
        print(f"Plots saved to {plots_dir}")
    
    def run_comparison(self, controllers: Optional[list] = None) -> Dict[str, Any]:
        """Run comparison between multiple controllers.
        
        Args:
            controllers: List of controller names to compare. If None, compares all.
            
        Returns:
            Dictionary containing comparison results
        """
        if controllers is None:
            controllers = list(self.controllers.keys())
        
        print(f"Running comparison between controllers: {controllers}")
        
        results = {}
        for controller_name in controllers:
            print(f"\n{'='*50}")
            print(f"Testing {controller_name.upper()} controller")
            print(f"{'='*50}")
            
            result = self.run_simulation(controller_name)
            results[controller_name] = result
            
            # Print metrics
            metrics = result['metrics']
            print(f"\nPerformance Metrics:")
            print(f"  Position RMSE: {metrics['position_rmse']:.4f} m")
            print(f"  Velocity RMSE: {metrics['velocity_rmse']:.4f} m/s")
            print(f"  Attitude RMSE: {metrics['attitude_rmse']:.4f} rad")
            print(f"  Control Effort: {metrics['control_effort']:.4f}")
            print(f"  Settling Time: {metrics['settling_time']:.2f} s")
            print(f"  Overshoot: {metrics['overshoot_percent']:.2f}%")
            print(f"  Final Position Error: {metrics['final_position_error']:.4f} m")
        
        # Generate comparison plot
        self._generate_comparison_plot(results)
        
        return results
    
    def _generate_comparison_plot(self, results: Dict[str, Any]) -> None:
        """Generate comparison plot between controllers.
        
        Args:
            results: Dictionary containing results from multiple controllers
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Position error comparison
        ax1 = axes[0, 0]
        for controller_name, result in results.items():
            position_error_magnitude = np.linalg.norm(result['position_errors'], axis=1)
            ax1.plot(result['time'], position_error_magnitude, label=controller_name.upper(), linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position Error (m)')
        ax1.set_title('Position Error Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Velocity comparison
        ax2 = axes[0, 1]
        for controller_name, result in results.items():
            velocity_magnitude = np.linalg.norm(result['velocities'], axis=1)
            ax2.plot(result['time'], velocity_magnitude, label=controller_name.upper(), linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity Magnitude (m/s)')
        ax2.set_title('Velocity Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Control effort comparison
        ax3 = axes[1, 0]
        for controller_name, result in results.items():
            control_effort = np.sum(result['control_signals']**2, axis=1)
            ax3.plot(result['time'], control_effort, label=controller_name.upper(), linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Control Effort')
        ax3.set_title('Control Effort Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance metrics comparison
        ax4 = axes[1, 1]
        controllers = list(results.keys())
        position_rmse = [results[c]['metrics']['position_rmse'] for c in controllers]
        velocity_rmse = [results[c]['metrics']['velocity_rmse'] for c in controllers]
        
        x = np.arange(len(controllers))
        width = 0.35
        
        ax4.bar(x - width/2, position_rmse, width, label='Position RMSE', alpha=0.8)
        ax4.bar(x + width/2, velocity_rmse, width, label='Velocity RMSE', alpha=0.8)
        
        ax4.set_xlabel('Controller')
        ax4.set_ylabel('RMSE')
        ax4.set_title('Performance Metrics Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels([c.upper() for c in controllers])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('assets/plots/controller_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Comparison plot saved to assets/plots/controller_comparison.png")


def main():
    """Main function to run simulations."""
    parser = argparse.ArgumentParser(description='Quadrotor Control Simulation')
    parser.add_argument('--controller', type=str, default='pid',
                       choices=['pid', 'lqr', 'mpc', 'geometric'],
                       help='Controller to use')
    parser.add_argument('--config', type=str, default='config/quadrotor.yaml',
                       help='Configuration file path')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison between all controllers')
    parser.add_argument('--target', type=str, default='5,5,5',
                       help='Target position as x,y,z')
    
    args = parser.parse_args()
    
    # Parse target position
    target_position = [float(x) for x in args.target.split(',')]
    
    # Initialize simulation
    sim = QuadrotorSimulation(args.config)
    
    # Update target position
    sim.target_position = np.array(target_position)
    
    if args.compare:
        # Run comparison
        results = sim.run_comparison()
        
        # Print summary
        print(f"\n{'='*60}")
        print("CONTROLLER COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        for controller_name, result in results.items():
            metrics = result['metrics']
            print(f"\n{controller_name.upper()} Controller:")
            print(f"  Position RMSE: {metrics['position_rmse']:.4f} m")
            print(f"  Settling Time: {metrics['settling_time']:.2f} s")
            print(f"  Control Effort: {metrics['control_effort']:.4f}")
            print(f"  Final Error: {metrics['final_position_error']:.4f} m")
        
        # Find best controller
        best_controller = min(results.keys(), 
                            key=lambda c: results[c]['metrics']['position_rmse'])
        print(f"\nBest Controller: {best_controller.upper()}")
        print(f"Position RMSE: {results[best_controller]['metrics']['position_rmse']:.4f} m")
        
    else:
        # Run single controller simulation
        result = sim.run_simulation(args.controller)
        
        # Print results
        metrics = result['metrics']
        print(f"\n{'='*50}")
        print(f"{args.controller.upper()} CONTROLLER RESULTS")
        print(f"{'='*50}")
        print(f"Target Position: {target_position}")
        print(f"Final Position: {result['positions'][-1]}")
        print(f"Position RMSE: {metrics['position_rmse']:.4f} m")
        print(f"Velocity RMSE: {metrics['velocity_rmse']:.4f} m/s")
        print(f"Attitude RMSE: {metrics['attitude_rmse']:.4f} rad")
        print(f"Control Effort: {metrics['control_effort']:.4f}")
        print(f"Settling Time: {metrics['settling_time']:.2f} s")
        print(f"Overshoot: {metrics['overshoot_percent']:.2f}%")
        print(f"Final Position Error: {metrics['final_position_error']:.4f} m")


if __name__ == "__main__":
    main()
