"""Visualization utilities for quadrotor control.

This module provides visualization functions for plotting trajectories, control
signals, and other data related to quadrotor control.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Tuple, Dict, Any
from numpy.typing import NDArray

from .math_utils import quaternion_to_euler


def plot_trajectory_3d(
    positions: NDArray[np.float64],
    target_position: Optional[NDArray[np.float64]] = None,
    title: str = "Quadrotor Trajectory",
    save_path: Optional[str] = None
) -> None:
    """Plot 3D trajectory of quadrotor.
    
    Args:
        positions: Position trajectory (N x 3) [x, y, z]
        target_position: Target position [x, y, z]. If None, not plotted.
        title: Plot title
        save_path: Path to save plot. If None, displays plot.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
    
    # Plot start and end points
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               color='green', s=100, label='Start', marker='o')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               color='red', s=100, label='End', marker='s')
    
    # Plot target if provided
    if target_position is not None:
        ax.scatter(target_position[0], target_position[1], target_position[2], 
                   color='orange', s=100, label='Target', marker='^')
    
    # Set labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(title)
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                         positions[:, 1].max() - positions[:, 1].min(),
                         positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_position_error(
    time: NDArray[np.float64],
    position_errors: NDArray[np.float64],
    title: str = "Position Error",
    save_path: Optional[str] = None
) -> None:
    """Plot position error over time.
    
    Args:
        time: Time vector
        position_errors: Position errors (N x 3) [ex, ey, ez]
        title: Plot title
        save_path: Path to save plot. If None, displays plot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    labels = ['X Error', 'Y Error', 'Z Error']
    colors = ['red', 'green', 'blue']
    
    for i in range(3):
        axes[i].plot(time, position_errors[:, i], color=colors[i], linewidth=2)
        axes[i].set_ylabel(f'{labels[i]} (m)')
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    axes[0].set_title(title)
    axes[2].set_xlabel('Time (s)')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_attitude(
    time: NDArray[np.float64],
    attitudes: NDArray[np.float64],
    target_attitudes: Optional[NDArray[np.float64]] = None,
    title: str = "Attitude",
    save_path: Optional[str] = None
) -> None:
    """Plot attitude over time.
    
    Args:
        time: Time vector
        attitudes: Attitude trajectory (N x 3) [roll, pitch, yaw]
        target_attitudes: Target attitude trajectory (N x 3). If None, not plotted.
        title: Plot title
        save_path: Path to save plot. If None, displays plot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    labels = ['Roll', 'Pitch', 'Yaw']
    colors = ['red', 'green', 'blue']
    
    for i in range(3):
        axes[i].plot(time, np.degrees(attitudes[:, i]), color=colors[i], linewidth=2, label='Actual')
        
        if target_attitudes is not None:
            axes[i].plot(time, np.degrees(target_attitudes[:, i]), 
                        color=colors[i], linestyle='--', alpha=0.7, label='Target')
        
        axes[i].set_ylabel(f'{labels[i]} (deg)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    axes[0].set_title(title)
    axes[2].set_xlabel('Time (s)')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_control_signals(
    time: NDArray[np.float64],
    control_signals: NDArray[np.float64],
    title: str = "Control Signals",
    save_path: Optional[str] = None
) -> None:
    """Plot control signals over time.
    
    Args:
        time: Time vector
        control_signals: Control signals (N x 4) [T1, T2, T3, T4]
        title: Plot title
        save_path: Path to save plot. If None, displays plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    motor_labels = ['Motor 1', 'Motor 2', 'Motor 3', 'Motor 4']
    colors = ['red', 'green', 'blue', 'orange']
    
    for i in range(4):
        row = i // 2
        col = i % 2
        
        axes[row, col].plot(time, control_signals[:, i], color=colors[i], linewidth=2)
        axes[row, col].set_ylabel(f'{motor_labels[i]} Thrust (N)')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_title(motor_labels[i])
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_velocity(
    time: NDArray[np.float64],
    velocities: NDArray[np.float64],
    target_velocities: Optional[NDArray[np.float64]] = None,
    title: str = "Velocity",
    save_path: Optional[str] = None
) -> None:
    """Plot velocity over time.
    
    Args:
        time: Time vector
        velocities: Velocity trajectory (N x 3) [vx, vy, vz]
        target_velocities: Target velocity trajectory (N x 3). If None, not plotted.
        title: Plot title
        save_path: Path to save plot. If None, displays plot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    labels = ['Vx', 'Vy', 'Vz']
    colors = ['red', 'green', 'blue']
    
    for i in range(3):
        axes[i].plot(time, velocities[:, i], color=colors[i], linewidth=2, label='Actual')
        
        if target_velocities is not None:
            axes[i].plot(time, target_velocities[:, i], 
                        color=colors[i], linestyle='--', alpha=0.7, label='Target')
        
        axes[i].set_ylabel(f'{labels[i]} (m/s)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    axes[0].set_title(title)
    axes[2].set_xlabel('Time (s)')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_angular_velocity(
    time: NDArray[np.float64],
    angular_velocities: NDArray[np.float64],
    target_angular_velocities: Optional[NDArray[np.float64]] = None,
    title: str = "Angular Velocity",
    save_path: Optional[str] = None
) -> None:
    """Plot angular velocity over time.
    
    Args:
        time: Time vector
        angular_velocities: Angular velocity trajectory (N x 3) [wx, wy, wz]
        target_angular_velocities: Target angular velocity trajectory (N x 3). If None, not plotted.
        title: Plot title
        save_path: Path to save plot. If None, displays plot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    labels = ['Wx', 'Wy', 'Wz']
    colors = ['red', 'green', 'blue']
    
    for i in range(3):
        axes[i].plot(time, np.degrees(angular_velocities[:, i]), color=colors[i], linewidth=2, label='Actual')
        
        if target_angular_velocities is not None:
            axes[i].plot(time, np.degrees(target_angular_velocities[:, i]), 
                        color=colors[i], linestyle='--', alpha=0.7, label='Target')
        
        axes[i].set_ylabel(f'{labels[i]} (deg/s)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    axes[0].set_title(title)
    axes[2].set_xlabel('Time (s)')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_control_metrics(
    time: NDArray[np.float64],
    metrics: Dict[str, NDArray[np.float64]],
    title: str = "Control Metrics",
    save_path: Optional[str] = None
) -> None:
    """Plot control performance metrics.
    
    Args:
        time: Time vector
        metrics: Dictionary of metrics to plot
        title: Plot title
        save_path: Path to save plot. If None, displays plot.
    """
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        row = i // n_cols
        col = i % n_cols
        
        axes[row, col].plot(time, metric_values, linewidth=2)
        axes[row, col].set_ylabel(metric_name)
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_title(metric_name)
    
    # Hide unused subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def animate_trajectory_3d(
    positions: NDArray[np.float64],
    attitudes: Optional[NDArray[np.float64]] = None,
    target_position: Optional[NDArray[np.float64]] = None,
    title: str = "Quadrotor Trajectory Animation",
    save_path: Optional[str] = None,
    interval: int = 50
) -> None:
    """Create animated 3D trajectory plot.
    
    Args:
        positions: Position trajectory (N x 3) [x, y, z]
        attitudes: Attitude trajectory (N x 3) [roll, pitch, yaw]. If None, not shown.
        target_position: Target position [x, y, z]. If None, not shown.
        title: Animation title
        save_path: Path to save animation. If None, displays animation.
        interval: Animation interval in milliseconds
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the plot
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(title)
    
    # Plot target if provided
    if target_position is not None:
        ax.scatter(target_position[0], target_position[1], target_position[2], 
                   color='orange', s=100, label='Target', marker='^')
    
    # Initialize trajectory line
    line, = ax.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
    point, = ax.plot([], [], [], 'ro', markersize=8, label='Current Position')
    
    # Set axis limits
    ax.set_xlim(positions[:, 0].min() - 1, positions[:, 0].max() + 1)
    ax.set_ylim(positions[:, 1].min() - 1, positions[:, 1].max() + 1)
    ax.set_zlim(positions[:, 2].min() - 1, positions[:, 2].max() + 1)
    
    ax.legend()
    
    def animate(frame):
        # Update trajectory line
        line.set_data_3d(positions[:frame+1, 0], positions[:frame+1, 1], positions[:frame+1, 2])
        
        # Update current position point
        point.set_data_3d([positions[frame, 0]], [positions[frame, 1]], [positions[frame, 2]])
        
        return line, point
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(positions), interval=interval, blit=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=20)
    else:
        plt.show()
    
    plt.close()


def create_summary_plot(
    time: NDArray[np.float64],
    positions: NDArray[np.float64],
    velocities: NDArray[np.float64],
    attitudes: NDArray[np.float64],
    control_signals: NDArray[np.float64],
    target_position: Optional[NDArray[np.float64]] = None,
    save_path: Optional[str] = None
) -> None:
    """Create a comprehensive summary plot.
    
    Args:
        time: Time vector
        positions: Position trajectory (N x 3) [x, y, z]
        velocities: Velocity trajectory (N x 3) [vx, vy, vz]
        attitudes: Attitude trajectory (N x 3) [roll, pitch, yaw]
        control_signals: Control signals (N x 4) [T1, T2, T3, T4]
        target_position: Target position [x, y, z]. If None, not shown.
        save_path: Path to save plot. If None, displays plot.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                color='green', s=100, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                color='red', s=100, label='End')
    if target_position is not None:
        ax1.scatter(target_position[0], target_position[1], target_position[2], 
                    color='orange', s=100, label='Target')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # Position over time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(time, positions[:, 0], 'r-', label='X')
    ax2.plot(time, positions[:, 1], 'g-', label='Y')
    ax2.plot(time, positions[:, 2], 'b-', label='Z')
    if target_position is not None:
        ax2.axhline(y=target_position[0], color='r', linestyle='--', alpha=0.7)
        ax2.axhline(y=target_position[1], color='g', linestyle='--', alpha=0.7)
        ax2.axhline(y=target_position[2], color='b', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Velocity over time
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(time, velocities[:, 0], 'r-', label='Vx')
    ax3.plot(time, velocities[:, 1], 'g-', label='Vy')
    ax3.plot(time, velocities[:, 2], 'b-', label='Vz')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Attitude over time
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(time, np.degrees(attitudes[:, 0]), 'r-', label='Roll')
    ax4.plot(time, np.degrees(attitudes[:, 1]), 'g-', label='Pitch')
    ax4.plot(time, np.degrees(attitudes[:, 2]), 'b-', label='Yaw')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Attitude (deg)')
    ax4.set_title('Attitude vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Control signals
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(time, control_signals[:, 0], 'r-', label='T1')
    ax5.plot(time, control_signals[:, 1], 'g-', label='T2')
    ax5.plot(time, control_signals[:, 2], 'b-', label='T3')
    ax5.plot(time, control_signals[:, 3], 'orange', label='T4')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Thrust (N)')
    ax5.set_title('Control Signals')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Position error
    ax6 = fig.add_subplot(2, 3, 6)
    if target_position is not None:
        position_errors = positions - target_position
        ax6.plot(time, position_errors[:, 0], 'r-', label='X Error')
        ax6.plot(time, position_errors[:, 1], 'g-', label='Y Error')
        ax6.plot(time, position_errors[:, 2], 'b-', label='Z Error')
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Position Error (m)')
        ax6.set_title('Position Error')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No target position\nprovided', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Position Error')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
