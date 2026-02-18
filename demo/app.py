"""Interactive Streamlit demo for quadrotor control.

This module provides an interactive web interface for testing different
quadrotor controllers and visualizing results in real-time.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import time
from pathlib import Path

# Import our modules
from src.dynamics.quadrotor import QuadrotorDynamics
from src.dynamics.parameters import QuadrotorParameters
from src.controllers.pid import PIDController, PIDGains, PIDLimits
from src.controllers.lqr import LQRController, LQRWeights
from src.controllers.mpc import MPCController, MPCConfig
from src.controllers.geometric import GeometricController, GeometricControllerGains


# Page configuration
st.set_page_config(
    page_title="Flying Robot Control Demo",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚁 Flying Robot Control Demo</h1>', unsafe_allow_html=True)

# Safety warning
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Safety Warning</h4>
    <p><strong>This is a simulation-only demo for educational purposes.</strong></p>
    <p>Do not use this code on real hardware without expert review and proper safety measures.</p>
    <p>See <a href="DISCLAIMER.md">DISCLAIMER.md</a> for important safety information.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("🎛️ Control Parameters")

# Controller selection
controller_type = st.sidebar.selectbox(
    "Controller Type",
    ["PID", "LQR", "MPC", "Geometric"],
    help="Select the control algorithm to use"
)

# Target position
st.sidebar.subheader("🎯 Target Position")
target_x = st.sidebar.slider("X Position (m)", -10.0, 10.0, 5.0, 0.1)
target_y = st.sidebar.slider("Y Position (m)", -10.0, 10.0, 5.0, 0.1)
target_z = st.sidebar.slider("Z Position (m)", 0.0, 10.0, 5.0, 0.1)

target_position = np.array([target_x, target_y, target_z])

# Simulation parameters
st.sidebar.subheader("⏱️ Simulation Parameters")
simulation_duration = st.sidebar.slider("Duration (s)", 1.0, 20.0, 10.0, 0.5)
time_step = st.sidebar.selectbox("Time Step (s)", [0.01, 0.02, 0.05, 0.1], index=0)

# Controller-specific parameters
st.sidebar.subheader("🔧 Controller Parameters")

if controller_type == "PID":
    st.sidebar.write("**Position Control Gains**")
    pid_kp_x = st.sidebar.slider("Kp X", 0.1, 5.0, 1.0, 0.1)
    pid_kp_y = st.sidebar.slider("Kp Y", 0.1, 5.0, 1.0, 0.1)
    pid_kp_z = st.sidebar.slider("Kp Z", 0.1, 5.0, 1.0, 0.1)
    
    pid_ki_x = st.sidebar.slider("Ki X", 0.0, 1.0, 0.1, 0.01)
    pid_ki_y = st.sidebar.slider("Ki Y", 0.0, 1.0, 0.1, 0.01)
    pid_ki_z = st.sidebar.slider("Ki Z", 0.0, 1.0, 0.1, 0.01)
    
    pid_kd_x = st.sidebar.slider("Kd X", 0.0, 2.0, 0.5, 0.1)
    pid_kd_y = st.sidebar.slider("Kd Y", 0.0, 2.0, 0.5, 0.1)
    pid_kd_z = st.sidebar.slider("Kd Z", 0.0, 2.0, 0.5, 0.1)

elif controller_type == "LQR":
    st.sidebar.write("**LQR Weights**")
    lqr_position_weight = st.sidebar.slider("Position Weight", 1.0, 50.0, 10.0, 1.0)
    lqr_velocity_weight = st.sidebar.slider("Velocity Weight", 0.1, 10.0, 1.0, 0.1)
    lqr_control_weight = st.sidebar.slider("Control Weight", 0.01, 1.0, 0.1, 0.01)

elif controller_type == "MPC":
    st.sidebar.write("**MPC Parameters**")
    mpc_horizon = st.sidebar.slider("Prediction Horizon", 5, 20, 10, 1)
    mpc_position_weight = st.sidebar.slider("Position Weight", 1.0, 50.0, 10.0, 1.0)
    mpc_control_weight = st.sidebar.slider("Control Weight", 0.01, 1.0, 0.1, 0.01)

elif controller_type == "Geometric":
    st.sidebar.write("**Geometric Controller Gains**")
    geo_kx = st.sidebar.slider("Kx", 0.1, 5.0, 1.0, 0.1)
    geo_kv = st.sidebar.slider("Kv", 0.1, 5.0, 1.0, 0.1)
    geo_kR = st.sidebar.slider("KR", 0.1, 5.0, 1.0, 0.1)

# Run simulation button
run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary")

# Main content area
if run_simulation:
    # Initialize simulation
    with st.spinner("Initializing simulation..."):
        # Quadrotor parameters
        quadrotor_params = QuadrotorParameters()
        
        # Initialize quadrotor
        quadrotor = QuadrotorDynamics(quadrotor_params)
        
        # Set initial state
        initial_state = np.zeros(13)
        initial_state[6] = 1.0  # Identity quaternion
        quadrotor.set_state(initial_state)
        
        # Initialize controller
        if controller_type == "PID":
            gains = PIDGains(
                position_kp=np.array([pid_kp_x, pid_kp_y, pid_kp_z]),
                position_ki=np.array([pid_ki_x, pid_ki_y, pid_ki_z]),
                position_kd=np.array([pid_kd_x, pid_kd_y, pid_kd_z])
            )
            controller = PIDController(gains, PIDLimits(), time_step)
            
        elif controller_type == "LQR":
            Q = np.diag([lqr_position_weight] * 3 + [lqr_velocity_weight] * 3 + 
                       [1.0] * 3 + [0.1] * 3)
            R = np.diag([lqr_control_weight] * 4)
            weights = LQRWeights(Q=Q, R=R)
            controller = LQRController(quadrotor_params, weights, time_step)
            
        elif controller_type == "MPC":
            config = MPCConfig(
                horizon=mpc_horizon,
                dt=time_step,
                position_weight=mpc_position_weight,
                control_weight=mpc_control_weight
            )
            controller = MPCController(quadrotor_params, config)
            
        elif controller_type == "Geometric":
            gains = GeometricControllerGains(kx=geo_kx, kv=geo_kv, kR=geo_kR)
            controller = GeometricController(quadrotor_params, gains)
    
    # Run simulation
    with st.spinner("Running simulation..."):
        n_steps = int(simulation_duration / time_step)
        
        # Initialize data storage
        time_data = np.zeros(n_steps)
        positions = np.zeros((n_steps, 3))
        velocities = np.zeros((n_steps, 3))
        attitudes = np.zeros((n_steps, 3))
        control_signals = np.zeros((n_steps, 4))
        position_errors = np.zeros((n_steps, 3))
        
        # Run simulation loop
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for step in range(n_steps):
            current_time = step * time_step
            time_data[step] = current_time
            
            # Get current state
            position = quadrotor.get_position()
            velocity = quadrotor.get_velocity()
            attitude = quadrotor.get_euler_angles()
            
            # Store state
            positions[step] = position
            velocities[step] = velocity
            attitudes[step] = attitude
            
            # Compute errors
            position_errors[step] = position - target_position
            
            # Compute control signal
            control_signal = controller.compute_control(
                quadrotor,
                target_position,
                np.zeros(3),  # target velocity
                np.zeros(3),  # target attitude
                np.zeros(3)   # target angular velocity
            )
            
            # Store control signal
            control_signals[step] = control_signal
            
            # Update quadrotor dynamics
            quadrotor.step(control_signal, time_step)
            
            # Update progress
            progress = (step + 1) / n_steps
            progress_bar.progress(progress)
            status_text.text(f"Step {step + 1}/{n_steps} - Time: {current_time:.2f}s")
        
        progress_bar.empty()
        status_text.empty()
    
    # Display results
    st.success("Simulation completed successfully!")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 3D Trajectory", "📈 Position Error", "🎛️ Control Signals", "📋 Metrics", "🔍 Analysis"])
    
    with tab1:
        st.subheader("3D Trajectory Visualization")
        
        # Create 3D plot
        fig = go.Figure()
        
        # Add trajectory
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines+markers',
            name='Trajectory',
            line=dict(color='blue', width=4),
            marker=dict(size=2)
        ))
        
        # Add start point
        fig.add_trace(go.Scatter3d(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            z=[positions[0, 2]],
            mode='markers',
            name='Start',
            marker=dict(size=10, color='green')
        ))
        
        # Add end point
        fig.add_trace(go.Scatter3d(
            x=[positions[-1, 0]],
            y=[positions[-1, 1]],
            z=[positions[-1, 2]],
            mode='markers',
            name='End',
            marker=dict(size=10, color='red')
        ))
        
        # Add target point
        fig.add_trace(go.Scatter3d(
            x=[target_position[0]],
            y=[target_position[1]],
            z=[target_position[2]],
            mode='markers',
            name='Target',
            marker=dict(size=10, color='orange')
        ))
        
        fig.update_layout(
            title=f"{controller_type} Controller - 3D Trajectory",
            scene=dict(
                xaxis_title='X Position (m)',
                yaxis_title='Y Position (m)',
                zaxis_title='Z Position (m)',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Position Error Analysis")
        
        # Position error magnitude
        position_error_magnitude = np.linalg.norm(position_errors, axis=1)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Position Error Magnitude', 'X Error', 'Y Error', 'Z Error'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Position error magnitude
        fig.add_trace(
            go.Scatter(x=time_data, y=position_error_magnitude, name='Error Magnitude', line=dict(color='red')),
            row=1, col=1
        )
        
        # Individual position errors
        fig.add_trace(
            go.Scatter(x=time_data, y=position_errors[:, 0], name='X Error', line=dict(color='blue')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=time_data, y=position_errors[:, 1], name='Y Error', line=dict(color='green')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time_data, y=position_errors[:, 2], name='Z Error', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Position Error Analysis")
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Error (m)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Control Signals")
        
        # Create subplots for control signals
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Motor 1', 'Motor 2', 'Motor 3', 'Motor 4'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['red', 'green', 'blue', 'orange']
        for i in range(4):
            row = i // 2 + 1
            col = i % 2 + 1
            fig.add_trace(
                go.Scatter(x=time_data, y=control_signals[:, i], 
                          name=f'Motor {i+1}', line=dict(color=colors[i])),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Motor Thrust Signals")
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Thrust (N)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Performance Metrics")
        
        # Compute metrics
        position_rmse = np.sqrt(np.mean(np.sum(position_errors**2, axis=1)))
        position_mae = np.mean(np.linalg.norm(position_errors, axis=1))
        position_max_error = np.max(np.linalg.norm(position_errors, axis=1))
        final_position_error = np.linalg.norm(position_errors[-1])
        
        control_effort = np.sum(np.sum(control_signals**2, axis=1)) * time_step
        
        # Settling time (time to reach within 5% of target)
        target_magnitude = np.linalg.norm(target_position)
        settling_threshold = 0.05 * target_magnitude
        settling_time = simulation_duration
        
        for i, error in enumerate(position_error_magnitude):
            if error < settling_threshold:
                settling_time = i * time_step
                break
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Position RMSE", f"{position_rmse:.4f} m")
            st.metric("Position MAE", f"{position_mae:.4f} m")
        
        with col2:
            st.metric("Max Position Error", f"{position_max_error:.4f} m")
            st.metric("Final Position Error", f"{final_position_error:.4f} m")
        
        with col3:
            st.metric("Settling Time", f"{settling_time:.2f} s")
            st.metric("Control Effort", f"{control_effort:.2f}")
        
        with col4:
            st.metric("Target Position", f"{target_position}")
            st.metric("Final Position", f"{positions[-1]}")
    
    with tab5:
        st.subheader("Detailed Analysis")
        
        # Position vs time
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Position vs Time', 'Velocity vs Time'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Position
        fig.add_trace(
            go.Scatter(x=time_data, y=positions[:, 0], name='X Position', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_data, y=positions[:, 1], name='Y Position', line=dict(color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_data, y=positions[:, 2], name='Z Position', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add target lines
        fig.add_hline(y=target_position[0], line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=target_position[1], line_dash="dash", line_color="green", row=1, col=1)
        fig.add_hline(y=target_position[2], line_dash="dash", line_color="blue", row=1, col=1)
        
        # Velocity
        fig.add_trace(
            go.Scatter(x=time_data, y=velocities[:, 0], name='Vx', line=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_data, y=velocities[:, 1], name='Vy', line=dict(color='green')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_data, y=velocities[:, 2], name='Vz', line=dict(color='blue')),
            row=2, col=1
        )
        
        fig.update_layout(height=800, title_text="Position and Velocity Analysis")
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Position (m)", row=1, col=1)
        fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Attitude analysis
        fig_attitude = go.Figure()
        
        fig_attitude.add_trace(go.Scatter(x=time_data, y=np.degrees(attitudes[:, 0]), 
                                        name='Roll', line=dict(color='red')))
        fig_attitude.add_trace(go.Scatter(x=time_data, y=np.degrees(attitudes[:, 1]), 
                                        name='Pitch', line=dict(color='green')))
        fig_attitude.add_trace(go.Scatter(x=time_data, y=np.degrees(attitudes[:, 2]), 
                                        name='Yaw', line=dict(color='blue')))
        
        fig_attitude.update_layout(
            title="Attitude vs Time",
            xaxis_title="Time (s)",
            yaxis_title="Angle (degrees)",
            height=400
        )
        
        st.plotly_chart(fig_attitude, use_container_width=True)

else:
    # Welcome message
    st.markdown("""
    ## Welcome to the Flying Robot Control Demo! 🚁
    
    This interactive demo allows you to test different control algorithms for quadrotor UAVs.
    
    ### Features:
    - **Multiple Controllers**: Test PID, LQR, MPC, and Geometric controllers
    - **Real-time Visualization**: See 3D trajectories and performance metrics
    - **Interactive Parameters**: Adjust controller gains and simulation settings
    - **Comprehensive Analysis**: Detailed performance metrics and error analysis
    
    ### How to Use:
    1. Select a controller type from the sidebar
    2. Adjust the target position using the sliders
    3. Modify controller parameters as needed
    4. Click "Run Simulation" to start the simulation
    5. Explore the results in the different tabs
    
    ### Safety Note:
    This is a simulation-only demo for educational purposes. Do not use this code on real hardware without expert review.
    
    ### Getting Started:
    Try running a simulation with the default PID controller and target position (5, 5, 5).
    """)
    
    # Display controller information
    st.subheader("Controller Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **PID Controller**
        - Proportional-Integral-Derivative control
        - Simple and robust
        - Good for basic position control
        - Easy to tune
        """)
        
        st.markdown("""
        **LQR Controller**
        - Linear Quadratic Regulator
        - Optimal control based on linearized dynamics
        - Minimizes quadratic cost function
        - Good performance near hover condition
        """)
    
    with col2:
        st.markdown("""
        **MPC Controller**
        - Model Predictive Control
        - Handles constraints explicitly
        - Optimal control over prediction horizon
        - More computationally intensive
        """)
        
        st.markdown("""
        **Geometric Controller**
        - Control on SE(3) manifold
        - Geometric guarantees
        - Robust to model uncertainties
        - Advanced control theory
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Flying Robot Control Demo - Educational Simulation Only</p>
    <p>⚠️ Do not use on real hardware without expert review</p>
</div>
""", unsafe_allow_html=True)
