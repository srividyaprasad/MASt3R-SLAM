import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pypose as pp
import os

def load_imu_data(csv_path):
    imu_data = pd.read_csv(csv_path)

    timestamps = torch.tensor(imu_data['time'].values, dtype=torch.float64)
    accel = torch.tensor(imu_data[['acc_x', 'acc_y', 'acc_z']].values, dtype=torch.float32)
    gyro = torch.tensor(imu_data[['gyro_x', 'gyro_y', 'gyro_z']].values, dtype=torch.float32)

    dt = torch.zeros_like(timestamps)
    dt[1:] = timestamps[1:] - timestamps[:-1]
    dt[0] = dt[1]

    return {
        'timestamps': timestamps,
        'accel': accel*9.81,
        'gyro': gyro*np.pi/180,
        'dt': dt
    }

def create_time_batches(imu_data, batch_duration=0.1):
    """Create batches of IMU data based on time intervals."""
    timestamps = imu_data['timestamps']
    accel = imu_data['accel']
    gyro = imu_data['gyro']
    dt = imu_data['dt']
    
    # Calculate batch boundaries
    start_time = timestamps[0]
    end_time = timestamps[-1]
    batch_boundaries = torch.arange(start_time, end_time + batch_duration, batch_duration)
    
    batches = []
    for i in range(len(batch_boundaries) - 1):
        start_idx = torch.searchsorted(timestamps, batch_boundaries[i])
        end_idx = torch.searchsorted(timestamps, batch_boundaries[i + 1])
        
        if start_idx < end_idx:  # Only create batch if there's data in this time window
            batch = {
                'timestamps': timestamps[start_idx:end_idx],
                'accel': accel[start_idx:end_idx],
                'gyro': gyro[start_idx:end_idx],
                'dt': dt[start_idx:end_idx]
            }
            batches.append(batch)
    
    return batches

def main():
    # Load IMU data
    imu_data = load_imu_data("/home/isaac-sim/mast3r/datasets/real_datasets/high-res-hall2-manual/2025-05-20-17-28-34-giga-ati-gg-hq-manual//imu.csv")

    # acc = imu_data['accel'].unsqueeze(0)  # (1, L, 3)
    # gyro = imu_data['gyro'].unsqueeze(0)  # (1, L, 3)
    # dt = imu_data['dt'].unsqueeze(0).unsqueeze(-1)      # (1, L, 1)

    # Create batches of 0.1 seconds
    batches = create_time_batches(imu_data, batch_duration=1/15)

    # Initial state
    pos = torch.zeros(1, 3, dtype=torch.float64)
    vel = torch.zeros(1, 3, dtype=torch.float64)
    rot = pp.identity_SO3(1).double()  # Shape: (1,)
    
    # Store all position estimates
    all_positions = []
    
    # Process each batch
    for batch in batches:
        acc = batch['accel'].unsqueeze(0)  # (1, L, 3)
        gyro = batch['gyro'].unsqueeze(0)  # (1, L, 3)
        dt = batch['dt'].unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
        
        # Preintegrator
        integrator = pp.module.IMUPreintegrator(pos, rot, vel, reset=False).double()
        
        # Run integration for this batch
        state = integrator(dt=dt, acc=acc, gyro=gyro, rot=rot)
        
        # Update state for next batch
        pos = state['pos'][:, -1:]  # Take last position
        vel = state['vel'][:, -1:]  # Take last velocity
        rot = state['rot'][-1:]     # Take last rotation
        
        # Store positions
        all_positions.append(state['pos'][0].cpu().numpy())
    
    # Combine all positions
    pos_est = np.concatenate(all_positions, axis=0)
    
    # Plot
    plt.figure()
    plt.plot(pos_est[:, 0], pos_est[:, 1], label='Integrated XY')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('IMU Trajectory (1/15s batches)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig("imu_traj.png")
    print("Saved to imu_traj.png")


if __name__ == "__main__":
    main()