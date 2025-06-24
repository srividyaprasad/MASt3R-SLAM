import numpy as np
import pandas as pd
import torch    

def load_imu_data(csv_path):
    imu_data = pd.read_csv(csv_path)
    
    timestamps = torch.tensor(imu_data['time'].values, dtype=torch.float64)
    print(f"timestamps: {timestamps}")
    accel = torch.tensor([
        imu_data['acc_x'].values,
        imu_data['acc_y'].values,
        imu_data['acc_z'].values
    ], dtype=torch.float32).T  # Shape: (N, 3)
    
    gyro = torch.tensor([
        imu_data['gyro_x'].values,
        imu_data['gyro_y'].values,
        imu_data['gyro_z'].values
    ], dtype=torch.float32).T  # Shape: (N, 3)
    
    hw_timestamps = {
        'diag': torch.tensor(imu_data['hw_time_diag'].values, dtype=torch.float64),
        'acc': torch.tensor(imu_data['hw_time_acc'].values, dtype=torch.float64),
        'gyro': torch.tensor(imu_data['hw_time_gyro'].values, dtype=torch.float64)
    }
    
    # Calculate dt between consecutive measurements
    dt = torch.zeros_like(timestamps)
    dt[1:] = timestamps[1:] - timestamps[:-1]
    dt[0] = dt[1]  # Use first dt for initial measurement
    
    return {
        'timestamps': timestamps,
        'accel': accel,
        'gyro': gyro,
        'hw_timestamps': hw_timestamps,
        'dt': dt
    }

def main():
    # Load IMU data
    imu_data = load_imu_data("/home/isaac-sim/mast3r/debug/2025-05-05-14-33-30-giga-ATI-Bangalore-Gorguntepalya-manual/imu.csv")
    
    print("Data shapes:")
    print(f"Timestamps: {imu_data['timestamps'].shape}")    
    print(f"Accelerometer: {imu_data['accel'].shape}")
    print(f"Gyroscope: {imu_data['gyro'].shape}")
    print(f"dt: {imu_data['dt'].shape}")
    
    print("\nFirst 5 measurements:")
    for i in range(5):
        print(f"\nMeasurement {i}:")
        print(f"Time: {imu_data['timestamps'][i]:.3f}")
        print(f"dt: {imu_data['dt'][i]:.3f}")
        print(f"Accel: {imu_data['accel'][i]}")
        print(f"Gyro: {imu_data['gyro'][i]}")

if __name__ == "__main__":
    main()