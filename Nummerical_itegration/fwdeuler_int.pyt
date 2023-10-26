import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Import Matplotlib

print("Running!")

# Open the CSV file and read the data
current_dir = os.getcwd()
csv_file_path = os.path.join(current_dir, 'pendel/recorded_data/45_lang_2.csv')
df = pd.read_csv(csv_file_path)

# Parameters
b = 0.9684993
L = 0.68509933
g = 9.81  # Gravitational acceleration

# Calculate the average time intervals for the datapoints
tid = []
for i in range(1, len(df)):
    dt = df['Time (s)'].iloc[i] - df['Time (s)'].iloc[i - 1]
    tid.append(dt)
dt = np.mean(tid)

# Define the system
def system(t, variables):
    y, v = variables
    
    dydt = v
    dvdt = -b * v - (g / L) * np.sin(y)
    return [dydt, dvdt]

# Forward Euler method
def forward_euler_step(y, v, dt):
    dydt, dvdt = system(0, [y, v])
    y_new = y + dydt * dt
    v_new = v + dvdt * dt
    return y_new, v_new

# Initialize arrays to store state values
num_steps = len(df['Time (s)'])
theta_est = np.zeros(num_steps)
theta_dot_est = np.zeros(num_steps)

# Initial conditions
y0 = np.radians(df['Vinkel'].iloc[0])  # Convert degrees to radians
v0 = np.radians(df['Vinkel_hastighet'].iloc[0])  # Convert degrees per sec to radians per sec

# Store initial conditions in the arrays
theta_est[0] = y0
theta_dot_est[0] = v0

# Perform the integration using Forward Euler
for i in range(1, num_steps):
    y, v = forward_euler_step(theta_est[i - 1], theta_dot_est[i - 1], dt)
    theta_est[i] = y
    theta_dot_est[i] = v

# Plot the results
time = df['Time (s)']
plt.figure(figsize=(12, 6))
plt.plot(time, np.degrees(theta_est), label='Angle (degrees)')
#plt.plot(time, np.degrees(theta_dot_est), label='Angular Velocity (degrees/s)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.title('Pendulum Motion')
plt.grid(True)
plt.show()

print("Done!")
