import numpy as np
import matplotlib.pyplot as plt
import time

# Initial conditions
theta0 = 0.3552124729248687  # Initial angle in radians
omega0 = 2.5600933573249716  # Initial angular velocity
t0 = 0.07683181362695312
b=0.03449173
g=9.81
L=0.6800446

y0 = [theta0, omega0]

# Time span for integration
t_end = 300
num_steps = 30000000  # Number of time steps
#dt = (t_end - t0) / num_steps
dt = 0.00001 #step size


# Measure execution time
start_time = time.time()

# Initialize arrays to store results
t_values = [t0]
theta_values = [theta0]

# Perform forward Euler integration
for _ in range(1, num_steps + 1):
    t_new = t_values[-1] + dt
    y_new = [0, 0]  # Initialize new state

    # Forward Euler update
    y_new[0] = y0[0] + dt * y0[1]
    y_new[1] = y0[1] + dt * (-b * y0[1] - g / L * y0[0])

    t_values.append(t_new)
    theta_values.append(y_new[0])

    y0 = y_new  # Update the state

# Calculate execution time
end_time = time.time()
execution_time = end_time - start_time

# Convert the lists to NumPy arrays
t = np.array(t_values)
theta_numerical = np.array(theta_values)

# Analytical solution
def analytical_solution(t):
    return (0.750286030314788 * np.sin(3.79805568991349 * t) + 0.146008031404678 * np.cos(3.79805568991349 * t)) * np.exp(-0.017245865 * t)

theta_analytical = analytical_solution(t)

# Calculate the error
error = np.abs(theta_numerical - theta_analytical)

# Plot both solutions and the error
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, theta_numerical, label='Forward Euler')
plt.plot(t, theta_analytical, label='Analytical Solution', linestyle='dashed')
plt.xlabel('Time [t]')
plt.ylabel('Angle [rad]')
plt.legend()
plt.grid()
plt.title(f'Analytical vs. Forward Euler, Execution time: {execution_time:.2f} seconds')

plt.subplot(2, 1, 2)
plt.plot(t, error, label='Error')
plt.xlabel('Time [s]')
plt.ylabel('Error')
plt.legend()
plt.grid()
plt.title('Error between Analytical and Forward Euler')

plt.tight_layout()
plt.show()