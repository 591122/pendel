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
t_end = 300  # Final time
h = 0.0001  # Step size

# Measure execution time
start_time = time.time()

# Initialize arrays
t = np.arange(t0, t_end, h)
u = np.zeros_like(t)
v = np.zeros_like(t)

# Set initial conditions
u[0] = omega0
v[0] = theta0

# Perform forward Euler integration
for i in range(1, len(t)):
    u[i] = (-h * b + 1) * u[i-1] - g * h / L * v[i-1]
    v[i] = v[i-1] + h * u[i]

# Calculate execution time
end_time = time.time()
execution_time = end_time - start_time


# Convert the lists to NumPy arrays
t = np.array(t)
theta_numerical = np.array(v)

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
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.legend()
plt.grid()
plt.title(f'Analytical vs. Forward Euler, Execution time: {execution_time:.2f} seconds. Step size: {h}')

plt.subplot(2, 1, 2)
plt.plot(t, error, label='Error [rad]')
plt.xlabel('Time [s]')
plt.ylabel('Error [rad]')
plt.legend()
plt.grid()
plt.title('Error between Analytical and Forward Euler')

plt.tight_layout()
plt.show()