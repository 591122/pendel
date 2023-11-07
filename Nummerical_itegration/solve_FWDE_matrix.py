import numpy as np
import matplotlib.pyplot as plt
import time

# Define the ODE function
def pendulum_ode(t, Y, b=0.03449173, g=9.81, L=0.6800446):
    theta, omega = Y
    dtheta_dt = omega
    domega_dt = -b * omega - g / L * theta
    return np.array([dtheta_dt, domega_dt])

# Initial conditions
theta0 = 0.3552124729248687  # Initial angle in radians
omega0 = 2.5600933573249716  # Initial angular velocity
t0 = 0.07683181362695312

Y0 = np.array([theta0, omega0])

# Time span for integration
t_start = t0
t_end = 300
num_steps = 1000000  # Number of time steps
dt = (t_end - t_start) / num_steps

# Measure execution time
start_time = time.time()

# Initialize arrays to store results
t_values = [t0]
Y_values = [Y0]

# Perform forward Euler integration
for _ in range(1, num_steps + 1):
    t_new = t_values[-1] + dt
    Y_new = np.zeros(2)  # Initialize new state as a column vector

    # Forward Euler update
    Y_new = Y_values[-1] + dt * pendulum_ode(t_values[-1], Y_values[-1])
    
    t_values.append(t_new)
    Y_values.append(Y_new)

# Calculate execution time
end_time = time.time()
execution_time = end_time - start_time

# Convert the lists to NumPy arrays
t = np.array(t_values)
Y_numerical = np.array(Y_values).T  # Transpose for convenient slicing

# Analytical solution
def analytical_solution(t):
    theta = (0.750286030314788 * np.sin(3.79805568991349 * t) + 0.146008031404678 * np.cos(3.79805568991349 * t)) * np.exp(-0.017245865 * t)
    omega = (-0.750286030314788 * 3.79805568991349 * np.cos(3.79805568991349 * t) - 0.146008031404678 * 3.79805568991349 * np.sin(3.79805568991349 * t) - 0.017245865 * (0.750286030314788 * np.sin(3.79805568991349 * t) + 0.146008031404678 * np.cos(3.79805568991349 * t))) * np.exp(-0.017245865 * t)
    return np.array([theta, omega])

theta_analytical, omega_analytical = analytical_solution(t)

# Calculate the error
error_theta = np.abs(Y_numerical[0] - theta_analytical)
error_omega = np.abs(Y_numerical[1] - omega_analytical)

# Plot both solutions and the error
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, Y_numerical[0], label='Numerical Solution (Theta)')
plt.plot(t, theta_analytical, label='Analytical Solution (Theta)', linestyle='dashed')
plt.xlabel('Time (t)')
plt.ylabel('Angle (θ)')
plt.legend()
plt.grid()
plt.title('Pendulum Motion with Linear Resistance (Analytical vs. Numerical)')

plt.subplot(2, 1, 2)
plt.plot(t, error_theta, label='Error (Theta)')
plt.xlabel('Time (t)')
plt.ylabel('Error')
plt.legend()
plt.grid()
plt.title('Error between Analytical and Numerical Solutions (Theta)')

plt.tight_layout()
plt.show()

# Print execution time
print(f"Execution time: {execution_time:.2f} seconds")
