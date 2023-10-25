import numpy as np
import matplotlib.pyplot as plt

# Define parameters
g = 9.81  # gravitational acceleration (m/s^2)
L = 0.684  # length of the pendulum (m)  66,1CM + 3mm+ 2cm
theta0 = np.pi/4  # initial angle (radians)
omega0 = 0.0  # initial angular velocity (radians/s)
t_max = 10.0  # simulation time (s)
dt = 0.000001  # time step (s)

# Define the equations of motion
def pendulum_derivatives(state, t):
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(g/L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Initialize time and state arrays
t = np.arange(0, t_max, dt)
theta = np.zeros_like(t)
omega = np.zeros_like(t)

# Set initial conditions
theta[0] = theta0
omega[0] = omega0

# Integrate using Euler's method
for i in range(1, len(t)):
    state = [theta[i-1], omega[i-1]]
    dtheta_dt, domega_dt = pendulum_derivatives(state, t[i-1])
    theta[i] = theta[i-1] + dtheta_dt * dt
    omega[i] = omega[i-1] + domega_dt * dt

# Convert theta to degrees for plotting
theta_degrees = np.degrees(theta)

# Plot the results
plt.figure(figsize=(10, 4))
plt.plot(t, theta_degrees)
plt.title('Harmonic Pendulum Simulation')
plt.xlabel('Time [s]')
plt.ylabel('Angle [degrees]')
plt.grid(True)
plt.show()
