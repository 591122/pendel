import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

# Define the ODE function
def pendulum_ode(t, y, b=0.03449173, g=9.81, L=0.6800446):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -b * omega - g / L * theta
    return [dtheta_dt, domega_dt]

# Initial conditions
theta0 = 0.3552124729248687  # Initial angle in radians
omega0 = 2.5600933573249716  # Initial angular velocity
t0 = 0.07683181362695312

y0 = [theta0, omega0]

# Time span for integration
t_span = (t0, 300)  # Adjust the time span as needed

# Measure execution time
start_time = time.time()

# Tolerance settings
atol = 1e-14  # Absolute tolerance
rtol = 1e-12  # Relative tolerance

# Solve the ODE using solve_ivp
solution = solve_ivp(pendulum_ode, t_span, y0, atol=atol, rtol=rtol, method='RK23', t_eval=np.linspace(t_span[0], t_span[1], 100000))

# Calculate execution time
end_time = time.time()
execution_time = end_time - start_time

# Extract the numerical solution
t = solution.t
theta_numerical = solution.y[0]

# Analytical solution
def analytical_solution(t):
    return (0.750286030314788 * np.sin(3.79805568991349 * t) + 0.146008031404678 * np.cos(3.79805568991349 * t)) * np.exp(-0.017245865 * t)

theta_analytical = analytical_solution(t)

# Calculate the error
error = np.abs(theta_numerical - theta_analytical)

# Plot both solutions and the error
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, theta_numerical, label='Numerical Solution')
plt.plot(t, theta_analytical, label='Analytical Solution', linestyle='dashed')
plt.xlabel('Time (t)')
plt.ylabel('Angle (θ)')
plt.legend()
plt.grid()
plt.title(f'Analytical vs. Numerical (RK23), Execution time: {execution_time:.2f} seconds, atol = {atol}, rtol = {rtol}')

plt.subplot(2, 1, 2)
plt.plot(t, error, label='Error')
plt.xlabel('Time (t)')
plt.ylabel('Error')
plt.legend()
plt.grid()
plt.title('Error between Analytical and Numerical Solutions')

plt.tight_layout()
plt.show()

# Print execution time
print(f"Execution time: {execution_time:.2f} seconds")
