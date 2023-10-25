import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Kjører!")

# Define the gravitational acceleration (m/s^2), length of the pendulum (m), initial angle (radians),
# initial angular velocity (radians/s), and time step (s)
g = 9.81
L = 0.684
theta0 = np.pi/4
omega0 = 0.0

# Load data
df = pd.read_csv('/Users/andreaswergeland/git/pendel/pendel/recorded_data/45_lang_2.csv')
df['Vinkel'] = np.radians(df['Vinkel'])
df['Vinkel_hastighet'] = np.radians(df['Vinkel_hastighet'])
t_values = df['Time (s)'].values
dt = t_values[1] - t_values[0]  # Calculate the time step from the data

# Initialize arrays to store state values
num_steps = len(t_values)
theta_est = np.zeros(num_steps)
theta_dot_est = np.zeros(num_steps)

# Define the second-order differential pendulum system
def system(t, variables, parameters):
    y, v = variables
    b, L = parameters
    dydt = v
    dvdt = -b * v - (g / L) * np.sin(y)
    return [dydt, dvdt]

# Forward Euler method
def forward_euler_step(y, v, b, L, dt):
    dydt, dvdt = system(0, [y, v], [b, L])
    y_new = y + dydt * dt
    v_new = v + dvdt * dt
    return y_new, v_new

# Define the cost function to be minimized
def C(params):
    initial_conditions = [df['Vinkel'].iloc[0], df['Vinkel_hastighet'].iloc[0]]
    
    # Initialize state values for each optimization iteration
    theta_est[0], theta_dot_est[0] = initial_conditions

    for i in range(1, num_steps):
        y_prev, v_prev = theta_est[i - 1], theta_dot_est[i - 1]
        b, L = params
        y_new, v_new = forward_euler_step(y_prev, v_prev, b, L, dt)
        theta_est[i] = y_new
        theta_dot_est[i] = v_new

    # Calculate the error using only the relevant part of theta_est
    error = np.sum((theta_est - df['Vinkel'])**2)
    return error

# Bounds for the parameters
bounds = [(0.1, 10), (0.1, 10)]

# Initial guess
initial_guess = [0.05, 0.68004458]

# Perform the minimization with the constraint
from scipy.optimize import minimize
res = minimize(C, initial_guess, bounds=bounds)

# Print the result
print("Optimal solution:", res.x)

# Calculate the error after the optimization
error = np.sum((theta_est - df['Vinkel'])**2)
print(f'squared error is:{error}')

# Plot the results
plt.plot(t_values, df['Vinkel'], 'o', label='Original Data')
plt.plot(t_values, theta_est, label='Estimated')
plt.xlabel('Time (s)')
plt.ylabel('Angle (radians)')
plt.legend()
plt.show()

print('FERDIG!')
