import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.integrate import solve_ivp, odeint

print("Running!")

# Define the gravitational acceleration (m/s^2), length of the pendulum (m), initial angle (radians),
# initial angular velocity (radians/s), and time step (s)
g = 9.81
L = 0.684
theta0 = np.pi/4
omega0 = 0.0

# Load data
# Get the current working directory
current_dir = os.getcwd()
csv_file_path = os.path.join(current_dir, 'recorded_data/45_lang_2.csv')
df = pd.read_csv(csv_file_path)
df['Vinkel'] = np.radians(df['Vinkel'])
df['Vinkel_hastighet'] = np.radians(df['Vinkel_hastighet'])
t_values = df['Time (s)'].values



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
        dt = df['Time (s)'].iloc[i] - df['Time (s)'].iloc[i-1]
        y_new, v_new = forward_euler_step(y_prev, v_prev, b, L, dt)
        theta_est[i] = y_new
        theta_dot_est[i] = v_new

    # Calculate the error using theta_est and the actual data
    error = np.sum((theta_est - df['Vinkel'])**2)
    return error

# Initial guess & bounds for the parameters
initial_guess = [1/3, 0.7]
bounds = [(0.01, 2), (0.1, 2)]

# Perform the minimization with the constraint and printing the result
from scipy.optimize import minimize
res = minimize(C, initial_guess, bounds=bounds)
print("Optimal solution:", res.x)


# Calculate the error after the optimization
error = np.sum((theta_est - df['Vinkel'])**2)
print(f'squared error is:{error}')


# Plot the results
plt.plot(t_values, df['Vinkel'], 'o', label='Original Data')
plt.plot(t_values, theta_est, 'o-', label='Estimated')
plt.title('Parameter estimation of L = ' + str(res.x[1])[:5] + ' and b = ' + str(res.x[0])[:5] + ' for pendulum system with linear damping. With Mean squard error = '  + str(error)[:5])
plt.xlabel('Time (s)')
plt.ylabel('Angle (radians)')
plt.legend()
plt.show()

print('Done!')