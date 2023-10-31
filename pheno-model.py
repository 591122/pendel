import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.integrate import solve_ivp, odeint
import os

print("Kjører!")

def contains_only_floats(lst):
    for element in lst:
        if not isinstance(element, float):
            return False
    return True


# Load data
# Get the current working directory
current_dir = os.getcwd()
csv_file_path = os.path.join(current_dir, 'pendel/recorded_data/45_lang_2.csv')
df = pd.read_csv(csv_file_path)

# Convert 'Vinkel' column from degrees to radians
df['Vinkel'] = np.radians(df['Vinkel'])
df['Vinkel_hastighet'] = np.radians(df['Vinkel_hastighet'])
print(len(df['Vinkel']))

# ============Define the second-order differential pendulumsystem============
def system(t, variables, parameters):
    y, v = variables
    b, L = parameters
    
    dydt = v
    dvdt = -b*(np.sign(v)*v**2) - (9.81/L)*np.sin(y)  
    #dvdt = -b*v - (9.81/L)*np.sin(y)  
    return [dydt, dvdt]


# ============Define the cost function to be minimized==============
def C(params):
    t = df['Time (s)']

    # Define initial conditions
    initial_conditions = [df['Vinkel'].iloc[0], df['Vinkel_hastighet'].iloc[0] ]  # y(0) = 0, dy/dt(0) = 1

    # Define the time span over which you want to solve the ODE
    t_span = [t.iloc[0], t.iloc[-1]]  

    # Call integrate:
    sol = solve_ivp(system, t_span, initial_conditions, args=(params,), t_eval=t)

    theta_est = sol.y[0]
    theta_dot_est = sol.y[1]
    t_values = sol.t
    #return np.sum(np.abs(theta_est - df['Vinkel']))
    #print(contains_only_floats(theta_est))
    return np.sum((theta_est - df['Vinkel'])**2)



#==================parameter estimation=======================

# Bounds for the parameters
bounds = [(0.001, 1), (0.6, 0.7)]

# Initial guess
initial_guess = [0.05, 0.68004458]  # Adjust these initial guesses as needed

# Perform the minimization with the constraint
res = minimize(C, initial_guess, bounds=bounds)

# Print the result
print("Optimal solution:", res.x)


#==============plotting the data:==================

t = df['Time (s)']  

# Define initial conditions
initial_conditions = [df['Vinkel'].iloc[0], df['Vinkel_hastighet'].iloc[0] ]  # y(0) = 0, dy/dt(0) = 1

# Define the time span over which you want to solve the ODE
t_span = [t.iloc[0], t.iloc[-1]]  # From t=0 to t=10

# Call integrate:
sol = solve_ivp(system, t_span, initial_conditions, args=(res.x,), t_eval=t)
#sol = solve_ivp(system, t_span, initial_conditions, args=([0.9, 0.67971812],), t_eval=t) # HERE IS THE PROBLEM (0.9)




theta_est = sol.y[0]
theta_dot_est = sol.y[1]
t_values = sol.t


error = np.sum((theta_est - df['Vinkel']))**2

print(f'squared error is:{error}')
# Plot the results
# Convert back to degrees for plotting
plt.plot(t, df['Vinkel'], 'o', label='Original Data')
plt.plot(t, theta_est, 'o-', label='Estimated')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
print('FERDIG!')