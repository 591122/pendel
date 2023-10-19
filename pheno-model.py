import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.integrate import solve_ivp

# Load data
df = pd.read_csv(r'C:\Users\vikto\Documents\Programming\Python codes\Mathematical mod and sim\Project\pendel\recorded_data\45_lang_1.csv')

# Convert 'Vinkel' column from degrees to radians
df['Vinkel'] = np.radians(df['Vinkel'])
df['Vinkel_hastighet'] = np.radians(df['Vinkel_hastighet'])





# ============Define the second-order differential pendulumsystem============
def system(t, variables, parameters):
    y, v = variables
    b, L = parameters
    
    dydt = v
    dvdt = -b*v - (9.81/L)*np.sin(y)  
    return [dydt, dvdt]




# ============Define the cost function to be minimized==============
def C(params):
    t = df['Time (s)']  # 

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
    return np.sum((theta_est - df['Vinkel'])**2)




#==================parameter estimation=======================

# Bounds for the parameters
bounds = [(0, None), (0, None)]

# Initial guess
initial_guess = [0.05, 0.684]  # Adjust these initial guesses as needed

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
#sol = solve_ivp(system, t_span, initial_conditions, args=([0.9, 0.684],), t_eval=t)

theta_est = sol.y[0]
theta_dot_est = sol.y[1]
t_values = sol.t


# Plot the results
# Convert back to degrees for plotting
plt.plot(t, df['Vinkel'], 'o', label='Original Data')
plt.plot(t, theta_est, label='Estimated')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()