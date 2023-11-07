import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.integrate import solve_ivp, odeint
import os
from numpy import cos, sin, exp

print("Kjører!")

# Load data
# Get the current working directory
current_dir = os.getcwd()
csv_file_path = os.path.join(current_dir, 'recorded_data/45_lang_1.csv')
df = pd.read_csv(csv_file_path)
#df = pd.read_csv(r'C:\Users\vikto\OneDrive - Høgskulen på Vestlandet\Documents\Programming\Python codes\Mathematical mod and sim\Project\pendel\recorded_data\45_lang_2.csv')

# Convert 'Vinkel' column from degrees to radians
df['Vinkel'] = np.radians(df['Vinkel'])
df['Vinkel_hastighet'] = np.radians(df['Vinkel_hastighet'])
print('Antall datapunkt: ',len(df['Vinkel']))

# ============Define the second-order differential pendulumsystem============
def system(t, variables, parameters):
    y, v = variables
    b, L = parameters
    
    dydt = v
    #dvdt = -b*(np.sign(v)*v**2) - (9.81/L)*np.sin(y)  
    dvdt = -b*v - (9.81/L)*np.sin(y)  
    return [dydt, dvdt]


# ============Define the cost function to be minimized==============
def C(params):
    t = df['Time (s)']

    # Define initial conditions
    initial_conditions = [df['Vinkel'].iloc[0], df['Vinkel_hastighet'].iloc[0] ]  # y(0) = 0, dy/dt(0) = 1

    # Define the time span over which you want to solve the ODE
    t_span = [t.iloc[0], t.iloc[-1]]  

    # Call integrate:
    # Tolerance settings
    atol = 1e-8  # Absolute tolerance
    rtol = 1e-6  # Relative tolerance

    sol = solve_ivp(system, t_span, initial_conditions, args=(params,), t_eval=t, atol=atol, rtol=rtol ,method='DOP853')

    theta_est = sol.y[0]
    theta_dot_est = sol.y[1]
    t_values = sol.t
    return np.sum((theta_est - df['Vinkel'])**2)



#==================parameter estimation=======================

# Bounds for the parameters
bounds = [(0.0001, 0.6), (0.6, 0.7)]

# Initial guess
#
initial_guess = [0.03, 0.68004458]  # For linear damping
#initial_guess = [0.02440196, 0.68004458]  #for quad 

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

theta_est = sol.y[0]
theta_dot_est = sol.y[1]
t_values = sol.t


error = np.sum((theta_est - df['Vinkel']))**2

y = (0.750286030314788*sin(3.79805568991349*t) + 0.146008031404678*cos(3.79805568991349*t))*exp(-0.017245865*t)

print(f'squared error is:{error}')
# Plot the results
# Convert back to degrees for plotting
#plt.plot(t, df['Vinkel'], 'o', label='Original Data')
plt.plot(t, theta_est, 'o-', label='Estimated')
plt.plot(t,y,label='analytical')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
print('FERDIG!')