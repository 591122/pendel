import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(r'C:\Users\vikto\Documents\Programming\Python codes\Mathematical mod and sim\Project\pendel\recorded_data\45_lang_1.csv')

# Convert 'Vinkel' column from degrees to radians
df['Vinkel'] = np.radians(df['Vinkel'])
df['Vinkel_hastighet'] = np.radians(df['Vinkel_hastighet'])

t_data =  df['Time (s)']  
y_data = df['Vinkel']
v_data = df['Vinkel_hastighet']



# ============Define the second-order differential pendulumsystem============
def second_order_ode_abL(t, variables, b,L):
    y, v = variables
    dydt = v
    dvdt = -b * v - (9.81 / L) * np.sin(y)  
    return [dydt, dvdt]


# Define an initial guess for the coefficients (b and L)
initial_guess_abL = (0.04, 0.684)

# Fit the ODE to the data, considering only b and L
data = np.column_stack((y_data, v_data))
params_abL, covariance_abL = curve_fit(second_order_ode_abL, t_data, data, p0=initial_guess_abL)

# Extract the estimated coefficients (b and L)
b_est_abL, L_est_abL = params_abL

# Print the estimated coefficients
print(f"Estimated coefficients: b = {b_est_abL}, L = {L_est_abL}")

# Define the ODE with the estimated coefficients (b and L)
def estimated_ode_abL(t, variables):
    return second_order_ode_abL(t, variables, b_est_abL, L_est_abL)

# Solve the ODE with the estimated coefficients for visualization
sol_abL = solve_ivp(estimated_ode_abL, (0, 10), [y_data[0], v_data[0]], t_eval=t_data)

# Plot the original data and the estimated solution

plt.figure(figsize=(10, 5))
plt.plot(t_data, y_data, label='Data')
plt.plot(sol_abL.t, sol_abL.y[0], label='Estimated Solution', linestyle='--')
plt.xlabel('t')
plt.ylabel('theta')
plt.legend()
plt.show()