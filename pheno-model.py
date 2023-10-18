from scipy.integrate import odeint

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt 
#sssrhts
from scipy.stats import linregress
from scipy.optimize import minimize
import plotly.graph_objects as go
# Define a function that represents the differential equation
def diff_equation(a, t, b, g, L):
    theta, theta_dot = a
    theta_double_dot = -b * theta_dot - (g / L) * np.sin(theta)
    return [theta_dot, theta_double_dot]

# Define a cost function that computes the error between the observed data and the solution of the differential equation
def C(a, months, data, b, L):
    solution = odeint(diff_equation, a, months, args=(b, 9.8, L))
    return np.sum((solution[:, 0] - data)**2)

# The bounds for optimization
bounds = [(0, 40), (0, 2 * np.pi / 12.5)]

# Load your data using pandas

data = pd.read_csv(r"pendel\recorded_data\45_lang_1.csv")

# Initial guess
initial_guess = [16, np.pi/6]

# Perform the minimization with the constraint
res = minimize(C, initial_guess, args=(data.index, data['Vinkel'],), bounds=bounds)

# Print the result
print("Optimal solution:", res.x)