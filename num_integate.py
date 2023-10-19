# Legg til dette i morra. 

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def system(t, variables, parameters):
    y, v = variables
    b, g, L = parameters
    
    dydt = v
    dvdt = -b*v - (g/L)*np.sin(y)  
    
    return [dydt, dvdt]

# Define initial conditions
initial_conditions = [np.pi/2, 0]  # y(0) = 0, dy/dt(0) = 1


# Define the time span over which you want to solve the ODE
t_span = [0, 100]  # From t=0 to t=10


# Define parameters:b , g, L 
parameters = [0.05, 9.81, 0.684]

# Call solve_ivp
sol = solve_ivp(system, t_span, initial_conditions, args=(parameters,), t_eval=np.linspace(0, 100, 1000))


# The solution is stored in sol.y
y_values = sol.y[0]
t_values = sol.t


plt.plot(t_values,y_values)
plt.show()
# Now you can use y_values and t_values to analyze the solution






