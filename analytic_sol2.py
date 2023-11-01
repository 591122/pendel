import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math




# ABC:
a = 1
b = 0.03449173
L = 0.6800446
c = 9.81/L


t, C1, C2 = sp.symbols('t C1 C2')
y = sp.Function('y')(t)

# Define the differential equation
ode = a*y.diff(t, t) + b*y.diff(t) + c*y

# Solve the differential equation
solution = sp.dsolve(ode)

# Display the solution
print('solution of ODE:')
print(solution)


# Solve the differential equation
solution = sp.dsolve(ode)

# Apply initial conditions
t_value = 0.07683181762695312
y0 = 0.3552124729248687
yp0 = 2.5600933573249716

# Substitute initial conditions to find C1 and C2
eq1 = solution.rhs.subs(t, t_value) - y0
eq2 = solution.rhs.diff(t).subs(t, t_value) - yp0

solutions = sp.solve([eq1, eq2], (C1, C2))

# Substitute C1 and C2 back into the solution
specific_solution = solution.subs(solutions)

# Display the specific solution
print(specific_solution)