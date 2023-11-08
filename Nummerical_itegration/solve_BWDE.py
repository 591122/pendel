import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Import Matplotlib
from numpy import cos, sin, exp
import time


print("Running!")

# Measure execution time
start_time = time.time()

def backward_euler(v0,u0,t0, h, t_end):
    #params:
    b = 0.03449173
    L = 0.6800446
    g = 9.81  # Gravitational acceleration

    t = [t0]
    v = [v0]
    u = [u0]
    A = np.array([    [1      ,        -h], 
                 [(h * g) / L, 1 + h * b]])

    invA = np.linalg.inv(A)
    while t[-1]<t_end:
        B = np.array([[v[-1]],[u[-1]]])
        #x = np.linalg.solve(A,B)
        x = np.dot(invA,B)
        v.append(x[0][0])
        u.append(x[1][0])
        t.append(t[-1]+h)
    return t, v, u

# Parameters
t_end = 300
t0 = 0.07683181762695312
v0 = 0.3552124729248687
u0 = 2.5600933573249716

h = 0.00001

t, v, u = backward_euler(v0,u0,t0, h, t_end)
t = np.array(t)
y = (0.750286030314788*sin(3.79805568991349*t) + 0.146008031404678*cos(3.79805568991349*t))*exp(-0.017245865*t)

# Calculate execution time
end_time = time.time()
execution_time = end_time - start_time

#params:
b = 0.03449173
L = 0.6800446
g = 9.81  # Gravitational acceleration
A = np.array([[1,-h], [(h * g) / L, 1 + h * b]])
print(A)

# Convert the lists to NumPy arrays
t = np.array(t)
theta_numerical = np.array(v)

# Analytical solution
def analytical_solution(t):
    return (0.750286030314788 * np.sin(3.79805568991349 * t) + 0.146008031404678 * np.cos(3.79805568991349 * t)) * np.exp(-0.017245865 * t)

theta_analytical = analytical_solution(t)


# Calculate the error
error = np.abs(theta_numerical - theta_analytical)

# Plot both solutions and the error
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, theta_numerical, label='Backwards Euler')
plt.plot(t, theta_analytical, label='Analytical Solution', linestyle='dashed')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.legend()
plt.grid()
plt.title(f'Analytical vs. Backwards Euler, Execution time: {execution_time:.2f} seconds. Step size: {h}')

plt.subplot(2, 1, 2)
plt.plot(t, error, label='Error [rad]')
plt.xlabel('Time [s]')
plt.ylabel('Error [rad]')
plt.legend()
plt.grid()
plt.title('Error between Analytical and Backwards Euler')

plt.tight_layout()
plt.show()