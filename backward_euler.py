
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Import Matplotlib
from numpy import cos, sin, exp


print("Running!")


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

    while t[-1]<t_end:
        B = np.array([[v[-1]],[u[-1]]])
        x = np.linalg.solve(A,B)
        v.append(x[0][0])
        u.append(x[1][0])
        t.append(t[-1]+h)
    return t, v, u 

# Parameters
t_end = 100
t0 = 0.07683181762695312
v0 = 0.3552124729248687
u0 = 2.5600933573249716

h = 0.00001

t, v, u = backward_euler(v0,u0,t0, h, t_end)
t = np.array(t)
y = (0.750286030314788*sin(3.79805568991349*t) + 0.146008031404678*cos(3.79805568991349*t))*exp(-0.017245865*t)

#params:
b = 0.03449173
L = 0.6800446
g = 9.81  # Gravitational acceleration
A = np.array([[1,-h], [(h * g) / L, 1 + h * b]])
print(A)

plt.figure(figsize=(10, 6))
plt.plot(t, v,label='backward')
plt.plot(t, y,label='analytic')
plt.title("Angle vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Angle")
plt.legend()
plt.show()