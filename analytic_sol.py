import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math

def quadratic_formula(a, b, c):
    discriminant = b**2 - 4*a*c
    
    if discriminant > 0:
        x1 = (-b + math.sqrt(discriminant)) / (2*a)
        x2 = (-b - math.sqrt(discriminant)) / (2*a)
        return x1, x2
    elif discriminant == 0:
        x = -b / (2*a)
        return x, None
    else:
        real_part = -b / (2*a)
        imag_part = math.sqrt(abs(discriminant)) / (2*a)
        x1 = complex(real_part, imag_part)
        x2 = complex(real_part, -imag_part)
        return x1, x2

# ABC:
a = 1
b = 0.03449173
L = 0.6800446
c = 9.81/L

#b = 0.03449173, L =  0.6800446 linear solution
#

x1, x2 = quadratic_formula(a, b, c)

print(f"Solutions: x1 = {x1}, x2 = {x2}")

# Define the symbolic variables
C1, C2 = sp.symbols('C1 C2')
beta = 3.7980556899134923 #x1.imag
alpha = -0.017245865 #x1.real
t0 = 0.07683181762695312
y0 = 20.352175529
y0_dot = 146.68254453419826

# Define the equations
eq1 = sp.Eq(y0, sp.exp(alpha*t0) * (C1*sp.cos(beta*t0) + C2*sp.sin(beta*t0)))
eq2 = sp.Eq(y0_dot, alpha*sp.exp(alpha*t0) * (C1*sp.cos(beta*t0) + C2*sp.sin(beta*t0)) - beta*sp.exp(alpha*t0) * (C1*sp.sin(beta*t0) - C2*sp.cos(beta*t0)))

# Solve the equations
solutions = sp.solve((eq1, eq2), (C1, C2))

# Print the solutions
print("Solutions:")
print(solutions[C1],solutions[C2])

alpha = -0.48424965
c1 = 8.36564397450163#solutions[C1]#7.94083085999242
c2 = 42.9882229646617#solutions[C2]
print(c1,c2)

t = np.linspace(0,291,4088)
y = np.exp(alpha*t) * (c1*np.cos(beta*t) + c2*np.sin(beta*t))


plt.plot(t,y)
plt.show()