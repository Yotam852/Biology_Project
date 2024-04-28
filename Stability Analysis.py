import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import scipy as sp
from scipy.integrate import odeint
import sympy as smp

# Define the variables
x, y, b, n, r = smp.symbols('x y b n r')

# Define the functions
x_s = 1/r * b/(1 + y**n)
y_s = 1/r * b/(1 + x**n)

# Calculate the derivatives
dx_s_dy = x_s.diff(y)
dy_s_dx = y_s.diff(x)

# Display the results
print("dx_s/dy:", dx_s_dy)
print("dy_s/dx:", dy_s_dx)

# Example calculation: Let's compute the derivative values for specific values of x, y, b, n, r
values = {x: 1, y: 1, b: 1, n: 2, r: 1}
print("dx_s/dy at x=1, y=1, b=1, n=2, r=1:", dx_s_dy.subs(values))
print("dy_s/dx at x=1, y=1, b=1, n=2, r=1:", dy_s_dx.subs(values))

