import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import scipy as sp
from scipy.integrate import odeint
import sympy as smp

# Define the variables

x, y = smp.symbols('x y')
beta = 10
gamma = 1
n = 1

# Define the functions

x_s = 1/gamma * beta/(1 + y**n)
y_s = 1/gamma * beta/(1 + x**n)

# Calculate the derivatives

values = {x: x_s, y: y_s}
dx_s_dx = x_s.diff(x).subs(values)
dx_s_dy = x_s.diff(y).subs(values)
dy_s_dx = y_s.diff(x).subs(values)
dy_s_dy = y_s.diff(y).subs(values)

# Display the results

print(dx_s_dx)
print(dx_s_dy)
print(dy_s_dx)
print(dy_s_dy)
