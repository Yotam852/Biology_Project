import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import scipy as sp
from scipy.integrate import odeint

# finding the S.S coordinates

x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
n = 1
beta = 10
gamma = 1
x_ss = beta / (gamma*(1+y**n))
y_ss = beta / (gamma*(1+x**n))

# calculating the jacobian matrix

