import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import scipy as sp
from scipy.integrate import odeint


# creating the function

def model(z, t, betaD, betaR, v, n, m, k):
    M = 0  # just for now
    D = z[:k]
    R = z[k:2*k]
    D_n = M@z[:k]
    dDdT = v * (betaD / (1 + R ** n) - D)
    dRdt = betaR * D_n**m / (1 + D_n**m) -R
    return [dDdT, dRdt]

# setting up the time axis

t = np.linspace(0, 10, 100)

# choosing parameters

n = 3
m = 3
k = 10
betaD = 10
betaR = 10
v = 1

# initial conditions


D0 = 1e-5 * np.random.random((k, 1))
R0 = np.zeros((k, 1))
z0 = [D0, R0]
# print(z0)

# solving the ODE

z = odeint(model, z0, t, args=(n, m, betaD, betaR, v, k))
# print(z)
D = z[:, :k]
R = z[:, k:2*k]

# plotting

