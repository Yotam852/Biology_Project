import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Creating the function

def model(z, t, betaD, betaR, v, n, m):
    D1, D2, R1, R2 = z
    dD1dt = v * (betaD / (1 + R1**n) - D1)
    dD2dt = v * (betaD / (1 + R2**n) - D2)
    dR1dt = betaR * D2**m / (1 + D2**m) - R1
    dR2dt = betaR * D1**m / (1 + D1**m) - R2
    return [dD1dt, dD2dt, dR1dt, dR2dt]


# initial conditions

z0 = [1e-5, 1e-5, 0, 0]

# setting up the time axis

t = np.linspace(0, 10, 100)

# choosing parameters

n = 1
m = 1
betaD = 1
betaR = 1
v = 1

# solving the ODE

z = odeint(model, z0, t, args=(n, m, betaD, betaR, v))

# print(z)

D1 = z[:, 0]
D2 = z[:, 1]
R1 = z[:, 2]
R2 = z[:, 3]

# plotting

plt.figure(1)
plt.plot(t, D1, 'r--')
plt.plot(t, D2, 'b-')
plt.legend(['D1', 'D2'], loc='best')
plt.figure(2)
plt.plot(t, R1, 'r--')
plt.plot(t, R2, 'b-')
plt.legend(['R1', 'R2'], loc='best')
plt.show()
