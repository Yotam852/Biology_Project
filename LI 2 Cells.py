import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Creating the function

def model(z, betaD, betaR, v, n, m):
    D1 = z[0]
    D2 = z[1]
    R1 = z[2]
    R2 = z[3]
    dD1dt = v * (-D1 + betaD / (1 + R1 ** n))
    dD2dt = v * (-D2 + betaD / (1 + R2 ** n))
    dR1dt = -R1 + betaR * D2 ** m / (1 + D2 ** m)
    dR2dt = -R2 + betaR * D1 ** m / (1 + D1 ** m)
    return [dD1dt, dD2dt, dR1dt, dR2dt]


# initial conditions

# e1 = np.random.rand()
# e2 = np.random.rand()
# print(e1, e2)
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
