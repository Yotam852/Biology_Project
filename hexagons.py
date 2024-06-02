import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scienceplots
from scipy.integrate import odeint

# Define the model and helper functions
def model(z, t, betaD, betaR, v, n, m, k, M, i, j):
    D = z[:k]
    R = z[k:2 * k]
    D_n = M @ D
    dDdT = v * (betaD * i ** n / (i ** n + R ** n) - D)
    dRdt = betaR * D_n ** m / (j ** m + D_n ** m) - R
    return np.ravel([dDdT, dRdt])

def get_connectivity_matrix(P, Q, w):
    k = P * Q
    M = np.zeros((k, k))
    for s in range(k):
        neighbors = find_neighbor_hex(s, P, Q)
        for neighbor in neighbors:
            M[s, neighbor] = w
    np.fill_diagonal(M, 0)
    return M

def find_neighbor_hex(ind, P, Q):
    p, q = ind2pq(ind, P)
    neighbors = [
        (p, (q - 1) % Q),  # top
        (p, (q + 1) % Q),  # bottom
        ((p + 1) % P, q if q % 2 == 0 else (q - 1) % Q),  # top-right / bottom-right
        ((p - 1) % P, q if q % 2 == 0 else (q - 1) % Q),  # top-left / bottom-left
        ((p + 1) % P, (q + 1) % Q if q % 2 != 0 else q),  # bottom-right / top-right
        ((p - 1) % P, (q + 1) % Q if q % 2 != 0 else q)  # bottom-left / top-left
    ]
    return [pq2ind(x, y, P) for x, y in neighbors]

def pq2ind(p, q, P):
    return p + q * P

def ind2pq(ind, P):
    q = ind // P
    p = ind % P
    return p, q

# Setting up the parameters
t = np.linspace(0, 30, 300)
n = 3
m = 3
P = 10
Q = 10
k = P * Q
w = 1 / 6
betaD = 10
betaR = 10
v = 1
M = get_connectivity_matrix(P, Q, w)
j = 1

# Initial conditions
D0 = 1e-5 * np.random.random(k)
R0 = np.zeros(k)
z0 = np.ravel([D0, R0])

# Varying i and plotting results
i_values = np.linspace(1e-30, 1e-20, 100)
inhibition_threshold = 0.01

R_final = []
for i in i_values:
    z = odeint(model, z0, t, args=(betaD, betaR, v, n, m, k, M, i, j))
    R = z[:, k:2 * k]
    R_final.append(np.mean(R[-1, :]))

plt.plot(i_values, R_final)
plt.axhline(y=inhibition_threshold, color='r', linestyle='--', label='Inhibition Threshold')
plt.xlabel('i')
plt.ylabel('Final mean R')
plt.title('Final mean R vs i')
plt.legend()
plt.show()

# Finding the lower limit of i where inhibition no longer occurs
lower_limit_i = None
for i, R_mean in zip(i_values, R_final):
    if R_mean < inhibition_threshold:
        lower_limit_i = i
        break

print(f"The lower limit of i where inhibition no longer occurs is approximately: {lower_limit_i}")
