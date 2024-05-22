import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import scienceplots
import scipy as sp
from scipy.integrate import odeint

# Creating the function for the ODE
def model(z, t, betaD, betaR, v, n, m, k, M, i, j):
    D = z[:k]
    R = z[k:2*k]
    D_n = M @ D
    dDdT = v * (betaD * i**n / (i**n + R**n) - D)
    dRdt = betaR * D_n**m / (j**m + D_n**m) - R
    return np.concatenate([dDdT, dRdt])

# Finding the Neighbours for each cell
def get_connectivity_matrix(P, Q, w):
    k = P * Q  # number of cells
    M = np.zeros((k, k))  # connectivity matrix

    for s in range(k):
        neighbors = find_neighbor_hex(s, P, Q)
        for neighbor in neighbors:
            M[s, neighbor] = w
    np.fill_diagonal(M, 0)
    return M

def find_neighbor_hex(ind, P, Q):
    p, q = ind2pq(ind, P)
    neighbors = [
        (p, (q + 1) % Q), ((p + 1) % P, q), ((p + 1) % P, (q - 1 + Q) % Q),
        (p, (q - 1 + Q) % Q), ((p - 1 + P) % P, q), ((p - 1 + P) % P, (q + 1) % Q)
    ]
    neighbors_ind = [pq2ind(np_p, np_q, P) for np_p, np_q in neighbors]
    return neighbors_ind

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
w = 1/6
betaD = 10
betaR = 10
v = 1
M = get_connectivity_matrix(P, Q, w)
i = 1
j = 1

# Initial conditions
D0 = 1e-5 * np.random.random(k)
R0 = np.zeros(k)
z0 = np.concatenate([D0, R0])

# Solving the ODE
z = odeint(model, z0, t, args=(betaD, betaR, v, n, m, k, M, i, j))
D = z[:, :k]
R = z[:, k:2*k]

# Plotting Concentrations
plt.figure(1)
plt.style.use(['science', 'notebook', 'grid'])
fig, ax = plt.subplots(1, 2, sharex=True, figsize=(8, 6))
ax[0].plot(t, D[:, :2])
ax[0].legend(['D1', 'D2'])
ax[1].plot(t, R[:, :2])
ax[1].legend(['R1', 'R2'])
fig.text(0.5, 0.04, 'Time [a.u]', ha='center')
fig.text(0.04, 0.5, 'Concentration [a.u]', va='center', rotation='vertical')
fig.suptitle('Lateral Inhibition Model for a Grid of Cells')
plt.show()

# Plotting hexagonal grids
def plot_hex_grid(values, P, Q, ax, title):
    hex_radius = 1.0
    hex_height = np.sqrt(3) * hex_radius  # height of the hexagon
    for q in range(Q):
        for p in range(P):
            x = p * 1.5 * hex_radius  # horizontal distance between centers
            y = q * hex_height + (p % 2) * hex_height / 2  # staggered vertically
            color = plt.cm.viridis(values[q * P + p])
            hex = RegularPolygon((x, y), numVertices=6, radius=hex_radius, orientation=np.radians(30),
                                 facecolor=color, edgecolor='k')
            ax.add_patch(hex)
    ax.set_xlim(-hex_radius, P * 1.5 * hex_radius)
    ax.set_ylim(-hex_radius, Q * hex_height)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
plot_hex_grid(D[-1], P, Q, ax1, 'Final D Concentrations')
plot_hex_grid(R[-1], P, Q, ax2, 'Final R Concentrations')
fig.suptitle('Hexagonal Grid of Cell Concentrations')
plt.show()
