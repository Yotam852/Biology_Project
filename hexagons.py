import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scienceplots
import scipy as sp
from scipy.integrate import odeint


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
            if neighbor != -1:
                M[s, neighbor] = w

    np.fill_diagonal(M, 0)
    return M


def find_neighbor_hex(ind, P, Q):
    p, q = ind2pq(ind, P)
    neighbors = []

    if q % 2 == 0:  # even column
        directions = [(-1, 0), (1, 0), (-1, -1), (0, -1), (-1, 1), (0, 1)]
    else:  # odd column
        directions = [(-1, 0), (1, 0), (0, -1), (1, -1), (0, 1), (1, 1)]

    for dp, dq in directions:
        np, nq = (p + dp) % P, (q + dq) % Q  # Apply periodic boundary conditions
        neighbors.append(pq2ind(np, nq, P))

    return neighbors


def pq2ind(p, q, P):
    return q * P + p


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
# print(M)
i = 1
j = 1

# Initial conditions
D0 = 1e-5 * np.random.random(k)
R0 = np.zeros(k)
z0 = np.ravel([D0, R0])

# Solving the ODE
z = odeint(model, z0, t, args=(betaD, betaR, v, n, m, k, M, i, j))
D = z[:, :k]
R = z[:, k:2 * k]

# Plotting Concentrations
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


# Plotting Hexagons
def draw_hexagonal_lattice(values, P, Q):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    hex_radius = 1
    hex_height = np.sqrt(3) * hex_radius
    hex_width = 2 * hex_radius

    def hexagon(x_center, y_center, color):
        hexagon = patches.RegularPolygon((x_center, y_center), numVertices=6, radius=hex_radius,
                                         orientation=np.radians(30), edgecolor='k')
        hexagon.set_facecolor(color)
        ax.add_patch(hexagon)

    norm = plt.Normalize(min(values), max(values))
    cmap = plt.get_cmap('viridis')

    index = 0
    for q in range(Q):
        for p in range(P):
            if index < len(values):
                x = q * 1.5 * hex_radius
                y = p * hex_height + (q % 2) * (hex_height / 2)
                color = cmap(norm(values[index]))
                hexagon(x, y, color)
                index += 1

    ax.autoscale()
    ax.axis('off')
    plt.show()


draw_hexagonal_lattice(D[-1, :], P, Q)
draw_hexagonal_lattice(R[-1, :], P, Q)
