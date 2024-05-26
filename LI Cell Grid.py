import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scienceplots
import scipy as sp
from scipy.integrate import odeint


# Creating the function for the ODE


def model(z, t, betaD, betaR, v, n, m, k, M, i, j):
    D = z[:k]
    R = z[k:2*k]
    D_n = M @ z[:k]
    dDdT = v * (betaD * i**n / (i**n + R**n) - D)
    dRdt = betaR * D_n**m / (j**m + D_n**m) - R
    return np.ravel([dDdT, dRdt])


# Finding the Neighbours for each cell


def get_connectivity_matrix(P, Q, w):
    k = P * Q  # number of cells
    M = np.zeros((k, k))  # connectivity matrix

    # calculating the connectivity matrix
    for s in range(k):
        kneighbor = find_neighbor_hex(s, P, Q)
        for r in range(6):
            M[s-1, kneighbor[r]-1] = w
    np.fill_diagonal(M, 0)
    return M

def find_neighbor_hex(ind, P, Q):
    # This function finds the 6 neighbors of cell ind
    p, q = ind2pq(ind, P)

    # above and below:
    out = [
        pq2ind((p % P) + 1, q, P),
        pq2ind((p - 2) % P + 1, q, P),
        # left and right sides:
        pq2ind(p if q % 2 != 0 else (p % P) + 1, (q - 2) % Q + 1, P),
        pq2ind((p - 2) % P + 1 if q % 2 != 0 else p, (q - 2) % Q + 1, P),
        pq2ind(p if q % 2 != 0 else (p % P) + 1, q % Q + 1, P),
        pq2ind((p - 2) % P + 1 if q % 2 != 0 else p, q % Q + 1, P)
    ]
    return out

def pq2ind(p, q, P):
    return p + (q - 1) * P

def ind2pq(ind, P):
    q = 1 + ((ind - 1) // P)
    p = ind - (q - 1) * P
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


D0 = 1e-5 * np.random.random((1, k))
R0 = np.zeros((1, k))
z0 = np.ravel([D0, R0])


# Solving the ODE


z = odeint(model, z0, t, args=(n, m, betaD, betaR, v, k, M, i, j))
D = z[:, :k]
R = z[:, k:2*k]
print(z)


# Plotting Concentrations


plt.style.use(['science', 'notebook', 'grid'])
fig, ax = plt.subplots(1, 2, sharex=True, figsize=(8, 6))
ax[0].plot(t, D[:, :2])
# lst1 = []
# for i in range(2*k):
#     lst1.append("D"+str(i))
ax[0].legend(['D1', 'D2'])
ax[1].plot(t, R[:, :2])
# lst2 = []
# for j in range(2*k):
#     lst2.append("R"+str(j))
ax[1].legend(['R1', 'R2'])
fig.text(0.5, 0.04, 'Time [a.u]', ha='center')
fig.text(0.04, 0.5, 'Concentration [a.u]', va='center', rotation='vertical')
fig.suptitle('Lateral Inhibition Model for a Grid of Cells')
plt.show()


# Plotting Hexagons


def draw_hexagonal_lattice(values, P, Q):
    """
    Draws a hexagonal lattice where each hexagon's color corresponds to a value in a list.

    :param values: List of values to determine the color of each hexagon.
    :param P: Number of rows in the hexagonal lattice.
    :param Q: Number of columns in the hexagonal lattice.
    """
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

    # Normalize values to range [0, 1] for color mapping
    norm = plt.Normalize(min(values), max(values))
    cmap = plt.get_cmap('viridis')

    index = 0
    for q in range(Q):
        for p in range(P):
            if index < len(values):
                x = q * hex_width * 0.75
                y = p * hex_height + (q % 2) * (hex_height / 2)
                color = cmap(norm(values[index]))
                hexagon(x, y, color)
                index += 1

    ax.autoscale()
    ax.axis('off')
    plt.show()


draw_hexagonal_lattice(D[-1, :], P, Q)

#