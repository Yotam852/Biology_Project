import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scienceplots
from scipy.integrate import odeint

def model(z, t, betaD, betaR, v, n, m, k, M, i_vec, j_vec):
    D = z[:k]
    R = z[k:2*k]
    D_n = M @ D
    f_R = betaD * i_vec ** n / (i_vec ** n + R ** n)
    g_D = betaR * D_n ** m / (j_vec ** m + D_n ** m)
    dDdT = v * f_R - D
    dRdt = g_D - R
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
mean_i = 4.25
std_i = 0.5  # Standard deviation for the normal distribution for i
i_vec = np.random.normal(mean_i, std_i, k)
mean_j = 6.25
std_j = 0.5  # Standard deviation for the normal distribution for j
# j_vec = np.random.normal(mean_j, std_j, k)
j_vec = 1

# Initial conditions
D0 = 1e-5 * np.random.random(k)
R0 = np.zeros(k)
z0 = np.ravel([D0, R0])

# Solving the ODE
z = odeint(model, z0, t, args=(betaD, betaR, v, n, m, k, M, i_vec, j_vec))
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
                x = q * hex_width * 0.75
                y = p * hex_height + (q % 2) * (hex_height / 2)
                color = cmap(norm(values[index]))
                hexagon(x, y, color)
                index += 1

    ax.autoscale()
    ax.axis('off')
    plt.show()

draw_hexagonal_lattice(D[-1, :], P, Q)
draw_hexagonal_lattice(R[-1, :], P, Q)
