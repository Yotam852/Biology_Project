import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.patches as patches
import matplotlib.animation as animation


# Default parameters
def default_params():
    return {
        'nu': 1,
        'betaD': 50,
        'betaR': 50,
        'h': 3,
        'm': 3,
        'sigma': 0.2,
        'P': 18,
        'Q': 18
    }


# Connectivity matrix
def get_connectivity_matrix(P, Q):
    k = P * Q
    M = np.zeros((k, k))
    w = 1 / 6
    for s in range(k):
        neighbors = find_neighbors(s, P, Q)
        for r in neighbors:
            M[s, r] = w
    return M


# Initial conditions
def get_initial_conditions(params, k):
    U = np.random.rand(k) - 0.5
    epsilon = 1e-5
    D0 = epsilon * params['betaD'] * (1 + params['sigma'] * U)
    R0 = np.zeros(k)
    return np.concatenate((D0, R0))


# Differential equations
def li(y, t, params):
    nu = params['nu']
    betaD = params['betaD']
    betaR = params['betaR']
    h = params['h']
    m = params['m']
    M = params['connectivity']
    k = len(M)

    D = y[:k]
    R = y[k:]
    D_neighbor = M.dot(D)

    dD = nu * (betaD / (1 + R ** h) - D)
    dR = betaR * D_neighbor ** m / (1 + D_neighbor ** m) - R

    return np.concatenate((dD, dR))


# Find neighbors
def find_neighbors(ind, P, Q):
    p, q = ind2pq(ind, P)

    neighbors = [
        pq2ind((p + 1) % P, q, P),
        pq2ind((p - 1) % P, q, P),
        pq2ind(p, (q - 1) % Q, P),
        pq2ind((p + 1) % P, (q - 1) % Q, P),
        pq2ind(p, (q + 1) % Q, P),
        pq2ind((p - 1) % P, (q + 1) % Q, P)
    ]
    return neighbors


# Index to PQ
def pq2ind(p, q, P):
    return p + q * P


# PQ to index
def ind2pq(ind, P):
    q = ind // P
    p = ind % P
    return p, q


# Plot two cells
def plot_two_cells(tout, yout, k):
    plt.figure()
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.plot(tout, yout[:, i], '-r', linewidth=2, label='d')
        plt.plot(tout, yout[:, k + i], '-b', linewidth=2, label='r')
        plt.title(f'Cell #{i + 1}')
        plt.xlabel('t [a.u]')
        plt.ylabel('concentration [a.u]')
        plt.legend()
    plt.show()


# Plot hexagon
def plot_hexagon(ax, p0, q0, color):
    s32 = np.sqrt(3) / 2
    q = q0 * 3 / 2
    p = p0 * s32
    if q0 % 2 == 1:
        p += s32 / 2

    x = [q - 0.5, q - 0.25, q + 0.25, q + 0.5, q + 0.25, q - 0.25]
    y = [p, p + s32 / 2, p + s32 / 2, p, p - s32 / 2, p - s32 / 2]

    hexagon = patches.Polygon(xy=list(zip(x, y)), closed=True, color=color, linewidth=2)
    ax.add_patch(hexagon)


# Movie lattice
def movie_lattice(tout, yout, P, Q, k):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    Cmax = np.max(yout[-1, :k])

    def update(frame):
        ax.clear()
        for i in range(P):
            for j in range(Q):
                ind = pq2ind(i, j, P)
                color = min([yout[frame, ind] / Cmax, 1])
                plot_hexagon(ax, i, j, [1 - color, 1 - color, 1])
        ax.axis('off')
        ax.relim()
        ax.autoscale_view()

    ani = animation.FuncAnimation(fig, update, frames=range(0, len(tout), 5), repeat=False)
    plt.show()
    return ani


# Main simulation function
def multicell_LI(params=None):
    if params is None:
        params = default_params()

    Tmax = 30
    tspan = np.linspace(0, Tmax, 500)

    P = params['P']
    Q = params['Q']
    k = P * Q

    params['connectivity'] = get_connectivity_matrix(P, Q)
    y0 = get_initial_conditions(params, k)

    yout = odeint(li, y0, tspan, args=(params,))
    plot_two_cells(tspan, yout, k)
    ani = movie_lattice(tspan, yout, P, Q, k)

    return yout, tspan, params, ani


# Run the simulation
if __name__ == '__main__':
    multicell_LI()

