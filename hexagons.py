import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


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


def get_connectivity_matrix(P, Q):
    k = P * Q
    M = np.zeros((k, k))
    w = 1 / 6

    for s in range(k):
        neighbors = find_neighbor_hex(s, P, Q)
        for r in range(6):
            M[s, neighbors[r]] = w
    return M


def find_neighbor_hex(ind, P, Q):
    p, q = ind2pq(ind, P)
    neighbors = [
        pq2ind((p % P) + 1, q, P),
        pq2ind((p - 2) % P + 1, q, P),
        pq2ind(p % P + 1, (q - 2) % Q + 1, P),
        pq2ind((p - 2) % P + 1, (q - 2) % Q + 1, P),
        pq2ind(p % P + 1, (q % Q) + 1, P),
        pq2ind((p - 2) % P + 1, (q % Q) + 1, P)
    ]
    return [n for n in neighbors if 0 <= n < P * Q]  # Ensure indices are within bounds


def pq2ind(p, q, P):
    return (p - 1) + (q - 1) * P


def ind2pq(ind, P):
    q = 1 + (ind // P)
    p = 1 + (ind % P)
    return p, q


def get_initial_conditions(params, k):
    U = np.random.rand(k) - 0.5
    epsilon = 1e-5
    D0 = epsilon * params['betaD'] * (1 + params['sigma'] * U)
    R0 = np.zeros(k)
    y0 = np.concatenate([D0, R0])
    return y0


def li(y, t, params):
    nu = params['nu']
    betaD = params['betaD']
    betaR = params['betaR']
    h = params['h']
    m = params['m']
    M = params['connectivity']
    k = len(M)

    D = y[:k]
    R = y[k:2 * k]
    Dneighbor = M.dot(D)

    dD = nu * (betaD / (1 + R ** h) - D)
    dR = betaR * Dneighbor ** m / (1 + Dneighbor ** m) - R

    return np.concatenate([dD, dR])


def plot_two_cells(t, y, k):
    plt.ioff()  # Ensure interactive mode is off
    fig, axes = plt.subplots(1, 2)
    for i in range(2):
        axes[i].plot(t, y[:, i], '-r', linewidth=2)
        axes[i].plot(t, y[:, k + i], '-b', linewidth=2)
        axes[i].set_title(f'cell #{i + 1}')
        axes[i].set_xlabel('t [a.u]')
        axes[i].set_ylabel('concentration [a.u]')
        axes[i].legend(['D', 'R'])
    plt.show()


def plot_hexagon(p0, q0, c, ax):
    s32 = np.sqrt(3) / 2
    q = q0 * 3 / 4
    p = p0 * s32
    if q0 % 2 == 0:
        p += s32 / 2

    x = [q - 0.5, q - 0.25, q + 0.25, q + 0.5, q + 0.25, q - 0.25]
    y = [p, p + s32, p + s32, p, p - s32, p - s32]

    polygon = patches.Polygon(list(zip(x, y)), closed=True, color=c, linewidth=2)
    ax.add_patch(polygon)


def movie_lattice(t, y, P, Q, k):
    fig, ax = plt.subplots()
    Cmax = np.max(y[-1, :k])

    def update(frame):
        ax.clear()
        for i in range(1, P + 1):
            for j in range(1, Q + 1):
                ind = pq2ind(i, j, P)
                if ind < k:  # Ensure the index is within bounds
                    mycolor = min(y[frame, ind] / Cmax, 1)
                    plot_hexagon(i, j, [1 - mycolor, 1 - mycolor, 1], ax)
        ax.set_aspect('equal')
        ax.axis('off')

    ani = FuncAnimation(fig, update, frames=range(0, len(t), 5), repeat=False)
    plt.show()
    return ani


def multicell_LI(params=None):
    if params is None:
        params = default_params()

    P = params['P']
    Q = params['Q']
    k = P * Q

    params['connectivity'] = get_connectivity_matrix(P, Q)
    y0 = get_initial_conditions(params, k)

    t = np.linspace(0, 30, 300)
    y = odeint(li, y0, t, args=(params,))

    plot_two_cells(t, y, k)
    ani = movie_lattice(t, y, P, Q, k)

    return y, t, params, ani


# Run the simulation
y, t, params, ani = multicell_LI()
