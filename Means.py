# A code that executes LI for a cell grid for multiple <k> values where k is drawn for each cell from a log-normal distribuition

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scienceplots

def multicell_LI(params=None, mean_k=1.0):
    Tmax = 30
    tspan = np.linspace(0, Tmax, 500)

    if params is None:
        params = defaultparams()

    P = params['P']
    Q = params['Q']
    n = P * Q

    params['connectivity'] = getconnectivityM(P, Q)
    params['k'] = np.random.lognormal(mean_k, 0.5, n)

    y0 = getIC(params, n)

    yout = odeint(li, y0, tspan, args=(params,))

    plot_final_lattice(tspan, yout, P, Q, n, mean_k)

    return yout, tspan, params


def li(y, t, params):
    nu = params['nu']
    betaD = params['betaD']
    betaR = params['betaR']
    h = params['h']
    m = params['m']
    g = params['g']
    beta0 = params['beta0']
    M = params['connectivity']
    k = params['k']
    n = len(M)

    D = y[:n]
    R = y[n:2 * n]
    Dneighbor = np.dot(M, y[:n])

    dD = nu * ((betaD * k ** h / (k ** h + R ** h)) - D)
    dR = beta0 + betaR * Dneighbor ** m / (g ** m + Dneighbor ** m) - R

    return np.concatenate((dD, dR))


def defaultparams():
    return {
        'nu': 1,
        'betaD': 10,
        'betaR': 10,
        'h': 3,
        'm': 3,
        'sigma': 0.2,
        'P': 10,
        'Q': 10,
        'g': 1,
        'beta0': 0.4
    }


def getconnectivityM(P, Q):
    n = P * Q
    M = np.zeros((n, n))
    w = 1 / 6

    for s in range(n):
        kneighbor = findneighborhex(s, P, Q)
        for r in range(6):
            M[s, kneighbor[r]] = w

    return M


def getIC(params, n):
    U = np.random.rand(n) - 0.5
    epsilon = 1e-5
    D0 = epsilon * params['betaD'] * (1 + params['sigma'] * U)
    R0 = np.zeros(n)

    return np.concatenate((D0, R0))


def plotHexagon(p0, q0, c, ax):
    s32 = np.sqrt(3) / 4
    q = q0 * 3 / 4
    p = p0 * 2 * s32

    if q0 % 2 == 0:
        p += s32

    x = [q - .5, q - .25, q + .25, q + .5, q + .25, q - .25]
    y = [p, p + s32, p + s32, p, p - s32, p - s32]

    polygon = patches.Polygon(np.c_[x, y], closed=True, edgecolor='black', facecolor=c)
    ax.add_patch(polygon)


def plot_final_lattice(tout, yout, P, Q, n, mean_k):
    fig, ax = plt.subplots()
    Cmax = np.max(yout[-1, :n])
    tind = -1  # last time point

    # Use a reversed color map 'Greens'
    cmap = plt.get_cmap('Greens').reversed()

    for i in range(1, P + 1):
        for j in range(1, Q + 1):
            ind = pq2ind(i, j, P)
            delta_level = yout[tind, ind] / Cmax  # Normalize D values to range [0, 1]
            color_cmap = cmap(0.7 * (delta_level - 1) + 1)  # Apply color mapping formula
            plotHexagon(i, j, color_cmap, ax)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(yout[tind, :n])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('D values')

    ax.axis('equal')
    ax.axis('off')
    plt.title(f'log(<k>) = {mean_k}')
    plt.show()


def findneighborhex(ind, P, Q):
    p, q = ind2pq(ind, P)

    out = [0] * 6
    out[0] = pq2ind((p % P) + 1, q, P)
    out[1] = pq2ind((p - 2) % P + 1, q, P)

    qleft = (q - 2) % Q + 1
    qright = q % Q + 1

    if q % 2 != 0:
        pup = p
        pdown = (p - 2) % P + 1
    else:
        pup = (p % P) + 1
        pdown = p

    out[2] = pq2ind(pup, qleft, P)
    out[3] = pq2ind(pdown, qleft, P)
    out[4] = pq2ind(pup, qright, P)
    out[5] = pq2ind(pdown, qright, P)

    return out


def pq2ind(p, q, P):
    return p + (q - 1) * P - 1


def ind2pq(ind, P):
    q = 1 + (ind // P)
    p = ind % P + 1
    return p, q


if __name__ == "__main__":
    low_means = np.linspace(-2, 2, 5)
    for mean_k in low_means:
        # print(f"Running simulation with mean k = {mean_k}")
        yout, tout, params = multicell_LI(mean_k=mean_k)
    # high_means = np.linspace(1, 10, 10)
    # for mean_k in high_means:
    #     # print(f"Running simulation with mean k = {mean_k}")
    #     yout, tout, params = multicell_LI(mean_k=mean_k)
