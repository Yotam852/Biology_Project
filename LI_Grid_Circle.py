import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scienceplots

def multicell_LI(params=None):
    Tmax = 30
    tspan = np.linspace(0, Tmax, 500)

    if params is None:
        params = defaultparams()

    P = params['P']
    Q = params['Q']
    n = P * Q

    params['connectivity'] = getconnectivityM(P, Q)

    y0 = getIC(params, n)

    yout = odeint(li, y0, tspan, args=(params,))

    plot2cells(tspan, yout, n)

    plot_final_lattice(tspan, yout, P, Q, n, radius=3)  # Specify the desired radius here

    return yout, tspan, params

def li(y, t, params):
    nu = params['nu']
    betaD = params['betaD']
    betaR = params['betaR']
    h = params['h']
    m = params['m']
    k = params['k']
    g = params['g']
    beta0 = params['beta0']
    M = params['connectivity']
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
        'k': 1,
        'g': 1,
        'beta0': 0.1
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

def plot2cells(tout, yout, n):
    plt.figure()
    plt.style.use(['science', 'notebook', 'grid'])
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.plot(tout, yout[:, i], '-r', linewidth=2)
        plt.plot(tout, yout[:, n + i], '-b', linewidth=2)
        plt.title(f'cell #{i + 1}')
        plt.xlabel('t [a.u]')
        plt.ylabel('concentration [a.u]')
        plt.legend(['D', 'R'])
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

def plot_final_lattice(tout, yout, P, Q, n, radius):
    fig, ax = plt.subplots()
    Cmax = np.max(yout[-1, :n])
    tind = -1  # last time point

    # Use a color map (e.g., 'viridis') to map D values to colors
    cmap = plt.get_cmap('viridis')

    for i in range(1, P + 1):
        for j in range(1, Q + 1):
            ind = pq2ind(i, j, P)
            mycolor = min([yout[tind, ind] / Cmax, 1])
            color = cmap(mycolor)  # Get color from the colormap
            plotHexagon(i, j, color, ax)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(yout[tind, :n])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('D values')

    # Calculate the center of the hexagon
    s32 = np.sqrt(3) / 4
    q_center = 5 * 3 / 4
    p_center = 5 * 2 * s32
    if 5 % 2 == 0:
        p_center += s32

    # Plot the circle
    circle = plt.Circle((q_center, p_center), radius, color='red', fill=False, linestyle='--')
    ax.add_patch(circle)

    ax.axis('equal')
    ax.axis('off')
    plt.show()

if __name__ == "__main__":
    yout, tout, params = multicell_LI()
