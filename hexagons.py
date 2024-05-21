import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation


def multicell_LI(params=None):
    # set time for simulation
    Tmax = 30
    tspan = np.linspace(0, Tmax, 300)

    # get the default parameters if none provided
    if params is None:
        params = defaultparams()

    P = params['P']  # number of cells per column
    Q = params['Q']  # number of columns - MUST BE EVEN
    k = P * Q  # number of cells

    # get the connectivity matrix
    params['connectivity'] = getconnectivityM(P, Q)

    # setting the initial conditions + noise
    y0 = getIC(params, k)

    # run simulation with lateral inhibition
    yout = odeint(li, y0, tspan, args=(params,))

    # show time traces of two cells with lateral inhibition
    plot2cells(tspan, yout, k)

    # show lattice simulation
    F = movielattice(tspan, yout, P, Q, k)

    return yout, tspan, params, F


def li(y, t, params):
    nu = params['nu']
    betaD = params['betaD']
    betaR = params['betaR']
    h = params['h']
    m = params['m']
    M = params['connectivity']
    k = len(M)

    D = y[:k]  # levels of Delta in cells 1 to k
    R = y[k:]  # levels of Repressor in cells 1 to k
    Dneighbor = M @ D  # average Delta level in the neighboring cells

    # differential equations for Delta and repressor levels
    dD = nu * (betaD / (1 + R ** h) - D)
    dR = betaR * Dneighbor ** m / (1 + Dneighbor ** m) - R
    dy = np.concatenate([dD, dR])

    return dy


def defaultparams():
    return {
        'nu': 1,  # ratio of degradation rates
        'betaD': 50,  # normalized Delta production
        'betaR': 50,  # normalized repressor production
        'h': 3,  # Hill coefficient repression function
        'm': 3,  # Hill coefficient activating function
        'sigma': 0.2,  # noise amplitude in initial conditions
        'P': 18,  # number of cells per column
        'Q': 18,  # number of columns - MUST BE EVEN
    }


def getconnectivityM(P, Q):
    k = P * Q  # number of cells
    M = np.zeros((k, k))  # connectivity matrix
    w = 1 / 6  # weight for interactions

    # calculating the connectivity matrix
    for s in range(k):
        kneighbor = findneighborhex(s, P, Q)
        for r in range(6):
            if kneighbor[r] is not None:  # Ensure neighbor is within bounds
                M[s, kneighbor[r]] = w
    return M


def getIC(params, k):
    U = np.random.rand(k) - 0.5  # a uniform random distribution
    epsilon = 1e-5  # multiplicative factor of Delta initial condition
    D0 = epsilon * params['betaD'] * (1 + params['sigma'] * U)  # initial Delta levels
    R0 = np.zeros(k)  # initial repressor levels
    y0 = np.concatenate([D0, R0])  # vector of initial conditions
    return y0


def plot2cells(tout, yout, k):
    plt.figure(figsize=(12, 5))
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.plot(tout, yout[:, i], '-r', linewidth=2)  # plot Delta levels
        plt.plot(tout, yout[:, k + i], '-b', linewidth=2)  # plot repressor levels
        plt.title(f'cell #{i + 1}')
        plt.xlabel('t [a.u]')
        plt.ylabel('concentration [a.u]')
        plt.legend(['d', 'r'])
    plt.show()


def findneighborhex(ind, P, Q):
    p, q = ind2pq(ind, P)

    # above and below:
    out = []
    out.append(pq2ind((p % P) + 1, q, P, Q))
    out.append(pq2ind((p - 2) % P + 1, q, P, Q))

    # left and right sides:
    qleft = (q - 2) % Q + 1
    qright = q % Q + 1

    if q % 2 != 0:
        pup = p
        pdown = (p - 2) % P + 1
    else:
        pup = (p % P) + 1
        pdown = p
    out.append(pq2ind(pup, qleft, P, Q))
    out.append(pq2ind(pdown, qleft, P, Q))
    out.append(pq2ind(pup, qright, P, Q))
    out.append(pq2ind(pdown, qright, P, Q))

    # Ensure neighbors are within bounds
    out = [neighbor if neighbor < P * Q and neighbor >= 0 else None for neighbor in out]
    return out


def pq2ind(p, q, P, Q):
    return (p - 1) + (q - 1) * P


def ind2pq(ind, P):
    q = 1 + ind // P
    p = ind % P + 1
    return p, q


def plotHexagon(ax, p0, q0, c):
    s32 = np.sqrt(3) / 4
    q = q0 * 3 / 4
    p = p0 * 2 * s32
    if q0 % 2 == 0:
        p = p + s32

    x = [q - .5, q - .25, q + .25, q + .5, q + .25, q - .25]
    y = [p, p + s32, p + s32, p, p - s32, p - s32]

    polygon = patches.Polygon(list(zip(x, y)), closed=True, color=c, linewidth=2)
    ax.add_patch(polygon)


def movielattice(tout, yout, P, Q, k):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    Cmax = np.max(yout[-1, :k])  # finds max(Delta) at the end point

    def update(tind):
        ax.clear()
        for i in range(P):
            for j in range(Q):
                ind = pq2ind(i + 1, j + 1, P, Q)
                if ind < k:
                    mycolor = min(yout[tind, ind] / Cmax, 1)
                    plotHexagon(ax, i + 1, j + 1, [1 - mycolor, 1 - mycolor, 1])
        ax.axis('off')
        ax.set_xlim(-1, Q * 0.75)
        ax.set_ylim(-1, P * np.sqrt(3) / 2)
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=range(0, len(tout), 5), blit=False)
    plt.show()
    return ani


# Run the simulation with default parameters
if __name__ == '__main__':
    yout, tout, params, F = multicell_LI()