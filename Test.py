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
    k = P * Q

    params['connectivity'] = getconnectivityM(P, Q)

    y0 = getIC(params, k)

    yout = odeint(li, y0, tspan, args=(params,))

    return yout, tspan, params


def li(y, t, params):
    nu = params['nu']
    betaD = params['betaD']
    betaR = params['betaR']
    h = params['h']
    m = params['m']
    f = params['f']
    g = params['g']
    M = params['connectivity']
    beta0 = params['beta0']
    k = len(M)

    D = y[:k]
    R = y[k:2 * k]
    Dneighbor = np.dot(M, y[:k])

    dD = nu * (beta0 + (betaD * f ** h / (f ** h + R ** h)) - D)
    dR = betaR * Dneighbor ** m / (g ** m + Dneighbor ** m) - R

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
        'f': 100,
        'g': 1,
        'beta0': 0.1
    }


def getconnectivityM(P, Q):
    k = P * Q
    M = np.zeros((k, k))
    w = 1 / 6

    for s in range(k):
        kneighbor = findneighborhex(s, P, Q)
        for r in range(6):
            M[s, kneighbor[r]] = w

    return M


def getIC(params, k):
    U = np.random.rand(k) - 0.5
    epsilon = 1e-5
    D0 = epsilon * params['betaD'] * (1 + params['sigma'] * U)
    R0 = np.zeros(k)

    return np.concatenate((D0, R0))


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


def plot_final_lattice(tout, yout, P, Q, k):
    fig, ax = plt.subplots()
    Cmax = np.max(yout[-1, :k])
    tind = -1  # last time point
    for i in range(1, P + 1):
        for j in range(1, Q + 1):
            ind = pq2ind(i, j, P)
            mycolor = min([yout[tind, ind] / Cmax, 1])
            plotHexagon(i, j, [1 - mycolor, 1 - mycolor, 1], ax)
    ax.axis('equal')
    ax.axis('off')
    plt.show()


def run_simulations():
    f_values = np.logspace(np.log(0.01), np.log(10), 50)
    D_ratios = []
    pattern_start = None
    pattern_end = None

    for f in f_values:
        params = defaultparams()
        params['f'] = f
        yout, tout, params = multicell_LI(params)

        D = yout[-1, :params['P'] * params['Q']]
        D_max = np.max(D)
        D_min = np.min(D)

        ratio = D_max / D_min
        D_ratios.append(ratio)

        if ratio > 2:
            if pattern_start is None:
                pattern_start = f
            pattern_end = f

    plt.figure()
    plt.style.use(['science', 'notebook', 'grid'])
    plt.semilogx(f_values, D_ratios, '-o')
    plt.xlabel('f [a.u]')
    plt.ylabel('$D_{max} / D_{min}$ [a.u]')
    plt.title('$D_{max} / D_{min}$ as a function of f')
    plt.show()

    print(f'Patterning starts at f = {pattern_start}')
    print(f'Patterning ends at f = {pattern_end}')


if __name__ == "__main__":
    run_simulations()
