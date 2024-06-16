# Copyright (C) 2014, David Sprinzak
# This program is part of Lateral Inhibition Tutorial.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

def multicell_LI(params=None):
    """
    multicell_LI simulates lateral inhibition in a hexagonal lattice.
    The structure params contains the model parameters of the system.
    TOUT is a vector containing the time points of the solution
    between 0 and Tmax. YOUT is a matrix containing the numerical
    solution for each variable for each time point. Each row in
    YOUT is a vector of the size of TOUT. F is a movie of the simulation.
    """
    Tmax = 30  # set time for simulation
    tspan = [0, Tmax]

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
    r = ode(li).set_integrator('vode', method='bdf')
    r.set_initial_value(y0, tspan[0])
    r.set_f_params(params)

    tout = []
    yout = []
    while r.successful() and r.t < tspan[1]:
        r.integrate(tspan[1])
        tout.append(r.t)
        yout.append(r.y)

    tout = np.array(tout)
    yout = np.array(yout)

    # show time traces of two cells with lateral inhibition
    plot2cells(tout, yout, k)

    # show lattice simulation
    F = movielattice(tout, yout, P, Q, k)

    return yout, tout, params, F

def li(t, y, params):
    """
    Differential equations for Delta and repressor levels.
    """
    nu = params['nu']
    betaD = params['betaD']
    betaR = params['betaR']
    h = params['h']
    m = params['m']
    f = params['f']
    g = params['g']
    M = params['connectivity']
    k = len(M)

    D = y[:k]  # levels of Delta in cells 1 to k
    R = y[k:]  # levels of Repressor in cells 1 to k
    Dneighbor = M @ y[:k]  # average Delta level in the neighboring cells

    dD = nu * (betaD * f**h / (f**h + R**h) - D)
    dR = betaR * Dneighbor**m / (g**m + Dneighbor**m) - R
    dy = np.concatenate((dD, dR))

    return dy

def defaultparams():
    """
    Default parameter values for the simulation.
    """
    params = {
        'nu': 1,  # ratio of degradation rates
        'betaD': 50,  # normalized Delta production
        'betaR': 50,  # normalized repressor production
        'h': 3,  # Hill coefficient repression function
        'm': 3,  # Hill coefficient activating function
        'sigma': 0.2,  # noise amplitude in initial conditions
        'P': 10,  # number of cells per column
        'Q': 10,  # number of columns - MUST BE EVEN
        'f': 1,
        'g': 1
    }
    return params

def getconnectivityM(P, Q):
    """
    Calculate the connectivity matrix for the hexagonal lattice.
    """
    k = P * Q  # number of cells
    M = np.zeros((k, k))  # connectivity matrix
    w = 1 / 6  # weight for interactions

    for s in range(k):
        kneighbor = findneighborhex(s, P, Q)
        for r in kneighbor:
            M[s, r-1] = w

    return M

def getIC(params, k):
    """
    Generate initial conditions for Delta and repressor levels.
    """
    U = np.random.rand(k) - 0.5  # a uniform random distribution
    epsilon = 1e-5  # multiplicative factor of Delta initial condition
    D0 = epsilon * params['betaD'] * (1 + params['sigma'] * U)  # initial Delta levels
    R0 = np.zeros(k)  # initial repressor levels
    y0 = np.concatenate((D0, R0))  # vector of initial conditions

    return y0

def plot2cells(tout, yout, k):
    """
    Plot the time traces of Delta and repressor levels for two cells.
    """
    plt.figure(21)
    plt.clf()
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.plot(tout, yout[:, i], '-r', linewidth=2)  # plot Delta levels
        plt.plot(tout, yout[:, k + i], '-b', linewidth=2)  # plot repressor levels
        plt.title(f'cell #{i + 1}')
        plt.xlabel('t [a.u]')
        plt.ylabel('concentration [a.u]')
        plt.legend(['d', 'r'])

def findneighborhex(ind, P, Q):
    """
    Find the 6 neighbors of a cell in the hexagonal lattice.
    """
    p, q = ind2pq(ind, P)

    # above and below
    out = [pq2ind(p % P + 1, q, P),
           pq2ind((p-2) % P + 1, q, P)]

    # left and right sides
    qleft = (q-2) % Q + 1
    qright = q % Q + 1

    if q // 2 != q / 2:
        pup = p
        pdown = (p-2) % P + 1
    else:
        pup = p % P + 1
        pdown = p

    out.extend([pq2ind(pup, qleft, P),
                pq2ind(pdown, qleft, P),
                pq2ind(pup, qright, P),
                pq2ind(pdown, qright, P)])

    return out

def pq2ind(p, q, P):
    """
    Convert (p, q) coordinates to a linear index.
    """
    return p + (q - 1) * P

def ind2pq(ind, P):
    """
    Convert a linear index to (p, q) coordinates.
    """
    q = 1 + (ind - 1) // P
    p = ind - (q - 1) * P
    return p, q

def plotHexagon(p0, q0, c):
    """
    Plot a hexagon centered at coordinates (p, q) with color c.
    """
    s32 = np.sqrt(3) / 4
    q = q0 * 3 / 4
    p = p0 * 2 * s32
    if q0 // 2 == q0 / 2:
        p = p + s32

    x = [q - 0.5, q - 0.25, q + 0.25, q + 0.5, q + 0.25, q - 0.25]
    y = [p, p + s32, p + s32, p, p - s32, p - s32]

    plt.fill(x, y, color=c, linewidth=2)

def movielattice(tout, yout, P, Q, k):
    """
    Generate a movie of patterning in the hexagonal lattice.
    The color represents the level of Delta.
    """
    Cmax = np.max(yout[:, :k])  # find max(Delta) at the end point
    frames = []

    for tind in range(0, len(tout), 5):  # show every 5th frame
        plt.figure(22)
        plt.clf()
        for i in range(P):
            for j in range(Q):
                ind = pq2ind(i + 1, j + 1, P)
                mycolor = min([yout[tind, ind] / Cmax, 1])
                plotHexagon(i + 1, j + 1, [1 - mycolor, 1 - mycolor, 1])

        plt.axis('image')
        plt.axis('off')
        plt.box(False)

        frames.append(plt.gcf().canvas.copy_from_bbox(plt.gcf().bbox))

    return frames

# Example usage
yout, tout, params, F = multicell_LI()
plt.figure()
P = params['P']
Q = params['Q']
k = P * Q
Cmax = np.max(yout[:, :k])  # find max(Delta) at the end point

for i in range(P):
    for j in range(Q):
        ind = pq2ind(i + 1, j + 1, P)
        mycolor = min([yout[-1, ind] / Cmax, 1])  # Use the last time point
        plotHexagon(i + 1, j + 1, [1 - mycolor, 1 - mycolor, 1])

plt.axis('image')
plt.axis('off')
plt.box(False)
plt.title('Final Delta Pattern')
plt.show()
