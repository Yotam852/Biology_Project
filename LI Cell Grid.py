"""
LI Cell Grid Simulation

This module simulates classic Lateral Inhibition on a hexagonal cell lattice. It includes functions to define the differential equations, generate initial conditions, create the connectivity matrix, and plot the results.

Modules:
    numpy: Used for numerical operations.
    scipy.integrate: Used for integrating the differential equations.
    matplotlib.pyplot: Used for plotting the results.
    matplotlib.patches: Used for creating hexagonal patches in the plots.
    scienceplots: Used for scientific plotting styles.

Functions:
    multicell_LI(params=None): Simulates the Lateral Inhibition on a hexagonal cell lattice.
    li(y, t, params): Defines the differential equations for the Lateral Inhibition model.
    defaultparams(): Returns the default parameters for the Lateral Inhibition simulation.
    getconnectivityM(P, Q): Generates the connectivity matrix for the hexagonal grid.
    getIC(params, n): Generates the initial conditions for the simulation.
    plot2cells(tout, yout, n): Plots the D and R values for the first two cells over time.
    findneighborhex(ind, P, Q): Finds the indices of the neighboring hexagons for a given cell.
    pq2ind(p, q, P): Converts (p, q) coordinates to a linear index.
    ind2pq(ind, P): Converts a linear index to (p, q) coordinates.
    plotHexagon(p0, q0, c, ax): Plots a single hexagon on the given axes.
    plot_final_lattice(tout, yout, P, Q, n): Plots the final lattice of hexagons with color mapping based on the normalized D values.

Usage:
    Run this module as a script to execute the multicell_LI function with default parameters and plot the results.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scienceplots


def multicell_LI(params=None):
    """
    Simulates the Lateral Inhibition on a hexagonal cell lattice.

    Parameters:
    params (dict, optional): Dictionary of parameters for the simulation. If None, default parameters are used.

    Returns:
    tuple: yout (array), tout (array), params (dict)
    """
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

    plot_final_lattice(tspan, yout, P, Q, n)

    return yout, tspan, params


def li(y, t, params):
    """
    Defines the differential equations for the Lateral Inhibition model.

    Parameters:
    y (array): Array of D and R values.
    t (float): Time variable.
    params (dict): Dictionary of parameters for the simulation.

    Returns:
    array: Concatenated array of derivatives dD and dR.
    """
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
    """
    Returns the default parameters for the Lateral Inhibition simulation.

    Returns:
    dict: Dictionary of default parameters.
    """
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
    """
    Generates the connectivity matrix for the hexagonal grid.

    Parameters:
    P (int): Number of rows in the lattice.
    Q (int): Number of columns in the lattice.

    Returns:
    array: Connectivity matrix.
    """
    n = P * Q
    M = np.zeros((n, n))
    w = 1 / 6

    for s in range(n):
        kneighbor = findneighborhex(s, P, Q)
        for r in range(6):
            M[s, kneighbor[r]] = w

    return M


def getIC(params, n):
    """
    Generates the initial conditions for the simulation.

    Parameters:
    params (dict): Dictionary of parameters for the simulation.
    n (int): Number of cells.

    Returns:
    array: Initial conditions array.
    """
    U = np.random.rand(n) - 0.5
    epsilon = 1e-5
    D0 = epsilon * params['betaD'] * (1 + params['sigma'] * U)
    R0 = np.zeros(n)

    return np.concatenate((D0, R0))


def plot2cells(tout, yout, n):
    """
    Plots the D and R values for the first two cells over time.

    Parameters:
    tout (array): Array of time points.
    yout (array): Array of D and R values for each cell at each time point.
    n (int): Number of cells.

    Returns:
    None
    """
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
    """
    Finds the indices of the neighboring hexagons for a given cell.

    Parameters:
    ind (int): Index of the cell.
    P (int): Number of rows in the lattice.
    Q (int): Number of columns in the lattice.

    Returns:
    list: List of indices of the neighboring cells.
    """
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
    """
    Converts (p, q) coordinates to a linear index.

    Parameters:
    p (int): Row index.
    q (int): Column index.
    P (int): Number of rows in the lattice.

    Returns:
    int: Linear index.
    """
    return p + (q - 1) * P - 1


def ind2pq(ind, P):
    """
    Converts a linear index to (p, q) coordinates.

    Parameters:
    ind (int): Linear index.
    P (int): Number of rows in the lattice.

    Returns:
    tuple: (p, q) coordinates.
    """
    q = 1 + (ind // P)
    p = ind % P + 1
    return p, q


def plotHexagon(p0, q0, c, ax):
    """
    Plots a single hexagon on the given axes.

    Parameters:
    p0 (int): Row index of the hexagon.
    q0 (int): Column index of the hexagon.
    c (str): Color of the hexagon.
    ax (matplotlib.axes.Axes): Axes on which to plot the hexagon.

    Returns:
    None
    """
    s32 = np.sqrt(3) / 4
    q = q0 * 3 / 4
    p = p0 * 2 * s32

    if q0 % 2 == 0:
        p += s32

    x = [q - .5, q - .25, q + .25, q + .5, q + .25, q - .25]
    y = [p, p + s32, p + s32, p, p - s32, p - s32]

    polygon = patches.Polygon(np.c_[x, y], closed=True, edgecolor='black', facecolor=c)
    ax.add_patch(polygon)


def plot_final_lattice(tout, yout, P, Q, n):
    """
    Plots the final lattice of hexagons with color mapping based on the normalized D values.

    Parameters:
    tout (array): Array of time points.
    yout (array): Array of D values for each cell at each time point.
    P (int): Number of rows in the lattice.
    Q (int): Number of columns in the lattice.
    n (int): Number of cells.

    Returns:
    None
    """
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
    plt.show()


if __name__ == "__main__":
    yout, tout, params = multicell_LI()