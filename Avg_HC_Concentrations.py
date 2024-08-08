import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import scienceplots


def multicell_LI(params=None, num_runs=100):
    Tmax = 100
    tspan = np.linspace(0, Tmax, 1000)

    if params is None:
        params = defaultparams()

    P = params['P']
    Q = params['Q']
    n = P * Q

    params['connectivity'] = getconnectivityM(P, Q)

    avg_high_D_concentration = np.zeros_like(tspan)
    avg_low_D_no_high_D_neighbors = np.zeros_like(tspan)
    avg_high_D_with_high_D_neighbors = np.zeros_like(tspan)

    for i in range(num_runs):
        y0, k_values = getIC(params, n)
        params['k_values'] = k_values  # Store initial k_values in params

        k_slope = params['k_slope']  # Define the rate at which k values change over time
        params['k_slope'] = k_slope

        yout = odeint(li, y0, tspan, args=(params,))

        avg_high_D_concentration += calculate_high_D_concentration(tspan, yout, P, Q)
        avg_low_D_no_high_D_neighbors += calculate_low_D_no_high_D_neighbors(tspan, yout, P, Q)
        avg_high_D_with_high_D_neighbors += calculate_high_D_with_high_D_neighbors(tspan, yout, P, Q)

    avg_high_D_concentration /= num_runs
    avg_low_D_no_high_D_neighbors /= num_runs
    avg_high_D_with_high_D_neighbors /= num_runs

    neighbors_df = analyze_neighbors(yout[-1, :n], params, P, Q)
    print(neighbors_df)

    # Export the DataFrame to a CSV file
    neighbors_df.to_csv('neighbors_analysis.csv', index=False)
    print('DataFrame exported to neighbors_analysis.csv')

    # Plot high D cell concentration over time across the whole lattice
    plot_average_high_D_concentration(tspan, avg_high_D_concentration)

    # Plot the amount of low D cells that do NOT neighbor any high D cells as a function of time
    plot_average_low_D_no_high_D_neighbors(tspan, avg_low_D_no_high_D_neighbors)

    # Plot the amount of high D cells that neighbor at least 1 high D cell as a function of time
    plot_average_high_D_with_high_D_neighbors(tspan, avg_high_D_with_high_D_neighbors)

    return yout, tspan, params, neighbors_df


def li(y, t, params):
    nu = params['nu']
    betaD = params['betaD']
    betaR = params['betaR']
    h = params['h']
    m = params['m']
    g = params['g']
    beta0 = params['beta0']
    M = params['connectivity']
    k_values = 10**(params['k_values'] + params['k_slope'] * t)  # Advance k values linearly with time
    n = len(M)

    D = y[:n]
    R = y[n:2 * n]
    Dneighbor = np.dot(M, y[:n])

    dD = nu * ((betaD * k_values ** h / (k_values ** h + R ** h)) - D)
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
        'k_mean': -1,  # Mean of k distribution
        'k_std': 1,  # Standard deviation of k distribution
        'g': 1,
        'beta0': 0.4,
        'k_slope': 0.01  # Slope of k values change over time
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

    k_values = np.random.lognormal(params['k_mean'], params['k_std'], n)  # Generate k values

    return np.concatenate((D0, R0)), k_values


def plot2cells(tout, yout, n):
    plt.style.use(['science', 'notebook', 'grid'])
    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(8, 6))
    ax[0].plot(tout, yout[:, :n])
    ax[0].title.set_text('D Values vs Time')
    ax[0].axhline(y=np.mean(yout[:, :n]), linestyle='--', color='black')
    ax[1].plot(tout, yout[:, n:])
    ax[1].title.set_text('R Values vs Time')
    ax[1].axhline(y=np.mean(yout[:, n:]), linestyle='--', color='black')
    fig.text(0.5, 0.04, 'time [a.u]', ha='center')
    fig.text(0.04, 0.5, 'concentration [a.u]', va='center', rotation='vertical')
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


def plot_final_lattice(tout, yout, P, Q, n):
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


def analyze_neighbors(D_values, params, P, Q):
    n = P * Q
    neighbor_types = {'p': [], 'q': [], 'high_D_neighbors': [], 'low_D_neighbors': [], 'cell_type': []}
    threshold = np.mean(D_values)  # Define a threshold to classify high and low D

    for ind in range(n):
        p, q = ind2pq(ind, P)
        neighbors = findneighborhex(ind, P, Q)
        high_D_count = sum(D_values[neighbor] > threshold for neighbor in neighbors)
        low_D_count = len(neighbors) - high_D_count
        cell_type = 'high_D' if D_values[ind] > threshold else 'low_D'
        neighbor_types['p'].append(p)
        neighbor_types['q'].append(q)
        neighbor_types['high_D_neighbors'].append(high_D_count)
        neighbor_types['low_D_neighbors'].append(low_D_count)
        neighbor_types['cell_type'].append(cell_type)

    df = pd.DataFrame(neighbor_types)
    return df


def plot_k_lattice(k_values, P, Q):
    fig, ax = plt.subplots()
    Cmax = 1.1 * np.max(k_values)

    cmap = plt.get_cmap('copper_r')

    for i in range(1, P + 1):
        for j in range(1, Q + 1):
            ind = pq2ind(i, j, P)
            mycolor = min([k_values[ind] / Cmax, 1])
            color = cmap(mycolor)
            plotHexagon(i, j, color, ax)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(k_values)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('k values')

    ax.axis('equal')
    ax.axis('off')
    plt.show()


def plot_D_vs_k(D_values, k_values):
    fig, ax = plt.subplots()
    ax.scatter(k_values, D_values, s=10, c='blue', alpha=0.5)
    ax.set_xlabel('k values')
    ax.set_ylabel('D values')
    ax.set_title('D values vs. k values')
    plt.show()


def plot_neighbor_distribution(neighbors_df):
    fig, ax = plt.subplots()

    # High D cells distribution
    high_D_cells = neighbors_df[neighbors_df['cell_type'] == 'high_D']['high_D_neighbors']
    high_D_cells.hist(bins=np.arange(0, 7) - 0.5, ax=ax, alpha=0.5, label='High D cells', color='green')

    # Low D cells distribution
    low_D_cells = neighbors_df[neighbors_df['cell_type'] == 'low_D']['high_D_neighbors']
    low_D_cells.hist(bins=np.arange(0, 7) - 0.5, ax=ax, alpha=0.5, label='Low D cells', color='red')

    ax.set_xlabel('Number of High D Neighbors')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of High D Neighbors')
    ax.legend()
    plt.show()


def calculate_high_D_concentration(tout, yout, P, Q):
    n = P * Q
    D_threshold = np.mean(yout[-1, :n])  # Threshold to classify high D cells

    high_D_concentration = []

    for t in range(len(tout)):
        D_values = yout[t, :n]
        high_D_count = np.sum(D_values > D_threshold)
        high_D_concentration.append(high_D_count / n)

    return np.array(high_D_concentration)


def plot_average_high_D_concentration(tout, avg_high_D_concentration):
    plt.figure(figsize=(10, 10))
    plt.plot(tout, avg_high_D_concentration, linestyle='-', color='blue', linewidth=5)
    plt.xlabel('Time [a.u]', fontsize=30)
    plt.ylabel('Concentration of HC [a.u]', fontsize=30)
    plt.show()


def calculate_low_D_no_high_D_neighbors(tout, yout, P, Q):
    n = P * Q
    D_threshold = np.mean(yout[-1, :n])  # Threshold to classify high D cells

    low_D_no_high_D_neighbors = []

    for t in range(len(tout)):
        D_values = yout[t, :n]
        count = 0
        for ind in range(n):
            if D_values[ind] <= D_threshold:
                neighbors = findneighborhex(ind, P, Q)
                if all(D_values[neighbor] <= D_threshold for neighbor in neighbors):
                    count += 1
        low_D_no_high_D_neighbors.append(count)

    return np.array(low_D_no_high_D_neighbors)


def plot_average_low_D_no_high_D_neighbors(tout, avg_low_D_no_high_D_neighbors):
    plt.figure(figsize=(10, 10))
    plt.plot(tout, avg_low_D_no_high_D_neighbors, linestyle='-', color='red', linewidth=5)
    plt.xlabel('Time [a.u]', fontsize=30)
    plt.ylabel('Number of SC', fontsize=30)
    plt.show()


def calculate_high_D_with_high_D_neighbors(tout, yout, P, Q):
    n = P * Q
    D_threshold = np.mean(yout[-1, :n])  # Threshold to classify high D cells

    high_D_with_high_D_neighbors = []

    for t in range(len(tout)):
        D_values = yout[t, :n]
        count = 0
        for ind in range(n):
            if D_values[ind] > D_threshold:
                neighbors = findneighborhex(ind, P, Q)
                if any(D_values[neighbor] > D_threshold for neighbor in neighbors):
                    count += 1
        high_D_with_high_D_neighbors.append(count)

    return np.array(high_D_with_high_D_neighbors)


def plot_average_high_D_with_high_D_neighbors(tout, avg_high_D_with_high_D_neighbors):
    plt.figure(figsize=(10, 10))
    plt.plot(tout, avg_high_D_with_high_D_neighbors, linestyle='-', color='green', linewidth=5)
    plt.xlabel('Time [a.u]', fontsize=30)
    plt.ylabel('Number of HC', fontsize=30)
    plt.show()


if __name__ == "__main__":
    multicell_LI()
