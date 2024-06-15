import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection

# Define the parameters
alpha = 10.0  # Activation rate of Delta
beta = 1.0  # Degradation rate of Delta
gamma = 10.0  # Activation rate of repressor
delta = 1.0  # Degradation rate of repressor
k = 2.0  # Hill coefficient


# Function to calculate the average Delta concentration of neighbors
def average_neighbors(delta, neighbors):
    return np.mean([delta[n] for n in neighbors])


# Define the lateral inhibition model on a hexagonal grid
def lateral_inhibition(y, t, P, Q, neighbors):
    D = y[:P * Q]
    R = y[P * Q:]

    dDdt = np.zeros(P * Q)
    dRdt = np.zeros(P * Q)

    for i in range(P * Q):
        avg_D_neighbors = average_neighbors(D, neighbors[i])
        dDdt[i] = alpha / (1 + R[i] ** k) - beta * D[i]
        dRdt[i] = gamma * avg_D_neighbors / (1 + avg_D_neighbors ** k) - delta * R[i]

    return np.concatenate([dDdt, dRdt])


# Create a hexagonal grid and define neighbors
def create_hexagonal_grid(P, Q):
    neighbors = []
    for p in range(P):
        for q in range(Q):
            index = p * Q + q
            n = []
            # North
            if p > 0:
                n.append(index - Q)
            # South
            if p < P - 1:
                n.append(index + Q)
            # West
            if q > 0:
                n.append(index - 1)
            # East
            if q < Q - 1:
                n.append(index + 1)
            # North-East
            if p > 0 and q < Q - 1:
                n.append(index - Q + 1)
            # South-West
            if p < P - 1 and q > 0:
                n.append(index + Q - 1)
            # South-East
            if p < P - 1 and q < Q - 1:
                n.append(index + Q + 1)
            # North-West
            if p > 0 and q > 0:
                n.append(index - Q - 1)
            neighbors.append(n)
    return neighbors


# Initial conditions and time span
P, Q = 10, 10
neighbors = create_hexagonal_grid(P, Q)

# Small random perturbations around a base level
base_level_D = 0.5
base_level_R = 0.5
perturbation_strength = 0.01

initial_D = base_level_D + perturbation_strength * np.random.randn(P * Q)
initial_R = base_level_R + perturbation_strength * np.random.randn(P * Q)
y0 = np.concatenate([initial_D, initial_R])

t = np.linspace(0, 500, 5000)  # Increased simulation time

# Solve the differential equations
solution = odeint(lateral_inhibition, y0, t, args=(P, Q, neighbors))

# Extract the final Delta concentrations
D_final = solution[-1, :P * Q]

# Create the hexagonal plot
fig, ax = plt.subplots(1, figsize=(10, 10))


# Function to convert grid coordinates to hexagonal coordinates
def hex_coords(P, Q):
    coords = []
    for p in range(P):
        for q in range(Q):
            x = q * 1.5
            y = p * np.sqrt(3) + (q % 2) * (np.sqrt(3) / 2)
            coords.append((x, y))
    return coords


hex_centers = hex_coords(P, Q)

# Create hexagons and assign colors based on Delta concentration
hexagons = []
colors = []

for i, (x, y) in enumerate(hex_centers):
    hexagon = RegularPolygon((x, y), numVertices=6, radius=0.9, orientation=np.radians(30), ec='k')
    hexagons.append(hexagon)
    colors.append(D_final[i])

collection = PatchCollection(hexagons, array=np.array(colors), cmap='viridis', edgecolor='k')
ax.add_collection(collection)
ax.set_xlim(-1, 1.5 * Q)
ax.set_ylim(-1, P * np.sqrt(3))
ax.set_aspect('equal')
plt.colorbar(collection, ax=ax, label='Delta Concentration')
plt.title('Hexagonal Lattice with Delta Concentration')
plt.show()
