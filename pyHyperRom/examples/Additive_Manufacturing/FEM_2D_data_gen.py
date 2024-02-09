# Import necessary libraries
import os
import sys
import dill as pickle

# Path configuration for module access
desired_path = os.path.join(os.path.dirname(__file__), '..', '..')
os.chdir(desired_path)
sys.path.append(desired_path)

# Import custom modules for simulation
from src.codes.basic import *
from src.codes.prob_classes.base_class_heat_conduction_AM import HeatConductionSimulationData


# Simulation parameters configuration
Torch_dia = 1e-4  # Diameter of the torch
params = 0.25 * np.pi * Torch_dia ** 2  # Area of the torch
feed_rate = 1.0  # Feed rate for the simulation
dt = 0.00001  # Time step for the simulation
tf = 0.1  # Final time for the simulation
cell_dim = Torch_dia / 3  # Cell dimension for the grid
L = [0.005 * 1.7, 7 * Torch_dia]  # Dimensions of the simulation domain
n_ref = np.array([np.ceil(L[0] / cell_dim), np.ceil(L[1] / cell_dim)], dtype=int)  # Reference points for the grid
num_snapshots = 1  # Number of snapshots to take during the simulation
pb_dim = 2  # Problem dimension
T_init = 298.0  # Initial temperature
t = np.arange(0, tf + dt, dt)  # Time array for the simulation


# Initialize the simulation with the specified parameters
simulation = HeatConductionSimulationData(n_ref, L, params, feed_rate, dt, t, quad_deg=5, num_snapshots=num_snapshots, pb_dim=pb_dim, T_init_guess=T_init)

# Execute the simulation
simulation.run_simulation()

# Data saving section
filename_dataC = 'pyHyperRom\\examples\\Additive_Manufacturing\\data\\DataClass_AM_M_2.dill'
with open(filename_dataC, 'wb') as file:
    pickle.dump(simulation, file, recurse=True)

print("Simulation data saved successfully.")