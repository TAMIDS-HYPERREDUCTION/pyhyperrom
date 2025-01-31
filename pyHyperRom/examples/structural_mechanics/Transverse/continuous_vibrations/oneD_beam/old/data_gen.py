# Import necessary libraries
import os
import sys
import numpy as np
import dill as pickle

# Environment setup: Adjust the working directory and Python path
desired_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..','..')
os.chdir(desired_path)  # Change the current working directory
print(os.getcwd())

sys.path.append(desired_path)  # Add the desired path to the Python search path
print(f"Changed working directory to: {desired_path}")

# Import simulation class from custom module
from src.codes.prob_classes.structural_mechanics.base_class_struc_mech_continuous_vibration import StructuralDynamicsSimulationData

#%%
# Simulation parameters
n_ref = 100  # Reference points

L = 1.0  # Length parameter
ep = 0.02  # Some parameter (epsilon)
T = 2 / (2 * np.pi)  # Period of the vibration
params = np.linspace(T / 25, T / 2, 50)  # Parameter range
# params=[params[25]]

cv = 1  # Damping coefficient
cm = 1e-3  # Mass coefficient
dt = T / 200  # Time step
t = np.r_[0:25 * T + dt:dt]  # Time array

num_snapshots = len(params)


# Initialize and run the simulation
simulation = StructuralDynamicsSimulationData(n_ref, L, T, params, dt, t, ep=ep, num_snapshots=num_snapshots, cv=cv, cm=cm)
simulation.run_simulation()

# Save simulation data
os.chdir(os.path.dirname(__file__))  # Ensure the current directory is set correctly for saving
filename_dataC = 'data\\DataClass_structMech_1p.dill'

with open(filename_dataC, 'wb') as file:
    pickle.dump(simulation, file, recurse=True)

print("Simulation data saved successfully.")
