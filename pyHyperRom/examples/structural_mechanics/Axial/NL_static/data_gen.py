# Import necessary libraries
import os
import sys
import numpy as np
import dill as pickle

# Environment setup: Adjust the working directory and Python path
desired_path = os.path.join(os.path.dirname(__file__), '..', '..', '..','..')
os.chdir(desired_path)  # Change the current working directory
sys.path.append(desired_path)  # Add the desired path to the Python search path
print(f"Changed working directory to: {desired_path}")

# Import simulation class from custom module
from src.codes.prob_classes.structural_mechanics.base_class_struc_mech_NL_static_axial import StructuralMechanicsSimulationData

# Simulation parameters
n_ref = 250  # Reference points
params = np.linspace(0.1,2,100)  # Parameter range
num_snapshots = len(params)


nk = 10
nq = 10
num_snapshots = nk*nq
q_param = np.linspace(-500, 500, nk)
k_param = np.linspace(-50, 50, nq)

K, Q = np.meshgrid(k_param, q_param)

# Each pair of k_param and q_param from the meshgrid is a row in the matrix
params = np.column_stack((K.ravel(), Q.ravel()))
# params = shuffle_matrix_rows(params)



# Initialize and run the simulation
simulation = StructuralMechanicsSimulationData(n_ref, params, num_snapshots=num_snapshots)
simulation.run_simulation()

# Save simulation data
os.chdir(os.path.dirname(__file__))  # Ensure the current directory is set correctly for saving
filename_dataC = 'data\\DataClass_structMech_axial.dill'

with open(filename_dataC, 'wb') as file:
    pickle.dump(simulation, file, recurse=True)

print("Simulation data saved successfully.")