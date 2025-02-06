# Restart the kernel
import os
import sys

desired_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
os.chdir(desired_path)
sys.path.append(desired_path)

from src.codes.basic import *
from src.codes.prob_classes.heat_conduction.base_class_heat_conduction import HeatConductionSimulationData
import dill as pickle


def shuffle_matrix_rows(matrix):
    # Create an array of indices from 0 to the number of rows in the matrix
    indices = np.arange(matrix.shape[0])
    # Shuffle these indices randomly
    np.random.shuffle(indices)
    # Use the shuffled indices to reorder the rows of the matrix
    shuffled_matrix = matrix[indices, :]
    return shuffled_matrix


#%%
# Define the reference points and widths
n_ref = np.array([40 * 10, 10 * 10], dtype=int)
w = np.array([0.4, 0.1])

nk = 10
nq = 10
num_snapshots = nk*nq
k_param = np.linspace(-0.5, 0.5, nk)
q_param = np.linspace(-50, 50, nq)

K, Q = np.meshgrid(k_param, q_param)

# Each pair of k_param and q_param from the meshgrid is a row in the matrix
params = np.column_stack((K.ravel(), Q.ravel()))
# params = shuffle_matrix_rows(params)

L = np.sum(w)
# Initialize the simulation class
simulation = HeatConductionSimulationData(n_ref, L , num_snapshots=num_snapshots, params=params)

# Run the simulation
simulation.run_simulation()


#%%
# Save Data ##

filename_dataC = 'examples//heat_conduction//oneD_heat_conduction//data//DataClass_UQ_NL_new.dill'   

with open(filename_dataC, 'wb') as f:
    pickle.dump(simulation, f, recurse=True)

print(f"Simulation data saved")