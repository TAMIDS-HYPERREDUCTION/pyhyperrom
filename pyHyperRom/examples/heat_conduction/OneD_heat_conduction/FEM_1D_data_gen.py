# Restart the kernel
import os
import sys

desired_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
os.chdir(desired_path)
sys.path.append(desired_path)

from src.codes.basic import *
from src.codes.prob_classes.base_class_heat_conduction import HeatConductionSimulationData
import dill as pickle


#%%
# Define the reference points and widths
n_ref = np.array([40 * 10, 10 * 10], dtype=int)
w = np.array([0.4, 0.1])
params = np.arange(1., 4.0, 0.01)
num_snapshots = 100
L = np.sum(w)
# Initialize the simulation class
simulation = HeatConductionSimulationData(n_ref, L , num_snapshots=num_snapshots, params=params)

# Run the simulation
simulation.run_simulation()


#%%
# Save Data ##

filename_dataC = 'examples\\heat_conduction\\oneD_heat_conduction\\data\\DataClass.dill'   

with open(filename_dataC, 'wb') as f:
    pickle.dump(simulation, f, recurse=True)

print(f"Simulation data saved")