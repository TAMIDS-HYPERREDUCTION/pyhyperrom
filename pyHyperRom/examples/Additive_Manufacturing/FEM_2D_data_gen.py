# Restart the kernel
import os
import sys

desired_path = os.path.join(os.path.dirname(__file__), '..', '..')
os.chdir(desired_path)
sys.path.append(desired_path)

from src.codes.basic import *
from src.codes.prob_classes.base_class_heat_conduction_AM_ms import HeatConductionSimulationData
import dill as pickle

#%%
# Define the reference points and widths

Torch_dia = 1e-4
params = 0.25*np.pi*Torch_dia**2  # Torch-area
feed_rate = 1.0;
dt = 0.000005;
tf=0.01;

cell_dim = Torch_dia/3
L = [0.005*1.7, 7*Torch_dia]
n_ref = np.array([np.ceil(L[0]/cell_dim),np.ceil(L[1]/cell_dim)], dtype=int)

num_snapshots = 1
pb_dim=2

T_init = 298.0
t = np.r_[0:tf+dt:dt]

# Initialize the simulation class
simulation = HeatConductionSimulationData(n_ref, L,params, feed_rate,  dt, t, quad_deg=5, num_snapshots=num_snapshots, pb_dim=pb_dim, T_init_guess=T_init)

# Run the simulation
simulation.run_simulation()

#%%
## Save Data ##
filename_dataC = 'pyHyperRom\\examples\\Additive_Manufacturing\\data\\DataClass_AM_M.dill'   

with open(filename_dataC, 'wb') as f:
    pickle.dump(simulation, f, recurse=True)

print(f"Simulation data saved")