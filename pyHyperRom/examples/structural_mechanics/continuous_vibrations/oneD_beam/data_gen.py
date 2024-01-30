# Restart the kernel
import os
import sys
import numpy as np

desired_path = os.path.join(os.path.dirname(__file__), '..', '..','..','..')
os.chdir(desired_path)
sys.path.append(desired_path)
print(desired_path)

# from src.codes.Utils.basic import *
from src.codes.prob_classes.base_class_struc_mech_continuous_vibration import StructuralDynamicsSimulationData
import dill as pickle

#%%
# Define the reference points and widths
n_ref = 100
num_snapshots = 2
L = 1.0
ep = 0.02
T=2/(2*np.pi)
params = np.arange(T/20, T/10, T/20)
cv = 1
cm = 1e-5

dt = T/200;
t = np.r_[0:25*T+dt:dt]

# Initialize the simulation class
simulation = StructuralDynamicsSimulationData(n_ref, L, T, params, dt, t, ep=ep, num_snapshots=num_snapshots, cv=cv, cm=cm)

# Run the simulation
simulation.run_simulation()

# %%
# Save Data ##
os.chdir(os.path.join(os.path.dirname(__file__)))
filename_dataC = 'data\\DataClass_structMech.dill' 


with open(filename_dataC, 'wb') as f:
    pickle.dump(simulation, f, recurse=True)

print(f"Simulation data saved")