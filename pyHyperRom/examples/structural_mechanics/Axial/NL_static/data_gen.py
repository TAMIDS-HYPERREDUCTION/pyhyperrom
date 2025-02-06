# Restart the kernel
import os
import sys

desired_path = os.path.join(os.path.dirname(__file__), '..', '..','..','..') # removed the third ..
os.chdir(desired_path)
sys.path.append(desired_path)

from src.codes.basic import *
from src.codes.prob_classes.structural_mechanics.base_class_struc_mech_NL_static_axial import StructuralMechanicsSimulationData
import dill as pickle

#%%

def data_gen(params, filename_dataC):

    # Simulation parameters
    n_ref = 250  # Reference points
    num_snapshots = len(params)
    
    # Initialize and run the simulation
    simulation = StructuralMechanicsSimulationData(n_ref, params, num_snapshots=num_snapshots)
    simulation.run_simulation()

    # Save simulation data
    os.chdir(os.path.dirname(__file__))  # Ensure the current directory is set correctly for saving
    
    with open(filename_dataC, 'wb') as file:
        pickle.dump(simulation, file, recurse=True)

    
    print("Simulation data saved successfully.")
    
    return simulation