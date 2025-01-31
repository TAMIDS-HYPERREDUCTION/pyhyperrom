# Restart the kernel
import os
import sys

desired_path = os.path.join(os.path.dirname(__file__), '..', '..','..') # removed the third ..
os.chdir(desired_path)
sys.path.append(desired_path)

from src.codes.basic import *
from src.codes.prob_classes.heat_conduction.base_class_heat_conduction import HeatConductionSimulationData
import dill as pickle


def data_gen(params, filename_dataC):

    #%%
    # Define the reference points and widths

    n_ref = np.array([40 * 20, 10 * 20], dtype=int)
    w = np.array([0.4, 0.1])
    # params = np.arange(1., 4.0, 0.01)
    num_snapshots = len(params)
    L = np.sum(w)

    # Initialize the simulation class
    simulation = HeatConductionSimulationData(n_ref, L , num_snapshots=num_snapshots, params=params)

    # Run the simulation
    simulation.run_simulation()

    #%%

    # Save simulation data
    os.chdir(os.path.dirname(__file__))  # Ensure the current directory is set correctly for saving
    
    with open(filename_dataC, 'wb') as f:
        pickle.dump(simulation, f, recurse=True)

    print(f"Simulation data saved")

    return simulation