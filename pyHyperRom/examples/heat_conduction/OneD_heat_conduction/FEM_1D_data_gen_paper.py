# Restart the kernel
import os
import sys

desired_path = os.path.join(os.path.dirname(__file__), '..', '..','..') # removed the third ..
os.chdir(desired_path)
sys.path.append(desired_path)

from src.codes.basic import *
import dill as pickle


def data_gen(params, filename_dataC, affine=False, train_mask=None, test_mask=None, ecm=False):

    #%%
    # Define the reference points and widths

    if not affine:
        from src.codes.prob_classes.heat_conduction.base_class_heat_conduction import HeatConductionSimulationData as simulate
    else:
        from src.codes.prob_classes.heat_conduction.base_class_heat_conduction import HeatConductionSimulationData_affine as simulate

    n_ref = np.array([40 * 100, 10 * 100], dtype=int)
    w = np.array([0.4, 0.1])
    # params = np.arange(1., 4.0, 0.01)
    num_snapshots = len(params)
    L = np.sum(w)

    # Initialize the simulation class
    simulation = simulate(n_ref, L , num_snapshots=num_snapshots, params=params,train_mask=train_mask,test_mask=test_mask,ecm=ecm)

    # Run the simulation
    simulation.run_simulation()

    #%%

    # Save simulation data
    # os.chdir(os.path.dirname(__file__))  # Ensure the current directory is set correctly for saving
    os.makedirs(os.path.dirname(filename_dataC), exist_ok=True)
    print(os.getcwd())
    with open(filename_dataC, 'wb') as f:
        pickle.dump(simulation, f, recurse=True)

    print(f"Simulation data saved")

    return simulation