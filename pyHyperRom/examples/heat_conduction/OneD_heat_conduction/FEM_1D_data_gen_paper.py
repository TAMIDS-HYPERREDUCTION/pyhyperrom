import os
import sys

# Define the desired directory path by moving up three levels relative to this file.
desired_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
# Change the current working directory to the desired path.
os.chdir(desired_path)
# Append the desired path to the system path so that modules from this directory can be imported.
sys.path.append(desired_path)

from src.codes.basic import *  # Import basic functionalities from the project source code.
import dill as pickle         # Import dill (a pickle alternative) for serializing objects.

def data_gen(params, filename_dataC, affine=False, train_mask=None, test_mask=None, ecm=False):
    """
    Generate simulation data for a heat conduction problem and save the simulation object to a file.
    
    Parameters:
        params (array-like): Array of parameter values for the simulation.
        filename_dataC (str): File path where the simulation data will be saved.
        affine (bool): Flag to determine whether to use the affine version of the simulation class.
        train_mask (optional): Training mask for selecting specific simulation data.
        test_mask (optional): Testing mask for selecting specific simulation data.
        ecm (bool): Flag indicating whether the Empirical Cubature Method (ECM) is applied.
    
    Returns:
        simulation (object): The simulation object containing generated simulation data.
    """
    
    # Choose the appropriate simulation class based on the 'affine' flag.
    if not affine:
        from src.codes.prob_classes.heat_conduction.base_class_heat_conduction import HeatConductionSimulationData as simulate
    else:
        from src.codes.prob_classes.heat_conduction.base_class_heat_conduction import HeatConductionSimulationData_affine as simulate

    # Define reference dimensions: first dimension is 40*100 and second dimension is 10*100.
    n_ref = np.array([4*1000, 1*1000], dtype=int)
    # Define widths or scaling factors for the simulation.
    w = np.array([0.4, 0.1])
    # Determine the number of snapshots from the length of the 'params' array.
    num_snapshots = len(params)
    # Compute the total length L as the sum of the widths.
    L = np.sum(w)

    # Initialize the simulation object with reference dimensions, total length, number of snapshots, parameters,
    # and optional training/test masks and ECM flag.
    if ecm:
        simulation = simulate(n_ref, L, num_snapshots=num_snapshots, params=params, train_mask=train_mask, test_mask=test_mask, ecm=ecm)
    else:
        simulation = simulate(n_ref, L, num_snapshots=num_snapshots, params=params, train_mask=train_mask, test_mask=test_mask)

    # Run the simulation to generate the simulation data.
    simulation.run_simulation()

    # Ensure the directory for the output file exists; create it if it does not.
    os.makedirs(os.path.dirname(filename_dataC), exist_ok=True)
    # Print the current working directory for verification.
    print(os.getcwd())
    # Open the specified file in binary write mode and serialize the simulation object using dill.
    with open(filename_dataC, 'wb') as f:
        pickle.dump(simulation, f, recurse=True)

    # Inform the user that the simulation data has been saved successfully.
    print(f"Simulation data saved")

    # Return the simulation object.
    return simulation
