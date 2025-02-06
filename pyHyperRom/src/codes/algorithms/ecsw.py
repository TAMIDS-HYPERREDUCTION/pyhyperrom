from src.codes.basic import *  # Import basic functionalities and definitions for the finite element model
from src.codes.algorithms.nnls_scipy import nnls as nnls_sp  # Import a custom non-negative least squares solver (aliased as nnls_sp)
from scipy.optimize import nnls  # Import the standard non-negative least squares solver from SciPy

def ecsw_red(d, V_sel, Le, data, n_sel, N_snap, NL_solutions, NL_solutions_mean, residual_func, tol=None, SS=False):
    """
    Executes Energy Conserving Sampling and Weighing (ECSW) hyper-reduction for nonlinear finite element analysis. This method
    integrates mesh information, selected vectors, and nonlinear solutions to solve least squares problems efficiently.

    Parameters:
    -----------
    d (object): Contains mesh and finite element model details.
    V_sel (array): Basis vectors selected for the reduction process.
    Le (matrix): Links elements to nodes in the mesh.
    data (object): General data related to the finite element model, including stiffness matrices and source terms.
    n_sel (int): The quantity of basis vectors chosen.
    N_snap (int): The total snapshots considered for analysis.
    NL_solutions (array): Adjusted nonlinear solutions for each snapshot.
    NL_solutions_mean (array): The mean of nonlinear solutions.
    residual_func (function): A custom function to compute residuals. Depends on the problem.
    tol (float, optional): Specifies the tolerance level for the non-negative least squares solver. Defaults to None.
    SS (bool, optional): Flag to indicate if a split strategy for the solution components should be used. Defaults to False.

    Returns:
    --------
    tuple: Contains the solution to the least squares problem and the normalized residual of the solution.
    """
    # Retrieve the number of cells (elements) from the finite element model.
    ncells = d.n_cells

    # Initialize matrix C to store contributions from each snapshot.
    # Dimensions: (n_sel * N_snap) rows x (ncells) columns.
    C = np.zeros((n_sel * N_snap, int(ncells)))

    # Create a local copy of the selected basis vectors.
    V_mask_ = V_sel

    # Compute the projection matrix from the selected basis (V_sel).
    P_sel = V_sel @ V_sel.T

    # Loop over each snapshot.
    for i in range(N_snap):
        if SS:
            # When SS is True, split the solution into two parts.
            dim = int(len(NL_solutions[0]) / 2)
            # Project the first half (e.g., displacement) and adjust with the mean.
            projected_sol_mask_d = np.dot(P_sel, NL_solutions[i, :dim] - NL_solutions_mean[:dim]) + NL_solutions_mean[:dim]
            # Project the second half (e.g., velocity or another component).
            projected_sol_mask_v = np.dot(P_sel, NL_solutions[i, dim:])

            # Process each cell individually.
            for j in range(ncells):
                # Determine the indices corresponding to the current cell using the connectivity matrix Le.
                col_indices = np.argmax(Le[j], axis=1)
                # Compute the residual for the current snapshot and cell using the split solution components.
                res = residual_func(i, j, projected_sol_mask_d[col_indices], projected_sol_mask_v[col_indices], data)
                # Project the residual onto the basis vectors for the current cell.
                Ce = np.dot(np.transpose(V_mask_[col_indices]), res)
                # Store the computed contribution in the appropriate block of C.
                C[i * n_sel: (i + 1) * n_sel, j] = Ce

        else:
            # When SS is False, use the full solution for projection.
            projected_sol_mask = np.dot(P_sel, NL_solutions[i]) + NL_solutions_mean

            # Process each cell individually.
            for j in range(ncells):
                # Determine the indices corresponding to the current cell.
                col_indices = np.argmax(Le[j], axis=1)
                # Compute the residual for the current snapshot and cell.
                res = residual_func(i, j, projected_sol_mask[col_indices], data)
                # Project the residual onto the basis vectors for the current cell.
                Ce = np.dot(np.transpose(V_mask_[col_indices]), res)
                # Store the computed contribution in matrix C.
                C[i * n_sel: (i + 1) * n_sel, j] = Ce

    # Compute the right-hand side vector (d_vec) by multiplying C with a vector of ones.
    d_vec = C @ np.ones((ncells, 1))
    # Compute the norm of d_vec for normalization of the residual.
    norm_d_vec = np.linalg.norm(d_vec)
    print(f"norm of rhs: {norm_d_vec}")

    # Solve the non-negative least squares problem.
    # Use the standard solver if no tolerance is provided; otherwise, use the custom solver.
    if tol is None:
        x, residual = nnls(C, d_vec.flatten(), maxiter=1e6)
    else:
        x, residual = nnls_sp(C, d_vec.flatten(), atol=tol, maxiter=1e6)

    # Return the NNLS solution vector and the normalized residual.
    return x, residual / norm_d_vec

def ecsw_red_SS_parametric(d, V_sel, Le, data, n_sel, NL_solutions, NL_solutions_mean, residual_func, train_mask_t, tol=None):
    """
    Executes a parametric variant of ECSW reduction for time-dependent FEA.

    Parameters:
    -----------
    d (object): Contains mesh and finite element model details.
    V_sel (array): Basis vectors selected for the reduction process.
    Le (matrix): Links elements to nodes in the mesh.
    data (object): General data related to the finite element model, including stiffness matrices and source terms.
    n_sel (int): The number of basis vectors selected.
    NL_solutions (array): A 3D array containing nonlinear solutions for different parameter sets.
    NL_solutions_mean (array): The mean nonlinear solution used for adjustment.
    residual_func (function): A custom function to compute residuals.
    train_mask_t: Training mask used by the residual function.
    tol (float, optional): Tolerance for the non-negative least squares solver. Defaults to None.

    Returns:
    --------
    tuple: Contains the solution to the least squares problem and the normalized residual.
    """
    # Retrieve the number of cells (elements) from the finite element model.
    ncells = d.n_cells

    # Compute the total number of snapshots across all parameter sets.
    total_snapshots = NL_solutions.shape[0] * NL_solutions.shape[1]
    # Initialize matrix C to store contributions from all snapshots.
    # Dimensions: (n_sel * total_snapshots) x (ncells).
    C = np.zeros((n_sel * total_snapshots, int(ncells)))

    # Create a local copy of the selected basis vectors.
    V_mask_ = V_sel

    # Compute the projection matrix from the selected basis.
    P_sel = V_sel @ V_sel.T

    # Loop over each parameter set (first dimension of NL_solutions).
    for k in range(NL_solutions.shape[0]):
        # Loop over each snapshot within the current parameter set.
        for i in range(NL_solutions.shape[1]):
            # Determine the dimension of the first half of the solution.
            dim = int(NL_solutions.shape[2] / 2)
            # Project the first half (e.g., displacement) and adjust with the mean.
            projected_sol_mask_d = np.dot(P_sel, NL_solutions[k][i, :dim] - NL_solutions_mean) + NL_solutions_mean
            # Project the second half (e.g., velocity or another component) of the solution.
            projected_sol_mask_v = np.dot(P_sel, NL_solutions[k][i, dim:])

            # Process each cell individually.
            for j in range(ncells):
                # Determine the indices corresponding to the current cell from the connectivity matrix Le.
                col_indices = np.argmax(Le[j], axis=1)
                # Compute the residual using the provided function.
                # A conditional branch exists for cell index 48, though both branches perform the same function call.
                if j == 48:
                    res = residual_func(i, j, k, projected_sol_mask_d[col_indices], projected_sol_mask_v[col_indices], data, train_mask_t)
                else:
                    res = residual_func(i, j, k, projected_sol_mask_d[col_indices], projected_sol_mask_v[col_indices], data, train_mask_t)
                # Project the residual onto the basis vectors for the current cell.
                Ce = np.dot(np.transpose(V_mask_[col_indices]), res)
                # Compute the appropriate row indices for inserting Ce into matrix C.
                C[(i + NL_solutions.shape[1] * k) * n_sel: (i + NL_solutions.shape[1] * k + 1) * n_sel, j] = Ce

    # Compute the right-hand side vector by multiplying C with a vector of ones.
    d_vec = C @ np.ones((ncells, 1))
    # Calculate the norm of d_vec for later normalization.
    norm_d_vec = np.linalg.norm(d_vec)
    print(f"norm of rhs: {norm_d_vec}")

    # Solve the non-negative least squares problem.
    if tol is None:
        x, residual = nnls(C, d_vec.flatten(), maxiter=1e6)
    else:
        x, residual = nnls_sp(C, d_vec.flatten(), atol=tol, maxiter=1e6)

    # Return the NNLS solution and the normalized residual.
    return x, residual / norm_d_vec
