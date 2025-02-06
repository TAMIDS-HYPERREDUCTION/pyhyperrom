from src.codes.basic import *
from src.codes.algorithms.empirical_cubature_method import EmpiricalCubatureMethod

def ECM(FOS, V_sel, Le, data, N_snap, NL_solutions, NL_solutions_mean, residual_func_ecm, tol=None, SS=False):
    """
    Executes Enhanced Compact Subspace-Wise (ECSW) reduction for nonlinear finite element analysis.
    This function integrates mesh information, selected basis vectors, and nonlinear solution snapshots 
    to set up and solve a least squares problem via an Empirical Cubature Method (ECM) approach.

    Parameters:
    -----------
    FOS: object
        Finite element model output structure containing mesh data and Gauss weights.
    V_sel: ndarray
        Selected basis vectors used for the projection/reduction process.
    Le: ndarray or matrix
        Connectivity matrix linking elements to nodes in the finite element mesh.
    data: dict or object
        Contains additional FEM data such as stiffness matrices or force vectors.
    N_snap: int
        The number of snapshots used in the analysis.
    NL_solutions: ndarray
        Array of nonlinear solutions for each snapshot.
    NL_solutions_mean: ndarray
        Mean value of the nonlinear solutions, used for adjustment.
    residual_func_ecm: function
        A user-defined function to compute the residual for a given element, Gauss point, and snapshot.
    tol: float, optional
        Tolerance for selecting the number of modes based on singular values (default is None).
    SS: bool, optional
        An optional flag parameter (usage not detailed in the code, default is False).

    Returns:
    --------
    tuple
        A tuple containing:
          - W: ndarray, the weights from the ECM process.
          - Z: ndarray, the selected indices (cubature points) from the ECM.
          - S: ndarray, the singular values from the SVD of the projection matrix.
    """
    # Retrieve FEM model details from the FOS object.
    d = FOS.data

    # Obtain the number of cells/elements in the mesh.
    ncells = d.n_cells

    # Create a copy of the selected basis vectors to avoid modifying the original.
    V_mask_ = np.copy(V_sel)

    # Compute the projection operator from the selected basis vectors.
    P_sel = V_sel @ V_sel.T

    # Determine the number of Gauss points from the length of the weight vector.
    num_gauss_points = len(FOS.w)

    # Retrieve the Gauss weights vector.
    wi = FOS.w

    # Get the number of modes (columns) in the selected basis.
    num_modes = V_sel.shape[-1]

    # Initialize the finite element projection matrix R_FE.
    # Its rows correspond to (number of cells * number of Gauss points)
    # and its columns correspond to (number of modes * number of snapshots).
    R_FE = np.zeros((ncells * num_gauss_points, num_modes * N_snap))

    # A variable 'p' is initialized (currently unused in the computation).
    p = 0

    # Loop over each snapshot index.
    for i in range(N_snap):
        # For each snapshot, project the nonlinear solution using the projection operator and add the mean.
        projected_sol_mask = np.dot(P_sel, NL_solutions[i]) + NL_solutions_mean

        # Loop over each cell/element in the mesh.
        for e in range(ncells):
            # Determine the relevant indices for the current element using its connectivity.
            col_indices = np.argmax(Le[e], axis=1)
            
            # Extract the force vector for the current snapshot and element.
            fe_ = data['fe_ecm_mus'][i][e]

            # This conditional appears to serve as a breakpoint or debug marker for the last cell.
            if e == ncells - 1:
                stop = 1

            # Loop over each Gauss point.
            for j in range(len(wi)):
                # Compute the residual for the given snapshot, element, and Gauss point.
                # The residual function uses the projected solution at the element nodes (indexed by col_indices) and additional data.
                res = residual_func_ecm(i, e, j, projected_sol_mask[col_indices], data)
                
                # Compute the contribution Ce by projecting the residual onto the basis at the selected indices,
                # and adding a force term scaled by the Gauss weight and the number of Gauss points.
                Ce = np.dot(np.transpose(V_mask_[col_indices]), res) + V_mask_[col_indices].T @ fe_ / (wi[j] * len(wi))
                
                # Insert the computed Ce vector into the appropriate block of the R_FE matrix.
                # The row index is determined by the current Gauss point and element.
                # The column block is determined by the current snapshot and the number of modes.
                R_FE[j + len(wi) * e, i * num_modes:(i + 1) * num_modes] = Ce.flatten()

    # Perform Singular Value Decomposition (SVD) on the assembled projection matrix R_FE.
    U_FE, S, _ = np.linalg.svd(R_FE)

    # Plot the singular values on a semilogarithmic scale to visualize their decay.
    plt.semilogy(S, 'o-')
    
    # Determine the number of finite element modes to retain based on the provided tolerance.
    # If a tolerance is specified, select the first singular value index below that tolerance;
    # otherwise, use a default tolerance value of 1e-5.
    if tol is not None:
        N_FE_sel = np.where(S < tol)[0][0]
    else:
        N_FE_sel = np.where(S < 1e-5)[0][0]

    # Build the weights vector for the finite elements by repeating the Gauss weights for each cell.
    W_FE = np.array([wi for _ in range(ncells)])
    # Reshape the weights array into a one-dimensional vector.
    W_FE = W_FE.reshape(-1, 1).flatten()

    # Instantiate the Empirical Cubature Method (ECM) object.
    ECM = EmpiricalCubatureMethod()

    # Set up the ECM problem using the reduced basis (from the SVD) corresponding to the selected number of modes.
    # The reduced basis is transposed and the Gauss weights are provided.
    # The option 'constrain_sum_of_weights' is set to False.
    ECM.SetUp(U_FE[:, :N_FE_sel].T, Weights=W_FE, constrain_sum_of_weights=False)

    # Execute the ECM algorithm to compute cubature weights and selected indices.
    ECM.Run()

    # Retrieve the computed weights (squeezed to remove extra dimensions) and indices (cubature points).
    W = np.squeeze(ECM.w)
    Z = ECM.z

    # Return the ECM weights, the selected indices, and the singular values from the SVD.
    return W, Z, S
