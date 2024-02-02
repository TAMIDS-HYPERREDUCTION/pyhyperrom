from src.codes.basic import *
from src.codes.algorithms.nnls_scipy import nnls as nnls_sp
from scipy.optimize import nnls

def ecsw_red_ms(d, V_sel, Le, K_mus, q_mus, n_sel, N_snap, mask, NL_solutions, NL_solutions_mean, tol=None):
    """
    Function: ecsw_red
    Overview: Perform (ECSW) reduction on the nonlinear FEM problems.
    
    Inputs:
    - d: Data object containing mesh and FEM details.
    - V_sel: Selected basis vectors.
    - Le: Element-node connectivity matrix.
    - K_mus: List of element stiffness matrices for each snapshot.
    - q_mus: List of element source terms for each snapshot.
    - n_sel: Number of selected basis vectors.
    - N_snap: Number of snapshots.
    - mask: Boolean mask for nodes without Dirichlet boundary conditions.
    - NL_solutions: Mean subtracted nonlinear solutions for all snapshots
    - tol: Tolerance for the non-negative least squares solver (optional).
    
    Outputs:
    - x: Solution to the least squares problem.
    - residual: Residual of the least squares problem.
    """
    
    # Initialize the number of cells in the mesh
    ncells = d.n_cells
    
    # Initialize the C matrix with zeros
    C = np.zeros((n_sel * N_snap, int(ncells)))

    # Apply mask to the selected basis vectors
    V_mask_ = V_sel
    
    # Compute the projection matrix P_sel
    P_sel = V_sel @ V_sel.T

    # Loop over all snapshots to populate C matrix
    for i in range(N_snap):
             
        # Project the solution onto the selected basis

        # projected_sol = np.dot(P_sel, NL_solutions[i])
        projected_sol_mask = np.dot(P_sel, NL_solutions[i]) + NL_solutions_mean

        # Mask and reshape the nonlinear solutions for the current snapshot
        # projected_sol_mask = projected_sol[mask]

        # Loop over all cells in the mesh
        for j in range(ncells):
            
            # Get the column indices for the current cell in Le
            col_indices = np.argmax(Le[j], axis=1)
            
            # Extract relevant stiffness matrices and source terms for the current snapshot and cell
            K_mus_ij = K_mus[i][j]
            q_mus_ij = np.array(q_mus[i][j])

            # Compute the entries of C matrix for the current cell and snapshot
            Ce = np.dot( np.transpose(V_mask_[col_indices]), (np.dot(K_mus_ij, projected_sol_mask[col_indices])))
            
            # Store the computed values in the C matrix
            C[i * n_sel : (i + 1) * n_sel, j] = Ce

    # Compute d_vec as C times a vector of ones
    d_vec = C @ np.ones((ncells, 1))
    norm_d_vec = np.linalg.norm(d_vec)
    print(f"norm of rhs: {norm_d_vec}")

    # Solve the non-negative least squares problem
    if tol is None:
        x, residual = nnls(C, d_vec.flatten(), maxiter=1e6)
        
    else:
        x, residual = nnls_sp(C, d_vec.flatten(), atol=tol, maxiter=1e6)

    
    return x, residual/norm_d_vec
