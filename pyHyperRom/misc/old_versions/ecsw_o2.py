from ..basic import *

def ecsw_red(d, V_sel, Le, K_mus, q_mus, P_sel, tol, n_sel, N_snap, mask, NL_solutions):
    """
    Function: ecsw_red
    Overview: Performs (ECSW) hyperreduction on nonlinear FEM problems.
    
    Inputs:
    - d: Mesh data object
    - V_sel: Selected basis vectors
    - Le: Local element matrices
    - K_mus: List of stiffness matrices for each snapshot
    - q_mus: List of source term vectors for each snapshot
    - P_sel: Projection matrix
    - tol: Tolerance for Fast Non-negative Least Squares
    - n_sel: Number of selected basis vectors
    - N_snap: Number of snapshots
    - mask: Mask for nodes without Dirichlet boundary conditions
    - NL_solutions: Nonlinear solutions for each snapshot
    
    Outputs:
    - x: Solution vector obtained from Fast Non-negative Least Squares
    - residual: Residual of the least squares problem
    
    """
    
    # Initialize the number of cells in the mesh
    ncells = d.n_cells
    
    # Initialize the C matrix with zeros
    C = np.zeros((n_sel * N_snap, ncells))

    # Apply mask to the selected basis vectors
    V_mask_ = V_sel[mask, :]

    # Loop over all snapshots to populate C matrix
    for i in range(N_snap):
        
        # Mask and reshape the nonlinear solutions for the current snapshot
        NL_solutions_i_mask = NL_solutions[i][mask].reshape(-1, 1)
        
        # Project the masked solution onto the selected basis
        projected_sol = P_sel @ NL_solutions_i_mask

        # Loop over all cells in the mesh
        for j in range(ncells):
            
            # Get the column indices for the current cell in Le
            col_indices = np.argmax(Le[j], axis=1)
            
            # Extract relevant stiffness matrices and source terms for the current snapshot and cell
            K_mus_ij = K_mus[i][j]
            q_mus_ij = np.array(q_mus[i][j]).reshape(-1, 1)

            # Compute the entries of C matrix for the current cell and snapshot
            Ce = np.transpose(V_mask_[col_indices]) @ (K_mus_ij @ projected_sol[col_indices] - q_mus_ij)
            
            # Store the computed values in the C matrix
            C[i * n_sel : (i + 1) * n_sel, j] = Ce.flatten()
    
    # Compute d_vec as C times a vector of ones
    d_vec = C @ np.ones((ncells, 1))

    # Solve for x using Fast Non-negative Least Squares
    x = fe.fnnls(C, d_vec.flatten(), tolerance=tol)
    
    # Compute the residual for the least squares problem
    residual = np.linalg.norm(d_vec.flatten() - np.dot(C, x)) / np.linalg.norm(d_vec.flatten())
    
    return x, residual
    