from src.codes.basic import *
from src.codes.algorithms.nnls_scipy import nnls as nnls_sp
from scipy.optimize import nnls

def ecsw_red(d, V_sel, Le, data, n_sel, N_snap, NL_solutions, NL_solutions_mean, residual_func, tol=None, SS=False):
    """
    Executes Enhanced Compact Subspace-Wise (ECSW) reduction for nonlinear finite element analysis. This method
    integrates mesh information, selected vectors, and nonlinear solutions to solve least squares problems efficiently.

    Parameters:
    d (object): Contains mesh and finite element model details.
    V_sel (array): Basis vectors selected for the reduction process.
    Le (matrix): Links elements to nodes in the mesh.
    data (object): General data related to the finite element model, including stiffness matrices and source terms.
    n_sel (int): The quantity of basis vectors chosen.
    N_snap (int): The total snapshots considered for analysis.
    NL_solutions (array): Adjusted nonlinear solutions for each snapshot.
    NL_solutions_mean (array): The mean of nonlinear solutions.
    residual_func (function): A custom function to compute residuals. Depends on the problem. Located in Base class.
    tol (float, optional): Specifies the tolerance level for the non-negative least squares solver. Defaults to None.

    Returns:
    tuple: Contains the solution to the least squares problem and the normalized residual of the solution.
    
    The function initializes with the mesh's cell count, sets up a zero matrix for projection and adjustment, and iterates
    through snapshots to project solutions and compute residuals. It concludes by solving a non-negative least squares
    problem to find the best fit solution and its corresponding residual, normalized by the right-hand side vector norm.
    """
    ncells = d.n_cells
    
    C = np.zeros((n_sel * N_snap, int(ncells)))

    V_mask_ = V_sel
    
    P_sel = V_sel @ V_sel.T

    for i in range(N_snap):

        if SS:

            dim = int(len(NL_solutions[0])/2)

            # Project the solution onto the selected basis
            projected_sol_mask_d = np.dot(P_sel, NL_solutions[i,:dim]-NL_solutions_mean[:dim]) + NL_solutions_mean[:dim]
            projected_sol_mask_v = np.dot(P_sel, NL_solutions[i,dim:])


            for j in range(ncells):

                col_indices = np.argmax(Le[j], axis=1)
                res = residual_func(i,j,projected_sol_mask_d[col_indices],projected_sol_mask_v[col_indices],data)
                Ce = np.dot( np.transpose(V_mask_[col_indices]), res)
                C[i * n_sel : (i + 1) * n_sel, j] = Ce


        else:
            projected_sol_mask = np.dot(P_sel, NL_solutions[i]) + NL_solutions_mean

            for j in range(ncells):

                col_indices = np.argmax(Le[j], axis=1)
                res = residual_func(i,j,projected_sol_mask[col_indices],data)
                Ce = np.dot( np.transpose(V_mask_[col_indices]), res)
                C[i * n_sel : (i + 1) * n_sel, j] = Ce


    d_vec = C @ np.ones((ncells, 1))
    norm_d_vec = np.linalg.norm(d_vec)
    print(f"norm of rhs: {norm_d_vec}")

    tic = time.time()

    if tol is None:
        x, residual = nnls(C, d_vec.flatten(), maxiter=1e6)
    else:
        x, residual = nnls_sp(C, d_vec.flatten(), atol=tol, maxiter=1e6)

    toc = time.time()

    return x, residual/norm_d_vec, C, d_vec, (toc-tic)



def ecsw_red_SS_parametric(d, V_sel, Le, data, n_sel, NL_solutions, NL_solutions_mean, residual_func, train_mask_t, tol=None):
    

    ncells = d.n_cells
    C = np.zeros((n_sel * NL_solutions.shape[0] * NL_solutions.shape[1], int(ncells)))
    V_mask_ = V_sel
    P_sel = V_sel @ V_sel.T

    
    for k in range(NL_solutions.shape[0]):
        
        for i in range(NL_solutions.shape[1]):
    
            dim = int(NL_solutions.shape[2]/2)
    
            # Project the solution onto the selected basis
            projected_sol_mask_d = np.dot(P_sel, NL_solutions[k][i,:dim]-NL_solutions_mean) + NL_solutions_mean
            projected_sol_mask_v = np.dot(P_sel, NL_solutions[k][i,dim:])


            for j in range(ncells):
    
                col_indices = np.argmax(Le[j], axis=1)
                if j==48:
                    res = residual_func(i,j,k, projected_sol_mask_d[col_indices],projected_sol_mask_v[col_indices],data, train_mask_t)
                else:
                    res = residual_func(i,j,k, projected_sol_mask_d[col_indices],projected_sol_mask_v[col_indices],data, train_mask_t)

                Ce = np.dot( np.transpose(V_mask_[col_indices]), res )
                C[(i+NL_solutions.shape[1]*k) * n_sel : (i+NL_solutions.shape[1]*k + 1) * n_sel, j] = Ce


    d_vec = C @ np.ones((ncells, 1))
    norm_d_vec = np.linalg.norm(d_vec)
    print(f"norm of rhs: {norm_d_vec}")


    if tol is None:
        x, residual = nnls(C, d_vec.flatten(), maxiter=1e6)
    else:
        x, residual = nnls_sp(C, d_vec.flatten(), atol=tol, maxiter=1e6)

    return x, residual/norm_d_vec


