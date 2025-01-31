from src.codes.basic import *
from src.codes.algorithms.empirical_cubature_method import EmpiricalCubatureMethod
from scipy.special import roots_legendre

def ECM(FOS, V_sel, Le, data, N_snap, NL_solutions, NL_solutions_mean, residual_func_ecm, tol=None, SS=False):
    
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

    d = FOS.data
    ncells = d.n_cells
    V_mask_ = np.copy(V_sel)
    P_sel = V_sel @ V_sel.T

    # def create_A_FE_reduced(basis, elements, K_each_gauss_p, F_each_gauss_p, num_elements, p_values, connectivity,num_modes):   
    num_gauss_points = len(FOS.w)
    wi = FOS.w
    # wi, _ = roots_legendre(num_gauss_points)

    # Calculate the number of independent elements in the stiffness matrix
    num_modes = V_sel.shape[-1]

    # independent_ele_per_stiffness_mat = int(num_modes * (num_modes + 1) / 2)
 
    # Initialize the A_FE matrix with appropriate dimensions
    R_FE = np.zeros((ncells * num_gauss_points, num_modes * N_snap))

    # Add zero rows to the basis at the top and bottom
    p=0
    # Loop through each parameter value (p_values) and element
    for i in range(N_snap):
        projected_sol_mask = np.dot(P_sel, NL_solutions[i]) + NL_solutions_mean

        for e in range(ncells):

            col_indices = np.argmax(Le[e], axis=1)            
            fe_ = data['fe_ecm_mus'][i][e]

            if e==ncells-1:
                stop = 1

            for j in range(len(wi)):                
                # Extract the corresponding rows from the basis using connectivity
                res = residual_func_ecm(i,e,j,projected_sol_mask[col_indices],data)
                Ce = np.dot(np.transpose(V_mask_[col_indices]), res) + V_mask_[col_indices].T@fe_/(wi[j]*len(wi))

                R_FE[j + len(wi) * e, i * num_modes:(i + 1) * num_modes] = Ce.flatten()



    U_FE, S, _ = np.linalg.svd(R_FE)

    plt.semilogy(S,'o-')
    
    if tol is not None:
        N_FE_sel = np.where(S < tol)[0][0]
    else:
        N_FE_sel = np.where(S < 1e-5)[0][0]

    W_FE = np.array([wi for _ in range(ncells)])
    W_FE=W_FE.reshape(-1,1).flatten()

    ECM = EmpiricalCubatureMethod() # Setting up Empirical Cubature Method problem
    ECM.SetUp(U_FE[:,:N_FE_sel].T, Weights = W_FE, constrain_sum_of_weights=False)
    # ECM.SetUp(U_FE.T, Weights = W_FE, constrain_sum_of_weights=False)
    ECM.Run()

    W = np.squeeze(ECM.w)
    Z = ECM.z

    return W,Z,S









    # for i in range(N_snap):

    #     if SS:

    #         dim = int(len(NL_solutions[0])/2)

    #         # Project the solution onto the selected basis
    #         projected_sol_mask_d = np.dot(P_sel, NL_solutions[i,:dim]-NL_solutions_mean[:dim]) + NL_solutions_mean[:dim]
    #         projected_sol_mask_v = np.dot(P_sel, NL_solutions[i,dim:])


    #         for j in range(ncells):

    #             col_indices = np.argmax(Le[j], axis=1)
    #             res = residual_func(i,j,projected_sol_mask_d[col_indices],projected_sol_mask_v[col_indices],data)
    #             Ce = np.dot( np.transpose(V_mask_[col_indices]), res)
    #             C[i * n_sel : (i + 1) * n_sel, j] = Ce


        # else:
        # projected_sol_mask = np.dot(P_sel, NL_solutions[i]) + NL_solutions_mean

        # for j in range(ncells):

        #     col_indices = np.argmax(Le[j], axis=1)
        #     res = residual_func(i,j,projected_sol_mask[col_indices],data)
        #     Ce = np.dot( np.transpose(V_mask_[col_indices]), res)
        #     C[i * n_sel : (i + 1) * n_sel, j] = Ce


    # d_vec = C @ np.ones((ncells, 1))
    # norm_d_vec = np.linalg.norm(d_vec)
    # print(f"norm of rhs: {norm_d_vec}")

    # if tol is None:
    #     x, residual = nnls(C, d_vec.flatten(), maxiter=1e6)
    # else:
    #     x, residual = nnls_sp(C, d_vec.flatten(), atol=tol, maxiter=1e6)


    # return x, residual/norm_d_vec



# def ecsw_red_SS_parametric(d, V_sel, Le, data, n_sel, NL_solutions, NL_solutions_mean, residual_func, train_mask_t, tol=None):
    

#     ncells = d.n_cells
#     C = np.zeros((n_sel * NL_solutions.shape[0] * NL_solutions.shape[1], int(ncells)))
#     V_mask_ = V_sel
#     P_sel = V_sel @ V_sel.T

    
#     for k in range(NL_solutions.shape[0]):
        
#         for i in range(NL_solutions.shape[1]):
    
#             dim = int(NL_solutions.shape[2]/2)
    
#             # Project the solution onto the selected basis
#             projected_sol_mask_d = np.dot(P_sel, NL_solutions[k][i,:dim]-NL_solutions_mean) + NL_solutions_mean
#             projected_sol_mask_v = np.dot(P_sel, NL_solutions[k][i,dim:])


#             for j in range(ncells):
    
#                 col_indices = np.argmax(Le[j], axis=1)
#                 if j==48:
#                     res = residual_func(i,j,k, projected_sol_mask_d[col_indices],projected_sol_mask_v[col_indices],data, train_mask_t)
#                 else:
#                     res = residual_func(i,j,k, projected_sol_mask_d[col_indices],projected_sol_mask_v[col_indices],data, train_mask_t)

#                 Ce = np.dot( np.transpose(V_mask_[col_indices]), res )
#                 C[(i+NL_solutions.shape[1]*k) * n_sel : (i+NL_solutions.shape[1]*k + 1) * n_sel, j] = Ce


#     d_vec = C @ np.ones((ncells, 1))
#     norm_d_vec = np.linalg.norm(d_vec)
#     print(f"norm of rhs: {norm_d_vec}")


#     if tol is None:
#         x, residual = nnls(C, d_vec.flatten(), maxiter=1e6)
#     else:
#         x, residual = nnls_sp(C, d_vec.flatten(), atol=tol, maxiter=1e6)

#     return x, residual/norm_d_vec


