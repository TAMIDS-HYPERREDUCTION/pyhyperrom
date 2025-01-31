from src.codes.prob_classes.heat_conduction.base_class_heat_conduction import FOS_FEM
from src.codes.utils.fem_utils_HC import *
from src.codes.utils.rom_utils import *
from src.codes.basic import *
from scipy.optimize import fsolve

def weighted_matrix_assembly(rom_cls, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):

    K_r, J_r = [np.zeros((V.shape[1],V.shape[1])) for _ in range(2)]
    rhs_r, K_r_mean = [np.zeros(V.shape[1]) for _ in range(2)]
    
    sol_prev = np.zeros(len(rom_cls.data.mask))


    if rom_cls.mean is not None:
        # Compute the full solution field from the reduced solution field
        T_mean = rom_cls.mean.flatten()
        sol_prev[rom_cls.data.mask] = np.dot(V,T_red) + T_mean

    else:

        sol_prev[rom_cls.data.mask] = np.dot(V,T_red)


    sol_prev[~rom_cls.data.mask] = rom_cls.data.T_dir


    # Loop through all elements in the domain
    for iel in range(rom_cls.data.n_cells):

        col_indices = np.argmax(rom_cls.data.Le[iel], axis=1)
        
        # Skip elements with zero importance
        if (xi[iel] == 0):
            continue

        # Retrieve global node numbers and equation IDs for the current element
        elem_glob_nodes = rom_cls.data.gn[iel, :]
        elem_glob_node_eqnId = rom_cls.data.glob_node_eqnId[iel]
        elem_glob_node_nonzero_eqnId = rom_cls.data.glob_node_nonzero_eqnId[iel]
        elem_local_node_nonzero_eqnId = rom_cls.data.local_node_nonzero_eqnId[iel]

        # Compute local element matrices and vectors
        Ke_, Je_, qe_ = compute_element_matrices(rom_cls, sol_prev.flatten(), iel)


        # Retrieve global and local indices for stiffness matrices
        # I_index, J_index = rom_cls.data.global_indices[iel][0], rom_cls.data.global_indices[iel][1]
        i_index, j_index = rom_cls.data.local_indices[iel][0], rom_cls.data.local_indices[iel][1]


        # Update global stiffness and Jacobian matrices with weighted local matrices
        K_r += V[col_indices].T @ (Ke_[i_index, j_index] * xi[iel]) @ V[col_indices]
        J_r += V[col_indices].T @ (Je_[i_index, j_index] * xi[iel]) @ V[col_indices]

        # Effect of the mean
        if rom_cls.mean is not None:
            K_r_mean += V[col_indices].T @ (Ke_[i_index, j_index] * xi[iel]) @ T_mean[col_indices]
        # else:
        #     K_r_mean = np.zeros((V[col_indices].T).shape[0])
            

        # Handle Dirichlet boundary conditions
        if np.isin(0, elem_glob_node_eqnId):
            elem_dof_values = dirichlet_bc(rom_cls, sol_dir, dir_nodes, elem_glob_nodes)
            fe = Ke_ @ elem_dof_values.reshape(-1, 1)
        else:
            fe = np.zeros((len(elem_glob_nodes), 1))

        # Compute local right-hand side vector
        rhs_e_ = qe_[elem_local_node_nonzero_eqnId] - fe[elem_local_node_nonzero_eqnId].flatten()

        # Update global right-hand side vector with weighted local vector
        rhs_r += V[col_indices].T @ (xi[iel] * rhs_e_)


    return K_r, J_r, K_r_mean, rhs_r

def weighted_matrix_assembly_affine(rom_cls, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):

    n_mat = int(np.max(rom_cls.data.cell2mat_layout)+1)
    n_src = int(np.max(rom_cls.data.cell2src_layout)+1)


    K_r = [np.zeros((V.shape[1],V.shape[1])) for _ in range(n_mat)]
    rhs_qe_ = [np.zeros(V.shape[1]) for _ in range(n_src)]
    rhs_fe_ = [np.zeros(V.shape[1]) for _ in range(n_src)]
    K_r_mean = [np.zeros(V.shape[1]) for _ in range(n_src)]
    
    if rom_cls.mean is not None:

        T_mean = rom_cls.mean.flatten()

    else:
        T_mean = 0.0

    # Loop through all elements in the domain
    for iel in range(rom_cls.data.n_cells):
        
        # Skip elements with zero importance
        if (xi[iel] == 0):
            continue

        cell_idx = tuple(e_n_2ij(rom_cls,iel))

        imat = rom_cls.data.cell2mat_layout[cell_idx].astype(int)
        isrc = rom_cls.data.cell2src_layout[cell_idx].astype(int)

        col_indices = np.argmax(rom_cls.data.Le[iel], axis=1)

        # Retrieve global node numbers and equation IDs for the current element
        elem_glob_nodes = rom_cls.data.gn[iel, :]
        elem_glob_node_eqnId = rom_cls.data.glob_node_eqnId[iel]
        elem_glob_node_nonzero_eqnId = rom_cls.data.glob_node_nonzero_eqnId[iel]
        elem_local_node_nonzero_eqnId = rom_cls.data.local_node_nonzero_eqnId[iel]
        sol_prev = np.zeros(V.shape[0]+len(rom_cls.data.T_dir))

        # Compute local element matrices and vectors
        Ke_, _, qe_ = compute_element_matrices(rom_cls, sol_prev, iel, affine=True)

        # Retrieve global and local indices for stiffness matrices
        i_index, j_index = rom_cls.data.local_indices[iel][0], rom_cls.data.local_indices[iel][1]

        # Update global stiffness and Jacobian matrices with weighted local matrices
        K_r[imat] += V[col_indices].T @ (Ke_[i_index, j_index] * xi[iel]) @ V[col_indices]

        # Effect of the mean
        if rom_cls.mean is not None:
            K_r_mean[imat] += V[col_indices].T @ (Ke_[i_index, j_index] * xi[iel]) @ T_mean[col_indices]
            

        # Handle Dirichlet boundary conditions
        if np.isin(0, elem_glob_node_eqnId):
            elem_dof_values = dirichlet_bc(rom_cls, sol_dir, dir_nodes, elem_glob_nodes)
            fe = Ke_ @ elem_dof_values.reshape(-1, 1)
        else:
            fe = np.zeros((len(elem_glob_nodes), 1))

        # Compute local right-hand side vector
        rhs_qe_[isrc] += V[col_indices].T @ (xi[iel] *qe_[elem_local_node_nonzero_eqnId])
        rhs_fe_[imat] += V[col_indices].T @ (xi[iel] *fe[elem_local_node_nonzero_eqnId].flatten())

        # Update global right-hand side vector with weighted local vector
        # rhs_qe_[isrc] += V[col_indices].T @ (xi[iel] * rhs_e_)

    return K_r, K_r_mean, rhs_qe_, rhs_fe_

def e_n_2ij(self, iel, el=True):

    # you can use node number in place of iel, and write el=False.
    dim_ = self.dim_
    indices = []
    divisor = 1
    for d in range(dim_):  # iterate from last dimension to first
        size = self.ncells[d]  # get size of current dimension
        if el==False:
            size = self.npts[d]
        idx = (iel//divisor)%size
        divisor *= size
        indices.append(idx)

    return indices

# def weighted_matrix_assembly_deim(rom_cls, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):

#     K, J, rhs = init_global_systems(max(node_eqnId))

#     sol_prev = np.zeros(len(rom_cls.data.mask))
#     # Compute the full solution field from the reduced solution field

#     T_mean = rom_cls.mean.flatten()

#     if rom_cls.mean is not None:
#         sol_prev[rom_cls.data.mask] = np.dot(V,T_red) + T_mean
#     else:
#         sol_prev[rom_cls.data.mask] = np.dot(V,T_red)

#     sol_prev[~rom_cls.data.mask] = rom_cls.data.T_dir

#     # Loop through all elements in the domain
#     for iel in range(rom_cls.data.n_cells):

#         if (xi[iel] != 0):

#             # Retrieve global node numbers and equation IDs for the current element
#             elem_glob_nodes = rom_cls.data.gn[iel, :]
#             elem_glob_node_eqnId = rom_cls.data.glob_node_eqnId[iel]
#             elem_glob_node_nonzero_eqnId = rom_cls.data.glob_node_nonzero_eqnId[iel]
#             elem_local_node_nonzero_eqnId = rom_cls.data.local_node_nonzero_eqnId[iel]

#             I_index, J_index = rom_cls.data.global_indices[iel][0], rom_cls.data.global_indices[iel][1]
#             i_index, j_index = rom_cls.data.local_indices[iel][0], rom_cls.data.local_indices[iel][1]

#             # if (xi[iel] != 0):
#             # Compute local element matrices and vectors
#             Ke_, Je_, qe_ = compute_element_matrices(rom_cls, sol_prev.flatten(), iel)

#             # Update global stiffness and Jacobian matrices with weighted local matrices
#             K[I_index, J_index] += Ke_[i_index, j_index]
#             J[I_index, J_index] += Je_[i_index, j_index]

#             # Handle Dirichlet boundary conditions
#             if np.isin(0, elem_glob_node_eqnId):
#                 elem_dof_values = dirichlet_bc(rom_cls, sol_dir, dir_nodes, elem_glob_nodes)
#                 fe = Ke_ @ elem_dof_values.reshape(-1, 1)
#             else:
#                 fe = np.zeros((len(elem_glob_nodes), 1))

#             # Compute local right-hand side vector
#             rhs_e_ = qe_[elem_local_node_nonzero_eqnId] - fe[elem_local_node_nonzero_eqnId].flatten()
#             rhs[elem_glob_node_nonzero_eqnId-1] += rhs_e_

#     return K, J, sol_prev, rhs

def weighted_matrix_assembly_deim(rom_cls, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):

    K, J, rhs = init_global_systems(max(node_eqnId))

    sol_prev = np.zeros(len(rom_cls.data.mask))
    # Compute the full solution field from the reduced solution field

    T_mean = rom_cls.mean.flatten()

    if rom_cls.mean is not None:
        sol_prev[rom_cls.data.mask] = np.dot(V,T_red) + T_mean
    else:
        sol_prev[rom_cls.data.mask] = np.dot(V,T_red)

    sol_prev[~rom_cls.data.mask] = rom_cls.data.T_dir

    # Loop through all elements in the domain
    for iel in range(rom_cls.data.n_cells):

        # Retrieve global node numbers and equation IDs for the current element
        elem_glob_nodes = rom_cls.data.gn[iel, :]
        elem_glob_node_eqnId = rom_cls.data.glob_node_eqnId[iel]
        elem_glob_node_nonzero_eqnId = rom_cls.data.glob_node_nonzero_eqnId[iel]
        elem_local_node_nonzero_eqnId = rom_cls.data.local_node_nonzero_eqnId[iel]


        if (xi[iel] != 0):
            # Compute local element matrices and vectors
            # Ke_, Je_, qe_ = compute_element_matrices(rom_cls, sol_prev.flatten(), iel)
            Ke_, Je_, _ = compute_element_matrices(rom_cls, sol_prev.flatten(), iel)


            # Update global stiffness and Jacobian matrices with weighted local matrices
            I_index, J_index = rom_cls.data.global_indices[iel][0], rom_cls.data.global_indices[iel][1]
            i_index, j_index = rom_cls.data.local_indices[iel][0], rom_cls.data.local_indices[iel][1]

            K[I_index, J_index] += Ke_[i_index, j_index]
            J[I_index, J_index] += Je_[i_index, j_index]

        _,_, qe_ = compute_element_matrices(rom_cls, sol_prev.flatten(), iel)

        # Handle Dirichlet boundary conditions
        if np.isin(0, elem_glob_node_eqnId):
            elem_dof_values = dirichlet_bc(rom_cls, sol_dir, dir_nodes, elem_glob_nodes)
            fe = Ke_ @ elem_dof_values.reshape(-1, 1)
        else:
            fe = np.zeros((len(elem_glob_nodes), 1))

        # Compute local right-hand side vector
        rhs_e_ = qe_[elem_local_node_nonzero_eqnId] - fe[elem_local_node_nonzero_eqnId].flatten()
        rhs[elem_glob_node_nonzero_eqnId-1] += rhs_e_

    return K, J, sol_prev, rhs


def solve_reduced(rom_cls, T_init, xi, V, tol=1e-5, max_iter=300,op=False):
    """
    Function: solve_reduced
    Overview: Solve the nonlinear system for the reduced-order model using Newton-Raphson iteration.

    Inputs:
    - cls: Refers to the FOS class instance containing mesh and finite element data.
    - T_init: Initial guess for the reduced temperature field.
    - xi: Element-wise importance weights.
    - V: Projection matrix.
    - tol: Tolerance for convergence (default is 1e-5).
    - max_iter: Maximum number of iterations for the solver (default is 300).

    Outputs:
    - Returns the updated temperature field after convergence or reaching max iterations.
    """

    # Get node equation IDs and create a mask for nodes with non-zero equation IDs
    node_eqnId = rom_cls.node_eqnId
    mask = node_eqnId != 0

    # Copy the initial temperature field
    T = np.copy(T_init)

    # Evaluate the reduced Jacobian and residual matrices for the initial guess
    Jac, res = rom_cls.eval_resJac_rom(mask, rom_cls.dir_nodes, rom_cls.sol_dir, T, node_eqnId, xi, V)

    # Compute and display the initial residual norm
    norm_ = np.linalg.norm(res)

    if op:
        print('initial residual =', norm_, "\n")

    # Initialize the iteration counter
    it = 0

    # Start the Newton-Raphson iteration loop
    while (it < max_iter) and (norm_ >= tol):
        
        # Solve the linear system to get the temperature increment (delta)
        delta = np.linalg.solve(Jac, -res)

        # Update the temperature field
        T += delta

        # Re-evaluate the reduced Jacobian and residual matrices
        Jac, res = rom_cls.eval_resJac_rom(mask, rom_cls.dir_nodes, rom_cls.sol_dir, T, node_eqnId, xi, V)

        # Compute the new residual norm
        norm_ = np.linalg.norm(res)

        # Display the current iteration details
        if op:
            print("iter {}, NL residual={}, delta={}".format(it, norm_, np.max(delta)))

        # Check for convergence
        if norm_ < tol:
            if op:
                print('Convergence !!!')
        elif it == max_iter - 1:
            print('\nWARNING: nonlinear solution has not converged')

        # Increment the iteration counter
        it += 1

    return T

def solve_reduced_fsolve(rom_cls, T_init, xi, V, tol=1e-5, max_iter=300):
    """
    Function: solve_reduced
    Overview: Solve the nonlinear system for the reduced-order model using Newton-Raphson iteration.

    Inputs:
    - cls: Refers to the FOS class instance containing mesh and finite element data.
    - T_init: Initial guess for the reduced temperature field.
    - xi: Element-wise importance weights.
    - V: Projection matrix.
    - tol: Tolerance for convergence (default is 1e-5).
    - max_iter: Maximum number of iterations for the solver (default is 300).

    Outputs:
    - Returns the updated temperature field after convergence or reaching max iterations.
    """
    print("fsolve")

    # Get node equation IDs and create a mask for nodes with non-zero equation IDs
    node_eqnId = rom_cls.node_eqnId
    mask = node_eqnId != 0

    # Copy the initial temperature field
    T = np.copy(T_init)

    # Evaluate the reduced Jacobian and residual matrices for the initial guess
    res = lambda T_: rom_cls.eval_res_fsolve_rom(mask, rom_cls.dir_nodes, rom_cls.sol_dir, T_, node_eqnId, xi, V).flatten()

    T_ans = fsolve(res, T)

    return T_ans

class rom(FOS_FEM):

    def __init__(self, data, quad_degree, mean=None):

        super().__init__(data, quad_degree)
        self.mean = mean


    def solve_rom(self, T_init, V):
        """
        Function: solve_rom
        Overview: Solve the nonlinear system for the reduced-order model.
        
        Inputs:
        - T_init: Initial guess for the reduced temperature field.
        - xi: Element-wise importance weights.
        - V: SVD matrix.

        Outputs:
        - Returns the updated temperature field after convergence or reaching max iterations.
        """
        
        T = solve_reduced(self, T_init, np.ones(self.data.n_cells), V)
        return T

    
    def eval_resJac_rom(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):

        K_r, J_r, K_r_mean, rhs_r = weighted_matrix_assembly(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V)
        res = (K_r @ T_red + K_r_mean - rhs_r)
        
        return K_r + J_r, res

class rom_ecsw(FOS_FEM):

    """
    Class: FEM_solver_rom_ecsw
    Overview: Inherits from the base class for finite element method (FEM) heat conduction simulations.
              This subclass focuses on reduced-order modeling with Element-based Coarse-Scale Weights (ECSW).
              
    Attributes:
    - data: Mesh and finite element data.
    - quad_degree: Degree for Gaussian quadrature integration.
    """

    def __init__(self, data, quad_degree, mean=None):
        """
        Function: __init__
        Overview: Constructor to initialize the reduced-order FEM solver.
        """
        super().__init__(data, quad_degree)
        self.mean = mean

    def solve_rom(self, T_init, xi, V):
        """
        Function: solve_rom
        Overview: Solve the nonlinear system for the reduced-order model.
        
        Inputs:
        - T_init: Initial guess for the reduced temperature field.
        - xi: Element-wise importance weights.
        - V: Projection matrix.

        Outputs:
        - Returns the updated temperature field after convergence or reaching max iterations.
        """
        T = solve_reduced(self, T_init, xi, V)
        return T
    
    def eval_resJac_rom(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):

        K_r, J_r, K_r_mean, rhs_r  = weighted_matrix_assembly(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V)
        res = (K_r @ T_red + K_r_mean - rhs_r)
        
        return K_r + J_r, res

class rom_deim(FOS_FEM):
    """
    Class: FEM_solver_rom_ecsw
    Overview: Inherits from the base class for finite element method (FEM) heat conduction simulations.
              This subclass focuses on reduced-order modeling with Element-based Coarse-Scale Weights (ECSW).
              
    Attributes:
    - data: Mesh and finite element data.
    - quad_degree: Degree for Gaussian quadrature integration.
    """

    def __init__(self, data, deim_cls, quad_degree, mean=None):
        """
        Function: __init__
        Overview: Constructor to initialize the reduced-order FEM solver.
        """
        super().__init__(data, quad_degree)
        self.deim_cls = deim_cls
        self.mean = mean

    def solve_rom(self, T_init, xi, V):
        """
        Function: solve_rom
        Overview: Solve the nonlinear system for the reduced-order model.
        
        Inputs:
        - T_init: Initial guess for the reduced temperature field.
        - xi: Element-wise importance weights.
        - V: Projection matrix.

        Outputs:
        - Returns the updated temperature field after convergence or reaching max iterations.
        """
        # T = solve_reduced(self, T_init, xi, V)
        T = solve_reduced_fsolve(self, T_init, xi, V)

        return T


    def eval_resJac_rom(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):

        K, J, sol_prev, rhs = weighted_matrix_assembly_deim(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V)

        LHS = K @ sol_prev[mask] 
        M = self.deim_cls.deim_mat
        deim_mask = self.deim_cls.bool_sampled
        res_projected = np.dot(M, LHS[deim_mask]) - np.dot(V.T, rhs)
        Jac_proj = (K + J) @ V
        
        return M @ Jac_proj[deim_mask] , res_projected 


    def eval_res_fsolve_rom(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):

        K, _, sol_prev, rhs = weighted_matrix_assembly_deim(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V)

        LHS = K @ sol_prev[mask]
        M = self.deim_cls.deim_mat
        deim_mask = self.deim_cls.bool_sampled
        res_projected = np.dot(M, LHS[deim_mask]) - np.dot(V.T, rhs)
        return res_projected
    

    """
    Class: FEM_solver_rom_ecsw
    Overview: Inherits from the base class for finite element method (FEM) heat conduction simulations.
              This subclass focuses on reduced-order modeling with Element-based Coarse-Scale Weights (ECSW).
              
    Attributes:
    - data: Mesh and finite element data.
    - quad_degree: Degree for Gaussian quadrature integration.
    """

    def __init__(self, data, deim_cls, quad_degree, mean=None):
        """
        Function: __init__
        Overview: Constructor to initialize the reduced-order FEM solver.
        """
        super().__init__(data, quad_degree)

        self.deim_cls = deim_cls
        self.mean = mean

    def solve_rom(self, T_init, xi, V):
        """
        Function: solve_rom
        Overview: Solve the nonlinear system for the reduced-order model.
        
        Inputs:
        - T_init: Initial guess for the reduced temperature field.
        - xi: Element-wise importance weights.
        - V: Projection matrix.

        Outputs:
        - Returns the updated temperature field after convergence or reaching max iterations.
        """
        T = solve_reduced(self, T_init, xi, V)
        # T = solve_reduced_fsolve(self, T_init, xi, V)

        return T

    def eval_resJac_rom(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):

        K, J, sol_prev, rhs = weighted_matrix_assembly_deim(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V)

        residual = K @ sol_prev[mask] - rhs 
        M = self.deim_cls.deim_mat
        deim_mask = self.deim_cls.bool_sampled
        res_projected = np.dot(M, residual[deim_mask])

        return M @ (K + J)[deim_mask] @ V, res_projected 

    def eval_res_fsolve_rom(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):

        K, _, sol_prev, rhs = weighted_matrix_assembly_deim(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V)

        residual = K @ sol_prev[mask] - rhs
        M = self.deim_cls.deim_mat
        deim_mask = self.deim_cls.bool_sampled
        res_projected = np.dot(M, residual[deim_mask])

        return res_projected
    
class rom_affine(FOS_FEM):

    def __init__(self, data, quad_degree, mean=None):

        super().__init__(data, quad_degree)
        self.mean = mean


    def solve_rom(self, V):
        """
        Function: solve_rom
        Overview: Solve the nonlinear system for the reduced-order model.
        
        Inputs:
        - T_init: Initial guess for the reduced temperature field.
        - xi: Element-wise importance weights.
        - V: SVD matrix.

        Outputs:
        - Returns the updated temperature field after convergence or reaching max iterations.
        """

        T_init = np.zeros(V.shape[1])
        K_r, K_r_mean, rhs_qe_, rhs_fe_ = self.reduced_affine(T_init, np.ones(self.data.n_cells), V)

        return  K_r, K_r_mean, rhs_qe_, rhs_fe_
    

    def reduced_affine(self, T_red, xi, V):

        node_eqnId = self.node_eqnId
        mask = node_eqnId != 0

        K_r, K_r_mean, rhs_qe_, rhs_fe_ = weighted_matrix_assembly_affine(self, mask, self.dir_nodes, self.sol_dir, T_red, node_eqnId, xi, V)
        
        return   K_r, K_r_mean, rhs_qe_, rhs_fe_