from src.codes.prob_classes.base_class_heat_conduction import FOS_FEM
from src.codes.utils.fem_utils_HC import *
from src.codes.utils.rom_utils import *
from src.codes.basic import *
from scipy.optimize import fsolve

def weighted_matrix_assembly(rom_cls, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):

    # Initialize global matrices and vectors
    K, J, rhs = init_global_systems(max(node_eqnId))
    
    sol_prev = np.zeros(len(rom_cls.data.mask))
    # Compute the full solution field from the reduced solution field
    
    if rom_cls.mean is not None:
        sol_prev[rom_cls.data.mask] = np.dot(V,T_red) + rom_cls.mean.flatten()
    else:
        sol_prev[rom_cls.data.mask] = np.dot(V,T_red)

        
    sol_prev[~rom_cls.data.mask] = rom_cls.data.T_dir

    # Loop through all elements in the domain
    for iel in range(rom_cls.data.n_cells):

        # Skip elements with zero importance
        if xi[iel] == 0:
            continue

        # Retrieve global node numbers and equation IDs for the current element
        elem_glob_nodes = rom_cls.data.gn[iel, :]
        elem_glob_node_eqnId = rom_cls.data.glob_node_eqnId[iel]
        elem_glob_node_nonzero_eqnId = rom_cls.data.glob_node_nonzero_eqnId[iel]
        elem_local_node_nonzero_eqnId = rom_cls.data.local_node_nonzero_eqnId[iel]

        # Compute local element matrices and vectors
        Ke_, Je_, qe_ = compute_element_matrices(rom_cls, sol_prev.flatten(), iel)

        # Retrieve global and local indices for stiffness matrices
        I_index, J_index = rom_cls.data.global_indices[iel][0], rom_cls.data.global_indices[iel][1]
        i_index, j_index = rom_cls.data.local_indices[iel][0], rom_cls.data.local_indices[iel][1]

        # Update global stiffness and Jacobian matrices with weighted local matrices
        K[I_index, J_index] += Ke_[i_index, j_index] * xi[iel]
        J[I_index, J_index] += Je_[i_index, j_index] * xi[iel]

        # Handle Dirichlet boundary conditions
        if np.isin(0, elem_glob_node_eqnId):
            elem_dof_values = dirichlet_bc(rom_cls, sol_dir, dir_nodes, elem_glob_nodes)
            fe = Ke_ @ elem_dof_values.reshape(-1, 1)
        else:
            fe = np.zeros((len(elem_glob_nodes), 1))

        # Compute local right-hand side vector
        rhs_e_ = qe_[elem_local_node_nonzero_eqnId] - fe[elem_local_node_nonzero_eqnId].flatten()

        # Update global right-hand side vector with weighted local vector
        rhs[elem_glob_node_nonzero_eqnId-1] += xi[iel] * rhs_e_

    return K, J, sol_prev.reshape(-1,1), rhs

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
        delta = linalg.spsolve(Jac, -res)

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

    # Get node equation IDs and create a mask for nodes with non-zero equation IDs
    node_eqnId = rom_cls.node_eqnId
    mask = node_eqnId != 0

    # Copy the initial temperature field
    T = np.copy(T_init)


    # Evaluate the reduced Jacobian and residual matrices for the initial guess
    res = lambda T_: rom_cls.eval_res_fsolve_rom(mask, rom_cls.dir_nodes, rom_cls.sol_dir, T_, node_eqnId, xi, V).flatten()


    T_ans = fsolve(res,T)


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

        K, J, sol_prev, rhs = weighted_matrix_assembly(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V)

        residual = K @ sol_prev[mask]  - rhs.reshape(-1, 1)
        
        # res = np.transpose(V[mask, :]) @ residual
        res = np.transpose(V) @ residual

        # return np.transpose(V[mask]) @ (K + J) @ V[mask], res
        return np.transpose(V) @ (K + J) @ V, res

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

        K, J, sol_prev, rhs = weighted_matrix_assembly(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V)

        residual = K @ sol_prev[mask]  - rhs.reshape(-1, 1)

        # res = np.transpose(V[mask, :]) @ residual
        res = np.transpose(V) @ residual
        
        # return np.transpose(V[mask]) @ (K + J) @ V[mask], res
        return np.transpose(V) @ (K + J) @ V, res
        
        # res = np.transpose(V[mask, :]) @ residual

        # return np.transpose(V[mask]) @ (K + J) @ V[mask], res

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
        T = solve_reduced(self, T_init, xi, V)
        # T = solve_reduced_fsolve(self, T_init, xi, V)

        return T


    def eval_resJac_rom(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):

        K, J, sol_prev, rhs = weighted_matrix_assembly(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V)

        residual = K @ sol_prev[mask] - rhs.reshape(-1, 1)

        M = self.deim_cls.deim_mat
    
        deim_mask = self.deim_cls.bool_sampled
    
        res_projected = np.dot(M, residual[deim_mask])

        return M @ (K + J)[deim_mask] @ V, res_projected 

        # return M @ (K + J)[deim_mask] @ V[mask], res_projected 


    def eval_res_fsolve_rom(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):

        K, J, sol_prev, rhs = weighted_matrix_assembly(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V)

        residual = K @ sol_prev[mask] - rhs.reshape(-1, 1)

        M = self.deim_cls.deim_mat
    
        deim_mask = self.deim_cls.bool_sampled
    
        res_projected = np.dot(M, residual[deim_mask])

        return res_projected