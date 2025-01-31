from src.codes.prob_classes.structural_mechanics.base_class_struc_mech_NL_static_axial import FOS_FEM
from src.codes.utils.fem_utils_StrucMech import *
from src.codes.utils.rom_utils import *
from src.codes.basic import *
from scipy import sparse


def weighted_matrix_assembly(rom_cls, mask, dir_nodes, sol_dir, sol_red, node_eqnId, xi, V):
    
    if xi is None:
        xi = np.ones(rom_cls.data.n_cells)

    K_r, J_r = [np.zeros((V.shape[1],V.shape[1])) for _ in range(2)]
    rhs_r, K_r_mean = [np.zeros(V.shape[1]) for _ in range(2)]
    
    sol_prev = np.zeros(len(rom_cls.data.mask))



    if rom_cls.mean is not None:
        
        # Compute the full solution field from the reduced solution field
        sol_mean = rom_cls.mean.flatten()
        sol_prev[rom_cls.data.mask] = np.dot(V,sol_red) + sol_mean
        
    else:
        
        sol_prev[rom_cls.data.mask] = np.dot(V,sol_red)


    sol_prev[~rom_cls.data.mask] = rom_cls.data.sol_dir
    

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
        Ke_, Je_, qe_ = compute_element_matrices_statics(rom_cls, sol_prev.flatten(), iel)


        # Retrieve global and local indices for stiffness matrices
        # I_index, J_index = rom_cls.data.global_indices[iel][0], rom_cls.data.global_indices[iel][1]
        i_index, j_index = rom_cls.data.local_indices[iel][0], rom_cls.data.local_indices[iel][1]


        # Update global stiffness and Jacobian matrices with weighted local matrices
        K_r += V[col_indices].T @ (Ke_[i_index, j_index] * xi[iel]) @ V[col_indices]
        J_r += V[col_indices].T @ (Je_[i_index, j_index] * xi[iel]) @ V[col_indices]


        # Effect of the mean
        if rom_cls.mean is not None:
            K_r_mean += V[col_indices].T @ (Ke_[i_index, j_index] * xi[iel]) @ sol_mean[col_indices]
        else:
            K_r_mean = np.zeros((V[col_indices].T).shape[0])


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


def solve_rom_statics(rom_cls, sol_init, V, xi, tol=1e-6, max_iter=3000,op=False):

    # Get node equation IDs and create a mask for nodes with non-zero equation IDs
    node_eqnId = rom_cls.node_eqnId
    mask = node_eqnId != 0

    # Copy the initial temperature field
    sol = np.copy(sol_init)

    # Evaluate the reduced Jacobian and residual matrices for the initial guess
    Jac, res = rom_cls.eval_resJac_rom(mask, rom_cls.dir_nodes, rom_cls.sol_dir, sol, node_eqnId, xi, V)

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
        sol += delta

        # Re-evaluate the reduced Jacobian and residual matrices
        Jac, res = rom_cls.eval_resJac_rom(mask, rom_cls.dir_nodes, rom_cls.sol_dir, sol, node_eqnId, xi, V)

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

    return sol


class rom(FOS_FEM):

    def __init__(self, f_cls, quad_degree, tau, mean, xi=None):
        ## xi = None implies **NO** ECSW hyperreduction

        super().__init__(f_cls.FOS.data, quad_degree, tau)
        
        self.f_cls = f_cls
        self.xi = xi
        self.mean = mean


    def solve_rom(self, solinit_rom, V):
        """
        Function: solve_rom
        Overview: Solve the nonlinear system for the reduced-order model.
        
        Inputs:
        - solinit: Initial guess for the reduced temperature field.
        - xi: Element-wise importance weights.
        - V: Projection matrix.

        Outputs:
        - Returns the updated temperature field after convergence or reaching max iterations.
        """
        
        sol = solve_rom_statics(self, solinit_rom, V, xi=self.xi)
        return sol


    def eval_resJac_rom(self, mask, dir_nodes, sol_dir, sol_red, node_eqnId, xi, V):

        K_r, J_r, K_r_mean, rhs_r  = weighted_matrix_assembly(self, mask, dir_nodes, sol_dir, sol_red, node_eqnId, xi, V)
        
        res = (K_r @ sol_red + K_r_mean - rhs_r)
        
        return K_r + J_r, res
    


###############################################################################


# class rom_deim(FOS_FEM):
#     """
#     Class: FEM_solver_rom_ecsw
#     Overview: Inherits from the base class for finite element method (FEM) heat conduction simulations.
#               This subclass focuses on reduced-order modeling with Element-based Coarse-Scale Weights (ECSW).
              
#     Attributes:
#     - data: Mesh and finite element data.
#     - quad_degree: Degree for Gaussian quadrature integration.
#     """

#     def __init__(self, data, deim_cls, quad_degree):
#         """
#         Function: __init__
#         Overview: Constructor to initialize the reduced-order FEM solver.
#         """
#         super().__init__(data, quad_degree)

#         self.deim_cls = deim_cls


#     def solve_rom(self, solinit, xi, V):
#         """
#         Function: solve_rom
#         Overview: Solve the nonlinear system for the reduced-order model.
        
#         Inputs:
#         - solinit: Initial guess for the reduced temperature field.
#         - xi: Element-wise importance weights.
#         - V: Projection matrix.

#         Outputs:
#         - Returns the updated temperature field after convergence or reaching max iterations.
#         """
#         sol = solve_reduced(self, solinit, xi, V)
#         # sol = solve_reduced_fsolve(self, solinit, xi, V)

#         return sol


#     def eval_resJac_rom(self, mask, dir_nodes, sol_dir, sol_red, node_eqnId, xi, V):

#         K, J, sol_prev, rhs = weighted_matrix_assembly(self, mask, dir_nodes, sol_dir, sol_red, node_eqnId, xi, V)

#         residual = K @ sol_prev[mask] - rhs.reshape(-1, 1)

#         M = self.deim_cls.deim_mat
    
#         deim_mask = self.deim_cls.bool_sampled
    
#         res_projected = np.dot(M, residual[deim_mask])

#         return M @ (K + J)[deim_mask] @ V[mask], res_projected 


#     def eval_res_fsolve_rom(self, mask, dir_nodes, sol_dir, sol_red, node_eqnId, xi, V):

#         K, J, sol_prev, rhs = weighted_matrix_assembly(self, mask, dir_nodes, sol_dir, sol_red, node_eqnId, xi, V)

#         residual = K @ sol_prev[mask] - rhs.reshape(-1, 1)

#         M = self.deim_cls.deim_mat
    
#         deim_mask = self.deim_cls.bool_sampled
    
#         res_projected = np.dot(M, residual[deim_mask])

#         return res_projected


# def solve_rom_dynamics(rom_cls, sol_init, V_, xi=None):

#     # Handle boundary conditions and get node equation IDs    
#     node_eqnId = rom_cls.node_eqnId

#     # Create a mask for nodes that do not have a Dirichlet boundary condition
#     mask = node_eqnId != 0

#     # Update initial temperature values for Dirichlet boundary nodes
#     K,M = global_KM_matrices(rom_cls, node_eqnId, xi)

#     K_r = V_.T@K@V_
#     M_r = V_.T@M@V_

#     C_r = rom_cls.cm*K_r + rom_cls.cv*M_r

#     dt = rom_cls.data.dt
#     t = rom_cls.data.t

#     _, rhs = global_F_matrix_t(rom_cls, node_eqnId, t, xi)
#     rhs_r = V_.T@rhs

#     A_rom, B_rom, C_rom, D_rom, U_rom = convert_second_to_first_order(sparse.csr_matrix(K_r), sparse.csr_matrix(M_r), sparse.csr_matrix(C_r), rhs_r, t)
    
    


#     # Create the state-space model
#     rom_sys = ctrl.ss(A_rom, B_rom, C_rom, D_rom)

#     # x0 = np.pad(sol_init[mask], (len(sol_init[mask]), 0), mode='constant', constant_values=0)
#     _,_,x_out_rom = ctrl.forced_response(rom_sys, T=t, U=U_rom, X0 = sol_init, return_x=True)

#     return x_out_rom

###############################################################################