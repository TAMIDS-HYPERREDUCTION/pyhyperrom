from src.codes.prob_classes.base_class_struc_mech_continuous_vibration import FOS_FEM
from src.codes.utils.fem_utils_StrucMech import *
from src.codes.utils.rom_utils import *
from src.codes.basic import *
from scipy import sparse

def solve_rom_dynamics(rom_cls, sol_init, V_, xi=None):

    # Handle boundary conditions and get node equation IDs    
    node_eqnId = rom_cls.node_eqnId

    # Create a mask for nodes that do not have a Dirichlet boundary condition
    mask = node_eqnId != 0

    # Update initial temperature values for Dirichlet boundary nodes
    K,M = global_KM_matrices(rom_cls, node_eqnId, xi)

    K_r = V_.T@K@V_
    M_r = V_.T@M@V_

    C_r = rom_cls.cm*K_r + rom_cls.cv*M_r

    dt = rom_cls.data.dt
    t = rom_cls.data.t

    _, rhs = global_F_matrix_t(rom_cls, node_eqnId, t, xi)
    rhs_r = V_.T@rhs

    A_rom, B_rom, C_rom, D_rom, U_rom = convert_second_to_first_order(sparse.csr_matrix(K_r), sparse.csr_matrix(M_r), sparse.csr_matrix(C_r), rhs_r, t)
    
    

    ### Modify C_rom for this problem to match with C_sys
    # A_rom, B_rom, C_rom, D_rom, _ = convert_second_to_first_order(sparse.csr_matrix(K_r), sparse.csr_matrix(M_r), sparse.csr_matrix(C_r), rhs_r, t)

    # fom = rom_cls.f_cls.fom

    # B_sys = fom.B

    # U_rom = np.zeros((B_sys.shape[0], len(t)))
    # U_rom[U_rom.shape[0]//2:, :] = rhs

    # V_Matrix_B = np.block([[np.zeros_like(V_.T),np.zeros_like(V_.T)],
    #                        [np.zeros_like(V_.T),V_.T]])

    # B_rom = B_rom@V_Matrix_B

    # C_sys = fom.C
    # V_Matrix_C = np.block([[V_,np.zeros_like(V_)],
    #                     [np.zeros_like(V_),V_]])
    # C_rom = C_sys@V_Matrix_C

    # D_rom = np.zeros((C_rom.shape[0],B_rom.shape[1]))
    ###


    # Create the state-space model
    rom_sys = ctrl.ss(A_rom, B_rom, C_rom, D_rom)

    # x0 = np.pad(sol_init[mask], (len(sol_init[mask]), 0), mode='constant', constant_values=0)
    _,_,x_out_rom = ctrl.forced_response(rom_sys, T=t, U=U_rom, X0 = sol_init, return_x=True)

    return x_out_rom


class rom(FOS_FEM):

    def __init__(self, f_cls, quad_degree,tau, ep, T, cv, cm, xi=None):
        ## xi = None implies **NO** ECSW hyperreduction

        super().__init__(f_cls.FOS.data, quad_degree, tau, ep, T)
        
        self.f_cls = f_cls
        self.cv = cv
        self.cm = cm
        self.xi = xi

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

        sol = solve_rom_dynamics(self, solinit_rom, V, xi=self.xi)
        return sol


class rom_deim(FOS_FEM):
    """
    Class: FEM_solver_rom_ecsw
    Overview: Inherits from the base class for finite element method (FEM) heat conduction simulations.
              This subclass focuses on reduced-order modeling with Element-based Coarse-Scale Weights (ECSW).
              
    Attributes:
    - data: Mesh and finite element data.
    - quad_degree: Degree for Gaussian quadrature integration.
    """

    def __init__(self, data, deim_cls, quad_degree):
        """
        Function: __init__
        Overview: Constructor to initialize the reduced-order FEM solver.
        """
        super().__init__(data, quad_degree)

        self.deim_cls = deim_cls


    def solve_rom(self, solinit, xi, V):
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
        sol = solve_reduced(self, solinit, xi, V)
        # sol = solve_reduced_fsolve(self, solinit, xi, V)

        return sol


    def eval_resJac_rom(self, mask, dir_nodes, sol_dir, sol_red, node_eqnId, xi, V):

        K, J, sol_prev, rhs = weighted_matrix_assembly(self, mask, dir_nodes, sol_dir, sol_red, node_eqnId, xi, V)

        residual = K @ sol_prev[mask] - rhs.reshape(-1, 1)

        M = self.deim_cls.deim_mat
    
        deim_mask = self.deim_cls.bool_sampled
    
        res_projected = np.dot(M, residual[deim_mask])

        return M @ (K + J)[deim_mask] @ V[mask], res_projected 


    def eval_res_fsolve_rom(self, mask, dir_nodes, sol_dir, sol_red, node_eqnId, xi, V):

        K, J, sol_prev, rhs = weighted_matrix_assembly(self, mask, dir_nodes, sol_dir, sol_red, node_eqnId, xi, V)

        residual = K @ sol_prev[mask] - rhs.reshape(-1, 1)

        M = self.deim_cls.deim_mat
    
        deim_mask = self.deim_cls.bool_sampled
    
        res_projected = np.dot(M, residual[deim_mask])

        return res_projected