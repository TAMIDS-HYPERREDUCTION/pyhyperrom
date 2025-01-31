from platform import node
from ...utils.fem_utils_StrucMech import *
from ...basic import *
import time

class probdata:

    def __init__(self, bc, mat_layout, src_layout, fdict, nref, L, dof_p_node, node_p_elem,expon):
        """
        Initializes the problem data.
        Args:
            bc: Boundary condition data.
            mat_layout: Material layout matrix.
            src_layout: Source layout matrix.1
            fdict: Dictionary of functions.
            nref: Not used, can be removed or repurposed.
            L (float): Length of the domain.
            dof_p_node (int): Degrees of freedom per node.
            node_p_elem (int): Nodes per element.
        """

        self.dim_ = 1  # Assuming a 1-dimensional problem
        self.dof_p_node = dof_p_node
        self.node_p_elem = node_p_elem
        self.elem_dof = dof_p_node * node_p_elem
        

        if self.dim_ > 2:
            raise ValueError("Dimension should be 1")

        self.cell2mat_layout = mat_layout
        self.cell2src_layout = src_layout
        self.expon = expon
        self.fdict = fdict
        self.L = L


        # Setting up mesh parameters
        self.ncells = [mat_layout.shape[0]]
        self.n_cells = mat_layout.shape[0]
        self.npts = [self.ncells[0] + 1]
        self.deltas = [L / self.ncells[0]]
        self.xi = [np.linspace(0, L, self.npts[0])]

        self.n_verts = self.npts[0] * dof_p_node

        # Create nodal connectivity for the continuous Finite Element Method (cFEM)
        self.connectivity()

        # Store the dirichlet nodes if any
        handle_boundary_conditions_statics(self, bc)

        # Determining the global equation numbers based on dirichlet nodes and storing in class
        get_glob_node_equation_id(self, self.dir_nodes)

        self.prepare_element_data()

        # Create a mask for nodes that do not have a Dirichlet boundary condition

        mask = self.node_eqnId != 0
        self.mask = mask

    def connectivity(self):
        """Determines the connectivity of nodes based on the problem dimension."""
        if self.dim_ == 1:
            self.connectivity_1d()
        else:
            raise ValueError("Unsupported dimension")

    def connectivity_1d(self):
        """
        Defines the nodal connectivity for each cell in a 1D mesh.
        """
        self.gn = np.zeros((self.ncells[0], self.elem_dof), dtype=int)
        self.gnodes = np.zeros((self.ncells[0], self.node_p_elem), dtype=int)

        for iel in range(self.ncells[0]):
            dof_start = self.dof_p_node * iel
            dofs = np.arange(dof_start, dof_start + self.elem_dof)
            
            self.gn[iel] = dofs 
            self.gnodes[iel] = dofs[::self.dof_p_node] // self.dof_p_node

    def prepare_element_data(self):
        """Prepares data for each element in the mesh."""

        # Get global node numbers and equation IDs for the current element
        self.glob_node_eqnId = []
        self.glob_node_nonzero_eqnId = []
        self.local_node_nonzero_eqnId = []
        self.Le = []
        self.global_indices = []
        self.local_indices = []

        for i in range(self.n_cells):
            get_element_global_nodes_and_nonzero_eqnId(self, i, self.node_eqnId)

class FOS_FEM:

    def __init__(self, data, quad_degree, tau):
        """
        Initialize the class with given data and quadrature degree.
        """
        self.data = data
        self.tau = tau
        self.expon=data.expon
        self.dim_ = data.dim_
        self.ncells = data.ncells.copy()
        self.npts = data.npts.copy()
        self.deltas = data.deltas.copy()
        
        self.n_nodes = data.n_verts
        self.sol_dir = data.sol_dir
        self.dir_nodes = data.dir_nodes
        self.node_eqnId = data.node_eqnId

        self.basis()
        self.basis_q(quad_degree)

    def basis(self):
        """
        Compute the continuous Finite Element (cFEM) basis functions.
        """
        x = sp.symbols('x')
        shape_functions = [1/2*(1-x), 1/2*(1+x)]
        
        self.b = [sp.lambdify(x, bf, 'numpy') for bf in shape_functions]
        self.dbdxi = [sp.lambdify(x, sp.diff(bf, x), 'numpy') for bf in shape_functions]

    def basis_q(self, quad_degree):
        """
        Compute the basis functions and their derivatives at the quadrature points.
        """
        xq, wq = np.polynomial.legendre.leggauss(quad_degree)
        self.xq = xq
        nq = len(xq)
        
        self.bq = np.array([[fi(xi) for fi in self.b] for xi in xq])
        self.dbdxiq = np.array([[dfi(xi) for dfi in self.dbdxi] for xi in xq])
        self.w = wq

    def eval_at_quadrature_points(self, U_prev, elem_glob_nodes):
        """
        Evaluate temperature and its derivative at the quadrature points using the FE basis functions.

        Parameters:
        - sol_prev: Previous temperature field
        - elem_glob_nodes: Global node numbers associated with the current element

        Returns:
        - sol_prev_q: Temperature values at the quadrature points
        - dsol_prev_q: Temperature derivative values at the quadrature points
        """
        dim_ = self.dim_

        # Evaluate temperature derivatives at the quadrature points using the FE basis function derivatives
        dU_prev_q = np.dot(self.dbdxiq, U_prev[elem_glob_nodes])
        
        
        if dU_prev_q.any()==0:
            dU_prev_q = 1e-4 + np.zeros_like(dU_prev_q)
            print("0 happened")
        else:
            dU_prev_q = np.abs(dU_prev_q)**(self.expon)


        return dU_prev_q  # Return temperature values and derivatives as a tuple

    def element_KM_matrices_fn_x(self, iel, U_prev, notconstant = False):
        """
        Compute the element matrices and vectors.
        """

        dx = self.data.deltas[0]
        mod_jac = dx/2
        elem_glob_nodes = self.data.gn[iel, :]

        imat = self.data.cell2mat_layout[iel].astype(int)
        # E, rho, A = self.data.fdict["E"][imat], self.data.fdict["rho"][imat], self.data.fdict["A"][imat]
        
        dU_prev_q = self.eval_at_quadrature_points(U_prev, self.data.gn[iel, :])

        n = len(elem_glob_nodes)
        Ke_, Je_ = np.zeros((n, n)), np.zeros((n, n))

        if notconstant:

            bq = self.bq
            self.bq_ = bq
            dbdxiq = np.copy(self.dbdxiq)

            # E_xq, rho_xq, A_xq = E(self.xq), rho(self.xq), A(self.xq)
            # Jac_inv = (dx/2)**(-1)

            for i in range(n):
                for j in range(n):
                    
                    # K_temp = (Jac_inv**2) * mod_jac * np.dot(self.w * dbdxiq[:,i], E_xq * A_xq * dbdxiq[:, j]*dU_prev_q)*(Jac_inv**(self.expon))
                    
                    K_temp =  np.dot(self.w * dbdxiq[:,i], dbdxiq[:, j]*dU_prev_q) # Enhanced num stability
                    # M_temp = mod_jac * np.dot(self.w * bq[:, i], rho_xq * A_xq * bq[:, j])
                    J_temp =  K_temp*(self.expon+1)
                    
                    Ke_[i, j] += K_temp
                    Je_[i, j] += J_temp

        return Ke_, Je_

    def element_F_matrices_fn_x(self, iel):
        """
        Compute the element source vector for a given temperature field.
        """
        isrc = self.data.cell2src_layout[iel].astype(int)
        fext = self.data.fdict["fext"][isrc]
           

        elem_glob_dofs = self.data.gn[iel, :]
        elem_glob_nodes = self.data.gnodes[iel, :]

        
        n = len(elem_glob_dofs)
        qe_ = np.zeros(n)

        x_iel = self.data.xi[0][elem_glob_nodes]
        fext_q = fext(self.xq, x_iel, self.tau[1], self.data.fdict["A"][isrc],self.data.fdict["rho"][isrc])
        mod_jac = self.deltas[0]/2

####### This is to enhance numerical stability

        imat = isrc 
        E, rho, A = self.data.fdict["E"][imat], self.data.fdict["rho"][imat], self.data.fdict["A"][imat]
        E_const, A_const = E(1,self.tau[0]), A(1)
        
        Jac_inv = (mod_jac)**(-1)
        Jac_inv_exp = (Jac_inv**(self.expon))
        
        Stiffness_factor = 1/((Jac_inv**2)*E_const*A_const*Jac_inv_exp)
        
########

        
        for i in range(n):

            # qe_[i] += mod_jac * np.dot(self.w * self.bq_[:, i], fext_q)
            qe_[i] += Stiffness_factor * np.dot(self.w * self.bq_[:, i], fext_q)
            
        return qe_

    def residual_func(self,i,j,p_sol,data):

        # Extract relevant stiffness matrices and source terms for the current snapshot and cell
        # Project the solution onto the selected basis
        
        K_mus = data['K_mus']
        q_mus = data['q_mus']

        K_mus_ij = K_mus[i][j]
        q_mus_ij = np.array(q_mus[i][j])
        
        res = np.dot(K_mus_ij, p_sol) - q_mus_ij

        return res
    
class StructuralMechanicsSimulationData:

    def __init__(self, n_ref, params, quad_deg=5, num_snapshots=1, pb_dim=1):
        """
        Initialize the simulation data class.
        """
        # Initialize layout instance

        from examples.structural_mechanics.Axial.NL_static.SystemProperties import SystemProperties

        self.properties = SystemProperties(n_ref)
        self.n_ref, self.L, self.expon = n_ref, self.properties.L, self.properties.expon
        self.quad_deg, self.num_snapshots = quad_deg, num_snapshots
        self.mat_layout, self.src_layout = self.properties.create_layouts()
        self.fdict = self.properties.define_properties()
        self.bc = self.properties.define_boundary_conditions()
        self.params = params

        self.NL_solutions, self.param_list = [], []
        self.fos_time, self.K_mus, self.q_mus = [], [], []
        self.dof_p_node, self.node_p_elem = 1, 2
        

    def run_simulation(self):
        """
        Run the simulation for the given number of snapshots.
        """

        for i in range(self.num_snapshots):
            print(f"Snap {i}")
            tau = self.params[i]
            self.param_list.append(tau)

            if i == 0:
                d = probdata(self.bc, self.mat_layout, self.src_layout, self.fdict, self.n_ref, self.L, self.dof_p_node, self.node_p_elem,self.expon)                
                self.FOS = FOS_FEM(d, self.quad_deg, tau)
                
                self.sol_init = abs(np.random.rand(d.n_verts)*15)
                self.sol_init[-1] = 30.06
                
            else:
                
                self.FOS.tau = tau
                ##################
                self.sol_init = NL_solutions_p
                ##################

            # sol_init = np.array([0,20,30])#abs(np.random.rand(d.n_verts)*15)/10000
            
            tic_fos = time.time()
            NL_solutions_p, Ke_d, rhs_e, _ = solve_fos_statics(self.FOS, self.sol_init)
            toc_fos = time.time()

            self.fos_time.append(toc_fos - tic_fos)
            self.NL_solutions.append(NL_solutions_p)
            self.q_mus.append(rhs_e)
            self.K_mus.append(Ke_d)

class ROM_simulation:
    
    def __init__(self, f_cls, test_data, param_list, Test_mask, V_sel, xi=None, deim=None,sol_init_guess = 0.1, N_rom_snap = None):

        """
        Initialize the Reduced Order Model (ROM) simulation class.
        """
        
        self.f_cls = f_cls
        self.FOS = f_cls.FOS
        self.sol_init_guess = sol_init_guess
        self.fos_test_data = test_data
        self.param_list_test = param_list[Test_mask]
        self.fos_test_time = np.asarray(f_cls.fos_time)[Test_mask]
        self.V_sel = V_sel
        self.deim = deim
        self.xi = xi
        self.quad_deg = f_cls.quad_deg
        self.NL_solutions_rom = []
        self.d = f_cls.FOS.data
        self.mean = f_cls.mean
        
        
        if N_rom_snap!=None:
            self.N_rom_snap = N_rom_snap
        else:
            self.N_rom_snap = len(self.param_list_test)
            

    def run_simulation(self):

        """
        Run the ROM simulation.
        """
        
        import src.codes.reductor.Struc_Mech.rom_class_StrucMech_axial as rom_class


        self.speed_up = []
        self.rom_error = []

        
        sol_fos_ = self.fos_test_data

        # Initial guess for temperature
        sol_init_fos = abs(np.random.rand(self.d.n_verts))
        sol_init_fos[-1] = self.sol_init_guess
        sol_init_rom = np.transpose(self.V_sel) @ sol_init_fos[self.FOS.data.mask]  # Initial guess in the reduced subspace
        
        sol_rom = np.zeros_like(sol_init_fos)
        

        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            

            tic_rom = time.time()
            
            ROM = rom_class.rom(self.f_cls, self.quad_deg, self.param_list_test[i], mean=self.mean, xi = self.xi)
            NL_solution_p_reduced = ROM.solve_rom(sol_init_rom, self.V_sel)

            sol_init_rom = NL_solution_p_reduced
                           
                
            toc_rom = time.time()
            rom_sim_time = toc_rom - tic_rom
            
            
            self.speed_up.append(self.fos_test_time[i]/rom_sim_time)
            

            sol_rom[self.FOS.data.mask] = np.dot(self.V_sel,NL_solution_p_reduced) + self.mean
            sol_rom[~self.FOS.data.mask] = self.FOS.data.sol_dir

            self.NL_solutions_rom.append(np.copy(sol_rom))

            sol_fos = sol_fos_[i]
            
            self.rom_error.append(np.linalg.norm(sol_rom[self.FOS.data.mask] - sol_fos) * 100 / np.linalg.norm(sol_fos))

class ROM_simulation_UQ:
    
    def __init__(self, f_cls, test_data, param_list, V_sel, xi=None, deim=None,sol_init_guess = 0.1, N_rom_snap = None, fos_comp= True):

        """
        Initialize the Reduced Order Model (ROM) simulation class.
        """
        
        self.f_cls = f_cls
        self.FOS = f_cls.FOS
        self.sol_init_guess = sol_init_guess
        self.fos_test_data = test_data
        self.param_list_test = param_list
        # self.fos_test_time = np.asarray(f_cls.fos_time)[Test_mask]
        self.V_sel = V_sel
        self.deim = deim
        self.xi = xi
        self.quad_deg = f_cls.quad_deg
        self.NL_solutions_rom = []
        self.d = f_cls.FOS.data
        self.mean = f_cls.mean
        self.fos_comp = fos_comp

        
        if N_rom_snap!=None:
            self.N_rom_snap = N_rom_snap
        else:
            self.N_rom_snap = len(self.param_list_test)
            

    def run_simulation(self):
        """
        Run the ROM simulation.
        """
        
        import src.codes.reductor.Struc_Mech.rom_class_StrucMech_axial as rom_class

        self.rom_error = []
        sol_fos_ = self.fos_test_data

        # Initial guess for temperature
        sol_init_fos = abs(np.random.rand(self.d.n_verts))
        sol_init_fos[-1] = self.sol_init_guess
        sol_init_rom = np.transpose(self.V_sel) @ sol_init_fos[self.FOS.data.mask]  # Initial guess in the reduced subspace
        
        sol_rom = np.zeros_like(sol_init_fos)

        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            
            if i%100 == 0:
                print(i)
            
            ROM = rom_class.rom(self.f_cls, self.quad_deg, self.param_list_test[i], mean=self.mean, xi = self.xi)
            NL_solution_p_reduced = ROM.solve_rom(sol_init_rom, self.V_sel)

            sol_init_rom = NL_solution_p_reduced
                           
            sol_rom[self.FOS.data.mask] = np.dot(self.V_sel,NL_solution_p_reduced) + self.mean
            sol_rom[~self.FOS.data.mask] = self.FOS.data.sol_dir

            self.NL_solutions_rom.append(np.copy(sol_rom))

            if self.fos_comp:
                sol_fos = sol_fos_[i]
                self.rom_error.append(np.linalg.norm(sol_rom[self.FOS.data.mask] - sol_fos) * 100 / np.linalg.norm(sol_fos))
