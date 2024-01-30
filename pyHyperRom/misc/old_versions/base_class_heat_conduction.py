from ..utils.fem_utils_HC import *
from ..basic import *
from ..utils.rom_utils import train_test_split
import time 

class probdata:

    def __init__(self, bc, mat_layout, src_layout, fdict, nref, L, mu, pb_dim):

        self.dim_ = pb_dim
        # refine the mesh and update material and source layouts

        if(pb_dim==1):
            
            self.cell2mat_layout = mat_layout.flatten()
            self.cell2src_layout = src_layout.flatten()

        else:

            repeats = np.asarray(nref, dtype=int)
            self.cell2mat_layout = self.repeat_array(mat_layout, repeats)
            self.cell2src_layout = self.repeat_array(src_layout, repeats)
	    
        ## change this mapping if needed.
        self.fdict = fdict

	    # mesh data cells
        self.ncells = [None] * pb_dim
        self.npts = [None] * pb_dim
        self.deltas = [None] * pb_dim
        self.xi = []

        for i in range(pb_dim):
            self.ncells[i] = self.cell2mat_layout.shape[i]
            self.npts[i] = self.ncells[i] + 1
            if pb_dim == 1:
                self.xi.append(np.linspace(0,L,self.npts[i]))
                self.deltas[i] = L/self.ncells[i]
            else:
                self.xi.append(np.linspace(0, L[i], self.npts[i]))
                self.deltas[i] = L[i] / self.ncells[i]

        if pb_dim == 1:
        	self.n_verts = self.npts[0]
        else:
       		self.n_verts = np.prod(np.array(self.npts))

        # Create nodal connectivity for the continuous Finite Element Method (cFEM)
        self.connectivity()

        # Store parameter value
        self.mu = mu

        # Store the dirichlet nodes if any
        handle_boundary_conditions(self, bc)

        # Determining the global equation numbers based on dirichlet nodes and storing in class
        get_glob_node_equation_id(self, self.dir_nodes)

        # Get global node numbers and equation IDs for the current element
        self.glob_node_eqnId = []
        self.glob_node_nonzero_eqnId = []
        self.local_node_nonzero_eqnId = []
        self.Le = []
        self.global_indices = []
        self.local_indices = []

        for i in range(self.n_cells):
            get_element_global_nodes_and_nonzero_eqnId(self, i, self.node_eqnId)  

                # Create a mask for nodes that do not have a Dirichlet boundary condition
        mask = self.node_eqnId != 0
        self.mask = mask

    def repeat_array(self, arr, repeats):
        for dim, n in enumerate(repeats):
            arr = np.repeat(arr, n, axis=dim)
        return arr

    def connectivity(self):
        """
        Define nodal connectivity for each cell in the mesh.
        """
        if self.dim_ == 1:
            self.connectivity_1d()
        elif self.dim_ == 2:
            self.connectivity_2d()
        elif self.dim_ == 3:
            self.connectivity_3d()
        else:
            raise ValueError("Unsupported dimension")

    def connectivity_1d(self):
        # Initialize the connectivity array for 1D
        self.n_cells = self.ncells[0]
        self.gn = np.zeros((self.n_cells, 2**self.dim_), dtype=int)
	    # Loop over all cells to define their nodal connectivity
        for iel in range(self.n_cells):
	    # For each cell, define the left and right nodes	
            self.gn[iel, 0] = iel
            self.gn[iel, 1] = iel + 1

    def connectivity_2d(self):
        # Define connectivity for 2D
        self.n_cells = np.prod(np.array(self.ncells))
        self.gn = np.zeros((self.n_cells, 2**self.dim_), dtype=int)

        node = lambda i, j: i + j * self.npts[0]

        iel = 0
        for j in range(self.ncells[1]):
            for i in range(self.ncells[0]):
                self.gn[iel, 0] = node(i, j)
                self.gn[iel, 1] = node(i + 1, j)
                self.gn[iel, 2] = node(i + 1, j + 1)
                self.gn[iel, 3] = node(i, j + 1)
                iel += 1

    def connectivity_3d(self):
        # Initialize the connectivity array for 3D
        self.n_cells = np.prod(np.array(self.ncells))
        self.gn = np.zeros((self.n_cells, 2**self.dim_), dtype=int)

        node = lambda i, j, k: i + j * self.npts[0] + k * self.npts[0] * self.npts[1]
        # Loop over all cells to define their nodal connectivity
        iel = 0
        for k in range(self.ncells[2]):
            for j in range(self.ncells[1]):
                for i in range(self.ncells[0]):
                     # counter-clockwise
                    self.gn[iel, 0] = node(i, j, k)
                    self.gn[iel, 1] = node(i + 1, j, k)
                    self.gn[iel, 2] = node(i + 1, j + 1, k)
                    self.gn[iel, 3] = node(i, j + 1, k)
                    self.gn[iel, 4] = node(i, j, k + 1)
                    self.gn[iel, 5] = node(i + 1, j, k + 1)
                    self.gn[iel, 6] = node(i + 1, j + 1, k + 1)
                    self.gn[iel, 7] = node(i, j + 1, k + 1)
                    iel += 1


class FOS_FEM:

    def __init__(self, data, quad_degree):
        """
        Initialize the class with given data and quadrature degree.

        Parameters:
        - data: Provided data object containing mesh information
        - quad_degree: Quadrature degree for numerical integration
        """

        # Store the provided data
        self.data = data
        
        self.mu = self.data.mu

        # Determine the dimension of the problem
        self.dim_ = data.dim_

        # Store some shortcuts for frequently accessed data attributes
        self.ncells = [data.ncells[i] for i in range(self.dim_)]
        self.npts = [data.npts[i] for i in range(self.dim_)]
        self.deltas = [data.deltas[i] for i in range(self.dim_)]

        self.n_nodes = data.n_verts

        # Compute the continuous Finite Element (cFEM) basis functions
        self.basis()

        # Compute the elemental matrices for the given quadrature degree
        self.basis_q(quad_degree)
        
        self.sol_dir = data.T_dir
        self.dir_nodes = data.dir_nodes
        self.node_eqnId = data.node_eqnId



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



    def basis(self):

        dim_ = self.dim_

        # Create symbolic variables
        symbols = sp.symbols(f'u:{dim_}')
        
        self.b = []  # List to hold basis functions
        self.dbdxi = [[] for _ in range(dim_)]  # List of lists to hold derivatives
        
        # Generate basis functions and their derivatives
        for i in range(2**dim_):  # There are 2**dim_ vertices in a hypercube
            factors = [(1 + (-1)**((i >> j) & 1) * symbols[j]) / 2 for j in range(dim_)]
            basis_function = sp.Mul(*factors)
            self.b.append(sp.lambdify(symbols, basis_function, 'numpy'))
            
            # Generate derivatives
            for j, symbol in enumerate(symbols):
                derivative = sp.diff(basis_function, symbol)
                self.dbdxi[j].append(sp.lambdify(symbols, derivative, 'numpy'))



    def basis_q(self, quad_degree):
        """
        Compute the basis functions and their derivatives at the quadrature points.

        Parameter:
        - quad_degree: Degree of the Gauss-Legendre quadrature
        """

        dim_ = self.dim_

        # Use Gauss-Legendre quadrature to get the quadrature points 'xq' and weights 'wq' for the given degree
        xq, wq = np.polynomial.legendre.leggauss(quad_degree)
        
        # Create a list of quadrature points and weights for each dimension
        quad_points = list(product(xq, repeat=dim_))
        quad_weights = list(product(wq, repeat=dim_))
        
        nq = len(quad_points)
        # Initialize arrays to store values of basis functions and their derivatives at the quadrature points
        self.bq = np.zeros((nq, len(self.b)))
        self.dbdxiq = [np.zeros_like(self.bq) for _ in range(dim_)]
        self.w = np.zeros(nq)

        # Evaluate the basis functions and their derivatives at each quadrature point
        for q, (point, weights) in enumerate(zip(quad_points, quad_weights)):
            for i, (fi, *dfi) in enumerate(zip(self.b, *self.dbdxi)):
                self.bq[q, i] = fi(*point)
                for dim, derivative in enumerate(dfi):
                    self.dbdxiq[dim][q, i] = derivative(*point)
            self.w[q] = np.prod(weights)

    

    def eval_at_quadrature_points(self, T_prev, elem_glob_nodes):
        """
        Evaluate temperature and its derivative at the quadrature points using the FE basis functions.

        Parameters:
        - T_prev: Previous temperature field
        - elem_glob_nodes: Global node numbers associated with the current element

        Returns:
        - T_prev_q: Temperature values at the quadrature points
        - dT_prev_q: Temperature derivative values at the quadrature points
        """
        dim_ = self.dim_

        # Evaluate temperature at the quadrature points using the FE basis functions
        T_prev_q = np.dot(self.bq, T_prev[elem_glob_nodes])

        # Evaluate temperature derivatives at the quadrature points using the FE basis function derivatives
        dT_prev_q = [np.dot(self.dbdxiq[k], T_prev[elem_glob_nodes]) for k in range(dim_)]

        return T_prev_q, dT_prev_q  # Return temperature values and derivatives as a tuple



    def element_matrices(self, T_prev, iel):
        """
        Compute the element matrices and vectors for a given temperature field.

        Parameters:
        - cond_arr: Conductivity array
        - qext_arr: External heat source array
        - T_prev: Previous temperature field
        - iel: Current element index

        Returns:
        - Ke_: Element stiffness matrix
        - Je_: Element Jacobian matrix
        - qe_: Element source vector
        - Le_: Element matrix
        """

        dim_ = self.dim_
        # Retrieve material and geometric data for the current element
        mu = self.mu
        cell_idx = tuple(self.e_n_2ij(iel))
        
        imat = self.data.cell2mat_layout[cell_idx].astype(int)
        isrc = self.data.cell2src_layout[cell_idx].astype(int)
        
        k    = self.data.fdict["cond"][imat]
        dkdT = self.data.fdict["dcond"][imat]
        qext = self.data.fdict["qext"][isrc]
        dqext = self.data.fdict["dqext"][isrc]
        

        # Evaluate temperature and its derivative at quadrature points
        T_prev_q, dT_prev_q = self.eval_at_quadrature_points(T_prev, self.data.gn[iel, :])
        cond_q = k(T_prev_q, mu)
        dcond_q = dkdT(T_prev_q, mu)
        qext_q = qext(T_prev_q, mu)
        dqext_q = dqext(T_prev_q, mu)


        # Get global node numbers associated with the current element
        elem_glob_nodes = self.data.gn[iel, :]

        # Initialize element matrices and vectors
        n = len(elem_glob_nodes)
        Ke_ = np.zeros((n, n))                           
        Je_ = np.zeros((n, n))                           
        qe_ = np.zeros(n)

        vol = np.prod(np.array(self.deltas))/(2 ** dim_)  # Compute the coefficient

        if dim_ == 1:
            stiff_J_coeff = [2 / self.deltas[0]]
        elif dim_ == 2:
            stiff_J_coeff = [self.deltas[1] / self.deltas[0], self.deltas[0] / self.deltas[1]]
        else:
            stiff_J_coeff = [self.deltas[1]*self.deltas[2] / self.deltas[0], self.deltas[0]*self.deltas[2] / self.deltas[1], self.deltas[0]*self.deltas[1] / self.deltas[2]]


        for i in range(n):
            
            qe_[i] += vol * np.dot(self.w*self.bq[:,i], qext_q)

            for j in range(n):
                # Compute stiffness matrix entry for the current pair of nodes
                qe_temp = vol*np.dot(self.w*self.bq[:,i], dqext_q*self.bq[:,j])

                for k_ in range(dim_):
                    K_temp = stiff_J_coeff[k_]*np.dot(self.w * self.dbdxiq[k_][:, i], cond_q * self.dbdxiq[k_][:, j])
                    J_temp = stiff_J_coeff[k_]*np.dot(self.w * self.dbdxiq[k_][:, i], dcond_q * self.bq[:, j] * dT_prev_q[k_])

                    Ke_[i, j] += K_temp
                    # Je_[i, j] += J_temp-qe_temp
                    Je_[i, j] += J_temp

                Je_[i, j]-= qe_temp


        return Ke_, Je_, qe_


class HeatConductionSimulationData:
    
    def __init__(self, n_ref, L, quad_deg=3, num_snapshots=15, pb_dim=1, params = np.arange(1., 4.0, 0.01), T_init_guess = 273.0):
        
        # Initialize layout instance
        if pb_dim==1:
            from examples.heat_conduction.OneD_heat_conduction.FEM_1D_system_properties import SystemProperties
        elif pb_dim==2:
            from examples.heat_conduction.TwoD_heat_conduction.FEM_2D_system_properties import SystemProperties
        else:
            from examples.heat_conduction.ThreeD_heat_conduction.FEM_3D_system_properties import SystemProperties
                
        self.layout = SystemProperties(n_ref, params)
        self.n_ref = n_ref
        self.L = L
        self.quad_deg = quad_deg
        self.num_snapshots = num_snapshots
        # Use methods from the HeatConductionLayout instance
        self.mat_layout, self.src_layout = self.layout.create_layouts()
        self.fdict = self.layout.define_properties()
        self.bc = self.layout.define_boundary_conditions()
        self.params = self.layout.params
        self.NL_solutions = []
        self.param_list = []
        self.pb_dim=pb_dim
        self.fos_time = []
        self.rhs = []
        self.K_mus = []
        self.q_mus = []
        self.T_init_guess = T_init_guess
        self.train_mask, self.test_mask = train_test_split(num_snapshots)


    def run_simulation(self):
        
        random.seed(25)
        
        
        for i in range(self.num_snapshots):
            print(f"Snap {i}")
            param = random.choice(self.params)  # Choose from parameter list
            self.param_list.append(param)
            
            if i == 0:
                d = probdata(self.bc, self.mat_layout, self.src_layout, self.fdict, self.n_ref, self.L, param, self.pb_dim)
                self.FOS = FOS_FEM(d, self.quad_deg)
            else:
                self.FOS.mu = param
            
            T_init = np.zeros(d.n_verts) + self.T_init_guess
            
            tic_fos = time.time()
            NL_solution_p, Ke, rhs_e, _, rhs_ = solve_fos(self.FOS, T_init)
            toc_fos = time.time()
            
            self.fos_time.append(toc_fos-tic_fos)
            self.NL_solutions.append(NL_solution_p.flatten())
            self.rhs.append(rhs_)
            self.K_mus.append(Ke)
            self.q_mus.append(rhs_e)
            
            
class ROM_simulation:
    
    def __init__(self, f_cls, test_data, param_list, Test_mask, V_sel, xi, deim=None,T_init_guess = 273.15, N_rom_snap = None):
                
        self.FOS = f_cls.FOS
        self.T_init_guess = T_init_guess
        self.fos_test_data = test_data
        self.param_list_test = param_list[Test_mask]
        self.fos_test_time = np.asarray(f_cls.fos_time)[Test_mask]
        self.V_sel = V_sel
        self.deim = deim
        self.xi = xi
        self.quad_deg = f_cls.quad_deg
        self.NL_solutions_rom = []
        self.d = f_cls.FOS.data
        if N_rom_snap!=None:
            self.N_rom_snap = N_rom_snap
        else:
            self.N_rom_snap = len(self.param_list_test)



    def run_simulation_h_deim(self):

        import src.codes.reductor.rom_class as rom_class

        self.speed_up_h = []
        self.rom_error = []
        
        sol_fos_ = self.fos_test_data

        # Initial guess for temperature
        T_init_fos = np.zeros(self.FOS.n_nodes) + self.T_init_guess
        T_init_rom = np.transpose(self.V_sel) @ T_init_fos[self.FOS.data.mask]  # Initial guess in the reduced subspace
        sol_rom = np.zeros_like(T_init_fos)

        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            
            self.FOS.data.mu = self.param_list_test[i]
            
            tic_rom = time.time()
            ROM_h = rom_class.rom_deim(self.FOS.data, self.deim, self.quad_deg)
            NL_solution_p_reduced = ROM_h.solve_rom(T_init_rom,self.xi,self.V_sel)
            toc_rom = time.time()
            
            rom_sim_time = toc_rom - tic_rom
            self.speed_up_h.append(self.fos_test_time[i]/rom_sim_time)

            sol_rom[self.FOS.data.mask] = np.dot(self.V_sel,NL_solution_p_reduced)
            sol_rom[~self.FOS.data.mask] = self.FOS.data.T_dir

            self.NL_solutions_rom.append(sol_rom)

            sol_fos = sol_fos_[i]
            self.rom_error.append(np.linalg.norm(sol_rom[self.FOS.data.mask] - sol_fos) * 100 / np.linalg.norm(sol_fos))
           
    def run_simulation_h_ecsw(self):

        import src.codes.reductor.rom_class as rom_class

        self.speed_up_h = []
        self.rom_error = []
        
        sol_fos_ = self.fos_test_data

        # Initial guess for temperature
        T_init_fos = np.zeros(self.FOS.n_nodes) + self.T_init_guess
        T_init_rom = np.transpose(self.V_sel) @ T_init_fos[self.FOS.data.mask]  # Initial guess in the reduced subspace
        sol_rom = np.zeros_like(T_init_fos)

        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            
            self.FOS.data.mu = self.param_list_test[i]
            
            tic_rom = time.time()
            ROM_h = rom_class.rom_ecsw(self.FOS.data, self.quad_deg)
            NL_solution_p_reduced = ROM_h.solve_rom(T_init_rom,self.xi,self.V_sel)
            toc_rom = time.time()
            
            rom_sim_time = toc_rom - tic_rom
            self.speed_up_h.append(self.fos_test_time[i]/rom_sim_time)

            sol_rom[self.FOS.data.mask] = np.dot(self.V_sel,NL_solution_p_reduced)
            sol_rom[~self.FOS.data.mask] = self.FOS.data.T_dir

            self.NL_solutions_rom.append(sol_rom)

            sol_fos = sol_fos_[i]
            self.rom_error.append(np.linalg.norm(sol_rom[self.FOS.data.mask] - sol_fos) * 100 / np.linalg.norm(sol_fos))

    def run_simulation(self):

        import src.codes.reductor.rom_class as rom_class

        self.speed_up = []
        self.rom_error = []
        
        sol_fos_ = self.fos_test_data

        # Initial guess for temperature
        T_init_fos = np.zeros(self.FOS.n_nodes) + self.T_init_guess
        T_init_rom = np.transpose(self.V_sel) @ T_init_fos[self.FOS.data.mask]  # Initial guess in the reduced subspace
        sol_rom = np.zeros_like(T_init_fos)

        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            
            self.FOS.data.mu = self.param_list_test[i]

            tic_rom = time.time()
            ROM = rom_class.rom(self.FOS.data, self.quad_deg)
            NL_solution_p_reduced = ROM.solve_rom(T_init_rom, self.V_sel)
            toc_rom = time.time()
            
            rom_sim_time = toc_rom - tic_rom
            self.speed_up.append(self.fos_test_time[i]/rom_sim_time)

            sol_rom[self.FOS.data.mask] = np.dot(self.V_sel,NL_solution_p_reduced)
            sol_rom[~self.FOS.data.mask] = self.FOS.data.T_dir

            self.NL_solutions_rom.append(sol_rom)

            sol_fos = sol_fos_[i]
            self.rom_error.append(np.linalg.norm(sol_rom[self.FOS.data.mask] - sol_fos) * 100 / np.linalg.norm(sol_fos))


   

####
## Calculate stiffness matrix through vector operation 
# Calculate J^T@J
# Le_ = np.zeros((n, self.n_nodes))
# cell_coords = np.zeros((dim_,2**dim_))

# for i in range(n):
#     qe_[i] += coeff * np.dot(self.w*self.bq[:,i], qext_q)

#     for j in range(n):
#         for w_loop in range(len(self.w)):
#             Cxy = cell_coords.T@cell_coords
#             B = self.dbdxiq[0:dim_][w_loop,:]
#             B_basis = np.repeat(self.bq[w_loop,j],(dim_,1))
#             JTJ = B@Cxy@B.T
#             J = cell_coords@B.T
#             detJ = np.linalg.det(J)
#             K_e += dcond_q[w_loop]* B[:,i].T @np.inv(JTJ)@ B[:,i] *detJ*self.w[w_loop]
#             J_e += dcond_q[w_loop]* dT_prev_q[w_loop]* B[:,i].T@  np.inv(JTJ)@   B_basis.T*  detJ*  self.w[w_loop]