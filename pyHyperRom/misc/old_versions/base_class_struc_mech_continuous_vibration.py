from platform import node
from ..utils.fem_utils_StrucMech import *
from ..basic import *
import time 

class probdata:

    def __init__(self, bc, mat_layout, src_layout, fdict, nref, L, dof_p_node,node_p_elem, cv, cm, dt, t):
        
        pb_dim = 1
        self.dim_ = pb_dim
        self.dof_p_node = dof_p_node
        self.node_p_elem = node_p_elem
        self.elem_dof = dof_p_node*node_p_elem
        self.cv = cv
        self.cm =cm
        
            
        self.dt = dt
        self.t = t

        # refine the mesh and update material and source layouts
        if pb_dim>2:
            raise ValueError("For now pb_dim < = 1")

        # if(pb_dim==1):
            
        self.cell2mat_layout = mat_layout#.flatten()
        self.cell2src_layout = src_layout#.flatten()

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

            self.xi.append(np.linspace(0,L,self.npts[i]))
            self.deltas[i] = L/self.ncells[i]

            self.n_verts = self.npts[0]*dof_p_node


        # Create nodal connectivity for the continuous Finite Element Method (cFEM)
        self.connectivity()

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

        self.n_cells = self.ncells[0]
        
        for i in range(self.n_cells):
            get_element_global_nodes_and_nonzero_eqnId(self, i, self.node_eqnId)
                # Create a mask for nodes that do not have a Dirichlet boundary condition
        
        mask = self.node_eqnId != 0
        self.mask = mask

    def connectivity(self):
        """
        Define nodal connectivity for each cell in the mesh.
        """
        if self.dim_ == 1:
            self.connectivity_1d()
        # elif self.dim_ == 2:
        #     self.connectivity_2d()
        else:
            raise ValueError("Unsupported dimension")

    def connectivity_1d(self):
        # Initialize the connectivity array for 1D
        self.n_cells = self.ncells[0]
        self.gn = np.zeros((self.n_cells, self.elem_dof), dtype=int)
        self.gnodes = np.zeros((self.n_cells, self.node_p_elem), dtype=int)
	    # Loop over all cells to define their nodal connectivity

        for iel in range(self.n_cells):
	    # For each cell, define the left and right nodes
        
            dof_start = self.dof_p_node*(iel)
            dofs = np.r_[dof_start:dof_start + self.elem_dof]
            
            self.gn[iel, 0:self.elem_dof] = dofs 
            self.gnodes[iel, 0:self.node_p_elem] = self.gn[iel,::self.dof_p_node]/self.dof_p_node

class FOS_FEM:

    def __init__(self, data, quad_degree, tau, ep, T):
        """
        Initialize the class with given data and quadrature degree.

        Parameters:
        - data: Provided data object containing mesh information
        - quad_degree: Quadrature degree for numerical integration
        """

        # Store the provided data
        self.data = data
        
        self.tau = tau
        self.ep = ep
        self.T = T

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
        
        self.sol_dir = data.sol_dir
        self.dir_nodes = data.dir_nodes
        self.node_eqnId = data.node_eqnId

    def basis(self):

        dim_ = self.dim_

        # Create symbolic variables
        x = sp.symbols(f'x')
        
        self.b = []  # List to hold basis functions
        self.dbdxi = []  # List of lists to hold derivatives
        self.d2bdx2i = [] 

        shape_functions = [1/4*((1-x)**2)*(2+x), 1/4*((1-x)**2)*(1+x), 1/4*((1+x)**2)*(2-x), 1/4*((1+x)**2)*(x-1)]
        
        # Generate basis functions and their derivatives
        for bf in shape_functions:
            self.b.append(sp.lambdify(x, bf, 'numpy'))
            dbdxi = sp.diff(bf, x)
            self.dbdxi.append(sp.lambdify(x, dbdxi, 'numpy'))
            d2bdx2i = sp.diff(dbdxi, x)
            self.d2bdx2i.append(sp.lambdify(x, d2bdx2i, 'numpy'))

    def basis_q(self, quad_degree):
        """
        Compute the basis functions and their derivatives at the quadrature points.

        Parameter:
        - quad_degree: Degree of the Gauss-Legendre quadrature
        """

        # Use Gauss-Legendre quadrature to get the quadrature points 'xq' and weights 'wq' for the given degree
        xq, wq = np.polynomial.legendre.leggauss(quad_degree)
        self.xq = xq
        
        # Initialize arrays to store values of basis functions and their derivatives at the quadrature points
        nq = len(xq)
        self.bq = np.zeros((nq, len(self.b)))
        self.dbdxiq = np.zeros_like(self.bq)
        self.d2bdx2iq = np.zeros_like(self.bq)
        self.w = np.zeros(nq)

        # Evaluate the basis functions and their derivatives at each quadrature point
        for q, xi in enumerate(xq):

            for i, (fi, dfi, d2fi) in enumerate(zip(self.b, self.dbdxi, self.d2bdx2i)):

                self.bq[q, i] = fi(xi)
                self.dbdxiq[q, i] = dfi(xi)
                self.d2bdx2iq[q, i] = d2fi(xi)

            self.w[q] = wq[q]

    def element_KM_matrices_fn_x(self, iel, notconstant = False):
        """
        Compute the element matrices and vectors for a given temperature field.

        Parameters:
        - cond_arr: Conductivity array
        - fext_arr: External heat source array
        - sol_prev: Previous solution field
        - iel: Current element index

        Returns:
        - Ke_: Element stiffness matrix
        - Je_: Element Jacobian matrix
        - qe_: Element source vector
        - Le_: Element matrix
        """
        dx = self.data.deltas[0]
        mod_jac = dx/2  # Compute the coefficient
        # Get global node numbers associated with the current element
        elem_glob_nodes = self.data.gn[iel, :]
        n = len(elem_glob_nodes)

        if notconstant:

            
            w_factor = np.array([1, dx/2, 1, dx/2])

            imat = self.data.cell2mat_layout[iel].astype(int)

            bq = w_factor * self.bq
            self.bq_ = bq
            d2bdx2iq = w_factor * self.d2bdx2iq

            E = self.data.fdict["E"][imat]
            I = self.data.fdict["I"][imat]
            rho = self.data.fdict["rho"][imat]
            A = self.data.fdict["A"][imat]

            E_xq = E(self.xq)
            I_xq = I(self.xq)
            rho_xq = rho(self.xq)
            A_xq = A(self.xq)

            # Initialize element matrices and vectors
            n = len(elem_glob_nodes)
            Ke_ = np.zeros((n, n))
            Me_ = np.zeros((n, n))                       

            Jac_inv = (dx/2)**(-4)

            for i in range(n):
                for j in range(n):
                    # Compute stiffness matrix entry for the current pair of nodes
                    K_temp = Jac_inv*mod_jac*np.dot(self.w * d2bdx2iq[:,i], E_xq * I_xq * d2bdx2iq[:, j])
                    M_temp = mod_jac*np.dot(self.w * bq[:, i], rho_xq * A_xq * bq[:, j])

                    Ke_[i, j] += K_temp
                    Me_[i,j]  += M_temp

        else:

            h = self.data.deltas[0]

            E = self.data.fdict["E"][imat]
            I = self.data.fdict["I"][imat]
            rho = self.data.fdict["rho"][imat]
            A = self.data.fdict["A"][imat]

            # Element stiffness matrix
            Ke_ = np.array([[12, 6*h, -12, 6*h],
                                        [6*h, 4*h**2, -6*h, 2*h**2],
                                        [-12, -6*h, 12, -6*h],
                                        [6*h, 2*h**2, -6*h, 4*h**2]]) * (E * I / h**3)

            # Element mass matrix
            Me_ = np.array([[156, 22*h, 54, -13*h],
                                    [22*h, 4*h**2, 13*h, -3*h**2],
                                    [54, 13*h, 156, -22*h],
                                    [-13*h, -3*h**2, -22*h, 4*h**2]]) * (h * rho * A / 420)


        Ce_ = self.data.cv*Ke_ + self.data.cm*Me_
        
        return Ke_, Me_, Ce_
    
    def element_F_matrices_fn_x(self, iel, t=None):

        isrc = self.data.cell2src_layout[iel].astype(int)
        fext = self.data.fdict["fext"][isrc]

        mod_jac = self.deltas[0]/2  # Compute the coefficient

        # Get global node numbers associated with the current element
        elem_glob_dofs = self.data.gn[iel, :]
        elem_glob_nodes = self.data.gnodes[iel, :]
        
        n = len(elem_glob_dofs)
        # qe_ = np.zeros(n)
        qe_ = np.zeros((n,len(t)))
        x_iel = [self.data.xi[0][ii] for ii in elem_glob_nodes]
        fext_q = fext(self.xq, x_iel, t, self.tau, self.ep, self.T)
        # fext_q=fext_q.reshape(-1,1)


        for i in range(n):
            
            if i%2 == 0:
                
                qe_[i,:] += mod_jac * np.dot(self.w*self.bq_[:,i], fext_q)
                
        return qe_

class StructuralDynamicsSimulationData:
    
    def __init__(self, n_ref, L, T, params, dt, t, ep = 0.02, quad_deg=3, num_snapshots=15, pb_dim=1, cv=1e-2, cm = 1e-4):
        
        # Initialize layout instance
        from examples.structural_mechanics.continuous_vibrations.oneD_beam.SystemProperties import SystemProperties

        self.properties = SystemProperties(n_ref, cv, cm)
        self.cv = cv
        self.cm = cm
        self.n_ref = n_ref
        self.L = L
        self.quad_deg = quad_deg
        self.num_snapshots = num_snapshots
        self.mat_layout, self.src_layout = self.properties.create_layouts()
        self.fdict = self.properties.define_properties()
        self.bc = self.properties.define_boundary_conditions()
        self.params = params
        self.T = T
        self.ep = ep
        self.dt = dt
        self.t = t

        self.NL_solutions = []
        self.param_list = []
        self.pb_dim=pb_dim
        self.fos_time = []
        # self.rhs = []
        self.K_mus = []
        self.C_mus = []

        self.dof_p_node = 2
        self.node_p_elem = 2

        self.q_mus = []
        self.fos_ss = []
        # self.sol_init_guess = sol_init_guess
        # self.train_mask, self.test_mask = train_test_split(num_snapshots)

    def run_simulation(self):
        
        random.seed(25)
        
        for i in range(self.num_snapshots):
            print(f"Snap {i}")
            tau = random.choice(self.params)  # Choose from parameter list
            self.param_list.append(tau)
            
            if i == 0:
                d = probdata(self.bc, self.mat_layout, self.src_layout, self.fdict, self.n_ref, self.L, self.dof_p_node, self.node_p_elem, self.cv, self.cm, self.dt, self.t)                
                self.FOS = FOS_FEM(d, self.quad_deg, tau, self.ep, self.T)
            else:
                self.FOS.tau = tau
            
            sol_init = np.zeros(d.n_verts)
            
            tic_fos = time.time()
            t_out, x_out, rhs_e, Ke_d, Ce_d, mask, U = solve_fos_dynamics(self.FOS, sol_init, self.cv, self.cm)
            toc_fos = time.time()
            
            self.fos_time.append(toc_fos-tic_fos)
            self.NL_solutions.append(x_out)

            self.q_mus.append(rhs_e)
            self.K_mus.append(Ke_d)
            self.C_mus.append(Ce_d)
            
        self.t = t_out

class ROM_simulation:
    
    def __init__(self, f_cls, V_sel, xi=None, deim=None, sol_init_guess = 0, N_rom_snap = None):
    
        self.f_cls = f_cls
        self.sol_init_guess = sol_init_guess
        # self.fos_test_data = test_data
        # self.param_list_test = param_list[Test_mask]
        # self.fos_test_time = np.asarray(f_cls.fos_time)[Test_mask]
        self.V_sel = V_sel
        self.deim = deim
        self.xi = xi
        self.quad_deg = f_cls.quad_deg
        self.NL_solutions_rom = []
        self.d = f_cls.FOS.data
        self.N_rom_snap = 1
        self.t = f_cls.FOS.data.t

        # if N_rom_snap!=None:
        #     self.N_rom_snap = N_rom_snap
        # else:
        #     self.N_rom_snap = len(self.param_list_test)


    def run_simulation_h_deim(self):

        import src.codes.reductor.rom_class_StrucMech as rom_class

        self.speed_up_h = []
        self.rom_error = []
        
        sol_fos_ = self.fos_test_data

        # Initial guess for temperature
        sol_init_fos = np.zeros(self.FOS.n_nodes) + self.sol_init_guess
        sol_init_rom = np.transpose(self.V_sel) @ sol_init_fos  # Initial guess in the reduced subspace

        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            
            self.FOS.data.mu = self.param_list_test[i]
            
            tic_rom = time.time()
            ROM_h = rom_class.rom_deim(self.FOS.data, self.deim, self.quad_deg)
            NL_solution_p_reduced = ROM_h.solve_rom(sol_init_rom,self.xi,self.V_sel)
            toc_rom = time.time()
            
            rom_sim_time = toc_rom - tic_rom
            self.speed_up_h.append(self.fos_test_time[i]/rom_sim_time)

            sol_rom = np.dot(self.V_sel,NL_solution_p_reduced)
            self.NL_solutions_rom.append(sol_rom)

            sol_fos = sol_fos_[i]
            self.rom_error.append(np.linalg.norm(sol_rom - sol_fos) * 100 / np.linalg.norm(sol_fos))

            
    def run_simulation_h_ecsw(self):

        import src.codes.reductor.rom_class_StrucMech as rom_class

        self.speed_up_h = []
        self.rom_error = []
        
        sol_fos_ = self.fos_test_data

        # Initial guess for temperature
        sol_init_fos = np.zeros(self.FOS.n_nodes) + self.sol_init_guess
        sol_init_rom = np.transpose(self.V_sel) @ sol_init_fos  # Initial guess in the reduced subspace

        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            
            self.FOS.data.mu = self.param_list_test[i]
            
            tic_rom = time.time()
            ROM_h = rom_class.rom_ecsw(self.FOS.data, self.quad_deg)
            NL_solution_p_reduced = ROM_h.solve_rom(sol_init_rom,self.xi,self.V_sel)
            toc_rom = time.time()
            
            rom_sim_time = toc_rom - tic_rom
            self.speed_up_h.append(self.fos_test_time[i]/rom_sim_time)

            sol_rom = np.dot(self.V_sel,NL_solution_p_reduced)
            self.NL_solutions_rom.append(sol_rom)

            sol_fos = sol_fos_[i]
            self.rom_error.append(np.linalg.norm(sol_rom - sol_fos) * 100 / np.linalg.norm(sol_fos))


    def run_simulation(self):

        import src.codes.reductor.rom_class_StrucMech as rom_class

        self.speed_up = []
        self.rom_error = []
        
        # sol_fos_ = self.fos_test_data

        N_dir = self.V_sel.shape[0]
        N_full = 2*self.f_cls.FOS.data.n_verts

        # Initial guess for temperature
        sol_init_fos = np.zeros(N_dir)
        sol_init_rom = np.append(np.transpose(self.V_sel) @ sol_init_fos, np.transpose(self.V_sel) @ sol_init_fos)  # Initial guess in the reduced subspace

        for i in range(self.N_rom_snap):
            
            # self.FOS.data.mu = self.param_list_test[i]

            tic_rom = time.time()
            ROM = rom_class.rom(self.f_cls.FOS, self.quad_deg, self.f_cls.params[0], self.f_cls.ep, self.f_cls.T, self.f_cls.cv, self.f_cls.cm, self.xi)
            NL_solution_p_reduced = ROM.solve_rom(sol_init_rom, self.V_sel)
            toc_rom = time.time()

            shape_0, shape_1 = NL_solution_p_reduced.shape
            
            rom_sim_time = toc_rom - tic_rom
            # self.speed_up.append(self.fos_test_time[i]/rom_sim_time)

            # sol_rom_dir = np.dot(self.V_sel,NL_solution_p_reduced) # state-space formulation takes care of this step

            sol_rom_d = np.zeros((int(N_full/2),len(self.t)))
            sol_rom_d[self.d.mask] = np.dot(self.V_sel, NL_solution_p_reduced[:int(shape_0/2),:] )

            sol_rom_v= np.zeros_like(sol_rom_d)      
            sol_rom_v[self.d.mask] = np.dot(self.V_sel, NL_solution_p_reduced[int(shape_0/2):,:] )


            # sol_rom_full = np.vstack([sol_rom_d, sol_rom_v])
            self.NL_solutions_rom.append([sol_rom_d, sol_rom_v])

            # sol_fos = sol_fos_[i]
            # self.rom_error.append(np.linalg.norm(sol_rom - sol_fos) * 100 / np.linalg.norm(sol_fos))
