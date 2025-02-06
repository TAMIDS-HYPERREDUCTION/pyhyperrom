from platform import node
from ...utils.fem_utils_StrucMech import *
from ...basic import *
import time

class probdata:
    def __init__(self, bc, mat_layout, src_layout, fdict, nref, L, dof_p_node, node_p_elem, cv, cm, dt, t):
        """
        Initializes the problem data for a structural dynamical system.
        
        Parameters:
            bc: Boundary condition data.
            mat_layout: Matrix defining the material layout (distribution of materials across elements).
            src_layout: Matrix defining the source layout (distribution of external sources).
            fdict: Dictionary of functions defining material properties (e.g., stiffness, damping functions).
            nref: Not used in this implementation (can be removed or repurposed if needed).
            L (float): Length of the domain.
            dof_p_node (int): Degrees of freedom per node.
            node_p_elem (int): Number of nodes per element.
            cv: Viscous damping coefficient.
            cm: Material damping coefficient.
            dt (float): Time step for the simulation.
            t (float): Current simulation time.
        """
        self.dim_ = 1  # This implementation assumes a 1-dimensional system.
        self.dof_p_node = dof_p_node          # Degrees of freedom per node.
        self.node_p_elem = node_p_elem          # Number of nodes per element.
        self.elem_dof = dof_p_node * node_p_elem  # Total degrees of freedom per element.
        self.cv = cv  # Viscous damping coefficient.
        self.cm = cm  # Material damping coefficient.
        self.dt = dt  # Time step.
        self.t = t    # Current simulation time.

        # For a 1D system, ensure the dimension is appropriate.
        if self.dim_ > 2:
            raise ValueError("Dimension should be < 2")

        self.cell2mat_layout = mat_layout
        self.cell2src_layout = src_layout
        self.fdict = fdict

        # Setup mesh parameters based on the material layout.
        self.ncells = [mat_layout.shape[0]]
        self.n_cells = mat_layout.shape[0]
        self.npts = [self.ncells[0] + 1]
        self.deltas = [L / self.ncells[0]]
        self.xi = [np.linspace(0, L, self.npts[0])]

        # Total number of vertices = number of nodal points multiplied by degrees of freedom per node.
        self.n_verts = self.npts[0] * dof_p_node

        # Create nodal connectivity for the continuous Finite Element Method (cFEM).
        self.connectivity()

        # Process and apply Dirichlet boundary conditions.
        handle_boundary_conditions(self, bc)

        # Determine global equation numbers for nodes, taking into account Dirichlet conditions.
        get_glob_node_equation_id(self, self.dir_nodes)

        # Prepare element-level data (global and local node indices, connectivity, etc.).
        self.prepare_element_data()

        # Create a mask to indicate nodes that are free (i.e., do not have Dirichlet BC applied).
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
        
        Constructs:
            - self.gn: Matrix where each row contains the global degrees of freedom (dof indices) for an element.
            - self.gnodes: Matrix where each row contains the corresponding global node numbers.
        """
        self.gn = np.zeros((self.ncells[0], self.elem_dof), dtype=int)
        self.gnodes = np.zeros((self.ncells[0], self.node_p_elem), dtype=int)

        for iel in range(self.ncells[0]):
            dof_start = self.dof_p_node * iel
            dofs = np.arange(dof_start, dof_start + self.elem_dof)
            
            self.gn[iel] = dofs 
            self.gnodes[iel] = dofs[::self.dof_p_node] // self.dof_p_node

    def prepare_element_data(self):
        """
        Prepares element-level data for the mesh.
        
        This method collects and stores for each element:
            - Global node equation IDs.
            - Global nonzero equation IDs.
            - Local nonzero equation IDs.
            - Element connectivity (Le).
            - Global and local node indices.
        
        It iterates over all elements and calls an external function 
        (get_element_global_nodes_and_nonzero_eqnId) to perform this mapping.
        """
        self.glob_node_eqnId = []             # Global equation IDs per element.
        self.glob_node_nonzero_eqnId = []       # Global nonzero equation IDs per element.
        self.local_node_nonzero_eqnId = []      # Local nonzero equation IDs per element.
        self.Le = []                          # Element connectivity information.
        self.global_indices = []              # Global node indices for each element.
        self.local_indices = []               # Local node indices within each element.

        for i in range(self.n_cells):
            get_element_global_nodes_and_nonzero_eqnId(self, i, self.node_eqnId)

class FOS_FEM:
    def __init__(self, data, quad_degree, tau, ep, T):
        """
        Initialize the full-order finite element (FEM) solver for a structural dynamical system.
        
        Parameters:
            data: A data object containing mesh parameters, property layouts, boundary conditions, and a function dictionary.
            quad_degree: Degree for Gauss-Legendre quadrature used in numerical integration.
            tau: A parameter (e.g., relaxation or time constant) used in the external force function.
            ep: A parameter (e.g., small perturbation or regularization term) for the force function.
            T: Temperature or another state variable relevant to the simulation.
        """
        self.data = data
        self.tau = tau
        self.ep = ep
        self.T = T

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
        Compute the continuous FEM basis functions and their first and second derivatives.
        
        The basis functions are defined symbolically in terms of 'x' and then converted
        to numerical functions using sympy.lambdify.
        """
        x = sp.symbols('x')
        shape_functions = [
            1/4 * ((1 - x)**2) * (2 + x),
            1/4 * ((1 - x)**2) * (1 + x),
            1/4 * ((1 + x)**2) * (2 - x),
            1/4 * ((1 + x)**2) * (x - 1)
        ]
        self.b = [sp.lambdify(x, bf, 'numpy') for bf in shape_functions]
        self.dbdxi = [sp.lambdify(x, sp.diff(bf, x), 'numpy') for bf in shape_functions]
        self.d2bdx2i = [sp.lambdify(x, sp.diff(bf, x, x), 'numpy') for bf in shape_functions]

    def basis_q(self, quad_degree):
        """
        Compute the basis functions and their derivatives at the quadrature points.
        """
        xq, wq = np.polynomial.legendre.leggauss(quad_degree)
        self.xq = xq
        nq = len(xq)
        
        self.bq = np.array([[fi(xi) for fi in self.b] for xi in xq])
        self.dbdxiq = np.array([[dfi(xi) for dfi in self.dbdxi] for xi in xq])
        self.d2bdx2iq = np.array([[d2fi(xi) for d2fi in self.d2bdx2i] for xi in xq])
        self.w = wq

    def element_KM_matrices_fn_x(self, iel, notconstant=False):
        """
        Compute the element stiffness (Ke_) and mass (Me_) matrices for element iel,
        and then form the combined damping matrix (Ce_) using material damping (cm)
        and viscous damping (cv).

        If notconstant is True, material properties (E, I, rho, A) vary with the spatial coordinate x,
        and numerical integration is performed using the evaluated basis functions.
        Otherwise, constant properties are assumed and analytical expressions are used.

        Returns:
            Ke_: Element stiffness matrix.
            Me_: Element mass matrix.
            Ce_: Damping matrix computed as: (cm * Ke_) + (cv * Me_).
                 Here, 'cv' represents viscous damping and 'cm' represents material damping.
        """
        dx = self.data.deltas[0]
        mod_jac = dx / 2
        elem_glob_nodes = self.data.gn[iel, :]

        imat = self.data.cell2mat_layout[iel].astype(int)
        E, I, rho, A = self.data.fdict["E"][imat], self.data.fdict["I"][imat], \
                         self.data.fdict["rho"][imat], self.data.fdict["A"][imat]

        n = len(elem_glob_nodes)
        Ke_, Me_ = np.zeros((n, n)), np.zeros((n, n))

        if notconstant:
            w_factor = np.array([1, dx / 2, 1, dx / 2])
            bq = w_factor * self.bq
            self.bq_ = bq
            d2bdx2iq = w_factor * self.d2bdx2iq

            E_xq, I_xq, rho_xq, A_xq = E(self.xq), I(self.xq), rho(self.xq), A(self.xq)
            Jac_inv = (dx / 2) ** (-4)

            for i in range(n):
                for j in range(n):
                    K_temp = Jac_inv * mod_jac * np.dot(self.w * d2bdx2iq[:, i], E_xq * I_xq * d2bdx2iq[:, j])
                    M_temp = mod_jac * np.dot(self.w * bq[:, i], rho_xq * A_xq * bq[:, j])
                    Ke_[i, j] += K_temp
                    Me_[i, j] += M_temp
        else:
            Ke_ = ((E * I / dx**3) * np.array([[12, 6 * dx, -12, 6 * dx],
                                                [6 * dx, 4 * dx**2, -6 * dx, 2 * dx**2],
                                                [-12, -6 * dx, 12, -6 * dx],
                                                [6 * dx, 2 * dx**2, -6 * dx, 4 * dx**2]]))
            Me_ = ((dx * rho * A / 420) * np.array([[156, 22 * dx, 54, -13 * dx],
                                                     [22 * dx, 4 * dx**2, 13 * dx, -3 * dx**2],
                                                     [54, 13 * dx, 156, -22 * dx],
                                                     [-13 * dx, -3 * dx**2, -22 * dx, 4 * dx**2]]))
        Ce_ = self.data.cm * Ke_ + self.data.cv * Me_
        return Ke_, Me_, Ce_

    def element_F_matrices_fn_x(self, iel, t=None):
        """
        Compute the element force vector (qe_) for element iel.
        
        This function evaluates the external force using the function fext.
        The evaluation is performed at quadrature points and integrated over the element.
        
        Parameters:
            iel: Index of the current element.
            t: Time vector at which the force is evaluated.
        
        Returns:
            qe_: Element force vector.
                 (Note: 'qe' is computed without applying Dirichlet boundary conditions.)
        """
        isrc = self.data.cell2src_layout[iel].astype(int)
        fext = self.data.fdict["fext"][isrc]
        mod_jac = self.deltas[0] / 2

        elem_glob_dofs = self.data.gn[iel, :]
        elem_glob_nodes = self.data.gnodes[iel, :]
        
        n = len(elem_glob_dofs)
        qe_ = np.zeros((n, len(t)))

        x_iel = self.data.xi[0][elem_glob_nodes]
       
        fext_q = fext(self.xq, x_iel, t, self.tau, self.ep, self.T)

        for i in range(n):
            if i % 2 == 0:
                qe_[i, :] += mod_jac * np.dot(self.w * self.bq_[:, i], fext_q)
                
        return qe_

    def residual_func(self, i, j, p_sol_d, p_sol_v, data):
        """
        Compute the residual vector for the dynamical system at an element level.
        
        The residual is computed as:
            res = K_mus * p_sol_d + C_mus * p_sol_v - q_mus,
        where:
            - K_mus is the element stiffness matrix,
            - C_mus is the element damping matrix,
            - q_mus is the external force vector,
            - p_sol_d is the displacement component of the solution, and
            - p_sol_v is the velocity component of the solution.
        
        Parameters:
            i: Snapshot (or time step) index.
            j: Element index.
            p_sol_d: Displacement component of the solution.
            p_sol_v: Velocity component of the solution.
            data: Dictionary containing full-order system matrices and force terms.
        
        Returns:
            res: Residual vector for the specified element and snapshot.
        """
        K_mus = data['K_mus']
        q_mus = data['q_mus']
        C_mus = data['C_mus']

        K_mus_ij = K_mus[0][j]
        C_mus_ij = C_mus[0][j]
        q_mus_ij = np.array(q_mus[0][j][:, i])
        
        res = np.dot(K_mus_ij, p_sol_d) + np.dot(C_mus_ij, p_sol_v) - q_mus_ij

        return res
    
    def residual_func_p(self, i, j, k, p_sol_d, p_sol_v, data, train_mask_t):
        """
        Compute the residual vector at a specific Gauss point for a given snapshot.
        
        The residual is computed as:
            res = K_mus * p_sol_d + C_mus * p_sol_v - q_mus,
        where q_mus is indexed using a training mask.
        
        Parameters:
            i: Index of the component (e.g., displacement) in the reduced system.
            j: Element index.
            k: Gauss point index within the element.
            p_sol_d: Displacement component of the solution.
            p_sol_v: Velocity component of the solution.
            data: Dictionary containing full-order system matrices and force terms.
            train_mask_t: Mask used to select specific entries from the force term.
        
        Returns:
            res: Residual vector at the specified Gauss point.
        """
        K_mus = data['K_mus']
        q_mus = data['q_mus']
        C_mus = data['C_mus']

        K_mus_ij = K_mus[k][j]
        C_mus_ij = C_mus[k][j]
        q_mus_ij_masked = np.array(q_mus[k][j][:, train_mask_t])
        q_mus_ij = q_mus_ij_masked[:, i]
        res = np.dot(K_mus_ij, p_sol_d) + np.dot(C_mus_ij, p_sol_v) - q_mus_ij

        return res

class StructuralDynamicsSimulationData:
    def __init__(self, n_ref, L, T, params, dt, t, ep=0.02, quad_deg=3, num_snapshots=1, pb_dim=1, cv=1e-2, cm=1e-4):
        """
        Initialize the simulation data for a structural dynamical system.

        Parameters:
            n_ref: Reference grid refinement for the problem.
            L (float): Length of the domain.
            T: Time period of the forcing function.
            params: List of parameters (tau) for different snapshots.
            dt (float): Time step size.
            t (float): Total simulation time.
            ep (float, optional): Small regularization parameter. Defaults to 0.02.
            quad_deg (int, optional): Quadrature degree for numerical integration. Defaults to 3.
            num_snapshots (int, optional): Number of snapshots for parameter variations. Defaults to 1.
            pb_dim (int, optional): Problem dimension. Defaults to 1.
            cv (float, optional): Viscous damping coefficient. Defaults to 1e-2.
            cm (float, optional): Material damping coefficient. Defaults to 1e-4.
        """
        from examples.structural_mechanics.Transverse.continuous_vibrations.oneD_beam.SystemProperties import SystemProperties

        # Initialize system properties
        self.properties = SystemProperties(n_ref, cv, cm)
        self.cv, self.cm = cv, cm
        self.n_ref, self.L, self.T = n_ref, L, T
        self.quad_deg, self.num_snapshots = quad_deg, num_snapshots

        # Generate material and force layouts
        self.mat_layout, self.src_layout = self.properties.create_layouts()
        self.fdict = self.properties.define_properties()
        self.bc = self.properties.define_boundary_conditions()

        self.params, self.ep, self.dt, self.t = params, ep, dt, t

        # Storage for simulation results
        self.NL_solutions = []
        self.param_list = []
        self.fos_time = []
        self.K_mus = []
        self.C_mus = []
        self.q_mus = []
        
        # Degrees of freedom settings
        self.dof_p_node = 2
        self.node_p_elem = 2

    def run_simulation(self):
        """
        Run the simulation for each snapshot with different parameters.
        """
        random.seed(25)

        for i in range(self.num_snapshots):
            print(f"Running snapshot {i}")

            # Store current parameter
            tau = self.params[i]
            self.param_list.append(tau)

            # Initialize problem data and FEM solver
            if i == 0:
                d = probdata(self.bc, self.mat_layout, self.src_layout, self.fdict,
                             self.n_ref, self.L, self.dof_p_node, self.node_p_elem,
                             self.cv, self.cm, self.dt, self.t)
                self.FOS = FOS_FEM(d, self.quad_deg, tau, self.ep, self.T)
            else:
                self.FOS.tau = tau  # Update tau for subsequent snapshots

            # Initial condition
            sol_init = np.zeros(d.n_verts)

            # Run full-order solver
            start_time = time.time()
            t_out, x_out, rhs_e, Ke_d, Ce_d, mask, U, fom = solve_fos_dynamics(self.FOS, sol_init, self.cv, self.cm)
            end_time = time.time()

            # Store results
            self.fos_time.append(end_time - start_time)
            self.NL_solutions.append(x_out)
            self.q_mus.append(rhs_e)
            self.K_mus.append(Ke_d)
            self.C_mus.append(Ce_d)
            self.fom = fom

        # Update simulation time with last computed value
        self.t = t_out

class ROM_simulation_non_p:
    
    def __init__(self, f_cls, V_sel, xi=None, deim=None, sol_init_guess=0, N_rom_snap=1):

        """
        Initialize the Reduced Order Model (ROM) simulation class.
        """
        self.f_cls = f_cls
        self.V_sel = V_sel
        self.deim = deim
        self.xi = xi
        self.quad_deg = f_cls.quad_deg
        self.NL_solutions_rom = []
        self.d = f_cls.FOS.data
        self.t = f_cls.FOS.data.t
        self.N_rom_snap = N_rom_snap
        self.sol_init_guess = sol_init_guess
        self.speed_up = []

    def run_simulation(self):
        """
        Run the ROM simulation.
        """
        import src.codes.reductor.rom_class_StrucMech as rom_class

        N_dir = self.V_sel.shape[0]
        N_full = 2 * self.d.n_verts

        sol_init_fos = np.zeros(N_dir)
        sol_init_rom = np.append(np.dot(self.V_sel.T, sol_init_fos), np.dot(self.V_sel.T, sol_init_fos))

        for i in range(self.N_rom_snap):
            tic_rom = time.time()
            ROM = rom_class.rom(self.f_cls, self.quad_deg, self.f_cls.params[0], self.f_cls.ep, self.f_cls.T, self.f_cls.cv, self.f_cls.cm, self.xi)
            NL_solution_p_reduced = ROM.solve_rom(sol_init_rom, self.V_sel)
            toc_rom = time.time()

            rom_sim_time = toc_rom - tic_rom

            shape_0 = NL_solution_p_reduced.shape[0]
            sol_rom_d = np.zeros((int(N_full / 2), len(self.t)))
            sol_rom_v =  np.zeros_like(sol_rom_d)
            sol_rom_d[self.d.mask] = np.dot(self.V_sel, NL_solution_p_reduced[:int(shape_0 / 2), :])
            sol_rom_v[self.d.mask] = np.dot(self.V_sel, NL_solution_p_reduced[int(shape_0 / 2):, :])
            
            self.NL_solutions_rom.append([sol_rom_d, sol_rom_v])
            self.speed_up.append(self.f_cls.fos_time[i]/rom_sim_time)

class ROM_simulation_p:
    """
    Class for running parametric Reduced Order Model (ROM) simulations for a structural
    dynamical system. This class projects the full-order system onto a reduced basis,
    solves the reduced system, and reconstructs the full-order displacement and velocity fields.
    """
    def __init__(self, f_cls, V_sel, xi=None, deim=None, sol_init_guess=0, N_rom_snap=1):
        """
        Initialize the ROM simulation class.

        Parameters:
            f_cls: Full-order simulation class instance (contains FOS and associated data).
            V_sel: Selected reduced basis matrix.
            xi: (Optional) Sampling indices (used in hyper-reduction techniques).
            deim: (Optional) DEIM operator for hyper-reduction (if applicable).
            sol_init_guess: Initial guess for the full-order solution (default is 0).
            N_rom_snap: Number of ROM snapshots to simulate.
        """
        self.f_cls = f_cls                          # Store the full-order simulation class instance.
        self.V_sel = V_sel                          # Store the reduced basis matrix.
        self.deim = deim                            # Store the DEIM operator, if provided.
        self.xi = xi                                # Store the sampling indices.
        self.quad_deg = f_cls.quad_deg              # Retrieve the quadrature degree from the full-order system.
        self.NL_solutions_rom = []                  # List to store reconstructed ROM solutions (displacement & velocity).
        self.d = f_cls.FOS.data                     # Full-order simulation data (mesh, mask, etc.).
        self.t = f_cls.FOS.data.t                   # Time array from the full-order simulation.
        self.N_rom_snap = N_rom_snap                # Number of ROM snapshots to simulate.
        self.sol_init_guess = sol_init_guess        # Initial guess for the full-order solution.
        self.speed_up = []                          # List to record speed-up factors for each ROM snapshot.

    def run_simulation(self):
        """
        Run the ROM simulation for the given number of snapshots.

        Steps:
            1. Determine the dimensions of the reduced and full-order systems.
            2. Project an initial full-order guess into the reduced space.
            3. For each snapshot:
                a. Instantiate the ROM solver with the appropriate parameter.
                b. Solve the reduced system from the projected initial guess.
                c. Reconstruct the full-order displacement and velocity fields.
                d. Compute and record the simulation speed-up.
                e. Store the reconstructed solution.
        """
        import time
        import src.codes.reductor.rom_class_StrucMech as rom_class

        # Determine the dimension of the reduced space (number of rows in V_sel)
        N_dir = self.V_sel.shape[0]
        # Full-order system dimension is twice the number of vertices (displacement and velocity)
        N_full = 2 * self.d.n_verts

        # Create an initial guess in the reduced space as a zero vector
        sol_init_fos = np.zeros(N_dir)
        # Project the initial guess into the reduced space for both displacement and velocity parts
        sol_init_rom = np.append(np.dot(self.V_sel.T, sol_init_fos),
                                 np.dot(self.V_sel.T, sol_init_fos))

        # Loop over the number of ROM snapshots
        for i in range(self.N_rom_snap):
            tic_rom = time.time()  # Start timer for ROM solve

            # Instantiate the ROM solver using the current parameter value from f_cls
            ROM = rom_class.rom(
                self.f_cls, self.quad_deg, self.f_cls.params[i],
                self.f_cls.ep, self.f_cls.T, self.f_cls.cv, self.f_cls.cm, self.xi
            )
            # Solve the reduced-order model starting from the projected initial guess
            NL_solution_p_reduced = ROM.solve_rom(sol_init_rom, self.V_sel)
            
            toc_rom = time.time()  # End timer for ROM solve
            rom_sim_time = toc_rom - tic_rom  # Compute the time taken for the ROM solve

            # Get the number of rows in the reduced solution (should be 2 * N_dir)
            shape_0 = NL_solution_p_reduced.shape[0]
            # Allocate arrays for full-order displacement (sol_rom_d) and velocity (sol_rom_v) fields
            sol_rom_d = np.zeros((int(N_full / 2), len(self.t)))
            sol_rom_v = np.zeros_like(sol_rom_d)
            # Reconstruct the displacement field by multiplying the reduced basis with the first half of NL_solution_p_reduced
            sol_rom_d[self.d.mask] = np.dot(self.V_sel, NL_solution_p_reduced[:int(shape_0 / 2), :])
            # Reconstruct the velocity field using the second half of NL_solution_p_reduced
            sol_rom_v[self.d.mask] = np.dot(self.V_sel, NL_solution_p_reduced[int(shape_0 / 2):, :])
            
            # Store the reconstructed solution (as a list containing displacement and velocity)
            self.NL_solutions_rom.append([sol_rom_d, sol_rom_v])
            # Compute and record the speed-up factor (full-order simulation time / ROM simulation time)
            self.speed_up.append(self.f_cls.fos_time[i] / rom_sim_time)

class ROM_simulation_p_UQ:
    """
    This class performs parametric Reduced Order Modeling (ROM) simulations under uncertainty quantification (UQ) conditions.
    For each parameter in the provided list, it solves the reduced system, reconstructs the full-order displacement and velocity fields,
    and optionally plots the results for the last period of the simulation.
    """
    def __init__(self, f_cls, V_sel, param_list, xi=None, deim=None, sol_init_guess=0, N_rom_snap=1, ax=None):
        """
        Initialize the ROM simulation class for UQ.

        Parameters:
            f_cls: Instance of the full-order simulation class (contains FOS and all associated data).
            V_sel: The reduced basis matrix.
            param_list: List or array of parameter values to be used for the ROM simulations.
            xi: (Optional) Sampling indices for hyper-reduction techniques (e.g., DEIM).
            deim: (Optional) DEIM operator if hyper-reduction is applied.
            sol_init_guess: Initial guess for the full-order solution (default is 0).
            N_rom_snap: Number of ROM snapshots to simulate.
            ax: (Optional) Array of matplotlib axes for plotting; if provided, ROM results for selected nodes
                over one period will be plotted.
        """
        self.f_cls = f_cls                                # Full-order simulation class instance.
        self.V_sel = V_sel                                # Reduced basis matrix.
        self.deim = deim                                  # DEIM operator, if applicable.
        self.xi = xi                                      # Sampling indices for hyper-reduction.
        self.quad_deg = f_cls.quad_deg                    # Quadrature degree from the full-order simulation.
        
        # Lists to store ROM simulation outputs for one period.
        self.NL_solutions_rom_UQ_1period_0p62 = []         # For a specific node (index 124) for one period.
        self.NL_solutions_rom_UQ_1p = []                   # For a subset of nodes (sampled every 2nd node) for one period.
        
        self.d = f_cls.FOS.data                           # Full-order simulation data (includes mesh, mask, etc.).
        self.t = f_cls.FOS.data.t                         # Time array from the full-order simulation.
        self.N_rom_snap = N_rom_snap                      # Number of ROM snapshots to simulate.
        self.sol_init_guess = sol_init_guess              # Initial guess for the solution.
        self.speed_up = []                                # List to record speed-up factors (if computed).
        self.param_list = param_list                      # List of parameters for the ROM simulation.
        self.ax = ax                                      # Optional axes for plotting.

    def run_simulation(self):
        """
        Run the parametric ROM simulation for UQ.
        
        Procedure:
          1. Determine dimensions: the reduced space dimension (N_dir) and full-order space dimension (N_full).
          2. Form an initial reduced guess by projecting a zero vector (or provided guess) into the reduced space.
          3. For each snapshot (for each parameter in the list):
             a. Instantiate the ROM solver with the current parameter.
             b. Solve the reduced system from the initial reduced guess.
             c. Reconstruct the full-order displacement and velocity fields by projecting the reduced solution
                back using the reduced basis.
             d. Extract the last period of the solution (number of time steps corresponding to one period)
                and, if plotting axes are provided, plot the results.
             e. Store the reconstructed solutions for later analysis.
        """
        import src.codes.reductor.rom_class_StrucMech as rom_class

        # Determine the reduced space dimension (N_dir = number of rows in V_sel)
        N_dir = self.V_sel.shape[0]
        # The full-order system includes displacement and velocity components.
        N_full = 2 * self.d.n_verts

        # Create an initial guess in the reduced space as a zero vector.
        sol_init_fos = np.zeros(N_dir)
        # Project the zero guess into the reduced space for both displacement and velocity parts.
        sol_init_rom = np.append(np.dot(self.V_sel.T, sol_init_fos),
                                 np.dot(self.V_sel.T, sol_init_fos))

        # Compute the number of time steps corresponding to one period.
        # The period is assumed to be given by f_cls.T; time step size is deduced from the time array.
        num_steps_period = int(self.f_cls.T / (self.t[1] - self.t[0]) + 1)

        # Loop over the number of ROM snapshots (each corresponding to a parameter in param_list)
        for i in range(self.N_rom_snap):
            # Instantiate the ROM solver with the current parameter.
            ROM = rom_class.rom(
                self.f_cls, self.quad_deg, self.param_list[i],
                self.f_cls.ep, self.f_cls.T, self.f_cls.cv, self.f_cls.cm, self.xi
            )
            # Solve the reduced system using the initial reduced guess.
            NL_solution_p_reduced = ROM.solve_rom(sol_init_rom, self.V_sel)
            
            # Determine the shape of the reduced solution; it should be 2*N_dir (first half: displacement, second half: velocity)
            shape_0 = NL_solution_p_reduced.shape[0]
            # Allocate arrays for full-order displacement and velocity fields over the full time horizon.
            sol_rom_d = np.zeros((int(N_full / 2), len(self.t)))
            sol_rom_v = np.zeros_like(sol_rom_d)
            # Reconstruct the displacement field from the reduced solution.
            sol_rom_d[self.d.mask] = np.dot(self.V_sel, NL_solution_p_reduced[:int(shape_0 / 2), :])
            # Reconstruct the velocity field from the reduced solution.
            sol_rom_v[self.d.mask] = np.dot(self.V_sel, NL_solution_p_reduced[int(shape_0 / 2):, :])

            # If plotting axes are provided, plot the reconstructed results for a selected node and a subset of nodes.
            if self.ax is not None:
                # Plot for node index 124: extract the last 'num_steps_period' time steps.
                self.ax[0].plot(self.t[-num_steps_period:], sol_rom_d[124, -num_steps_period:], color='grey', linewidth=0.05)
                self.ax[0].set_xlabel('$t$')
                self.ax[0].set_ylabel('$w(0.25,t)$')

                self.ax[1].plot(self.t[-num_steps_period:], sol_rom_v[124, -num_steps_period:], color='grey', linewidth=0.05)
                self.ax[1].set_xlabel('$t$')
                self.ax[1].set_ylabel('$\dot{w}(0.25,t)$')

                # Store the last period's displacement and velocity for node 124.
                self.NL_solutions_rom_UQ_1period_0p62.append([
                    sol_rom_d[124, -num_steps_period:], 
                    sol_rom_v[124, -num_steps_period:]
                ])
                # Store the reconstructed solutions for a subset of nodes (sample every 2nd node) for the last period.
                self.NL_solutions_rom_UQ_1p.append([
                    sol_rom_d[::2, -num_steps_period:], 
                    sol_rom_v[::2, -num_steps_period:]
                ])
            else:
                # If no plotting axes are provided, still store the solutions for further analysis.
                self.NL_solutions_rom_UQ_1period_0p62.append([
                    sol_rom_d[124, -num_steps_period:], 
                    sol_rom_v[124, -num_steps_period:]
                ])
                self.NL_solutions_rom_UQ_1p.append([
                    sol_rom_d[::2, -num_steps_period:], 
                    sol_rom_v[::2, -num_steps_period:]
                ])

            # Optionally, speed-up factors could be computed if full-order simulation times are available.
            # (This line is commented out as per the original code.)
            # self.speed_up.append(self.f_cls.fos_time[i] / rom_sim_time)
