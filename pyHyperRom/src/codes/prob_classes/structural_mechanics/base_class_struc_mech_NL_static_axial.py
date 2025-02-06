from platform import node
from ...utils.fem_utils_StrucMech import *
from ...basic import *
import time
class probdata:
    def __init__(self, bc, mat_layout, src_layout, fdict, nref, L, dof_p_node, node_p_elem, expon):
        """
        Initializes the problem data for a nonlinear structural mechanics problem.
        
        Parameters:
            bc: Boundary condition data.
            mat_layout: Matrix defining the material layout (distribution of materials over the domain).
            src_layout: Matrix defining the force layout (distribution of applied forces).
            fdict: Dictionary of functions defining material properties and nonlinear behavior.
            nref: (Not used) Reference parameter that can be repurposed.
            L (float): Length of the domain.
            dof_p_node (int): Degrees of freedom per node.
            node_p_elem (int): Number of nodes per element.
            expon: Exponent parameter for the nonlinearity.
        """
        self.dim_ = 1  # This implementation assumes a 1-dimensional problem.
        self.dof_p_node = dof_p_node          # Store the degrees of freedom per node.
        self.node_p_elem = node_p_elem          # Store the number of nodes per element.
        self.elem_dof = dof_p_node * node_p_elem  # Compute the total DOFs per element.

        # For a 1D problem, ensure that the dimension is 1.
        if self.dim_ > 2:
            raise ValueError("Dimension should be 1")

        # Store the material and force layout matrices.
        self.cell2mat_layout = mat_layout
        self.cell2src_layout = src_layout
        # Store the nonlinearity exponent.
        self.expon = expon
        # Store the dictionary of property functions.
        self.fdict = fdict
        # Store the length of the domain.
        self.L = L

        # Set up the mesh parameters.
        # The number of cells is determined by the first dimension (number of rows) of the material layout.
        self.ncells = [mat_layout.shape[0]]
        self.n_cells = mat_layout.shape[0]
        # The number of nodal points is one more than the number of cells.
        self.npts = [self.ncells[0] + 1]
        # Compute the size of each cell.
        self.deltas = [L / self.ncells[0]]
        # Create a 1D array of nodal coordinates linearly spaced from 0 to L.
        self.xi = [np.linspace(0, L, self.npts[0])]

        # Compute the total number of vertices in the mesh (each node has dof_p_node degrees of freedom).
        self.n_verts = self.npts[0] * dof_p_node

        # Establish the nodal connectivity for the continuous FEM formulation.
        self.connectivity()

        # Apply the boundary conditions for the static problem.
        handle_boundary_conditions_statics(self, bc)

        # Determine and assign global equation numbers for nodes, taking into account Dirichlet conditions.
        get_glob_node_equation_id(self, self.dir_nodes)

        # Prepare element-specific data such as global and local node indices and connectivity.
        self.prepare_element_data()

        # Create a mask to indicate nodes that are free (i.e., do not have Dirichlet boundary conditions).
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
        Defines the nodal connectivity for a 1D mesh.
        
        Constructs two matrices:
            - self.gn: Contains the global DOF indices for each element.
            - self.gnodes: Contains the global node numbers for each element.
        """
        # Initialize connectivity matrices with zeros.
        self.gn = np.zeros((self.ncells[0], self.elem_dof), dtype=int)
        self.gnodes = np.zeros((self.ncells[0], self.node_p_elem), dtype=int)

        # Loop over each cell (element) in the 1D domain.
        for iel in range(self.ncells[0]):
            # Compute the starting DOF index for the current element.
            dof_start = self.dof_p_node * iel
            # Create an array of consecutive DOF indices for the element.
            dofs = np.arange(dof_start, dof_start + self.elem_dof)
            
            # Assign the DOF indices to the connectivity matrix.
            self.gn[iel] = dofs
            # Determine the global node numbers by selecting every dof_p_node-th index and normalizing.
            self.gnodes[iel] = dofs[::self.dof_p_node] // self.dof_p_node

    def prepare_element_data(self):
        """
        Prepares element-level data required for assembling the FEM system.
        
        This method collects, for each element, the following:
            - Global node equation IDs.
            - Global nonzero equation IDs.
            - Local nonzero equation IDs.
            - Element connectivity (Le).
            - Global and local node indices.
        It iterates over all elements and populates these data structures by calling an external function.
        """
        self.glob_node_eqnId = []             # List to store global equation IDs per element.
        self.glob_node_nonzero_eqnId = []       # List to store global nonzero equation IDs per element.
        self.local_node_nonzero_eqnId = []      # List to store local nonzero equation IDs per element.
        self.Le = []                          # List to store element connectivity information.
        self.global_indices = []              # List to store global node indices for each element.
        self.local_indices = []               # List to store local node indices for each element.

        # Loop over each element in the mesh.
        for i in range(self.n_cells):
            # Populate the element data by calling an external function.
            get_element_global_nodes_and_nonzero_eqnId(self, i, self.node_eqnId)

class FOS_FEM:
    def __init__(self, data, quad_degree, tau):
        """
        Initialize the full-order solver (FOS) for the nonlinear structural mechanics problem.

        Parameters:
            data: Data object containing mesh information, layouts, boundary conditions, and function dictionary.
            quad_degree: Degree for Gauss-Legendre quadrature.
            tau: Parameter array used for the nonlinear functions.
        """
        self.data = data                      # Store the problem data.
        self.tau = tau                        # Parameter (e.g., for material or force nonlinearity).
        self.expon = data.expon               # Exponent for the nonlinearity.
        self.dim_ = data.dim_                 # Problem dimension (assumed 1D here).
        self.ncells = data.ncells.copy()      # Number of cells in the mesh.
        self.npts = data.npts.copy()          # Number of nodal points.
        self.deltas = data.deltas.copy()      # Element sizes.
        self.n_nodes = data.n_verts           # Total number of vertices.
        self.sol_dir = data.sol_dir           # Prescribed Dirichlet solution (if any).
        self.dir_nodes = data.dir_nodes       # Nodes with Dirichlet boundary conditions.
        self.node_eqnId = data.node_eqnId     # Global equation numbering for nodes.

        # Compute the FE basis functions and their derivatives.
        self.basis()
        self.basis_q(quad_degree)

    def basis(self):
        """
        Compute the continuous FEM basis functions for a 1D element and their derivatives.

        Here we use linear shape functions:
            N1(x) = 1/2 * (1 - x)
            N2(x) = 1/2 * (1 + x)
        """
        x = sp.symbols('x')
        shape_functions = [1/2 * (1 - x), 1/2 * (1 + x)]
        # Convert symbolic expressions to numerical functions using lambdify.
        self.b = [sp.lambdify(x, bf, 'numpy') for bf in shape_functions]
        # Compute and store the first derivatives.
        self.dbdxi = [sp.lambdify(x, sp.diff(bf, x), 'numpy') for bf in shape_functions]

    def basis_q(self, quad_degree):
        """
        Compute the basis functions and their derivatives at quadrature points.

        Parameters:
            quad_degree: Degree for Gauss-Legendre quadrature.
        """
        # Obtain quadrature points and weights.
        xq, wq = np.polynomial.legendre.leggauss(quad_degree)
        self.xq = xq
        nq = len(xq)
        # Evaluate the basis functions at each quadrature point.
        self.bq = np.array([[fi(xi) for fi in self.b] for xi in xq])
        # Evaluate the derivatives at each quadrature point.
        self.dbdxiq = np.array([[dfi(xi) for dfi in self.dbdxi] for xi in xq])
        self.w = wq

    def eval_at_quadrature_points(self, U_prev, elem_glob_nodes):
        """
        Evaluate the derivative of the solution U_prev at the quadrature points for the current element.

        The solution derivative is computed by projecting U_prev (at the global nodes of the element)
        using the derivative of the FE basis functions.

        If the computed derivative is zero, it is replaced by a small positive value for numerical stability.
        The derivative is then raised elementwise to the power 'expon' (to incorporate nonlinearity).

        Parameters:
            U_prev: The solution field (e.g., displacement) from the previous iteration.
            elem_glob_nodes: Global node numbers corresponding to the current element.

        Returns:
            dU_prev_q: Array of the (possibly nonlinear) derivative values at the quadrature points.
        """
        # Compute the derivative at quadrature points by multiplying the derivative of the basis functions with U_prev.
        dU_prev_q = np.dot(self.dbdxiq, U_prev[elem_glob_nodes])
        
        # If the derivative is zero everywhere, replace with a small number to ensure numerical stability.
        if not np.any(dU_prev_q):
            dU_prev_q = 1e-4 + np.zeros_like(dU_prev_q)
            print("Zero derivative encountered; using 1e-4 for stability.")
        else:
            # Apply the nonlinearity by taking the absolute value raised to the given exponent.
            dU_prev_q = np.abs(dU_prev_q) ** self.expon

        return dU_prev_q

    def element_KM_matrices_fn_x(self, iel, U_prev, notconstant=False):
        """
        Compute the element stiffness (Ke_) and Jacobian (Je_) matrices for element iel.

        The computation uses the FE basis functions and the evaluated solution derivative
        at quadrature points. This derivative is used to incorporate the nonlinearity.

        Parameters:
            iel: Element index.
            U_prev: Current solution field (used to compute the derivative).
            notconstant (bool): If True, assume properties vary with x; otherwise, use constant approximations.

        Returns:
            Ke_: Element stiffness matrix.
            Je_: Element Jacobian matrix (typically proportional to the derivative of the stiffness).
        """
        dx = self.data.deltas[0]
        mod_jac = dx / 2
        elem_glob_nodes = self.data.gn[iel, :]

        imat = self.data.cell2mat_layout[iel].astype(int)
        # Evaluate the nonlinear factor based on the previous solution.
        dU_prev_q = self.eval_at_quadrature_points(U_prev, self.data.gn[iel, :])

        n = len(elem_glob_nodes)
        Ke_, Je_ = np.zeros((n, n)), np.zeros((n, n))

        if notconstant:
            # Use the precomputed basis function values at quadrature points.
            bq = self.bq
            self.bq_ = bq  # Store for later use in force vector computation.
            # Use a copy of the derivative evaluations.
            dbdxiq = np.copy(self.dbdxiq)

            for i in range(n):
                for j in range(n):
                    # Compute an entry of the stiffness matrix; the product includes the nonlinear derivative factor.
                    K_temp = np.dot(self.w * dbdxiq[:, i], dbdxiq[:, j] * dU_prev_q)
                    # For the Jacobian matrix, scale the stiffness entry by (expon + 1).
                    J_temp = K_temp * (self.expon + 1)
                    Ke_[i, j] += K_temp
                    Je_[i, j] += J_temp

        return Ke_, Je_

    def element_F_matrices_fn_x(self, iel):
        """
        Compute the element force vector (qe_) for element iel.

        The force vector is computed by evaluating the external force function fext at the quadrature points,
        then integrating over the element. Numerical stability is enhanced by scaling with a stiffness factor.

        Parameters:
            iel: Element index.

        Returns:
            qe_: Element force vector (computed without applying Dirichlet boundary conditions).
        """
        # Get the index for the external force properties.
        isrc = self.data.cell2src_layout[iel].astype(int)
        fext = self.data.fdict["fext"][isrc]
        mod_jac = self.deltas[0] / 2

        elem_glob_dofs = self.data.gn[iel, :]
        elem_glob_nodes = self.data.gnodes[iel, :]
        n = len(elem_glob_dofs)
        qe_ = np.zeros(n)

        # Get the physical coordinates of the element nodes.
        x_iel = self.data.xi[0][elem_glob_nodes]
        # Evaluate the external force function at the quadrature points.
        fext_q = fext(self.xq, x_iel, self.tau[1], self.data.fdict["A"][isrc], self.data.fdict["rho"][isrc])
        
        # Enhance numerical stability by computing a stiffness factor.
        imat = isrc
        E, rho, A = self.data.fdict["E"][imat], self.data.fdict["rho"][imat], self.data.fdict["A"][imat]
        E_const, A_const = E(1, self.tau[0]), A(1)
        Jac_inv = (mod_jac) ** (-1)
        Jac_inv_exp = (Jac_inv ** self.expon)
        Stiffness_factor = 1 / ((Jac_inv ** 2) * E_const * A_const * Jac_inv_exp)

        # Integrate the force function weighted by the basis functions.
        for i in range(n):
            qe_[i] += Stiffness_factor * np.dot(self.w * self.bq_[:, i], fext_q)
            
        return qe_

    def residual_func(self, i, j, p_sol, data):
        """
        Compute the residual vector for the nonlinear system at the element level.

        The residual is defined as:
            res = K_mus * p_sol - q_mus,
        where:
            - K_mus is the element stiffness matrix.
            - q_mus is the force vector.
            - p_sol is the solution (e.g., displacement) vector.

        Parameters:
            i: Snapshot (or time step) index.
            j: Element index.
            p_sol: Current solution vector.
            data: Dictionary containing full-order system matrices and force vectors.

        Returns:
            res: Residual vector for the specified element and snapshot.
        """
        K_mus = data['K_mus']
        q_mus = data['q_mus']

        K_mus_ij = K_mus[i][j]
        q_mus_ij = np.array(q_mus[i][j])
        
        res = np.dot(K_mus_ij, p_sol) - q_mus_ij

        return res
 
class StructuralMechanicsSimulationData:
    def __init__(self, n_ref, params, quad_deg=5, num_snapshots=1, pb_dim=1):
        """
        Initialize the simulation data class for a nonlinear structural mechanics problem.

        Parameters:
            n_ref: Reference grid dimensions or cell counts for the mesh.
            params: Array or list of parameter values (tau values) for different snapshots.
            quad_deg (int): Degree for Gauss-Legendre quadrature; higher degrees yield more accurate integration.
            num_snapshots (int): Number of snapshots to simulate.
            pb_dim (int): Problem dimension (assumed 1D in this implementation).
        """
        # Import the SystemProperties class for axial nonlinear static analysis from the appropriate module.
        from examples.structural_mechanics.Axial.NL_static.SystemProperties import SystemProperties

        # Instantiate system properties using the provided reference dimensions.
        self.properties = SystemProperties(n_ref)
        # Store reference dimensions, the total domain length (L), and the nonlinearity exponent (expon) from properties.
        self.n_ref, self.L, self.expon = n_ref, self.properties.L, self.properties.expon
        # Store the quadrature degree and the number of snapshots.
        self.quad_deg, self.num_snapshots = quad_deg, num_snapshots
        
        # Create the material layout and force (source) layout arrays using the system properties.
        self.mat_layout, self.src_layout = self.properties.create_layouts()
        # Retrieve the dictionary of functions that define material behavior and other properties.
        self.fdict = self.properties.define_properties()
        # Retrieve the boundary condition definitions.
        self.bc = self.properties.define_boundary_conditions()
        # Store the parameter values for simulation.
        self.params = params

        # Initialize lists to store simulation outputs for each snapshot.
        self.NL_solutions = []  # List to store the computed nonlinear solution for each snapshot.
        self.param_list = []    # List to record the parameter (tau) used in each snapshot.
        self.fos_time = []      # List to record computation times of the full-order solver.
        self.K_mus = []         # List to store stiffness matrices from each snapshot.
        self.q_mus = []         # List to store force (or "source") vectors from each snapshot.
        
        # Set degrees of freedom per node and nodes per element for this problem.
        self.dof_p_node, self.node_p_elem = 1, 2

    def run_simulation(self):
        """
        Run the simulation for the specified number of snapshots.

        For each snapshot:
            - The parameter tau is selected from the params list.
            - For the first snapshot, the problem data is initialized, including the mesh, layouts, and BCs.
            - The full-order solver (FOS) is constructed using the problem data and quadrature degree.
            - An initial solution is generated randomly (with absolute values scaled by 15) and the last node
              is assigned a fixed value (30.06) to enforce a boundary condition.
            - For subsequent snapshots, the FOS parameter tau is updated and the previous solution is used as the new initial guess.
            - The function 'solve_fos_statics' is called to compute the nonlinear solution, stiffness matrix, and force vector.
            - The computation time, solution, stiffness, and force data are stored.
        """
        import time

        # Loop over the number of snapshots.
        for i in range(self.num_snapshots):
            print(f"Snap {i}")
            # Select the current parameter value tau from the provided parameter list.
            tau = self.params[i]
            self.param_list.append(tau)

            if i == 0:
                # For the first snapshot, initialize the problem data (mesh, BCs, etc.)
                d = probdata(
                    self.bc, self.mat_layout, self.src_layout, self.fdict,
                    self.n_ref, self.L, self.dof_p_node, self.node_p_elem, self.expon
                )
                # Create the full-order FEM solver instance with the initial parameter tau.
                self.FOS = FOS_FEM(d, self.quad_deg, tau)
                
                # Generate an initial guess for the solution:
                # Use random values scaled by 15 (absolute values) for each vertex.
                self.sol_init = abs(np.random.rand(d.n_verts) * 15)
                # Set the last vertex to a fixed value (e.g., representing a prescribed displacement or temperature).
                self.sol_init[-1] = 30.06
            else:
                # For subsequent snapshots, update the parameter tau in the full-order solver.
                self.FOS.tau = tau
                ##################
                # In subsequent snapshots, the previous nonlinear solution (NL_solutions_p) may be used as the new initial guess.
                self.sol_init = NL_solutions_p  
                ##################

            # Uncomment or adjust the following line if a specific initial condition is desired.
            # sol_init = np.array([0,20,30])  # Example initial condition

            # Record the start time for the full-order simulation.
            tic_fos = time.time()
            # Call the full-order solver for static analysis.
            # The function solve_fos_statics returns:
            #   NL_solutions_p: Computed nonlinear solution for the snapshot.
            #   Ke_d: Computed stiffness matrix.
            #   rhs_e: Computed force vector.
            #   _ : Unused additional output.
            NL_solutions_p, Ke_d, rhs_e, _ = solve_fos_statics(self.FOS, self.sol_init)
            # Record the end time for the simulation.
            toc_fos = time.time()

            # Store the computation time for this snapshot.
            self.fos_time.append(toc_fos - tic_fos)
            # Save the computed nonlinear solution.
            self.NL_solutions.append(NL_solutions_p)
            # Save the computed force vector.
            self.q_mus.append(rhs_e)
            # Save the computed stiffness matrix.
            self.K_mus.append(Ke_d)

class ROM_simulation:
    """
    Reduced Order Model (ROM) simulation class for a nonlinear structural mechanics problem.
    This class runs the ROM simulation by projecting the full-order model (FOM) solution onto a
    reduced basis, solving the reduced system, and reconstructing the full-order solution.
    It also compares the ROM solution with the full-order solution for error analysis and computes
    the speed-up factor.
    """
    def __init__(self, f_cls, test_data, param_list, Test_mask, V_sel, xi=None, deim=None, sol_init_guess=0.1, N_rom_snap=None):
        """
        Initialize the ROM simulation class.
        
        Parameters:
            f_cls: Full-order simulation class instance (includes FOS and related data).
            test_data: Full-order simulation results (reference solutions) for error computation.
            param_list: List/array of parameters for the ROM simulation.
            Test_mask: Mask (boolean or index) to select test parameters from param_list.
            V_sel: Reduced basis matrix.
            xi: (Optional) Sampling indices for hyper-reduction methods.
            deim: (Optional) DEIM operator if using hyper-reduction.
            sol_init_guess: Initial guess for the solution at the Dirichlet boundary (default is 0.1).
            N_rom_snap: (Optional) Number of ROM snapshots to simulate. If not provided, use all parameters from the test mask.
        """
        # Store the full-order simulation class instance and its FOS object.
        self.f_cls = f_cls
        self.FOS = f_cls.FOS

        # Store the initial guess for the solution (used in the reduced space initialization).
        self.sol_init_guess = sol_init_guess
        
        # Store the full-order test data for later error analysis.
        self.fos_test_data = test_data
        # Select the parameters corresponding to the test mask.
        self.param_list_test = param_list[Test_mask]
        # Extract full-order simulation times corresponding to the test mask.
        self.fos_test_time = np.asarray(f_cls.fos_time)[Test_mask]
        
        # Store the reduced basis and any hyper-reduction parameters.
        self.V_sel = V_sel
        self.deim = deim
        self.xi = xi
        
        # Retrieve the quadrature degree from the full-order simulation.
        self.quad_deg = f_cls.quad_deg
        # Initialize an empty list to store ROM solutions.
        self.NL_solutions_rom = []
        # Store reference to full-order simulation data.
        self.d = f_cls.FOS.data
        # Store the mean field (used in reconstruction).
        self.mean = f_cls.mean
        
        # Determine the number of ROM snapshots to simulate.
        if N_rom_snap is not None:
            self.N_rom_snap = N_rom_snap
        else:
            self.N_rom_snap = len(self.param_list_test)

    def run_simulation(self):
        """
        Run the ROM simulation for the specified number of snapshots.
        
        Procedure:
            1. Generate an initial full-order solution guess using random values.
            2. Enforce the Dirichlet boundary condition at the last node using sol_init_guess.
            3. Project the initial full-order guess onto the reduced space (only for free nodes as per the mask).
            4. For each snapshot (each parameter in the test set):
                a. Instantiate the ROM solver with the current parameter.
                b. Solve the reduced system starting from the current reduced initial guess.
                c. Update the reduced initial guess with the newly computed reduced solution.
                d. Reconstruct the full-order solution from the reduced solution by mapping via the basis and adding the mean.
                e. Enforce Dirichlet boundary conditions in the reconstructed solution.
                f. Record the ROM computation time and compute the speed-up factor.
                g. Compute the ROM error as the relative difference between the ROM and full-order solution.
        """
        import time
        import src.codes.reductor.Struc_Mech.rom_class_StrucMech_axial as rom_class

        # Determine the dimension of the reduced space (N_dir) from the basis matrix.
        N_dir = self.V_sel.shape[0]
        # Full-order system has both displacement and velocity components,
        # so total degrees of freedom equals 2 * number of vertices.
        N_full = 2 * self.d.n_verts

        # Generate an initial full-order guess for the free nodes using random values.
        # Here we use absolute values to ensure non-negative initial conditions.
        sol_init_fos = abs(np.random.rand(self.d.n_verts))
        # Enforce the Dirichlet condition at the last node.
        sol_init_fos[-1] = self.sol_init_guess
        
        # Project the initial guess onto the reduced space:
        # Multiply the transpose of the basis by the full-order guess (only at free nodes).
        sol_init_rom = np.transpose(self.V_sel) @ sol_init_fos[self.FOS.data.mask]
        
        # Initialize an array to hold the reconstructed full-order solution.
        sol_rom = np.zeros_like(sol_init_fos)

        # Loop over the snapshots to simulate.
        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            tic_rom = time.time()  # Start timer for the ROM solve
            
            # Instantiate the ROM solver for the current parameter.
            ROM = rom_class.rom(self.f_cls, self.quad_deg, self.param_list_test[i], mean=self.mean, xi=self.xi)
            # Solve the reduced system starting from the current reduced initial guess.
            NL_solution_p_reduced = ROM.solve_rom(sol_init_rom, self.V_sel)
            
            # Update the reduced initial guess with the new solution for possible iterative improvement.
            sol_init_rom = NL_solution_p_reduced
                           
            toc_rom = time.time()  # End timer for the ROM solve
            rom_sim_time = toc_rom - tic_rom  # Compute ROM simulation time
            
            # Record the speed-up factor as the ratio of full-order time to ROM simulation time.
            self.speed_up.append(self.fos_test_time[i] / rom_sim_time)
            
            # Reconstruct the full-order solution from the reduced solution:
            # The reduced solution is expected to have 2*N_dir rows (first half for displacement, second half for velocity).
            shape_0 = NL_solution_p_reduced.shape[0]
            # Reconstruct displacement:
            sol_rom[self.FOS.data.mask] = np.dot(self.V_sel, NL_solution_p_reduced[:int(shape_0 / 2), :]) + self.mean
            # For nodes with Dirichlet boundary conditions, use the prescribed Dirichlet values.
            sol_rom[~self.FOS.data.mask] = self.FOS.data.sol_dir

            # Save a copy of the reconstructed ROM solution.
            self.NL_solutions_rom.append(np.copy(sol_rom))

            # Retrieve the corresponding full-order solution for error computation.
            sol_fos = self.fos_test_data[i]
            # Compute the relative ROM error (percentage) using the L2 norm.
            error = np.linalg.norm(sol_rom[self.FOS.data.mask] - sol_fos) * 100 / np.linalg.norm(sol_fos)
            self.rom_error.append(error)

class ROM_simulation_UQ:
    """
    Reduced Order Model (ROM) simulation class for uncertainty quantification (UQ)
    in a nonlinear structural mechanics problem. This class runs the ROM simulation
    for each parameter in the test set, reconstructs the full-order solution from the 
    reduced solution, and (optionally) compares it to the full-order solution.
    """
    def __init__(self, f_cls, test_data, param_list, V_sel, xi=None, deim=None, sol_init_guess=0.1, N_rom_snap=None, fos_comp=True):
        """
        Initialize the ROM simulation class.

        Parameters:
            f_cls: Full-order simulation class instance, which contains the FOS and its data.
            test_data: Full-order simulation results (reference solutions) for error comparison.
            param_list: List/array of parameter values for the ROM simulation.
            V_sel: Reduced basis matrix used to project the full-order model.
            xi: (Optional) Sampling indices for hyper-reduction methods (e.g., DEIM).
            deim: (Optional) DEIM operator if hyper-reduction is applied.
            sol_init_guess: Initial guess for the solution at the Dirichlet boundary (default 0.1).
            N_rom_snap: (Optional) Number of ROM snapshots to simulate. If None, use all parameters.
            fos_comp: Boolean flag indicating whether to compare ROM results with full-order solutions.
        """
        # Store the full-order simulation class and extract the FOS (full-order solver) instance.
        self.f_cls = f_cls
        self.FOS = f_cls.FOS
        
        # Store the initial guess for the solution.
        self.sol_init_guess = sol_init_guess
        
        # Store the full-order simulation test data (used later for error computation).
        self.fos_test_data = test_data
        
        # In this UQ setup, we use all parameters provided (no masking is applied here).
        self.param_list_test = param_list
        
        # (Commented out: if a mask were provided, full-order simulation times could be extracted.)
        # self.fos_test_time = np.asarray(f_cls.fos_time)[Test_mask]
        
        # Store the reduced basis matrix.
        self.V_sel = V_sel
        # Store DEIM operator and sampling indices if available.
        self.deim = deim
        self.xi = xi
        
        # Retrieve the quadrature degree from the full-order simulation.
        self.quad_deg = f_cls.quad_deg
        
        # Initialize list to store ROM solutions (reconstructed full-order solutions).
        self.NL_solutions_rom = []
        
        # Get a reference to the full-order simulation data (mesh, boundary conditions, etc.).
        self.d = f_cls.FOS.data
        
        # Store the mean field (used in the reconstruction of the ROM solution).
        self.mean = f_cls.mean
        
        # Flag to indicate if full-order comparison is performed.
        self.fos_comp = fos_comp
        
        # Determine the number of ROM snapshots to simulate.
        if N_rom_snap is not None:
            self.N_rom_snap = N_rom_snap
        else:
            self.N_rom_snap = len(self.param_list_test)

    def run_simulation(self):
        """
        Run the ROM simulation for all specified snapshots.
        
        Procedure:
            1. Generate an initial full-order solution guess using random values.
            2. Set the last entry of the full-order guess to the prescribed initial value.
            3. Project the free-node part of the full-order guess onto the reduced space using the basis matrix.
            4. For each snapshot:
               a. Instantiate the ROM solver with the current parameter.
               b. Solve the reduced system starting from the current reduced initial guess.
               c. Update the reduced initial guess for potential iterative improvement.
               d. Reconstruct the full-order solution by mapping the reduced solution back via the basis 
                  and adding the mean field.
               e. For nodes with Dirichlet conditions, assign the prescribed Dirichlet values.
               f. Save the reconstructed ROM solution.
               g. If full-order comparison is enabled, compute the relative error.
        """
        import time
        import src.codes.reductor.Struc_Mech.rom_class_StrucMech_axial as rom_class

        self.rom_error = []
        sol_fos_ = self.fos_test_data  # Full-order solution data for error comparison

        # Generate an initial full-order guess for free nodes as random nonnegative values.
        sol_init_fos = abs(np.random.rand(self.d.n_verts))
        # Enforce the prescribed Dirichlet condition at the last node.
        sol_init_fos[-1] = self.sol_init_guess
        
        # Project the initial guess from full-order space to the reduced subspace.
        # Only the degrees of freedom corresponding to free nodes (self.FOS.data.mask) are used.
        sol_init_rom = np.transpose(self.V_sel) @ sol_init_fos[self.FOS.data.mask]
        
        # Prepare an array to store the reconstructed full-order solution.
        sol_rom = np.zeros_like(sol_init_fos)

        # Loop over each ROM snapshot (each parameter value in param_list_test).
        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            # Optionally, print the iteration index every 100 iterations for progress tracking.
            if i % 100 == 0:
                print(i)
            
            # Instantiate the ROM solver with the current parameter.
            ROM = rom_class.rom(self.f_cls, self.quad_deg, self.param_list_test[i], mean=self.mean, xi=self.xi)
            
            # Solve the reduced system starting from the current reduced initial guess.
            NL_solution_p_reduced = ROM.solve_rom(sol_init_rom, self.V_sel)
            
            # Update the reduced initial guess with the new reduced solution (for iterative improvement if desired).
            sol_init_rom = NL_solution_p_reduced
                           
            # Reconstruct the full-order solution:
            # Multiply the reduced basis with the reduced solution and add the mean field.
            sol_rom[self.FOS.data.mask] = np.dot(self.V_sel, NL_solution_p_reduced) + self.mean
            # For nodes with Dirichlet boundary conditions, use the prescribed full-order Dirichlet values.
            sol_rom[~self.FOS.data.mask] = self.FOS.data.sol_dir

            # Store a copy of the reconstructed ROM solution.
            self.NL_solutions_rom.append(np.copy(sol_rom))

            # If full-order comparison is enabled, compute the relative error.
            if self.fos_comp:
                sol_fos = sol_fos_[i]
                error = np.linalg.norm(sol_rom[self.FOS.data.mask] - sol_fos) * 100 / np.linalg.norm(sol_fos)
                self.rom_error.append(error)
