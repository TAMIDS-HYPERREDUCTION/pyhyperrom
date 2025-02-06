from ...utils.fem_utils_HC import *
from ...basic import *
from ...utils.rom_utils import train_test_split
import time

class probdata:
    def __init__(self, bc, mat_layout, src_layout, fdict, nref, L, mu, pb_dim):
        """
        Initialize the problem data for finite element analysis, including mesh refinement,
        property functions, boundary conditions, and nodal connectivity.
        
        Parameters:
            bc: Boundary conditions for the problem.
            mat_layout: Array defining the material layout.
            src_layout: Array defining the source layout.
            fdict: Dictionary containing function definitions for material properties.
            nref: Reference dimensions used to repeat the layout in higher dimensions.
            L: Domain length(s); scalar for 1D or array for higher dimensions.
            mu: Parameter value(s) associated with the problem.
            pb_dim: Problem dimension (1 for 1D, 2 for 2D, 3 for 3D).
        """
        # Store the problem dimension.
        self.dim_ = pb_dim
        
        # Refine the mesh layouts based on the problem dimension.
        if pb_dim == 1:
            # For 1D problems, flatten the input arrays.
            self.cell2mat_layout = mat_layout.flatten()
            self.cell2src_layout = src_layout.flatten()
        else:
            # For multidimensional problems, repeat the layout arrays using the reference dimensions.
            repeats = np.asarray(nref, dtype=int)
            self.cell2mat_layout = self.repeat_array(mat_layout, repeats)
            self.cell2src_layout = self.repeat_array(src_layout, repeats)
        
        # Store the function dictionary that defines material and source properties.
        self.fdict = fdict

        # Initialize mesh data containers for each spatial dimension.
        self.ncells = [None] * pb_dim  # Number of cells per dimension.
        self.npts = [None] * pb_dim    # Number of nodal points per dimension.
        self.deltas = [None] * pb_dim  # Cell sizes per dimension.
        self.xi = []                   # Coordinates of the nodal points in each dimension.

        # Determine mesh metrics for each dimension.
        for i in range(pb_dim):

            # The number of cells is determined from the shape of the refined layout.
            self.ncells[i] = self.cell2mat_layout.shape[i]

            # The number of nodes is one more than the number of cells.
            self.npts[i] = self.ncells[i] + 1

            if pb_dim == 1:
                # For 1D, generate a linear space from 0 to L.
                self.xi.append(np.linspace(0, L, self.npts[i]))
                self.deltas[i] = L / self.ncells[i]
            else:
                # For higher dimensions, L is an array; generate a linear space for dimension i.
                self.xi.append(np.linspace(0, L[i], self.npts[i]))
                self.deltas[i] = L[i] / self.ncells[i]

        # Determine the total number of vertices in the mesh.
        if pb_dim == 1:
            self.n_verts = self.npts[0]
        else:
            self.n_verts = np.prod(np.array(self.npts))

        # Establish the nodal connectivity for the mesh.
        self.connectivity()

        # Store the parameter.
        self.mu = mu

        # Apply boundary conditions to update the mesh with Dirichlet nodes.
        handle_boundary_conditions(self, bc)

        # Assign global equation numbers to nodes considering the Dirichlet conditions.
        get_glob_node_equation_id(self, self.dir_nodes)

        # Initialize containers for element-level data.
        self.glob_node_eqnId = []             # Global node equation IDs for each element.
        self.glob_node_nonzero_eqnId = []       # Global nonzero equation IDs per element.
        self.local_node_nonzero_eqnId = []      # Local nonzero equation IDs per element.
        self.Le = []                          # Element connectivity information.
        self.global_indices = []              # Global node indices for elements.
        self.local_indices = []               # Local node indices within an element.

        # For each element, determine the global node numbers and associated nonzero equation IDs.
        for i in range(self.n_cells):
            get_element_global_nodes_and_nonzero_eqnId(self, i, self.node_eqnId)

        # Create a boolean mask indicating nodes that are not subject to Dirichlet boundary conditions.
        mask = self.node_eqnId != 0
        self.mask = mask

    def repeat_array(self, arr, repeats):
        """
        Repeat elements of the input array along each axis according to specified repeat counts.
        
        Parameters:
            arr: Input array to be repeated.
            repeats: Array of repeat counts for each axis.
        
        Returns:
            The repeated array.
        """
        for dim, n in enumerate(repeats):
            arr = np.repeat(arr, n, axis=dim)
        return arr

    def connectivity(self):
        """
        Define the nodal connectivity for the mesh based on the problem dimension.
        Calls the appropriate method for 1D, 2D, or 3D connectivity.
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
        """
        Define nodal connectivity for a 1D mesh.
        Each cell (element) is connected by two nodes (left and right).
        """
        # Number of cells is determined by the first dimension of ncells.
        self.n_cells = self.ncells[0]
        # Initialize connectivity array: each cell has 2 nodes (2^1 = 2).
        self.gn = np.zeros((self.n_cells, 2**self.dim_), dtype=int)
        # Assign node indices for each cell.
        for iel in range(self.n_cells):
            self.gn[iel, 0] = iel         # Left node index.
            self.gn[iel, 1] = iel + 1     # Right node index.


    def connectivity_2d(self):
        """
        Define nodal connectivity for a 2D mesh.
        Each cell (quadrilateral element) is defined by 4 nodes.
        """
        # Total number of cells is the product of cells in both dimensions.
        self.n_cells = np.prod(np.array(self.ncells))
        # Initialize connectivity array: each cell has 4 nodes (2^2 = 4).
        self.gn = np.zeros((self.n_cells, 2**self.dim_), dtype=int)

        # Define a helper function to compute a node's global index from its (i, j) position.
        node = lambda i, j: i + j * self.npts[0]

        iel = 0
        # Loop over the vertical (j) and horizontal (i) indices.
        for j in range(self.ncells[1]):
            for i in range(self.ncells[0]):
                self.gn[iel, 0] = node(i, j)       # Lower-left node.
                self.gn[iel, 1] = node(i + 1, j)   # Lower-right node.
                self.gn[iel, 2] = node(i + 1, j + 1)  # Upper-right node.
                self.gn[iel, 3] = node(i, j + 1)   # Upper-left node.
                iel += 1

    def connectivity_3d(self):
        """
        Define nodal connectivity for a 3D mesh.
        Each cell (hexahedral element) is defined by 8 nodes.
        """
        # Total number of cells is the product of cells in all three dimensions.
        self.n_cells = np.prod(np.array(self.ncells))
        # Initialize connectivity array: each cell has 8 nodes (2^3 = 8).
        self.gn = np.zeros((self.n_cells, 2**self.dim_), dtype=int)

        # Define a helper function to compute a node's global index from its (i, j, k) position.
        node = lambda i, j, k: i + j * self.npts[0] + k * self.npts[0] * self.npts[1]
        iel = 0
        # Loop over the depth (k), vertical (j), and horizontal (i) indices.
        for k in range(self.ncells[2]):
            for j in range(self.ncells[1]):
                for i in range(self.ncells[0]):
                    self.gn[iel, 0] = node(i, j, k)         # Node 0.
                    self.gn[iel, 1] = node(i + 1, j, k)     # Node 1.
                    self.gn[iel, 2] = node(i + 1, j + 1, k)  # Node 2.
                    self.gn[iel, 3] = node(i, j + 1, k)     # Node 3.
                    self.gn[iel, 4] = node(i, j, k + 1)     # Node 4.
                    self.gn[iel, 5] = node(i + 1, j, k + 1)  # Node 5.
                    self.gn[iel, 6] = node(i + 1, j + 1, k + 1)  # Node 6.
                    self.gn[iel, 7] = node(i, j + 1, k + 1)  # Node 7.
                    iel += 1

class FOS_FEM:
    def __init__(self, data, quad_degree):
        """
        Initialize the full-order finite element model (FEM) simulation class with mesh data,
        material properties, and a specified quadrature degree for numerical integration.

        Parameters:
            data: An object containing the mesh, material properties, and boundary condition information.
            quad_degree: The degree of Gauss-Legendre quadrature to be used for numerical integration.
        """
        # Store the provided simulation data.
        self.data = data
        
        # Extract the material parameter vector (mu) from the data.
        self.mu = self.data.mu

        # Determine the spatial dimension of the problem from the data.
        self.dim_ = data.dim_

        # Create shortcut lists for the number of cells, nodal points, and cell sizes (deltas)
        # in each spatial dimension.
        self.ncells = [data.ncells[i] for i in range(self.dim_)]
        self.npts = [data.npts[i] for i in range(self.dim_)]
        self.deltas = [data.deltas[i] for i in range(self.dim_)]

        # Total number of nodes (vertices) in the mesh.
        self.n_nodes = data.n_verts

        # Compute the continuous Finite Element (cFEM) basis functions and their derivatives.
        self.basis()

        # Evaluate the basis functions and their derivatives at the quadrature points.
        self.basis_q(quad_degree)
        
        # Store the Dirichlet boundary condition solution and node information.
        self.sol_dir = data.T_dir
        self.dir_nodes = data.dir_nodes
        self.node_eqnId = data.node_eqnId

        # Initialize the ECM (Empirical Cubature Method) flag to False by default.
        self.ecm = False

    def e_n_2ij(self, iel, el=True):
        """
        Convert a single global element index (or node index) into multi-dimensional indices,
        one for each spatial dimension, using modular arithmetic.

        Parameters:
            iel: The global element index (or node index if el is False).
            el (bool): Flag indicating whether 'iel' is an element index (True) or a node index (False).

        Returns:
            indices (list): A list of indices corresponding to each dimension.
        """
        dim_ = self.dim_
        indices = []
        divisor = 1
        # Iterate over each spatial dimension.
        for d in range(dim_):
            # Choose the appropriate size: number of cells if el is True, else number of nodal points.
            size = self.ncells[d]
            if el == False:
                size = self.npts[d]
            # Calculate the index for the current dimension using integer division and modulo.
            idx = (iel // divisor) % size
            divisor *= size
            indices.append(idx)
        return indices

    def basis(self):
        """
        Compute the continuous Finite Element (cFEM) basis functions for the element and
        their derivatives using symbolic computation. The basis functions correspond to the
        vertices of a hypercube (e.g., linear basis functions on an element).

        The symbolic expressions are converted to numerical functions via lambdify.
        """
        dim_ = self.dim_

        # Create symbolic variables: u0, u1, ..., u{dim_-1}.
        symbols = sp.symbols(f'u:{dim_}')
        
        # Initialize lists to store basis functions and their derivatives.
        self.b = []  # List to hold the basis functions.
        self.dbdxi = [[] for _ in range(dim_)]  # List of lists to hold the derivative functions for each variable.
        
        # There are 2**dim_ basis functions corresponding to the vertices of the hypercube.
        for i in range(2**dim_):
            # For each basis function, determine factors based on the binary representation of i.
            factors = [(1 + (-1)**((i >> j) & 1) * symbols[j]) / 2 for j in range(dim_)]
            # Multiply the factors to construct the basis function.
            basis_function = sp.Mul(*factors)
            # Convert the symbolic basis function into a numerical function using numpy.
            self.b.append(sp.lambdify(symbols, basis_function, 'numpy'))
            
            # Compute and store the derivative of the basis function with respect to each symbolic variable.
            for j, symbol in enumerate(symbols):
                derivative = sp.diff(basis_function, symbol)
                self.dbdxi[j].append(sp.lambdify(symbols, derivative, 'numpy'))

    def basis_q(self, quad_degree):
        """
        Compute the values of the basis functions and their derivatives at the Gauss-Legendre
        quadrature points for numerical integration.

        Parameters:
            quad_degree: The degree of the Gauss-Legendre quadrature.

        Results stored in:
            self.bq: Array of basis function values at each quadrature point.
            self.dbdxiq: List of arrays of the derivatives of the basis functions (one per spatial dimension).
            self.w: Array of quadrature weights for each quadrature point.
        """
        dim_ = self.dim_

        # Get 1D Gauss-Legendre quadrature points (xq) and weights (wq) for the specified degree.
        xq, wq = np.polynomial.legendre.leggauss(quad_degree)
        
        # Create a multi-dimensional grid of quadrature points and corresponding weights.
        quad_points = list(product(xq, repeat=dim_))
        quad_weights = list(product(wq, repeat=dim_))
        
        nq = len(quad_points)  # Total number of quadrature points.
        # Initialize arrays for the basis function evaluations and their derivatives.
        self.bq = np.zeros((nq, len(self.b)))
        self.dbdxiq = [np.zeros_like(self.bq) for _ in range(dim_)]
        self.w = np.zeros(nq)

        # Evaluate the basis functions and derivatives at each quadrature point.
        for q, (point, weights) in enumerate(zip(quad_points, quad_weights)):
            # Loop over each basis function.
            for i, (fi, *dfi) in enumerate(zip(self.b, *self.dbdxi)):
                # Evaluate the i-th basis function at the quadrature point.
                self.bq[q, i] = fi(*point)
                # Evaluate the derivative of the i-th basis function for each dimension.
                for d, derivative in enumerate(dfi):
                    self.dbdxiq[d][q, i] = derivative(*point)
            # The overall quadrature weight is the product of the 1D weights.
            self.w[q] = np.prod(weights)

    def eval_at_quadrature_points(self, T_prev, elem_glob_nodes):
        """
        Evaluate the temperature field and its spatial derivatives at the quadrature points
        for a given element.

        Parameters:
            T_prev: The temperature field vector (nodal values) from the previous iteration.
            elem_glob_nodes: Global node indices corresponding to the current element.

        Returns:
            T_prev_q: Array of temperature values at the quadrature points.
            dT_prev_q: List of arrays of temperature derivatives for each spatial dimension.
        """
        dim_ = self.dim_

        # Evaluate the temperature at the quadrature points by projecting nodal values using the basis.
        T_prev_q = np.dot(self.bq, T_prev[elem_glob_nodes])

        # Compute the derivative of the temperature field in each dimension at the quadrature points.
        dT_prev_q = [np.dot(self.dbdxiq[k], T_prev[elem_glob_nodes]) for k in range(dim_)]

        return T_prev_q, dT_prev_q

    def element_matrices(self, T_prev, iel, affine=False):
        """
        Compute the element stiffness matrix (Ke_), the Jacobian matrix (Je_), and the source vector (qe_)
        for a given element using the finite element method.

        Parameters:
            T_prev: The temperature field from the previous iteration.
            iel: The index of the current element.
            affine (bool): Flag to determine whether to use affine material properties.

        Returns:
            Ke_: Element stiffness matrix.
            Je_: Element Jacobian matrix.
            qe_: Element source vector.
        """
        dim_ = self.dim_
        # Retrieve the material parameter vector.
        mu = self.mu
        # Convert the element index to multi-dimensional indices.
        cell_idx = tuple(self.e_n_2ij(iel))
        
        # Retrieve material and source layout indices for the current element.
        imat = self.data.cell2mat_layout[cell_idx].astype(int)
        isrc = self.data.cell2src_layout[cell_idx].astype(int)
        
        # Depending on the 'affine' flag, select either the standard nonlinear or a simplified affine model.
        if not affine:
            k    = self.data.fdict["cond"][imat]
            dkdT = self.data.fdict["dcond"][imat]
            qext = self.data.fdict["qext"][isrc]
            dqext = self.data.fdict["dqext"][isrc]
        else:
            k    = lambda T, mu: 0.0 * T + 1
            dkdT = lambda T, mu: 0.0 * T
            qext = lambda T, mu: 0.0 * T + 1
            dqext = lambda T, mu: 0.0 * T

        # Evaluate the temperature and its derivatives at the quadrature points for the current element.
        T_prev_q, dT_prev_q = self.eval_at_quadrature_points(T_prev, self.data.gn[iel, :])
        # Compute the conductivity and its temperature derivative at quadrature points.
        cond_q = k(T_prev_q, mu[0])
        dcond_q = dkdT(T_prev_q, mu[0])
        # Compute the external heat source and its temperature derivative at quadrature points.
        qext_q = qext(T_prev_q, mu[1])
        dqext_q = dqext(T_prev_q, mu[1])

        # Get the global node indices for the current element.
        elem_glob_nodes = self.data.gn[iel, :]

        # Initialize the element matrices and source vector.
        n = len(elem_glob_nodes)
        Ke_ = np.zeros((n, n))
        Je_ = np.zeros((n, n))
        qe_ = np.zeros(n)

        # Compute the volume (or area/length) scaling factor for the element.
        vol = np.prod(np.array(self.deltas)) / (2 ** dim_)

        # Define stiffness Jacobian coefficients based on the problem's spatial dimension.
        if dim_ == 1:
            stiff_J_coeff = [2 / self.deltas[0]]
        elif dim_ == 2:
            stiff_J_coeff = [self.deltas[1] / self.deltas[0], self.deltas[0] / self.deltas[1]]
        else:
            stiff_J_coeff = [self.deltas[1] * self.deltas[2] / self.deltas[0],
                             self.deltas[0] * self.deltas[2] / self.deltas[1],
                             self.deltas[0] * self.deltas[1] / self.deltas[2]]

        # Compute the element matrices without using ECM hyperreduction.
        if not self.ecm:
            for i in range(n):
                # Compute the contribution to the element source vector for node i.
                qe_[i] += vol * np.dot(self.w * self.bq[:, i], qext_q)

                for j in range(n):
                    # Compute an intermediate term representing the external source derivative effect.
                    qe_temp = vol * np.dot(self.w * self.bq[:, i], dqext_q * self.bq[:, j])

                    # Loop over each spatial dimension to compute stiffness and Jacobian contributions.
                    for k_ in range(dim_):
                        # Stiffness contribution: involves the derivative of basis functions and conductivity.
                        K_temp = stiff_J_coeff[k_] * np.dot(self.w * self.dbdxiq[k_][:, i], cond_q * self.dbdxiq[k_][:, j])
                        # Jacobian contribution: involves the derivative of basis functions, conductivity derivative, and temperature derivative.
                        J_temp = stiff_J_coeff[k_] * np.dot(self.w * self.dbdxiq[k_][:, i], dcond_q * self.bq[:, j] * dT_prev_q[k_])
                        # Accumulate contributions to the stiffness and Jacobian matrices.
                        Ke_[i, j] += K_temp
                        Je_[i, j] += J_temp
                    # Subtract the intermediate source term contribution from the Jacobian.
                    Je_[i, j] -= qe_temp
        
        else:
            # For ECM hyperreduction, initialize temporary storage for Gauss point contributions.
            Ke_gp = np.zeros((len(self.w), n, n))
            Je_gp = np.zeros((len(self.w), n, n))
            qe_gp = np.zeros((len(self.w), n))

            # Loop over each Gauss point.
            for gp in range(len(self.w)):
                # Extract material properties at the current Gauss point.
                qext_gp = qext_q[gp]
                dqext_gp = dqext_q[gp]
                cond_gp = cond_q[gp]
                # Compute the derivative of conductivity at the Gauss point.
                dcond_gp = dkdT(T_prev_q[gp], mu[0])
                # Retrieve the temperature derivatives at the current Gauss point.
                dT_prev_gp = [dT_prev_q[d][gp] for d in range(dim_)]
                
                for i in range(n):
                    # Compute the Gauss point contribution to the source vector.
                    qe_gp[gp, i] = vol * self.bq[gp, i] * qext_gp

                    for j in range(n):
                        # Compute the intermediate external source term at the Gauss point.
                        qe_temp = vol * self.bq[gp, i] * dqext_gp * self.bq[gp, j]
                        
                        for k_ in range(dim_):
                            # Compute the stiffness matrix contribution at the current Gauss point.
                            K_temp = stiff_J_coeff[k_] * self.dbdxiq[k_][gp, i] * cond_gp * self.dbdxiq[k_][gp, j]
                            # Compute the Jacobian matrix contribution at the current Gauss point.
                            J_temp = stiff_J_coeff[k_] * self.dbdxiq[k_][gp, i] * dcond_gp * self.bq[gp, j] * dT_prev_gp[k_]
                            Ke_gp[gp, i, j] += K_temp
                            Je_gp[gp, i, j] += J_temp
                        # Adjust the Jacobian by subtracting the source term contribution.
                        Je_gp[gp, i, j] -= qe_temp

            # Append the Gauss point matrices to the corresponding class attributes.
            self.Ke_gauss.append(Ke_gp)
            self.rhs_e_gauss.append(qe_gp)
            self.Je_gauss.append(Je_gp)

            # Sum up the contributions from all Gauss points, weighted by the quadrature weights.
            for gp in range(len(self.w)):
                Ke_ += self.w[gp] * Ke_gp[gp]
                Je_ += self.w[gp] * Je_gp[gp]
                qe_ += self.w[gp] * qe_gp[gp]

        # Return the computed element stiffness matrix, Jacobian matrix, and source vector.
        return Ke_, Je_, qe_

    def residual_func(self, i, j, p_sol, data):
        """
        Compute the residual vector for a given snapshot and cell (non-ECM case) by subtracting
        the source term from the product of the stiffness matrix and the solution.

        Parameters:
            i: Index of the snapshot.
            j: Index of the cell.
            p_sol: The solution vector (e.g., temperature) for the current snapshot.
            data: A dictionary containing the stiffness matrices ('K_mus') and source vectors ('q_mus').

        Returns:
            res: The residual vector for the specified cell.
        """
        # Extract the stiffness matrices and source vectors from the data.
        K_mus = data['K_mus']
        q_mus = data['q_mus']
        # Retrieve the stiffness matrix and source vector for snapshot i and cell j.
        K_mus_ij = K_mus[i][j]
        q_mus_ij = np.array(q_mus[i][j])
        # Compute the residual as the difference between the product of the stiffness matrix and p_sol, and q_mus_ij.
        res = np.dot(K_mus_ij, p_sol) - q_mus_ij
        return res
    
    def residual_func_ecm(self, i, j, k, p_sol, data_ecm):
        """
        Compute the residual at a specific Gauss point for ECM hyperreduction.

        Parameters:
            i: Snapshot index.
            j: Cell index.
            k: Gauss point index within the cell.
            p_sol: The solution vector (e.g., temperature) for the current snapshot.
            data_ecm: A dictionary containing Gauss point stiffness matrices ('Ke_gauss_mus') and 
                      source vectors ('rhs_e_gauss_mus').

        Returns:
            res_k: The residual vector computed at the k-th Gauss point.
        """
        # Extract Gauss point stiffness matrices and source vectors.
        K_mus = data_ecm['Ke_gauss_mus']
        q_mus = data_ecm['rhs_e_gauss_mus']
        # Retrieve the stiffness matrix and source vector for snapshot i, cell j, at Gauss point k.
        K_mus_ij = K_mus[i][j][k]
        q_mus_ij = np.array(q_mus[i][j][k])
        # Compute the residual at the Gauss point.
        res_k = np.dot(K_mus_ij, p_sol) - q_mus_ij
        return res_k

class HeatConductionSimulationData:
    
    def __init__(self, n_ref, L, quad_deg=3, num_snapshots=15, pb_dim=1, params=np.arange(1., 4.0, 0.01),
                 T_init_guess=273.0, train_mask=None, test_mask=None, ecm=False):
        """
        Initialize the simulation data for a heat conduction problem.
        
        Parameters:
            n_ref: Array or list defining reference dimensions (e.g., number of cells in each region).
            L: Domain length (scalar for 1D or array for multi-D problems).
            quad_deg: Degree of the quadrature used for numerical integration.
            num_snapshots: Total number of simulation snapshots to generate.
            pb_dim: Problem dimension (1 for 1D, 2 for 2D, 3 for 3D).
            params: Array of parameter values for each simulation snapshot.
            T_init_guess: Initial guess for the temperature field.
            train_mask: (Optional) Boolean or index mask for training snapshots.
            test_mask: (Optional) Boolean or index mask for testing snapshots.
            ecm: Flag to indicate whether ECM (Empirical Cubature Method) hyperreduction is used.
        """
        
        # Based on the problem dimension, import the corresponding SystemProperties class.
        if pb_dim == 1:
            from examples.heat_conduction.OneD_heat_conduction.FEM_1D_system_properties import SystemProperties
        elif pb_dim == 2:
            from examples.heat_conduction.TwoD_heat_conduction.FEM_2D_system_properties import SystemProperties
        else:
            from examples.heat_conduction.ThreeD_heat_conduction.FEM_3D_system_properties import SystemProperties
                
        # Instantiate the system properties with the provided reference dimensions and parameter list.
        self.layout = SystemProperties(n_ref, params)
        self.n_ref = n_ref                  # Store reference dimensions.
        self.L = L                          # Store domain length(s).
        self.quad_deg = quad_deg            # Store the quadrature degree.
        self.num_snapshots = num_snapshots  # Store the number of snapshots.
        
        # Create material and source layouts using the system properties instance.
        self.mat_layout, self.src_layout = self.layout.create_layouts()
        # Define the property functions (e.g., conductivity, heat source) using the layout.
        self.fdict = self.layout.define_properties()
        # Define the boundary conditions for the simulation.
        self.bc = self.layout.define_boundary_conditions()
        # Store the complete list of parameters.
        self.params = self.layout.params
        
        # Initialize lists to store simulation outputs.
        self.NL_solutions = []   # To store nonlinear solution snapshots.
        self.param_list = []     # To store the parameter value used for each snapshot.
        self.pb_dim = pb_dim     # Store the problem dimension.
        self.fos_time = []       # To record the computational time for each snapshot.
        self.rhs = []            # To store right-hand side vectors for each snapshot.
        self.K_mus = []          # To store stiffness matrices for each snapshot.
        self.q_mus = []          # To store source vectors for each snapshot.

        # For ECM hyperreduction outputs, initialize empty lists.
        self.Ke_gauss_mus = []
        self.fe_ecm_mus = []
        self.rhs_e_gauss_mus = []
        self.ecm = ecm         # Flag indicating whether ECM hyperreduction is enabled.

        # Set the initial guess for the temperature field.
        self.T_init_guess = T_init_guess

        # If training or testing masks are provided (non-empty), store them.
        # Otherwise, generate a default train/test split based on the number of snapshots.
        if train_mask.any() or test_mask.any() is not None:
            self.train_mask, self.test_mask = train_mask, test_mask
        else:
            self.train_mask, self.test_mask = train_test_split(num_snapshots)

    def run_simulation(self):
        """
        Execute the simulation across all snapshots. For each snapshot:
          - Select the appropriate parameter.
          - Initialize or update the full-order FEM system.
          - Solve the FEM system to obtain the nonlinear solution.
          - Record timing and solution data.
        """
        # Set a fixed random seed for reproducibility.
        random.seed(25)       

        # Loop over the number of snapshots.
        for i in range(self.num_snapshots):
            print(f"Snap {i}")
            # Select the parameter for the current snapshot from the parameter list.
            param = self.params[i]  # Could alternatively choose randomly from params.
            self.param_list.append(param)
            
            if i == 0:
                # For the first snapshot, initialize the problem data using the current boundary conditions,
                # material/source layouts, property functions, and the current parameter.
                d = probdata(self.bc, self.mat_layout, self.src_layout, self.fdict, self.n_ref, self.L, param, self.pb_dim)
                # Initialize the full-order FEM system using the problem data and specified quadrature degree.
                self.FOS = FOS_FEM(d, self.quad_deg)
                # Initialize the temperature field with the initial guess for all vertices.
                T_init = np.zeros(d.n_verts) + self.T_init_guess
            else:
                # For subsequent snapshots, update the FEM system with the new parameter.
                self.FOS.mu = param
                # Reset the initial guess for the temperature field.
                T_init = np.zeros(d.n_verts) + self.T_init_guess
            
            # Record the start time for solving the FEM system.
            tic_fos = time.perf_counter()
            # Set the ECM flag in the FEM system according to the simulation setting.
            self.FOS.ecm = self.ecm

            # Solve the full-order FEM system with the current temperature field guess.
            # solve_fos returns the nonlinear solution (NL_solution_p), the stiffness matrix (Ke), the ECM-adjusted
            # source vector (rhs_e), an unused output (_), and the standard right-hand side vector (rhs_).
            NL_solution_p, Ke, rhs_e, _, rhs_ = solve_fos(self.FOS, T_init)
            # Record the end time after the solution is obtained.
            toc_fos = time.perf_counter()
            
            # Print progress message indicating completion of the current snapshot.
            print("1 finished")
            # Record the elapsed time for this snapshot.
            self.fos_time.append(toc_fos - tic_fos)
            # Store the flattened nonlinear solution.
            self.NL_solutions.append(NL_solution_p.flatten())
            # Store the right-hand side vector.
            self.rhs.append(rhs_)
            # Store the stiffness matrix.
            self.K_mus.append(Ke)
            # Store the source vector.
            self.q_mus.append(rhs_e)

            # For ECM hyperreduction, store the Gauss point based stiffness matrices, force vectors,
            # and ECM-specific right-hand side vectors.
            if self.ecm:
                self.Ke_gauss_mus.append(self.FOS.Ke_gauss)
                self.fe_ecm_mus.append(self.FOS.fe_ecm)
                self.rhs_e_gauss_mus.append(self.FOS.rhs_e_gauss)

class HeatConductionSimulationData_affine:
    def __init__(self, n_ref, L, quad_deg=3, num_snapshots=15, pb_dim=1,
                 params=np.arange(1., 4.0, 0.01), train_mask=None, test_mask=None):
        """
        Initialize the simulation data for the affine heat conduction problem.
        
        Parameters:
            n_ref: Reference dimensions for mesh refinement.
            L: Domain length (scalar for 1D or array for multi-dimensional problems).
            quad_deg: Degree of quadrature for numerical integration.
            num_snapshots: Number of simulation snapshots.
            pb_dim: Problem dimension (1 for 1D, 2 for 2D, 3 for 3D).
            params: Array of parameter values to use for the simulation.
            train_mask: Optional mask for training snapshots.
            test_mask: Optional mask for testing snapshots.
        """
        # Import the appropriate SystemProperties class based on the problem dimension.
        if pb_dim == 1:
            from examples.heat_conduction.OneD_heat_conduction.FEM_1D_system_properties import SystemProperties
        elif pb_dim == 2:
            from examples.heat_conduction.TwoD_heat_conduction.FEM_2D_system_properties import SystemProperties
        else:
            from examples.heat_conduction.ThreeD_heat_conduction.FEM_3D_system_properties import SystemProperties
        
        # Instantiate the system properties using the provided reference dimensions and parameter list.
        self.layout = SystemProperties(n_ref, params)
        self.n_ref = n_ref                    # Store the reference dimensions.
        self.L = L                            # Store the domain length(s).
        self.quad_deg = quad_deg              # Store the quadrature degree.
        self.num_snapshots = num_snapshots    # Store the number of snapshots.
        
        # Generate material and source layouts using methods from the system properties instance.
        self.mat_layout, self.src_layout = self.layout.create_layouts()
        # Define the property functions (e.g., conductivity, heat source) using the layout.
        self.fdict = self.layout.define_properties()
        # Define boundary conditions based on the layout.
        self.bc = self.layout.define_boundary_conditions()
        # Store the parameter values.
        self.params = self.layout.params
        
        # Initialize lists to store simulation outputs.
        self.NL_solutions = []  # Nonlinear solution snapshots.
        self.param_list = []    # List to record parameter values used for each snapshot.
        self.pb_dim = pb_dim    # Store the problem dimension.
        self.fos_time = []      # To record the time taken for each snapshot.
        self.rhs = []           # To store right-hand side vectors for each snapshot.
        self.K_mus = []         # To store stiffness matrices for each snapshot.
        self.q_mus = []         # To store source vectors for each snapshot.
        
        # If training and testing masks are provided, store them; otherwise, create default splits.
        if train_mask.any() or test_mask.any() is not None:
            self.train_mask, self.test_mask = train_mask, test_mask
        else:
            self.train_mask, self.test_mask = train_test_split(num_snapshots)

    def run_simulation(self):
        """
        Execute the simulation over all snapshots for the affine heat conduction problem.
        For each snapshot:
          - Select the parameter from the list.
          - For the first snapshot, initialize the full-order FEM system.
          - For subsequent snapshots, update the parameter in the FEM system.
          - Evaluate the affine system matrices and vectors.
          - Assemble the effective stiffness matrix and right-hand side.
          - Solve the linear system to obtain the nonlinear solution.
          - Record the computation time and solution.
        """
        for i in range(self.num_snapshots):
            print(f"Snap {i}")
            # Select the parameter for the current snapshot.
            param = self.params[i]  # Could also select randomly if desired.
            self.param_list.append(param)
            
            if i == 0:
                # For the first snapshot, create a problem data instance with boundary conditions,
                # layouts, property functions, and the current parameter.
                d = probdata(self.bc, self.mat_layout, self.src_layout, self.fdict,
                             self.n_ref, self.L, param, self.pb_dim)
                # Initialize the full-order FEM system with the problem data and quadrature degree.
                self.FOS = FOS_FEM(d, self.quad_deg)
            else:
                # For subsequent snapshots, update the material parameter in the FEM system.
                self.FOS.mu = param
            
            # Record the start time for the current snapshot simulation.
            tic_fos = time.perf_counter()

            # For the first snapshot (or when i < 1), evaluate the affine system matrices and vectors.
            if i < 1:
                K_aff, rhs_qe_, rhs_fe_, sol, mask = eval_KF_affine(self.FOS)

            # Create a copy of the solution computed from the affine evaluation.
            NL_solution_p = np.copy(sol)

            # Initialize the full-order stiffness matrix and the force contribution vector.
            K_FOS = 0 * K_aff[0]  # Creates a zero matrix with the same shape as K_aff[0].
            RHS_fe = np.zeros_like(rhs_fe_[0])

            # Accumulate contributions from the conductivity property for each condition.
            for j in range(len(self.FOS.data.fdict["cond"])):
                k = self.FOS.data.fdict["cond"][j]
                # Scale the affine stiffness matrix and force vector by the conductivity function evaluated at T=0.
                K_FOS += k(0, param[0]) * K_aff[j]
                RHS_fe += k(0, param[0]) * rhs_fe_[j]

            # Initialize vectors for external heat source contributions.
            rhs = np.zeros_like(rhs_qe_[0])
            RHS_qe = np.zeros_like(rhs_qe_[0])

            # Accumulate contributions from the external heat source property.
            for j in range(len(self.FOS.data.fdict["qext"])):
                q = self.FOS.data.fdict["qext"][j]
                RHS_qe += q(0, param[1]) * rhs_qe_[j]

            # Compute the effective right-hand side by subtracting the force contribution from the external source.
            rhs = RHS_qe - RHS_fe

            # Solve the linear system for the degrees of freedom indicated by the mask.
            NL_solution_p[mask] = linalg.spsolve(K_FOS.tocsc(), rhs)

            # Record the end time for the current snapshot simulation.
            toc_fos = time.perf_counter()
            
            print("1 finished")
            # Record the elapsed time.
            self.fos_time.append(toc_fos - tic_fos)
            # Store the computed nonlinear solution (flattened) for the current snapshot.
            self.NL_solutions.append(NL_solution_p.flatten())

class ROM_simulation:
    def __init__(self, f_cls, test_data, param_list, Test_mask, V_sel, xi=None, deim=None, T_init_guess=573.15, N_rom_snap=None):
        """
        Initialize the Reduced Order Model (ROM) simulation class with the necessary data, parameters, 
        and reduced basis information.

        Parameters:
            f_cls: An object containing the full-order solver (FOS) and related simulation data.
            test_data: Full-order simulation results used for testing the ROM.
            param_list: List or array of parameters used in the simulation.
            Test_mask: Boolean or index mask to select test parameters from param_list.
            V_sel: The selected reduced basis matrix.
            xi: (Optional) Sampling vector or indicator for hyper-reduction.
            deim: (Optional) DEIM object for hyper-reduction (if applicable).
            T_init_guess: Initial temperature guess for the simulation.
            N_rom_snap: (Optional) Number of ROM snapshots to process; if None, use all test parameters.
        """
        # Extract the full-order FEM system from the provided full-order solver object.
        self.FOS = f_cls.FOS
        # Store the initial temperature guess.
        self.T_init_guess = T_init_guess
        # Store the full-order solution test data.
        self.fos_test_data = test_data
        # Select test parameters using the provided test mask.
        self.param_list_test = param_list[Test_mask]
        # Select full-order simulation times corresponding to the test mask.
        self.fos_test_time = np.asarray(f_cls.fos_time)[Test_mask]
        # Store the reduced basis matrix.
        self.V_sel = V_sel
        # Store the DEIM object for hyper-reduction (if provided).
        self.deim = deim
        # Store the sampling indicator vector (if provided).
        self.xi = xi
        # Extract the quadrature degree from the full-order solver.
        self.quad_deg = f_cls.quad_deg
        # Initialize a list to store the ROM solution snapshots.
        self.NL_solutions_rom = []
        # Store the full-order data object.
        self.d = f_cls.FOS.data
        # Store the mean field (used for ROM projection and reconstruction).
        self.mean = f_cls.mean

        # Determine the number of ROM snapshots to process.
        if N_rom_snap is not None:
            self.N_rom_snap = N_rom_snap
        else:
            self.N_rom_snap = len(self.param_list_test)

    def run_simulation_h_deim(self):
        """
        Run the ROM simulation using the DEIM hyper-reduction technique.
        For each test parameter, solve the reduced system and reconstruct the full-order solution,
        recording the simulation speed-up and ROM error.
        """
        import src.codes.reductor.rom_class_ms as rom_class

        self.speed_up_h = []  # List to store speed-up factors relative to the full-order simulation.
        self.rom_error = []   # List to store the relative errors (in percentage) of the ROM solutions.

        sol_fos_ = self.fos_test_data  # Retrieve full-order solution data for comparison.
        print(self.T_init_guess)
        # Create the initial full-order temperature guess using the prescribed initial temperature.
        T_init_fos = np.zeros(self.FOS.n_nodes) + self.T_init_guess
        # Project the full-order initial guess onto the reduced subspace using the mask.
        T_init_rom = np.transpose(self.V_sel) @ T_init_fos[self.FOS.data.mask]
        # Initialize an array to store the reconstructed ROM solution in full-order space.
        sol_rom = np.zeros_like(T_init_fos)

        # Iterate over the test parameters (up to the specified number of ROM snapshots).
        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            print(i)
            # Update the full-order system's material parameter with the current test parameter.
            self.FOS.data.mu = self.param_list_test[i]
            
            tic_rom = time.perf_counter()  # Start timing the ROM solve.
            # Instantiate the ROM solver using the DEIM method.
            ROM_h = rom_class.rom_deim(self.FOS.data, self.deim, self.quad_deg, mean=self.mean)
            # Solve the ROM system starting from the reduced initial guess.
            NL_solution_p_reduced = ROM_h.solve_rom(T_init_rom, self.xi, self.V_sel)
            toc_rom = time.perf_counter()  # End timing the ROM solve.
            
            # Compute the simulation time for the ROM and calculate the speed-up factor.
            rom_sim_time = toc_rom - tic_rom
            self.speed_up_h.append(self.fos_test_time[i] / rom_sim_time)

            # Reconstruct the full-order solution by projecting the reduced solution back using the basis
            # and adding the mean field.
            sol_rom[self.FOS.data.mask] = np.dot(self.V_sel, NL_solution_p_reduced) + self.mean
            # For nodes with Dirichlet conditions, assign the prescribed temperature.
            sol_rom[~self.FOS.data.mask] = self.FOS.data.T_dir

            # Store the ROM solution for this snapshot.
            self.NL_solutions_rom.append(np.copy(sol_rom))

            sol_fos = sol_fos_[i]  # Retrieve the corresponding full-order solution.
            # Compute and record the relative error (percentage) between the ROM and full-order solutions.
            self.rom_error.append(
                np.linalg.norm(sol_rom[self.FOS.data.mask] - sol_fos) * 100 / np.linalg.norm(sol_fos)
            )

    def run_simulation_h_ecsw(self):
        """
        Run the ROM simulation using the ECSW hyper-reduction method.
        For each test parameter, solve the reduced system and reconstruct the full-order solution,
        while recording speed-up factors and ROM errors.
        """
        import src.codes.reductor.rom_class_ms as rom_class

        self.speed_up_h = []  # List to store speed-up factors.
        self.rom_error = []   # List to store ROM errors.

        sol_fos_ = self.fos_test_data  # Full-order solution data.

        # Create the initial full-order temperature guess.
        T_init_fos = np.zeros(self.FOS.n_nodes) + self.T_init_guess
        # Project the initial guess onto the reduced subspace.
        T_init_rom = np.transpose(self.V_sel) @ T_init_fos[self.FOS.data.mask]
        sol_rom = np.zeros_like(T_init_fos)  # Initialize the reconstructed ROM solution.

        # Loop over each test parameter (up to the specified ROM snapshots).
        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            # Update the full-order data with the current parameter.
            self.FOS.data.mu = self.param_list_test[i]
            
            tic_rom = time.perf_counter()  # Start the timer.
            # Instantiate the ROM solver using the ECSW method.
            ROM_h = rom_class.rom_ecsw(self.FOS.data, self.quad_deg, mean=self.mean)
            # Solve the reduced system starting from the initial reduced guess.
            NL_solution_p_reduced = ROM_h.solve_rom(T_init_rom, self.xi, self.V_sel)
            toc_rom = time.perf_counter()  # Stop the timer.
            
            # Calculate the ROM simulation time and the corresponding speed-up factor.
            rom_sim_time = toc_rom - tic_rom
            self.speed_up_h.append(self.fos_test_time[i] / rom_sim_time)

            # Reconstruct the full-order solution from the reduced solution.
            sol_rom[self.FOS.data.mask] = np.dot(self.V_sel, NL_solution_p_reduced) + self.mean
            sol_rom[~self.FOS.data.mask] = self.FOS.data.T_dir  # Enforce Dirichlet boundary conditions.
            store = np.copy(sol_rom)  # Make a copy of the reconstructed solution.
            self.NL_solutions_rom.append(store)

            sol_fos = sol_fos_[i]  # Retrieve the full-order solution for comparison.
            # Compute and store the relative error (percentage) of the ROM solution.
            self.rom_error.append(
                np.linalg.norm(sol_rom[self.FOS.data.mask] - sol_fos) * 100 / np.linalg.norm(sol_fos)
            )

    def run_simulation_h_ecm(self):
        """
        Run the ROM simulation using the ECM hyper-reduction technique.
        For each test parameter, solve the reduced system, reconstruct the full-order solution,
        and record speed-up factors and relative errors.
        """
        import src.codes.reductor.rom_class_ms as rom_class

        self.speed_up_h = []  # List to record speed-up factors.
        self.rom_error = []   # List to record ROM errors.
        
        sol_fos_ = self.fos_test_data  # Retrieve full-order solution data.

        # Initialize the full-order temperature guess.
        T_init_fos = np.zeros(self.FOS.n_nodes) + self.T_init_guess
        # Project the guess onto the reduced basis subspace.
        T_init_rom = np.transpose(self.V_sel) @ T_init_fos[self.FOS.data.mask]
        sol_rom = np.zeros_like(T_init_fos)  # Initialize the ROM solution array.

        # Loop over test parameters for ROM simulation.
        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            # Update the full-order data parameter.
            self.FOS.data.mu = self.param_list_test[i]
            
            tic_rom = time.perf_counter()  # Start timing the ROM computation.
            # Instantiate the ROM solver using the ECM method.
            ROM_h = rom_class.rom_ecm(self.FOS.data, self.quad_deg, mean=self.mean)
            # Solve the ROM system with the reduced initial guess.
            NL_solution_p_reduced = ROM_h.solve_rom(T_init_rom, self.xi, self.V_sel)
            toc_rom = time.perf_counter()  # End timing.
            
            # Determine the ROM simulation time and compute the speed-up factor.
            rom_sim_time = toc_rom - tic_rom
            self.speed_up_h.append(self.fos_test_time[i] / rom_sim_time)

            # Reconstruct the full-order solution by mapping the reduced solution back using the reduced basis.
            sol_rom[self.FOS.data.mask] = np.dot(self.V_sel, NL_solution_p_reduced) + self.mean
            sol_rom[~self.FOS.data.mask] = self.FOS.data.T_dir  # Apply prescribed boundary conditions.
            store = np.copy(sol_rom)  # Copy the reconstructed solution.
            self.NL_solutions_rom.append(store)

            sol_fos = sol_fos_[i]  # Retrieve the corresponding full-order solution.
            # Calculate and record the relative error (percentage) of the ROM reconstruction.
            self.rom_error.append(
                np.linalg.norm(sol_rom[self.FOS.data.mask] - sol_fos) * 100 / np.linalg.norm(sol_fos)
            )

    def run_simulation(self):
        """
        Run the standard ROM simulation without additional hyper-reduction.
        For each test parameter, solve the reduced system, reconstruct the full-order solution,
        and record the speed-up factor and relative ROM error.
        """
        import src.codes.reductor.rom_class_ms as rom_class

        self.speed_up = []  # List to store speed-up factors for the standard ROM simulation.
        self.rom_error = []  # List to store the relative errors of the ROM reconstructions.
        
        sol_fos_ = self.fos_test_data  # Full-order solution data for reference.

        # Create the initial temperature guess for the full-order model.
        T_init_fos = np.zeros(self.FOS.n_nodes) + self.T_init_guess
        # Project this guess into the reduced space using the selected basis and the mask.
        T_init_rom = np.transpose(self.V_sel) @ T_init_fos[self.FOS.data.mask]
        sol_rom = np.zeros_like(T_init_fos)  # Initialize the full-order ROM solution array.

        # Iterate over the test parameters for the ROM simulation.
        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            # Update the full-order data parameter with the current test parameter.
            self.FOS.data.mu = self.param_list_test[i]

            tic_rom = time.perf_counter()  # Start timing the ROM solve.
            # Instantiate the standard ROM solver.
            ROM = rom_class.rom(self.FOS.data, self.quad_deg, mean=self.mean)
            # Solve the ROM system starting from the reduced initial guess.
            NL_solution_p_reduced = ROM.solve_rom(T_init_rom, self.V_sel)
            toc_rom = time.perf_counter()  # End timing the ROM solve.
            
            # Calculate the ROM simulation time and the corresponding speed-up factor.
            rom_sim_time = toc_rom - tic_rom
            self.speed_up.append(self.fos_test_time[i] / rom_sim_time)

            # Reconstruct the full-order solution by mapping the reduced solution back 
            # and adding the mean field.
            sol_rom[self.FOS.data.mask] = np.dot(self.V_sel, NL_solution_p_reduced) + self.mean
            # Enforce Dirichlet boundary conditions on the remaining nodes.
            sol_rom[~self.FOS.data.mask] = self.FOS.data.T_dir

            # Store the reconstructed ROM solution for this snapshot.
            self.NL_solutions_rom.append(np.copy(sol_rom))

            sol_fos = sol_fos_[i]  # Retrieve the corresponding full-order solution.
            # Compute and record the relative ROM error (percentage) using the L2 norm.
            self.rom_error.append(
                np.linalg.norm(sol_rom[self.FOS.data.mask] - sol_fos) * 100 / np.linalg.norm(sol_fos)
            )

class ROM_simulation_UQ:
    def __init__(self, f_cls, test_data, param_list, V_sel, xi=None, deim=None, T_init_guess=273.15, N_rom_snap=None, fos_comp=True):
        """
        Initialize the ROM simulation class for uncertainty quantification (UQ).
        This class manages the reduced order simulation for a range of parameter values.
        
        Parameters:
            f_cls: An object that contains the full-order solver (FOS) and associated simulation data.
            test_data: Full-order simulation results used as reference data for the ROM.
            param_list: List or array of parameter values used in the simulation.
            V_sel: The reduced basis matrix used to project the full-order model.
            xi: (Optional) A sampling indicator vector used in hyper-reduction methods.
            deim: (Optional) A DEIM object for hyper-reduction if DEIM is used.
            T_init_guess: Initial guess for the temperature field (default is 273.15).
            N_rom_snap: (Optional) Number of ROM snapshots to simulate; if not provided, uses all parameters.
            fos_comp: Boolean flag indicating if full-order simulation comparison should be performed.
        """
        # Extract the full-order solver from the provided f_cls object.
        self.FOS = f_cls.FOS
        # Store the initial temperature guess.
        self.T_init_guess = T_init_guess
        # Save the full-order test data (reference solutions).
        self.fos_test_data = test_data
        # Store the parameter list to be used in the ROM simulation.
        self.param_list_test = param_list
        # (Commented out: full-order simulation times based on a test mask can be extracted if needed.)
        # self.fos_test_time = np.asarray(f_cls.fos_time)[Test_mask]
        # Store the reduced basis matrix.
        self.V_sel = V_sel
        # Save the DEIM object if hyper-reduction via DEIM is applied.
        self.deim = deim
        # Save the sampling indicator vector if provided.
        self.xi = xi
        # Extract the quadrature degree from the full-order solver.
        self.quad_deg = f_cls.quad_deg
        # Initialize an empty list to store ROM solutions.
        self.NL_solutions_rom = []
        # Store the full-order data object.
        self.d = f_cls.FOS.data
        # Save the mean field used for ROM reconstruction.
        self.mean = f_cls.mean
        # Flag to determine whether to compare with full-order simulation solutions.
        self.fos_comp = fos_comp

        # Determine the number of ROM snapshots to simulate.
        if N_rom_snap is not None:
            self.N_rom_snap = N_rom_snap
        else:
            self.N_rom_snap = len(self.param_list_test)

    def run_simulation_h_deim(self):
        """
        Run the ROM simulation using the DEIM hyper-reduction method.
        For each parameter in the test set, the method:
          - Updates the full-order data with the current parameter.
          - Constructs a ROM solver using DEIM.
          - Solves the reduced system starting from an initial guess projected onto the reduced subspace.
          - Reconstructs the full-order solution from the reduced solution.
          - Computes the relative error compared to the full-order solution.
        """
        # Import the ROM classes for hyper-reduction from the reductor module.
        import src.codes.reductor.rom_class_ms as rom_class

        # Initialize a list to store ROM errors (relative errors in percentage).
        self.rom_error = []
        
        # Retrieve the full-order test solutions for later error computation.
        sol_fos_ = self.fos_test_data

        # Create the initial full-order temperature guess (all nodes set to T_init_guess).
        T_init_fos = np.zeros(self.FOS.n_nodes) + self.T_init_guess
        # Project the full-order initial guess into the reduced subspace using the selected basis and mask.
        T_init_rom = np.transpose(self.V_sel) @ T_init_fos[self.FOS.data.mask]
        # Initialize an array to hold the reconstructed full-order solution.
        sol_rom = np.zeros_like(T_init_fos)

        # Loop over each ROM snapshot, up to the specified number.
        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            # Update the full-order data's parameter (mu) with the current test parameter.
            self.FOS.data.mu = self.param_list_test[i]
            
            # Instantiate the ROM solver using the DEIM method.
            ROM_h = rom_class.rom_deim(self.FOS.data, self.deim, self.quad_deg, mean=self.mean)
            # Solve the reduced system starting from the initial reduced guess.
            NL_solution_p_reduced = ROM_h.solve_rom(T_init_rom, self.xi, self.V_sel)
            
            # Reconstruct the full-order solution from the reduced solution:
            # Multiply the reduced basis with the reduced solution and add the mean field.
            sol_rom[self.FOS.data.mask] = np.dot(self.V_sel, NL_solution_p_reduced) + self.mean
            # For nodes with Dirichlet boundary conditions, set the prescribed temperature.
            sol_rom[~self.FOS.data.mask] = self.FOS.data.T_dir

            # Append a copy of the reconstructed solution to the list of ROM solutions.
            self.NL_solutions_rom.append(np.copy(sol_rom))

            # Retrieve the corresponding full-order solution for error computation.
            sol_fos = sol_fos_[i]
            # Compute the relative error (in percentage) using the L2 norm over the free nodes.
            self.rom_error.append(
                np.linalg.norm(sol_rom[self.FOS.data.mask] - sol_fos) * 100 / np.linalg.norm(sol_fos)
            )
           
    def run_simulation_h_ecsw(self):
        """
        Run the ROM simulation using the ECSW hyper-reduction method.
        For each test parameter, the method:
          - Updates the full-order data parameter.
          - Constructs a ROM solver using the ECSW method.
          - Solves the reduced system and reconstructs the full-order solution.
          - If full-order comparison is enabled, computes the relative error.
        """
        import src.codes.reductor.rom_class_ms as rom_class

        # Initialize the list for ROM error metrics.
        self.rom_error = []
        
        # Retrieve the full-order test solutions.
        sol_fos_ = self.fos_test_data

        # Initialize the full-order temperature guess.
        T_init_fos = np.zeros(self.FOS.n_nodes) + self.T_init_guess
        # Project the initial guess onto the reduced space.
        T_init_rom = np.transpose(self.V_sel) @ T_init_fos[self.FOS.data.mask]
        # Create an array to hold the reconstructed full-order solution.
        sol_rom = np.zeros_like(T_init_fos)

        # Loop over the ROM snapshots.
        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            # Update the full-order parameter with the current test value.
            self.FOS.data.mu = self.param_list_test[i]
            
            # Instantiate the ROM solver using the ECSW method.
            ROM_h = rom_class.rom_ecsw(self.FOS.data, self.quad_deg, mean=self.mean)
            # Solve the reduced system from the reduced initial guess.
            NL_solution_p_reduced = ROM_h.solve_rom(T_init_rom, self.xi, self.V_sel)
            
            # (Optional: update T_init_rom with the current reduced solution for subsequent iterations.)
            # T_init_rom = NL_solution_p_reduced

            # Reconstruct the full-order solution by projecting back with the reduced basis and adding the mean.
            sol_rom[self.FOS.data.mask] = np.dot(self.V_sel, NL_solution_p_reduced) + self.mean
            sol_rom[~self.FOS.data.mask] = self.FOS.data.T_dir
            
            # Store a copy of the reconstructed solution.
            self.NL_solutions_rom.append(np.copy(sol_rom))
            
            # If full-order comparison is enabled, compute the relative error.
            if self.fos_comp:
                sol_fos = sol_fos_[i]
                self.rom_error.append(
                    np.linalg.norm(sol_rom[self.FOS.data.mask] - sol_fos) * 100 / np.linalg.norm(sol_fos)
                )

    def run_simulation(self):
        """
        Run the standard ROM simulation (without additional hyper-reduction techniques).
        For each test parameter, the method:
          - Updates the full-order data with the current parameter.
          - Constructs a ROM solver (standard method).
          - Solves the reduced system and reconstructs the full-order solution.
          - Computes the relative error compared to the full-order solution.
        """
        import src.codes.reductor.rom_class_ms as rom_class

        # Initialize the ROM error list.
        self.rom_error = []
        
        # Retrieve the full-order test solutions.
        sol_fos_ = self.fos_test_data

        # Create the initial temperature guess for the full-order model.
        T_init_fos = np.zeros(self.FOS.n_nodes) + self.T_init_guess
        # Project this guess onto the reduced subspace.
        T_init_rom = np.transpose(self.V_sel) @ T_init_fos[self.FOS.data.mask]
        # Initialize an array to hold the reconstructed full-order solution.
        sol_rom = np.zeros_like(T_init_fos)

        # Iterate over each test parameter for the ROM simulation.
        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            # Update the parameter in the full-order data.
            self.FOS.data.mu = self.param_list_test[i]

            # Instantiate the standard ROM solver.
            ROM = rom_class.rom(self.FOS.data, self.quad_deg, mean=self.mean)
            # Solve the ROM system starting from the reduced initial guess.
            NL_solution_p_reduced = ROM.solve_rom(T_init_rom, self.V_sel)
            
            # (Optional: update T_init_rom with the current solution for iterative improvement.)
            # T_init_rom = NL_solution_p_reduced
            
            # Reconstruct the full-order solution from the reduced solution.
            sol_rom[self.FOS.data.mask] = np.dot(self.V_sel, NL_solution_p_reduced) + self.mean
            # Enforce Dirichlet boundary conditions on nodes not in the free set.
            sol_rom[~self.FOS.data.mask] = self.FOS.data.T_dir

            # Append the reconstructed solution to the ROM solution list.
            self.NL_solutions_rom.append(np.copy(sol_rom))

            # Retrieve the full-order solution for error computation.
            sol_fos = sol_fos_[i]
            # Compute and store the relative error (percentage) between ROM and full-order solutions.
            self.rom_error.append(
                np.linalg.norm(sol_rom[self.FOS.data.mask] - sol_fos) * 100 / np.linalg.norm(sol_fos)
            )

class ROM_simulation_affine:
    def __init__(self, f_cls, test_data, param_list, Test_mask, V_sel, N_rom_snap=None):
        """
        Initialize the ROM simulation class for the affine case.
        
        Parameters:
            f_cls: Object containing the full-order solver (FOS) and simulation data.
            test_data: Full-order simulation results for comparison.
            param_list: Array or list of parameter values.
            Test_mask: Boolean or index mask to select test parameters from param_list.
            V_sel: The selected reduced basis matrix.
            N_rom_snap: (Optional) Number of ROM snapshots to simulate; if not provided, use all test parameters.
        """
        self.FOS = f_cls.FOS
        self.fos_test_data = test_data
        self.param_list_test = param_list[Test_mask]
        self.fos_test_time = np.asarray(f_cls.fos_time)[Test_mask]
        self.V_sel = V_sel
        self.quad_deg = f_cls.quad_deg
        self.NL_solutions_rom = []
        self.d = f_cls.FOS.data
        self.mean = f_cls.mean

        if N_rom_snap is not None:
            self.N_rom_snap = N_rom_snap
        else:
            self.N_rom_snap = len(self.param_list_test)

    def run_simulation(self):
        """
        Run the ROM simulation for the affine case.
        For each selected test parameter, the method:
          - Updates the full-order data parameter.
          - Constructs the ROM solver for the affine model.
          - Computes the reduced system matrices and right-hand side.
          - Solves for the reduced solution.
          - Reconstructs the full-order solution from the reduced solution.
          - Computes the speed-up factor and ROM error.
        
        Notes:
          - The external heat source contributions 'qe' are computed without applying Dirichlet BC.
          - The corresponding contributions 'fe' are computed with Dirichlet BC at the element level.
          - The effective right-hand side for the reduced system is obtained as the difference between 
            qe (without Dirichlet BC) and fe (with Dirichlet BC).
        """
        import src.codes.reductor.rom_class_ms as rom_class

        self.speed_up = []
        self.rom_error = []
        
        sol_fos_ = self.fos_test_data

        for i in range(len(self.param_list_test[:self.N_rom_snap])):
            sol_rom = np.zeros(self.FOS.n_nodes)
            
            self.FOS.data.mu = self.param_list_test[i]

            tic_rom = time.perf_counter()

            ROM = rom_class.rom_affine(self.FOS.data, self.quad_deg, mean=self.mean)
            
            if i < 1:
                # ROM.solve_rom returns:
                #   K_r: List of reduced stiffness matrices associated with the conductivity terms.
                #   K_r_mean: List of mean contributions from the stiffness matrices.
                #   rhs_qe_: List of external heat source contributions (qe) computed without Dirichlet BC.
                #   rhs_fe_: List of corresponding contributions (fe) computed with Dirichlet BC at the element level.
                K_r, K_r_mean, rhs_qe_, rhs_fe_ = ROM.solve_rom(self.V_sel)

            # Accumulate the reduced stiffness matrices and their mean contributions using the conductivity properties.
            K_rom = np.zeros_like(K_r[0])
            K_rom_mean = np.zeros_like(K_r_mean[0])
            RHS_fe = np.zeros_like(rhs_fe_[0])

            for j in range(len(self.FOS.data.fdict["cond"])):
                k = self.FOS.data.fdict["cond"][j]
                K_rom += k(0, self.param_list_test[i][0]) * K_r[j]
                K_rom_mean += k(0, self.param_list_test[i][0]) * K_r_mean[j]
                # 'fe' here represents the contribution computed with Dirichlet BC at the element level.
                RHS_fe += k(0, self.param_list_test[i][0]) * rhs_fe_[j]

            # Compute the external heat source contributions without Dirichlet BC (qe).
            rhs_rom = np.zeros_like(rhs_qe_[0])
            RHS_qe = np.zeros_like(rhs_qe_[0])

            for j in range(len(self.FOS.data.fdict["qext"])):
                q = self.FOS.data.fdict["qext"][j]
                RHS_qe += q(0, self.param_list_test[i][1]) * rhs_qe_[j]

            # The effective reduced right-hand side is the difference between the contribution without
            # Dirichlet BC (qe) and the one with Dirichlet BC (fe).
            rhs_rom = RHS_qe - RHS_fe

            NL_solution_p_reduced = np.linalg.solve(K_rom, rhs_rom - K_rom_mean)

            toc_rom = time.perf_counter()
            
            rom_sim_time = toc_rom - tic_rom
            self.speed_up.append(self.fos_test_time[i] / rom_sim_time)

            sol_rom[self.FOS.data.mask] = np.dot(self.V_sel, NL_solution_p_reduced) + self.mean
            sol_rom[~self.FOS.data.mask] = self.FOS.data.T_dir

            self.NL_solutions_rom.append(np.copy(sol_rom))

            sol_fos = sol_fos_[i]
            self.rom_error.append(
                np.linalg.norm(sol_rom[self.FOS.data.mask] - sol_fos) * 100 / np.linalg.norm(sol_fos)
            )
