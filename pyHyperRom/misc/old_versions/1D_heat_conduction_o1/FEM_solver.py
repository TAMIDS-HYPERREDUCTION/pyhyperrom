from utils import *

class FEM_solver:

    def __init__(self, data, quad_degree):
        """
        Initialize the class with given data and quadrature degree.

        Parameters:
        - data: Provided data object containing mesh information
        - quad_degree: Quadrature degree for numerical integration
        """

        # Store the provided data
        self.data = data

        # Store some shortcuts for frequently accessed data attributes
        self.n_cells = data.n_cells
        self.n_nodes = data.n_nodes
        self.dx = np.copy(data.dx)

        # Compute the continuous Finite Element (cFEM) basis functions
        self.basis()

        # Compute the elemental matrices for the given quadrature degree
        self.basis_q(quad_degree)


    def basis(self):
        """
        Define the basis functions and their derivatives for the interval [-1, 1].
        """

        # List of basis functions defined on the interval [-1, 1] in counter-clockwise ordering
        self.b = []
        self.b.append(lambda u: (1 - u) / 2)
        self.b.append(lambda u: (1 + u) / 2)

        # Derivatives of the basis functions with respect to the local coordinate 'u'
        self.dbdx = []
        self.dbdx.append(lambda u: -0.5)
        self.dbdx.append(lambda u:  0.5)


    def basis_q(self, quad_degree, verbose=False):
        """
        Compute the basis functions and their derivatives at the quadrature points.

        Parameters:
        - quad_degree: Degree of the Gauss-Legendre quadrature
        - verbose: Flag to control print output (default is False)
        """

        # Use Gauss-Legendre quadrature to get the quadrature points 'xq' and weights 'wq' for the given degree
        [xq, self.wq] = np.polynomial.legendre.leggauss(quad_degree)

        # Initialize arrays to store values of basis functions and their derivatives at the quadrature points
        self.bq  = np.zeros((len(xq), len(self.b)))
        self.dbq = np.zeros_like(self.bq)

        # Evaluate the basis functions and their derivatives at each quadrature point
        for q, uq in enumerate(xq):
            for i, (fi, fxi) in enumerate(zip(self.b, self.dbdx)):
                self.bq[q, i] = fi(uq)
                self.dbq[q, i] = fxi(uq)

        
    def handle_boundary_conditions(self, bc):
        """
        Identify nodes corresponding to Dirichlet boundary conditions and their values.

        Parameters:
        - bc: Dictionary containing boundary conditions. 
              Expected keys are 'xmin' and 'xmax', with each having subkeys 'type' and 'value'.

        Returns:
        - dir_nodes: List of node indices that have Dirichlet boundary conditions
        - T_dir: List of boundary condition values for the corresponding nodes in dir_nodes
        """

        # Lists to store nodes with Dirichlet boundary conditions and their corresponding values
        dir_nodes = []
        T_dir = []

        # Check the boundary condition at the left end ('xmin') of the domain
        if bc['xmin']['type'] != 'refl':
            dir_nodes.append(0)
            T_dir.append(bc['xmin']['value'])

        # Check the boundary condition at the right end ('xmax') of the domain
        if bc['xmax']['type'] != 'refl':
            dir_nodes.append(self.n_nodes - 1)
            T_dir.append(bc['xmax']['value'])

        return np.asarray(dir_nodes), T_dir

    
    def dirichlet_bc(self, T_dir, dir_nodes, elem_glob_nodes):
        """
        Assign Dirichlet boundary condition values to the local degrees of freedom of an element.

        Parameters:
        - T_dir: List of boundary condition values
        - dir_nodes: List of node indices that have Dirichlet boundary conditions
        - elem_glob_nodes: Global node numbers associated with the current element

        Returns:
        - z: Array containing the Dirichlet boundary condition values for the local DOFs of the element
        """

        # Initialize an array to store the Dirichlet boundary condition values for the local DOFs
        z = np.zeros(len(elem_glob_nodes)).astype(int)

        # Identify which nodes of the current element have Dirichlet boundary conditions
        mask = np.isin(elem_glob_nodes, dir_nodes)

        # Assign the Dirichlet boundary condition values to the local DOFs
        for idx in np.where(mask)[0]:
            dir_index = np.searchsorted(dir_nodes, elem_glob_nodes[idx])
            z[idx] = T_dir[dir_index]

        return z

              
    def get_glob_node_equation_id(self, dir_nodes):
        """
        Assign a global equation number to each degree of freedom (DOF) of the mesh.
        For DOFs corresponding to Dirichlet boundary conditions, the equation number is set to 0.
        Only non-zero equation numbers will be solved.

        Parameters:
        - dir_nodes: Nodes corresponding to Dirichlet boundary conditions

        Returns:
        - node_eqnId: Array of global equation numbers for each node in the mesh
        """

        # Initialize an array to store global equation numbers for each node
        node_eqnId = np.zeros(self.n_nodes).astype(int)

        # Get a list of unique global nodes in the mesh
        glob_nodes = np.unique(self.data.gn)

        # Loop over all nodes in the mesh
        for i in range(len(node_eqnId)):
            # Check if the current global node corresponds to a Dirichlet boundary condition
            if np.isin(glob_nodes[i], dir_nodes):
                # Set equation number to 0 for nodes with Dirichlet boundary conditions
                node_eqnId[i] = 0
            else:
                # Assign the next available equation number to the current node
                node_eqnId[i] = int(max(node_eqnId)) + 1

        return node_eqnId

        
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

        # Initialize arrays to store temperature and its derivative at the quadrature points
        T_prev_q = np.zeros(len(self.bq))
        dT_prev_q = np.zeros(len(self.bq))

        # Loop over all nodes associated with the current element
        for i, ind_i in enumerate(elem_glob_nodes):
            # Evaluate temperature at the quadrature points using the FE basis functions
            T_prev_q += self.bq[:, i] * T_prev[ind_i]

            # Evaluate temperature derivative at the quadrature points using the FE basis function derivatives
            dT_prev_q += self.dbq[:, i] * T_prev[ind_i]

        return T_prev_q, dT_prev_q
 
    
    def element_matrices(self, cond_arr, qext_arr, T_prev, iel, node_eqnId):
        """
        Compute the element matrices and vectors for a given temperature field.

        Parameters:
        - cond_arr: Conductivity array
        - qext_arr: External heat source array
        - T_prev: Previous temperature field
        - iel: Current element index
        - node_eqnId: Node equation IDs

        Returns:
        - Ke_: Element stiffness matrix
        - Je_: Element Jacobian matrix
        - qe_: Element source vector
        - Le_: Element matrix
        """

        # Retrieve material and geometric data for the current element
        mu = self.data.mu
        imat = self.data.cell2mat[iel]
        k, dkdT = cond_arr[imat]
        qext = qext_arr[imat]

        # Evaluate temperature and its derivative at quadrature points
        T_prev_q, dT_prev_q = self.eval_at_quadrature_points(T_prev, self.data.gn[iel, :])
        cond_q = k(T_prev_q, mu)
        dcond_q = dkdT(T_prev_q, mu)
        qext_q = qext(T_prev_q, mu)

        # Get global node numbers associated with the current element
        elem_glob_nodes = self.data.gn[iel, :]

        # Initialize element matrices and vectors
        n = len(elem_glob_nodes)
        Ke_ = np.zeros((n, n))
        Je_ = np.zeros((n, n))
        qe_ = np.zeros(n)
        Le_ = np.zeros((n, self.n_nodes))

        # Fill Le_ matrix using the global node numbers
        for i, ind_i in enumerate(elem_glob_nodes):
            Le_[i, ind_i] = 1

        # Compute the element matrices Ke_ and Je_ using nested loops
        for i in range(n):
            # Compute source vector for the current node
            qe_[i] = (self.dx[iel] / 2) * np.dot(self.wq * self.bq[:, i], qext_q)

            for j in range(n):
                
                # Compute stiffness matrix entry for the current pair of nodes
                Ke_[i, j] = 2 / self.dx[iel] * np.dot(self.wq * self.dbq[:, i], cond_q * self.dbq[:, j])

                # Compute Jacobian matrix entry for the current pair of nodes
                Je_[i, j] = 2 / self.dx[iel] * np.dot(self.wq * self.dbq[:, i], dcond_q * self.bq[:, j] * dT_prev_q)

        return Ke_, Je_, qe_, Le_

       
    def eval_resJac(self, cond_arr, qext_arr, bc, T_prev, node_eqnId):
        """
        Evaluate the residual and Jacobian matrix for a given temperature field.

        Parameters:
        - cond_arr: Conductivity array
        - qext_arr: External heat source array
        - bc: Boundary conditions
        - T_prev: Previous temperature field
        - node_eqnId: Node equation IDs

        Returns:
        - K + J: Sum of stiffness matrix and Jacobian
        - res: Residual vector
        - Le: List of element matrices
        - Ke: List of element stiffness matrices
        - rhs_e: List of right-hand side element vectors
        """

        # Handle boundary conditions and get Dirichlet nodes and their temperature values
        dir_nodes, T_dir = self.handle_boundary_conditions(bc)

        # Create a mask for non-zero node equation IDs
        mask = node_eqnId != 0

        # Initialize matrices and vectors for global system
        K = sparse.lil_matrix((max(node_eqnId), max(node_eqnId)))
        J = sparse.lil_matrix((max(node_eqnId), max(node_eqnId)))
        rhs = np.zeros(max(node_eqnId))

        # Lists to store element matrices and vectors
        Le = []
        Ke = []
        rhs_e = []

        # Loop over all elements (cells) in the domain
        for iel in range(self.n_cells):
            # Get the global node numbers associated with this element
            elem_glob_nodes = self.data.gn[iel, :]

            # Get the equation IDs associated with these nodes
            elem_glob_node_eqnId = node_eqnId[elem_glob_nodes]

            # Find nodes of the element that are not associated with Dirichlet boundaries
            nonzero_mask = elem_glob_node_eqnId != 0
            elem_glob_node_nonzero_eqnId = elem_glob_node_eqnId[nonzero_mask]
            elem_local_node_nonzero_eqnId = np.nonzero(nonzero_mask)[0]

            # Compute the element matrices for the current element
            Ke_, Je_, qe_, Le_ = self.element_matrices(cond_arr, qext_arr, T_prev, iel, node_eqnId)

            # Mapping from local to global DOFs
            I_index, J_index = np.meshgrid(elem_glob_node_nonzero_eqnId-1, elem_glob_node_nonzero_eqnId-1)
            i_index, j_index = np.meshgrid(elem_local_node_nonzero_eqnId, elem_local_node_nonzero_eqnId)

            # Assemble the global matrices
            K[I_index, J_index] += Ke_[i_index, j_index]
            J[I_index, J_index] += Je_[i_index, j_index]

            # Check and handle Dirichlet boundary conditions
            if np.isin(0, elem_glob_node_eqnId):
                elem_dof_values = self.dirichlet_bc(T_dir, dir_nodes, elem_glob_nodes)
                fe = Ke_ @ elem_dof_values.reshape(-1, 1)
            else:
                fe = np.zeros((len(elem_glob_nodes), 1))

            # Compute the right-hand side for the element
            rhs_e_ = qe_[elem_local_node_nonzero_eqnId] - fe[elem_local_node_nonzero_eqnId].flatten()
            rhs[elem_glob_node_nonzero_eqnId-1] += rhs_e_

            # Append the element matrices and vectors to the lists
            rhs_e.append(rhs_e_)
            Le.append(Le_[elem_local_node_nonzero_eqnId][:,mask])
            Ke.append(Ke_[i_index, j_index])

        # Compute the global residual
        res = K @ T_prev[mask] - rhs

        return K + J, res, Le, Ke, rhs_e

    
    def solve_system(self, cond_arr, qext_arr, bc, T_init, tol=1e-5, max_iter=300):
        """
        Solve the nonlinear system using a Newton-Raphson method.

        Parameters:
        - cond_arr: Conductivity array
        - qext_arr: External heat source array
        - bc: Dictionary containing boundary conditions
        - T_init: Initial guess for the temperature field
        - tol: Tolerance for convergence (default is 1e-5)
        - max_iter: Maximum number of iterations (default is 300)

        Returns:
        - T: Solved temperature field
        - Le: List of element matrices
        - Ke: List of element stiffness matrices
        - rhs_e: Right-hand side element vectors
        - mask: Mask indicating which nodes have non-zero equation IDs
        - T_dir: Temperature values at the Dirichlet boundary nodes
        """

        # Handle boundary conditions and get node equation IDs
        dir_nodes, T_dir = self.handle_boundary_conditions(bc)
        node_eqnId = self.get_glob_node_equation_id(dir_nodes)

        # Create a mask for nodes that do not have a Dirichlet boundary condition
        mask = node_eqnId != 0

        # Update initial temperature values for Dirichlet boundary nodes
        T_init[~mask] = T_dir

        # Sanity checks
        if len(bc) != 2:
            raise ValueError('bc dictionary must have 2 keys')
        if self.data.n_zones != len(cond_arr):
            raise ValueError('len(cond_arr) /= n_zones')
        if self.data.n_zones != len(qext_arr):
            raise ValueError('len(qext_arr) /= n_zones')

        # Copy the initial temperature field
        T = np.copy(T_init)

        # Evaluate the Jacobian, residual, and other relevant matrices/vectors
        Jac, res, Le, Ke, rhs_e = self.eval_resJac(cond_arr, qext_arr, bc, T, node_eqnId)

        # Compute the initial norm of the residual
        norm_ = np.linalg.norm(res)
        print('initial residual =', norm_, "\n")

        it = 0

        # Start the Newton-Raphson iterative process
        while (it < max_iter) and not(norm_ < tol):
            # Solve for the temperature increment (delta) using the current Jacobian and residual
            delta = linalg.spsolve(Jac.tocsc(), -res)

            # Update the temperature field (excluding Dirichlet) using the computed increment
            T[mask] += delta

            # Re-evaluate the Jacobian, residual, and other relevant matrices/vectors
            Jac, res, Le, Ke, rhs_e = self.eval_resJac(cond_arr, qext_arr, bc, T, node_eqnId)

            # Compute the current norm of the residual
            norm_ = np.linalg.norm(res)

            # Print current iteration details
            print("iter {}, NL residual={}, delta={}".format(it, norm_, np.max(delta)))

            # Check convergence
            if norm_ < tol:
                print('Convergence !!!')
            else:
                if it == max_iter - 1:
                    print('\nWARNING: nonlinear solution has not converged')

            # Increment the iteration counter
            it += 1

        return T.reshape(-1, 1), Le, Ke, rhs_e, mask, T_dir
