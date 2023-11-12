from utils import *
from FEM_solver import FEM_solver

class FEM_solver_rom(FEM_solver):
    def __init__(self, data, quad_degree):
        """Initialize the reduced-order FEM solver."""
        super().__init__(data, quad_degree)
    
    
    def eval_resJac_rom(self, cond_arr, qext_arr, bc, T_red, xi, V, node_eqnId):
        """
        Evaluate the residual and Jacobian matrices for the reduced-order system.
        """
        print(self.data.mu)
        # Handle boundary conditions
        dir_nodes, T_dir = self.handle_boundary_conditions(bc)

        # Create a mask for nodes with non-zero equation IDs
        mask = node_eqnId != 0

        # Initialize sparse matrices for stiffness (K), Jacobian (J), and rhs
        K = sparse.lil_matrix((max(node_eqnId), max(node_eqnId)))
        J = sparse.lil_matrix((max(node_eqnId), max(node_eqnId)))
        rhs = np.zeros(max(node_eqnId))
        
        # Compute the previous temperature field from the reduced temperature field
        T_prev = V @ T_red.reshape(-1, 1)

        # Loop through all the elements
        for iel in range(self.n_cells):

            # Skip elements with zero importance (xi value)
            if xi[iel] == 0:
                continue

            # Obtain global nodes and their equation IDs for the current element
            elem_glob_nodes = self.data.gn[iel, :]
            elem_glob_node_eqnId = node_eqnId[elem_glob_nodes]

            # Create a mask for non-zero equation IDs and filter the global nodes accordingly
            nonzero_mask = elem_glob_node_eqnId != 0
            elem_glob_node_nonzero_eqnId = elem_glob_node_eqnId[nonzero_mask]
            elem_local_node_nonzero_eqnId = np.nonzero(nonzero_mask)[0]

            # Compute the element matrices for the current element
            Ke_, Je_, qe_, Le_ = self.element_matrices(cond_arr, qext_arr, T_prev, iel, node_eqnId)

            # Define indices for the global and local nodes
            I_index, J_index = np.meshgrid(elem_glob_node_nonzero_eqnId-1, elem_glob_node_nonzero_eqnId-1)
            i_index, j_index = np.meshgrid(elem_local_node_nonzero_eqnId, elem_local_node_nonzero_eqnId)

            # Update the global matrices by adding the weighted element matrices
            K[I_index, J_index] += Ke_[i_index, j_index] * xi[iel]
            J[I_index, J_index] += Je_[i_index, j_index] * xi[iel]

            # Handle Dirichlet boundary conditions
            if np.isin(0, elem_glob_node_eqnId):
                elem_dof_values = self.dirichlet_bc(T_dir, dir_nodes, elem_glob_nodes)
                fe = Ke_ @ elem_dof_values.reshape(-1, 1) # np.zeros((len(elem_glob_nodes), 1))  # 
            else:
                fe = np.zeros((len(elem_glob_nodes), 1))

            # Compute the right-hand side element vector
            rhs_e_ = qe_[elem_local_node_nonzero_eqnId] - fe[elem_local_node_nonzero_eqnId].flatten()

            # Update the global right-hand side vector
            rhs[elem_glob_node_nonzero_eqnId-1] += xi[iel] * rhs_e_

        # Compute the reduced residual using the projection matrix V
        res = np.transpose(V[mask, :]) @ K @ T_prev[mask] - np.transpose(V[mask, :]) @ rhs.reshape(-1, 1)

        # Return the reduced Jacobian and residual
        return np.transpose(V[mask]) @ (K + J) @ V[mask], res


    def solve_reduced_system(self, cond_arr, qext_arr, bc, T_init, xi, V, tol=1e-5, max_iter=300):
        """
        Solve the nonlinear system for the reduced-order model using Newton-Raphson iteration.
        """

        # Handle boundary conditions
        dir_nodes, T_dir = self.handle_boundary_conditions(bc)

        # Obtain global equation IDs for nodes
        node_eqnId = self.get_glob_node_equation_id(dir_nodes)

        # Create a mask for nodes with non-zero equation IDs
        mask = node_eqnId != 0

        # Ensure boundary conditions and material arrays have the correct dimensions
        if len(bc) != 2:
            raise ValueError('bc dictionary must have 2 keys')
        if self.data.n_zones != len(cond_arr):
            raise ValueError('len(cond_arr) /= n_zones')
        if self.data.n_zones != len(qext_arr):
            raise ValueError('len(qext_arr) /= n_zones')

        # Initialize the temperature field
        T = np.copy(T_init)

        # Evaluate the reduced Jacobian and residual for the initial guess
        Jac, res = self.eval_resJac_rom(cond_arr, qext_arr, bc, T, xi, V, node_eqnId)

        # Compute the initial residual norm
        norm_ = np.linalg.norm(res)
        print('initial residual =', norm_, "\n")

        # Initialize iteration counter
        it = 0

        # Newton-Raphson iteration loop
        while (it < max_iter) and (norm_ >= tol):

            # Solve the linear system for the update (delta)
            delta = linalg.spsolve(Jac, -res)

            # Update the temperature field
            T += delta

            # Re-evaluate the reduced Jacobian and residual after updating the temperature field
            Jac, res = self.eval_resJac_rom(cond_arr, qext_arr, bc, T, xi, V, node_eqnId)

            # Compute the new residual norm
            norm_ = np.linalg.norm(res)

            # Print iteration details
            print("iter {}, NL residual={}, delta={}".format(it, norm_, np.max(delta)))

            # Check for convergence
            if norm_ < tol:
                print('Convergence !!!')
            elif it == max_iter-1:
                print('\nWARNING: nonlinear solution has not converged')

            # Increment the iteration counter
            it += 1

        # Return the updated temperature field after convergence or reaching max iterations
        return T