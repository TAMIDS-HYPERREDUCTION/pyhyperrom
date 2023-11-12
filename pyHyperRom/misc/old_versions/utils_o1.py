## Generic FEM functions. These are applicable for all kinds of FEM problems. ##
from .basic import *


def handle_boundary_conditions(self, bc):
    # self refers to data class
    """
    Identify nodes corresponding to Dirichlet boundary conditions and their values for n-dimensional grids.
    Parameters:
    - bc: Dictionary containing boundary conditions. 
            Expected keys depend on the dimensionality and should include 'min' and 'max' for each dimension, 
            with each having subkeys 'type' and 'value'.
    Returns:
    - dir_nodes: List of node indices that have Dirichlet boundary conditions
    - T_dir: List of boundary condition values for the corresponding nodes in dir_nodes
    """
    dir_nodes = []
    T_dir = []

    def node(*args):
        index = 0
        multiplier = 1
        for i, n in enumerate(args):
            index += n * multiplier
            if i < len(self.npts) - 1:
                multiplier *= self.npts[i]
        return index


    if self.dim_ == 1:
        if bc['x_min']['type'] != 'refl':
            dir_nodes.append(node(0))
            T_dir.append(bc['x_min']['value'])
        if bc['x_max']['type'] != 'refl':
            dir_nodes.append(node(self.npts[0]-1))
            T_dir.append(bc['x_max']['value'])
    
    elif self.dim_ == 2:
        for i in range(self.npts[0]):
            if bc['y_min']['type'] != 'refl':
                dir_nodes.append(node(i, 0))
                T_dir.append(bc['y_min']['value'])
            if bc['y_max']['type'] != 'refl':
                dir_nodes.append(node(i, self.npts[1]-1))
                T_dir.append(bc['y_max']['value'])
        for j in range(self.npts[1]):
            if bc['x_min']['type'] != 'refl':
                dir_nodes.append(node(0, j))
                T_dir.append(bc['x_min']['value'])
            if bc['x_max']['type'] != 'refl':
                dir_nodes.append(node(self.npts[0]-1, j))
                T_dir.append(bc['x_max']['value'])
    
    elif self.dim_ == 3:
        if bc['z_min']['type'] != 'refl':
            for i in range(self.npts[0]):
                for j in range(self.npts[1]):
                    dir_nodes.append( node(i,j,0) )
                    T_dir.append(bc['z_min']['value'])
        if bc['z_max']['type'] != 'refl':
            for i in range(self.npts[0]):
                for j in range(self.npts[1]):
                    dir_nodes.append( node(i,j,self.npts[2]-1) )
                    T_dir.append(bc['z_max']['value'])
        if bc['y_min']['type'] != 'refl':
            for i in range(self.npts[0]):
                for k in range(self.npts[2]):
                    dir_nodes.append( node(i,0,k) )
                    T_dir.append(bc['y_min']['value'])
        if bc['y_max']['type'] != 'refl':
            for i in range(self.npts[0]):
                for k in range(self.npts[2]):
                    dir_nodes.append( node(i,self.npts[1]-1,k) )
                    T_dir.append(bc['y_max']['value'])
        if bc['x_min']['type'] != 'refl':
            for j in range(self.npts[1]):
                for k in range(self.npts[2]):
                    dir_nodes.append( node(0,j,k) )
                    T_dir.append(bc['x_min']['value'])
        if bc['x_max']['type'] != 'refl':
            for j in range(self.npts[1]):
                for k in range(self.npts[2]):
                    dir_nodes.append( node(self.npts[0]-1,j,k) )
                    T_dir.append(bc['x_max']['value'])


    dir_nodes = np.asarray(dir_nodes)
    T_dir = np.asarray(T_dir)
    
    dir_nodes, index = np.unique(dir_nodes, return_index=True)
    T_dir = T_dir[index]

    indx = np.argsort(dir_nodes)

    dir_nodes = dir_nodes[indx]
    T_dir = T_dir[indx]
    
    self.dir_nodes = dir_nodes
    self.T_dir = T_dir


def get_glob_node_equation_id(self, dir_nodes):
    # self refers to data class
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
    node_eqnId = np.zeros(self.n_verts).astype(int)

    # Get a list of unique global nodes in the mesh
    glob_nodes = np.unique(self.gn)

    # Loop over all nodes in the mesh
    for i in range(len(node_eqnId)):
        # Check if the current global node corresponds to a Dirichlet boundary condition
        if np.isin(glob_nodes[i], dir_nodes):
            # Set equation number to 0 for nodes with Dirichlet boundary conditions
            node_eqnId[i] = 0
        else:
            # Assign the next available equation number to the current node
            node_eqnId[i] = int(max(node_eqnId)) + 1

    self.node_eqnId = node_eqnId


def get_element_global_nodes_and_nonzero_eqnId(self, iel, node_eqnId):
    # self refers to data class

    # Function to get global node numbers, equation IDs, and non-zero equation IDs for an element

    # Get the global node numbers associated with this element
    elem_glob_nodes = self.gn[iel, :]
    Le_ = np.zeros((len(elem_glob_nodes), self.n_verts))

    # Get the equation IDs associated with these nodes
    elem_glob_node_eqnId = node_eqnId[elem_glob_nodes]
  
    # Find nodes of the element that are not associated with Dirichlet boundaries
    nonzero_mask = elem_glob_node_eqnId != 0
    elem_glob_node_nonzero_eqnId = elem_glob_node_eqnId[nonzero_mask]
    elem_local_node_nonzero_eqnId = np.nonzero(nonzero_mask)[0]

    for i, ind_i in enumerate(elem_glob_nodes):
        Le_[i, ind_i] = 1

    mask = node_eqnId != 0

    self.Le.append(Le_[elem_local_node_nonzero_eqnId][:,mask])        
    self.glob_node_eqnId.append(elem_glob_node_eqnId)
    self.glob_node_nonzero_eqnId.append(elem_glob_node_nonzero_eqnId)
    self.local_node_nonzero_eqnId.append(elem_local_node_nonzero_eqnId)


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


# Initialize global matrices and vectors
def init_global_systems(max_node_eqnId):
    K = sparse.lil_matrix((max_node_eqnId, max_node_eqnId))
    J = sparse.lil_matrix((max_node_eqnId, max_node_eqnId))
    rhs = np.zeros(max_node_eqnId)
    return K, J, rhs


# Compute element matrices for each element
def compute_element_matrices(self, sol_prev, iel):
    return self.element_matrices(sol_prev, iel)


# Compute element matrices for each element
def assemble_global_matrices(K, J, Ke_, Je_, elem_glob_node_nonzero_eqnId, elem_local_node_nonzero_eqnId):
    I_index, J_index = np.meshgrid(elem_glob_node_nonzero_eqnId-1, elem_glob_node_nonzero_eqnId-1)
    i_index, j_index = np.meshgrid(elem_local_node_nonzero_eqnId, elem_local_node_nonzero_eqnId)
    K[I_index, J_index] += Ke_[i_index, j_index]
    J[I_index, J_index] += Je_[i_index, j_index]
    Ke_d_ = Ke_[i_index, j_index]
    return K, J, Ke_d_


# Main function to evaluate residual and Jacobian
def eval_resJac(self, mask, dir_nodes, sol_dir, sol_prev, node_eqnId):
    # self denotes the FOS class
    # Initialize global matrices and vectors
    K, J, rhs = init_global_systems(max(node_eqnId))

    # Lists to store element matrices and vectors
    Ke = []
    rhs_e = []

    # Loop over all elements in the domain
    for iel in range(self.data.n_cells):

        elem_glob_nodes = self.data.gn[iel, :]

        # Get global node numbers and equation IDs for the current element
        elem_glob_node_eqnId = self.data.glob_node_eqnId[iel]
        elem_glob_node_nonzero_eqnId = self.data.glob_node_nonzero_eqnId[iel]
        elem_local_node_nonzero_eqnId = self.data.local_node_nonzero_eqnId[iel]
        
        # Compute element matrices for the current element
        Ke_, Je_, qe_ = compute_element_matrices(self, sol_prev, iel)

        # Assemble global matrices
        K, J, Ke_d_ = assemble_global_matrices(K, J, Ke_, Je_, elem_glob_node_nonzero_eqnId, elem_local_node_nonzero_eqnId)

        # Check and handle Dirichlet boundary conditions
        if np.isin(0, elem_glob_node_eqnId):
            elem_dof_values = dirichlet_bc(self, sol_dir, dir_nodes, elem_glob_nodes)
            fe = Ke_ @ elem_dof_values.reshape(-1, 1)
        else:
            fe = np.zeros((len(elem_glob_nodes), 1))

        # Compute the right-hand side for the element
        rhs_e_ = qe_[elem_local_node_nonzero_eqnId] - fe[elem_local_node_nonzero_eqnId].flatten()
        rhs[elem_glob_node_nonzero_eqnId-1] += rhs_e_

        # Append the element matrices and vectors to the lists
        rhs_e.append(rhs_e_)
        Ke.append(Ke_d_)

    res = K @ sol_prev[mask] - rhs

    return K + J, res, Ke, rhs_e


def solve_fos(self, sol_init, tol=1e-5, max_iter=300):
    
    """
    Solve the nonlinear system using a Newton-Raphson method.

    Parameters:
    - cond_arr: Conductivity array
    - qext_arr: External heat source array
    - sol_init: Initial guess for the temperature field
    - tol: Tolerance for convergence (default is 1e-5)
    - max_iter: Maximum number of iterations (default is 300)

    Returns:
    - T: Solved temperature field
    - Le: List of element matrices
    - Ke: List of element stiffness matrices
    - rhs_e: Right-hand side element vectors
    - mask: Mask indicating which nodes have non-zero equation IDs
    - sol_dir: Temperature values at the Dirichlet boundary nodes
    """

    # Handle boundary conditions and get node equation IDs    
    node_eqnId = self.node_eqnId

    # Create a mask for nodes that do not have a Dirichlet boundary condition
    mask = node_eqnId != 0

    # Update initial temperature values for Dirichlet boundary nodes
    sol_init[~mask] = self.sol_dir

    # Copy the initial temperature field
    sol = np.copy(sol_init)

    # Evaluate the Jacobian, residual, and other relevant matrices/vectors
    Jac, res, Ke, rhs_e = eval_resJac(self, mask, self.dir_nodes, self.sol_dir, sol, node_eqnId)

    # Compute the initial norm of the residual
    norm_ = np.linalg.norm(res)
    print('initial residual =', norm_, "\n")

    it = 0

    # Start the Newton-Raphson iterative process
    
    while (it < max_iter) and not(norm_ < tol):
        # Solve for the temperature increment (delta) using the current Jacobian and residual
        delta = linalg.spsolve(Jac.tocsc(), -res)

        # Update the temperature field (excluding Dirichlet) using the computed increment
        sol[mask] += delta

        # Re-evaluate the Jacobian, residual, and other relevant matrices/vectors
        Jac, res, Ke, rhs_e = eval_resJac(self, mask, self.dir_nodes, self.sol_dir, sol, node_eqnId)

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

    return sol.reshape(-1, 1), Ke, rhs_e, mask


def makedirs(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)
 
        
def plot3D(x,y,z,Z,hmap=False, sz = 1.0, clr = 'b', save_file=False):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    Z3d = np.reshape(Z,(len(z),len(y),len(x))).T
    
    xx, yy, zz = np.meshgrid(x,y,z)
    
    if hmap==True:
        sc = ax.scatter(xx.flatten(), yy.flatten(), zz.flatten(), c=Z3d, cmap='hot',s=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylabel('z')
        plt.colorbar(sc)
        plt.show()
        
    else:
        # Second scatter plot
        ax.scatter(xx.flatten(), yy.flatten(), zz.flatten(), c=clr, s=sz)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylabel('z')
        
    
    if save_file:
        from pyevtk.hl import gridToVTK
        gridToVTK("./structured"+f"{i}"+"100",  x,  y,  z, \
                  pointData = {"temp" : Z3d})
            #     cellData = {"cond" : d.cell2mat_layout, "qext" : d.cell2src_layout},\

    

# def eval_resJac_old(self, mask, dir_nodes, sol_dir, sol_prev, node_eqnId):
#     """
#     Evaluate the residual and Jacobian matrix for a given temperature field.

#     Parameters:
        
#     - cond_arr: Conductivity array
#     - qext_arr: External heat source array
#     - T_prev: Previous temperature field
#     - node_eqnId: Node equation IDs

#     Returns:
#     - K + J: Sum of stiffness matrix and Jacobian
#     - res: Residual vector
#     - Le: List of element matrices
#     - Ke: List of element stiffness matrices
#     - rhs_e: List of right-hand side element vectors
#     """
    
#     # Initialize matrices and vectors for global system
#     K = sparse.lil_matrix((max(node_eqnId), max(node_eqnId)))
#     J = sparse.lil_matrix((max(node_eqnId), max(node_eqnId)))
#     rhs = np.zeros(max(node_eqnId))

#     # Lists to store element matrices and vectors
#     Le = []
#     Ke = []
#     rhs_e = []

#     # Loop over all elements (cells) in the domain
#     for iel in range(self.data.n_cells):
#         # Get the global node numbers associated with this element
#         elem_glob_nodes = self.data.gn[iel, :]

#         # Get the equation IDs associated with these nodes
#         elem_glob_node_eqnId = node_eqnId[elem_glob_nodes]

#         # Find nodes of the element that are not associated with Dirichlet boundaries
#         nonzero_mask = elem_glob_node_eqnId != 0
#         elem_glob_node_nonzero_eqnId = elem_glob_node_eqnId[nonzero_mask]
#         elem_local_node_nonzero_eqnId = np.nonzero(nonzero_mask)[0]

#         # Compute the element matrices for the current element
#         Ke_, Je_, qe_, Le_ = self.element_matrices(sol_prev, iel)

#         # Mapping from local to global DOFs
#         I_index, J_index = np.meshgrid(elem_glob_node_nonzero_eqnId-1, elem_glob_node_nonzero_eqnId-1)
#         i_index, j_index = np.meshgrid(elem_local_node_nonzero_eqnId, elem_local_node_nonzero_eqnId)

#         # Assemble the global matrices
#         K[I_index, J_index] += Ke_[i_index, j_index]
#         J[I_index, J_index] += Je_[i_index, j_index]

#         # Check and handle Dirichlet boundary conditions
#         if np.isin(0, elem_glob_node_eqnId):
#             elem_dof_values = dirichlet_bc(self,sol_dir, dir_nodes, elem_glob_nodes)
#             fe = Ke_ @ elem_dof_values.reshape(-1, 1)
#         else:
#             fe = np.zeros((len(elem_glob_nodes), 1))

#         # Compute the right-hand side for the element
#         rhs_e_ = qe_[elem_local_node_nonzero_eqnId] - fe[elem_local_node_nonzero_eqnId].flatten()
#         rhs[elem_glob_node_nonzero_eqnId-1] += rhs_e_

#         # Append the element matrices and vectors to the lists
#         rhs_e.append(rhs_e_)
#         Le.append(Le_[elem_local_node_nonzero_eqnId][:,mask])
#         Ke.append(Ke_[i_index, j_index])

#     # Compute the global residual
#     res = K @ sol_prev[mask] - rhs

#     return K + J, res, Le, Ke, rhs_e