from ..basic import *

def handle_boundary_conditions(self, bc):
    """
    Function: handle_boundary_conditions
    Overview: This function handles the boundary conditions for finite element models in 1D, 2D, or 3D spaces.
    It identifies the nodes that are subject to Dirichlet boundary conditions and computes their associated values.
    
    Inputs:
    - self: Refers to the data class that contains the mesh and finite element information.
    - bc: A dictionary containing boundary conditions. The keys are dimension names ('x_min', 'x_max', etc.)
          and the values are dictionaries with 'type' and 'value' fields.
    
    Outputs:
    - Modifies the following class attributes:
        - self.dir_nodes: Sets the global node numbers subject to Dirichlet boundary conditions.
        - self.T_dir: Sets the associated values for the nodes specified in dir_nodes.
    
    Example usage:
    obj.handle_boundary_conditions(bc)
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
    """
    Function: get_glob_node_equation_id
    Overview: This function assigns equation IDs to the global nodes in a finite element mesh.
    Nodes that correspond to Dirichlet boundary conditions are assigned an ID of 0.
    
    Inputs:
    - self: Refers to the data class that contains the mesh and finite element information.
    - dir_nodes: A list of global node numbers that correspond to Dirichlet boundary conditions.
    
    Outputs:
    - Modifies the following class attribute:
        - self.node_eqnId: Sets an array of equation IDs corresponding to each global node in the mesh.
    
    Example usage:
    obj.get_glob_node_equation_id(dir_nodes)
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
    """
    Function: get_element_global_nodes_and_nonzero_eqnId
    Overview: This function extracts the global and local node numbers and equation IDs for a 
              given element specified by its index (iel). It also identifies nodes not associated 
              with Dirichlet boundaries. The function computes indices used in assembling global 
              and local stiffness matrices.
    
    Inputs:
    - self: Refers to the class instance containing mesh and finite element data.
    - iel: Index of the element for which information is to be retrieved.
    - node_eqnId: Array containing equation IDs for all nodes in the mesh.
    
    Outputs:
    - Modifies several class attributes to append the computed data:
        - self.Le: List of local element matrices.
        - self.glob_node_eqnId: List of global node equation IDs for each element.
        - self.glob_node_nonzero_eqnId: List of global node equation IDs not associated with Dirichlet boundaries.
        - self.local_node_nonzero_eqnId: List of local node equation IDs not associated with Dirichlet boundaries.
        - self.global_indices: List of global indices for each element, used in global stiffness matrices.
        - self.local_indices: List of local indices for each element, used in local stiffness matrices.
    
    Example usage:
    obj.get_element_global_nodes_and_nonzero_eqnId(iel, node_eqnId)
    """
    
    elem_glob_nodes = self.gn[iel, :]
    Le_ = np.zeros((len(elem_glob_nodes), self.n_verts))

    # Get the equation IDs associated with these nodes
    elem_glob_node_eqnId = node_eqnId[elem_glob_nodes]
  
    # Find nodes of the element that are not associated with Dirichlet boundaries
    nonzero_mask = elem_glob_node_eqnId != 0
    elem_glob_node_nonzero_eqnId = elem_glob_node_eqnId[nonzero_mask]
    elem_local_node_nonzero_eqnId = np.nonzero(nonzero_mask)[0]

    elem_global_indices = np.meshgrid(elem_glob_node_nonzero_eqnId-1, elem_glob_node_nonzero_eqnId-1)
    elem_local_indices = np.meshgrid(elem_local_node_nonzero_eqnId, elem_local_node_nonzero_eqnId)


    for i, ind_i in enumerate(elem_glob_nodes):
        Le_[i, ind_i] = 1

    mask = node_eqnId != 0

    self.Le.append(Le_[elem_local_node_nonzero_eqnId][:,mask])        
    self.glob_node_eqnId.append(elem_glob_node_eqnId)
    self.glob_node_nonzero_eqnId.append(elem_glob_node_nonzero_eqnId)
    self.local_node_nonzero_eqnId.append(elem_local_node_nonzero_eqnId)
    self.global_indices.append(elem_global_indices)
    self.local_indices.append(elem_local_indices)

def dirichlet_bc(self, T_dir, dir_nodes, elem_glob_nodes):
    """
    Function: dirichlet_bc
    Overview: This function applies Dirichlet boundary conditions to a given element in the mesh.
    It identifies the nodes of the element that correspond to Dirichlet boundaries and assigns
    the associated temperature values to them.
    
    Inputs:
    - self: Refers to the data class that contains the mesh and finite element information.
    - T_dir: An array containing the temperature values associated with Dirichlet boundary nodes.
    - dir_nodes: A list of global node numbers that correspond to Dirichlet boundary conditions.
    - elem_glob_nodes: A list of global node numbers associated with the current element.
    
    Outputs:
    - Returns an array 'z' containing the Dirichlet boundary condition values for the local degrees of freedom (DOFs) of the element.
    
    Example usage:
    z = obj.dirichlet_bc(T_dir, dir_nodes, elem_glob_nodes)
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

def init_global_systems(max_node_eqnId):
    """
    Initializes global matrices and arrays for finite element computations.
    
    Parameters:
    - max_node_eqnId: Determines the size of the system based on the highest equation ID.
    
    Returns:
    - K: Sparse stiffness matrix.
    - J: Sparse Jacobian matrix.
    - rhs: Zero-initialized right-hand side array.
    """
    K = sparse.lil_matrix((max_node_eqnId, max_node_eqnId))
    J = sparse.lil_matrix((max_node_eqnId, max_node_eqnId))
    rhs = np.zeros(max_node_eqnId)
    return K, J, rhs

def compute_element_matrices(self, sol_prev, iel, affine=False):
    """
    Computes element matrices using a separate method.
    
    Parameters:
    - sol_prev: Solution from the previous step.
    - iel: Index of the element being processed.
    - affine: Optional flag to modify computation behavior.
    
    Returns:
    - Output of the element_matrices method.
    """
    return self.element_matrices(sol_prev, iel, affine=affine)

def assemble_global_matrices(self, iel, K, J, Ke_, Je_):
    """
    Integrates local element stiffness and Jacobian matrices into global matrices.
    
    Parameters:
    - iel: Element index.
    - K: Global stiffness matrix.
    - J: Global Jacobian matrix.
    - Ke_: Local stiffness matrix.
    - Je_: Local Jacobian matrix.
    
    Returns:
    - Updated K and J matrices.
    - Derived local stiffness matrix adjusted for boundary conditions.
    """
    I_index, J_index = self.data.global_indices[iel][0], self.data.global_indices[iel][1]
    i_index, j_index = self.data.local_indices[iel][0] , self.data.local_indices[iel][1] 
    
    K[I_index, J_index] += Ke_[i_index, j_index]
    J[I_index, J_index] += Je_[i_index, j_index]
    
    Ke_d_ = Ke_[i_index, j_index]
    
    if self.ecm:
        self.Ke_gauss[iel] = self.Ke_gauss[iel][:,i_index, j_index]
    
    return K, J, Ke_d_

def assemble_global_forces(self, iel, qe_, Ke_, rhs, rhs_nl):
    """
    Assembles global force vectors considering boundary conditions.
    
    Parameters:
    - iel: Element index.
    - qe_: Element force vector.
    - Ke_: Element stiffness matrix.
    - rhs: Global right-hand side vector.
    - rhs_nl: Nonlinear right-hand side vector.
    
    Returns:
    - Updated rhs and rhs_nl vectors.
    - Element-level force contribution.
    """
    elem_glob_nodes = self.data.gn[iel, :]
    elem_glob_node_nonzero_eqnId = self.data.glob_node_nonzero_eqnId[iel]
    elem_glob_node_eqnId = self.data.glob_node_eqnId[iel]
    elem_local_node_nonzero_eqnId = self.data.local_node_nonzero_eqnId[iel]

    if np.isin(0, elem_glob_node_eqnId):
        elem_dof_values = dirichlet_bc(self, self.sol_dir, self.dir_nodes, elem_glob_nodes)
        fe = Ke_ @ elem_dof_values.reshape(-1, 1)
    else:
        fe = np.zeros((len(elem_glob_nodes), 1))
    
    rhs_e_ = qe_[elem_local_node_nonzero_eqnId] - fe[elem_local_node_nonzero_eqnId].flatten()
    
    if self.ecm:
        self.fe_ecm.append(fe[elem_local_node_nonzero_eqnId].flatten())
    
    rhs[elem_glob_node_nonzero_eqnId-1] += rhs_e_
    rhs_nl[elem_glob_node_nonzero_eqnId-1] += qe_[elem_local_node_nonzero_eqnId]
    
    if self.ecm:
        self.rhs_e_gauss[iel]=self.rhs_e_gauss[iel][:,elem_local_node_nonzero_eqnId]
    
    return rhs, rhs_e_, rhs_nl

def eval_resJac(self, mask, dir_nodes, sol_dir, sol_prev, node_eqnId):
    """
    Computes the residual and Jacobian matrices for finite element analysis.
    
    Parameters:
    - mask: Filters nodes based on conditions.
    - dir_nodes: Global node numbers with Dirichlet boundary conditions.
    - sol_dir: Dirichlet boundary condition values.
    - sol_prev: Previous solution field.
    - node_eqnId: Equation IDs for all nodes.
    
    Returns:
    - Global stiffness and Jacobian matrices.
    - Residual vector.
    - Local stiffness matrices.
    - Right-hand side vectors for each element.
    """
    K, J, rhs = init_global_systems(max(node_eqnId))
    rhs_nl = np.copy(rhs)
    Ke, rhs_e = [], []

    if self.ecm:
        self.Ke_gauss, self.Je_gauss, self.rhs_e_gauss, self.fe_ecm = [], [], [], []
    
    for iel in range(self.data.n_cells):
        Ke_, Je_, qe_ = compute_element_matrices(self, sol_prev, iel)
        K, J, Ke_d_ = assemble_global_matrices(self, iel, K, J, Ke_, Je_)
        rhs, rhs_e_, rhs_nl = assemble_global_forces(self, iel, qe_, Ke_, rhs, rhs_nl)
        rhs_e.append(rhs_e_)
        Ke.append(Ke_d_)
    
    res = K @ sol_prev[mask] - rhs
    
    return K, J, res, Ke, rhs_e, rhs_nl

def solve_fos(self, sol_init, tol=1e-5, max_iter=3000, op=False):
    """
    Uses the Newton-Raphson method to solve the finite element system.
    
    Parameters:
    - sol_init: Initial solution guess.
    - tol: Residual norm tolerance for convergence.
    - max_iter: Maximum allowed iterations.
    
    Returns:
    - Final solution field.
    - Local stiffness matrices.
    - Right-hand side vectors.
    - Mask array for filtering nodes.
    """
    node_eqnId = self.node_eqnId
    mask = node_eqnId != 0
    self.mask = mask
    sol_init[~mask] = self.sol_dir
    sol = np.copy(sol_init)
    
    K, J, res, Ke, rhs_e, _ = eval_resJac(self, mask, self.dir_nodes, self.sol_dir, sol, node_eqnId)
    Jac = J + K
    norm_ = np.linalg.norm(res)
    
    if op:
        print('initial residual =', norm_)
    
    it = 0
    while (it < max_iter) and not (norm_ < tol):
        delta = linalg.spsolve(Jac.tocsc(), -res)
        sol[mask] += delta
        K, J, res, Ke, rhs_e, rhs = eval_resJac(self, mask, self.dir_nodes, self.sol_dir, sol, node_eqnId)
        Jac = J + K
        norm_ = np.linalg.norm(res)
        
        if op:
            print(f"iter {it}, NL residual={norm_}, delta={np.max(delta)}")
        
        if norm_ < tol:
            if op:
                print('Convergence achieved!')
        else:
            if it == max_iter - 1:
                print('WARNING: Nonlinear solution did not converge.')
        
        it += 1
    
    return sol.reshape(-1, 1), Ke, rhs_e, mask, rhs

def assemble_global_forces_affine(self, iel, qe_, Ke_, rhs_qe_, rhs_fe_):
    """
    Assembles affine global force vectors while handling boundary conditions.
    
    Parameters:
    - iel: Element index.
    - qe_: Element force vector.
    - Ke_: Element stiffness matrix.
    - rhs_qe_: Global force vector from element forces.
    - rhs_fe_: Global force vector from boundary conditions.
    
    Returns:
    - Updated global force vectors.
    """
    elem_glob_nodes = self.data.gn[iel, :]
    elem_glob_node_nonzero_eqnId = self.data.glob_node_nonzero_eqnId[iel]
    elem_glob_node_eqnId = self.data.glob_node_eqnId[iel]
    elem_local_node_nonzero_eqnId = self.data.local_node_nonzero_eqnId[iel]
    
    if np.isin(0, elem_glob_node_eqnId):
        elem_dof_values = dirichlet_bc(self, self.sol_dir, self.dir_nodes, elem_glob_nodes)
        fe = Ke_ @ elem_dof_values.reshape(-1, 1)
    else:
        fe = np.zeros((len(elem_glob_nodes), 1))
    
    rhs_qe_[elem_glob_node_nonzero_eqnId-1] += qe_[elem_local_node_nonzero_eqnId]
    rhs_fe_[elem_glob_node_nonzero_eqnId-1] += fe[elem_local_node_nonzero_eqnId].flatten()
    
    return rhs_qe_, rhs_fe_

def eval_KF_affine(self):
    """
    Evaluates affine global stiffness matrices and force vectors.
    
    This function initializes the required global structures, iterates over all elements,
    computes element-level contributions, and assembles them into corresponding global matrices.
    
    Returns:
    - K_aff: List of affine global stiffness matrices for different material properties.
    - rhs_qe_: List of force vectors due to external sources.
    - rhs_fe_: List of force vectors due to fixed boundary conditions.
    - sol: Initial solution vector.
    - mask: Boolean array indicating free degrees of freedom.
    """
    node_eqnId = self.node_eqnId
    mask = node_eqnId != 0
    self.mask = mask
    
    # Initialize global stiffness matrices and force vectors
    K, _, rhs = init_global_systems(max(node_eqnId))
    n_mat = int(np.max(self.data.cell2mat_layout) + 1)
    n_src = int(np.max(self.data.cell2src_layout) + 1)
    
    K_aff = [K.copy() for _ in range(n_mat)]
    rhs_qe_ = [np.copy(rhs) for _ in range(n_src)]
    rhs_fe_ = [np.copy(rhs) for _ in range(n_src)]
    
    # Initialize the solution vector
    sol = np.zeros(self.data.n_verts)
    
    # Iterate through all elements and assemble matrices
    for iel in range(self.data.n_cells):
        cell_idx = tuple(self.e_n_2ij(iel))
        imat = self.data.cell2mat_layout[cell_idx].astype(int)
        isrc = self.data.cell2src_layout[cell_idx].astype(int)
        
        # Compute element matrices
        Ke_, _, qe_ = compute_element_matrices(self, sol, iel, affine=True)
        
        # Assemble affine global stiffness and force contributions
        K_aff[imat], _, _ = assemble_global_matrices(self, iel, K_aff[imat], K.copy(), Ke_, np.zeros_like(Ke_))
        rhs_qe_[isrc], rhs_fe_[isrc] = assemble_global_forces_affine(self, iel, qe_, Ke_, rhs_qe_[isrc], rhs_fe_[isrc])
    
    # Apply Dirichlet boundary conditions
    sol[~mask] = self.sol_dir
    
    return K_aff, rhs_qe_, rhs_fe_, sol, mask
