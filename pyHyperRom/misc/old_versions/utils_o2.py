## Generic FEM functions. These are applicable for all kinds of FEM problems. ##
from .basic import *

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
    Function: init_global_systems
    Overview: This function initializes the global systems for the finite element analysis.
    It creates sparse matrices for the stiffness matrix (K) and the Jacobian matrix (J),
    as well as a zero-initialized array for the right-hand side (rhs) of the equations.
    
    Inputs:
    - max_node_eqnId: The maximum equation ID among all nodes, which determines the size of the global systems.
    
    Outputs:
    - Returns the initialized K, J, and rhs:
        - K: Stiffness matrix, represented as a sparse lil_matrix.
        - J: Jacobian matrix, represented as a sparse lil_matrix.
        - rhs: Right-hand side array, initialized to zeros.
    
    Example usage:
    K, J, rhs = init_global_systems(max_node_eqnId)
    """
    K = sparse.lil_matrix((max_node_eqnId, max_node_eqnId))
    J = sparse.lil_matrix((max_node_eqnId, max_node_eqnId))
    rhs = np.zeros(max_node_eqnId)
    return K, J, rhs

def compute_element_matrices(self, sol_prev, iel):
    """
    Function: compute_element_matrices
    Overview: This function serves as a wrapper for computing element matrices in the finite element analysis.
    It delegates the task to another method, 'element_matrices', which performs the actual calculations.
    
    Inputs:
    - self: Refers to the FOS class that contains the mesh and finite element information.
    - sol_prev: Previous solution, used for computing the new element matrices.
    - iel: Index of the element for which the matrices are to be computed.
    
    Outputs:
    - Returns the output from the 'element_matrices' method, which is typically an object or tuple
      containing the element matrices.
    
    Example usage:
    matrices = obj.compute_element_matrices(sol_prev, iel)
    """
    return self.element_matrices(sol_prev, iel)

def assemble_global_matrices(self, iel, K, J, Ke_, Je_):
    """
    Function: assemble_global_matrices
    Overview: This function assembles the global stiffness and Jacobian matrices for a given
              element specified by its index (iel). The function uses precomputed local and 
              global indices to map local element stiffness and Jacobian matrices to their 
              corresponding locations in the global matrices.
              
    Inputs:
    - self: Refers to the class instance containing mesh and finite element data.
    - iel: Index of the element for which the global matrices are to be assembled.
    - K: Global stiffness matrix.
    - J: Global Jacobian matrix.
    - Ke_: Local stiffness matrix for the element.
    - Je_: Local Jacobian matrix for the element.
    
    Outputs:
    - Returns updated global stiffness and Jacobian matrices, and a derived local stiffness matrix:
        - K: Updated global stiffness matrix.
        - J: Updated global Jacobian matrix.
        - Ke_d_: Derived local stiffness matrix based on global mapping that takes into account dirichlet nodes.
    
    Example usage:
    K, J, Ke_d_ = obj.assemble_global_matrices(iel, K, J, Ke_, Je_)
    """

    I_index, J_index = self.data.global_indices[iel][0], self.data.global_indices[iel][1]
    i_index, j_index = self.data.local_indices[iel][0] , self.data.local_indices[iel][1] 
    
    K[I_index, J_index] += Ke_[i_index, j_index]
    J[I_index, J_index] += Je_[i_index, j_index]
    Ke_d_ = Ke_[i_index, j_index]
    return K, J, Ke_d_

def assemble_global_forces(self, iel, qe_, Ke_, rhs):

    elem_glob_nodes = self.data.gn[iel, :]
    elem_glob_node_nonzero_eqnId = self.data.glob_node_nonzero_eqnId[iel]
    elem_glob_node_eqnId = self.data.glob_node_eqnId[iel]
    elem_local_node_nonzero_eqnId = self.data.local_node_nonzero_eqnId[iel]

    # Check and handle Dirichlet boundary conditions
    if np.isin(0, elem_glob_node_eqnId):
        elem_dof_values = dirichlet_bc(self, self.sol_dir, self.dir_nodes, elem_glob_nodes)
        fe = Ke_ @ elem_dof_values.reshape(-1, 1)
    else:
        fe = np.zeros((len(elem_glob_nodes), 1))

    # Compute the right-hand side for the element
    rhs_e_ = qe_[elem_local_node_nonzero_eqnId] - fe[elem_local_node_nonzero_eqnId].flatten()
    rhs[elem_glob_node_nonzero_eqnId-1] += rhs_e_

    return rhs, rhs_e_

def eval_resJac(self, mask, dir_nodes, sol_dir, sol_prev, node_eqnId):
    """
    Function: eval_resJac
    Overview: This function evaluates the residual and Jacobian matrices for the finite element analysis.
    It loops through all elements, assembles the global matrices, and handles Dirichlet boundary conditions.
    
    Inputs:
    - self: Refers to the FOS class, containing the mesh and finite element information.
    - mask: An array used to filter nodes based on certain conditions.
    - dir_nodes: A list of global node numbers corresponding to Dirichlet boundary conditions.
    - sol_dir: An array of Dirichlet boundary condition values.
    - sol_prev: Previous solution, used for computing the new element matrices.
    - node_eqnId: Array containing equation IDs for all nodes in the mesh.
    
    Outputs:
    - Returns the updated global matrices and vectors:
        - K + J: Sum of the global stiffness and Jacobian matrices.
        - res: Residual vector.
        - Ke: List of local stiffness matrices for each element.
        - rhs_e: List of right-hand side vectors for each element.
    
    Example usage:
    K_J, res, Ke, rhs_e = obj.eval_resJac(mask, dir_nodes, sol_dir, sol_prev, node_eqnId)
    """
    
    # Initialize global matrices and vectors
    K, J, rhs = init_global_systems(max(node_eqnId))

    # Lists to store element matrices and vectors
    Ke = []
    rhs_e = []

    # Loop over all elements in the domain
    for iel in range(self.data.n_cells):
        
        # Compute element matrices for the current element
        Ke_, Je_, qe_ = compute_element_matrices(self, sol_prev, iel)

        # Assemble global matrices
        K, J, Ke_d_ = assemble_global_matrices(self, iel, K, J, Ke_, Je_)

        rhs, rhs_e_ = assemble_global_forces(self, iel, qe_, Ke_, rhs)

        # Append the element matrices and vectors to the lists
        rhs_e.append(rhs_e_)
        Ke.append(Ke_d_)

    res = K @ sol_prev[mask] - rhs
    
    return K + J, res, Ke, rhs_e

def solve_fos(self, sol_init, tol=1e-5, max_iter=300, op=False):
    """
    Function: solve_fos
    Overview: This function employs the Newton-Raphson method to solve the finite element system
    of equations for a Field of Study (FOS). It iteratively updates the solution field until the residual
    norm falls below a specified tolerance or the maximum number of iterations is reached.
    
    Inputs:
    - self: Refers to the FOS class that contains the mesh and finite element information.
    - sol_init: Initial guess for the solution field.
    - tol: Tolerance for the residual norm to check for convergence (default is 1e-5).
    - max_iter: Maximum number of iterations for the Newton-Raphson process (default is 300).
    
    Outputs:
    - Returns the final solution field, local stiffness matrices, right-hand side vectors, and a mask array:
        - sol.reshape(-1, 1): Final solution field as a column vector.
        - Ke: List of local stiffness matrices for each element.
        - rhs_e: List of right-hand side vectors for each element.
        - mask: An array used to filter nodes based on certain conditions.
    
    Example usage:
    sol, Ke, rhs_e, mask = obj.solve_fos(sol_init, tol=1e-5, max_iter=300)
    """


    # Handle boundary conditions and get node equation IDs    
    node_eqnId = self.node_eqnId

    # Create a mask for nodes that do not have a Dirichlet boundary condition
    mask = node_eqnId != 0

    self.mask = mask

    # Update initial temperature values for Dirichlet boundary nodes
    sol_init[~mask] = self.sol_dir

    # Copy the initial temperature field
    sol = np.copy(sol_init)

    # Evaluate the Jacobian, residual, and other relevant matrices/vectors
    Jac, res, Ke, rhs_e = eval_resJac(self, mask, self.dir_nodes, self.sol_dir, sol, node_eqnId)

    # Compute the initial norm of the residual
    norm_ = np.linalg.norm(res)
    # norm_ = max(abs(res))

    if op:
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
        if op:
            print("iter {}, NL residual={}, delta={}".format(it, norm_, np.max(delta)))

        # Check convergence
        if norm_ < tol:
            if op:
                print('Convergence !!!')
        else:
            if it == max_iter - 1:
                print('\nWARNING: nonlinear solution has not converged')

        # Increment the iteration counter
        it += 1

    return sol.reshape(-1, 1), Ke, rhs_e, mask

def plot3D(x, y, z, Z, hmap=False, sz=1.0, clr='b', save_file=False):
    """
    Function: plot3D
    Overview: This function generates a 3D scatter plot for a given data set. It offers the option
    to either color the points based on a heatmap or use a uniform color.
    
    Inputs:
    - x, y, z: 1D arrays representing the x, y, and z coordinates of the data points.
    - Z: 1D array representing the values at each data point for the heatmap.
    - hmap: Boolean flag to enable heatmap coloring (default is False).
    - sz: Size of the scatter points (default is 1.0).
    - clr: Color for the scatter points when not using heatmap (default is 'b' for blue).
    - save_file: Boolean flag to enable saving the plot to a file (default is False).
    
    Outputs:
    - Generates a 3D scatter plot and displays it.
    - Optionally saves the plot to a VTK file if save_file is True.
    
    Example usage:
    plot3D(x, y, z, Z, hmap=True, sz=1.0, clr='b', save_file=False)
    """
   
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    Z3d = np.reshape(Z,(len(z),len(y),len(x))).T
    
    xx, yy, zz = np.meshgrid(x,y,z)
    
    if hmap==True:
        sc = ax.scatter(xx.flatten(), yy.flatten(), zz.flatten(), c=Z3d, cmap=cm.coolwarm,s=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.colorbar(sc)
        plt.show()
        
    else:
        # Second scatter plot
        ax.scatter(xx.flatten(), yy.flatten(), zz.flatten(), c=clr, s=sz)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
    
    if save_file:
        from pyevtk.hl import gridToVTK
        gridToVTK("/structured"+f"{i}"+"100",  x,  y,  z, \
                  pointData = {"temp" : Z3d})



def plot2D(x, y, Z_flat, scattr=False, clr='b', sz=1.0):
    """
    Function: plot2D_surface
    Overview: This function generates a 2D surface plot utilizing heatmap coloring based on a given dataset.
    
    Inputs:
    - x, y: 1D arrays for the x and y coordinates of the data points.
    - Z_flat: Flattened 1D array corresponding to the values at each data point for the heatmap.
    
    Outputs:
    - Produces and displays a 2D surface plot with heatmap coloring.
    
    Example usage:
    plot2D_surface(x, y, Z_flat)
    
    """    
    from matplotlib import rcParams
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.family'] = 'STIXGeneral'
    rcParams['font.size'] = 15
    plt.rc('text', usetex=True)

    X, Y = np.meshgrid(x, y)

    if scattr == False:
        sx = 4
        sy = sx * (np.max(y)-np.min(y))/ (np.max(x)-np.min(x))
        
        # Reshape the flattened Z array to 2D
        Z = Z_flat.reshape(len(y), len(x))

        fig, ax = plt.subplots(figsize=(2*sx, 2*sy),subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=1)
        ax.view_init(elev=15, azim=30)

        ax.scatter(X, Y, Z,s=0.1)
        ax.set_zlim(np.min(Z_flat),np.max(Z_flat))
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')
        # rotate as needed
        # ax.view_init(30,-60)
        ax.set_box_aspect(aspect=(1, sy/sx, 1))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$T$')


        plt.colorbar(surf)

    else:

        fig, ax = plt.subplots()
        ax.scatter((X).flatten(), (Y).flatten(), c=clr, s=sz)
    
    plt.show()



def plot1D(x, Z_flat, ax=None, scattr=False, clr='b', sz=1.0):
    """
    Function: plot1D
    Overview: This function generates a 1D plot with color mapping based on the provided dataset.
              If scattr is True, a scatter plot is produced. Otherwise, a line plot is generated.
    
    Inputs:
    - x: 1D array for the x coordinates of the data points.
    - Z_flat: Flattened 1D array corresponding to the values at each data point for the color mapping.
    - scattr: Boolean flag to switch between scatter plot (True) and line plot (False).
    - clr: Color for the line or scatter points.
    - sz: Size of the scatter points.
    
    Outputs:
    - Generates and displays a 1D plot with color mapping.
    
    Example usage:
    plot1D(x, Z_flat, scattr=True)
    """

    import random
    cmap = plt.get_cmap("tab10")

    if ax is None:
        fig, ax = plt.subplots()

    if scattr:
        scatter = ax.scatter(x,Z_flat,c=clr, s=sz)

    else:
        line_color = cmap(random.choice(range(10)))  # Randomly choose an index from 0 to 9
        line = ax.plot(x, Z_flat, color=line_color)
   
    ax.set_xlabel('x')
    ax.set_ylabel('T')
    
    # Example usage with multiple plots on the same axis
    # x = np.linspace(0, 10, 100)
    #Z_flat1 = np.sin(x)
    #Z_flat2 = np.cos(x)

    # fig, ax = plt.subplots()
    # plot1D(x, Z_flat1, ax=ax, scattr=False)
    # plot1D(x, Z_flat2, ax=ax, scattr=False)
    # plt.show()

def makedirs(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)

def svd_mode_selector(Data, tol_f=1e-3):
    
    Data = np.asarray(Data)
    U_, S_, _ = np.linalg.svd(Data.T, full_matrices=False)
    S_cumsum = np.cumsum(S_)/np.sum(S_)

    n_sel = np.where((1.0-S_cumsum)<tol_f)[0]
    n_sel = n_sel[0] if n_sel[0]!=0 else 1

    plt.figure(figsize = (6,4))
    plt.semilogy(1.0-S_cumsum,'s-')
    plt.axhline(y=tol_f, color="black", linestyle="--")

    plt.show()

    return n_sel, U_

#def node(*args):
#    index = 0
#    multiplier = 1
#    for i, n in enumerate(args):
#        index += n * multiplier
#        if i < len(self.npts) - 1:
#            multiplier *= self.npts[i]
#    return index

