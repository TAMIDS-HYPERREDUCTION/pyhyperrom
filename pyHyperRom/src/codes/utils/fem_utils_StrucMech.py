from ..basic import *
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spalg
import control as ctrl
from pylab import *

def handle_boundary_conditions(cls_data, bc):
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
        - cls_data.dir_nodes: Sets the global node numbers subject to Dirichlet boundary conditions.
        - cls_data.sol_dir: Sets the associated values for the nodes specified in dir_nodes.
    
    Example usage:
    obj.handle_boundary_conditions(bc)
    """
    
    dir_nodes = []
    sol_dir = []


    if cls_data.dim_ == 1:

        if bc['x_min']['type'] != 'refl':
            dir_nodes.append(0)
            sol_dir.append(bc['x_min']['value'])

        if bc['x_min_theta']['type'] != 'refl':
            dir_nodes.append(1)
            sol_dir.append(bc['x_min_theta']['value'])

        if bc['x_max']['type'] != 'refl':
            dir_nodes.append(cls_data.n_verts-2)
            sol_dir.append(bc['x_max']['value'])

        if bc['x_min_theta']['type'] != 'refl':
            dir_nodes.append(cls_data.n_verts-1)
            sol_dir.append(bc['x_min_theta']['value'])
    


    dir_nodes = np.asarray(dir_nodes)
    sol_dir = np.asarray(sol_dir)
    
    dir_nodes, index = np.unique(dir_nodes, return_index=True)
    sol_dir = sol_dir[index]

    indx = np.argsort(dir_nodes)

    dir_nodes = dir_nodes[indx]
    sol_dir = sol_dir[indx]
    
    cls_data.dir_nodes = dir_nodes
    cls_data.sol_dir = sol_dir

def get_glob_node_equation_id(cls_data, dir_nodes):
    """
    Function: get_glob_node_equation_id
    Overview: This function assigns equation IDs to the global nodes in a finite element mesh.
    Nodes that correspond to Dirichlet boundary conditions are assigned an ID of 0.
    
    Inputs:
    - self: Refers to the data class that contains the mesh and finite element information.
    - dir_nodes: A list of global node numbers that correspond to Dirichlet boundary conditions.
    
    Outputs:
    - Modifies the following class attribute:
        - cls_data.node_eqnId: Sets an array of equation IDs corresponding to each global node in the mesh.
    
    Example usage:
    obj.get_glob_node_equation_id(dir_nodes)
    """
    # Initialize an array to store global equation numbers for each node
    node_eqnId = np.zeros(cls_data.n_verts).astype(int)

    # Get a list of unique global nodes in the mesh
    glob_nodes = np.unique(cls_data.gn)

    # Loop over all nodes in the mesh
    for i in range(len(node_eqnId)):
        # Check if the current global node corresponds to a Dirichlet boundary condition
        if np.isin(glob_nodes[i], dir_nodes):
            # Set equation number to 0 for nodes with Dirichlet boundary conditions
            node_eqnId[i] = 0
        else:
            # Assign the next available equation number to the current node
            node_eqnId[i] = int(max(node_eqnId)) + 1

    cls_data.node_eqnId = node_eqnId

def get_element_global_nodes_and_nonzero_eqnId(cls_data, iel, node_eqnId):
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
        - cls_data.Le: List of local element matrices.
        - cls_data.glob_node_eqnId: List of global node equation IDs for each element.
        - cls_data.glob_node_nonzero_eqnId: List of global node equation IDs not associated with Dirichlet boundaries.
        - cls_data.local_node_nonzero_eqnId: List of local node equation IDs not associated with Dirichlet boundaries.
        - cls_data.global_indices: List of global indices for each element, used in global stiffness matrices.
        - cls_data.local_indices: List of local indices for each element, used in local stiffness matrices.
    
    Example usage:
    obj.get_element_global_nodes_and_nonzero_eqnId(iel, node_eqnId)
    """
    
    elem_glob_nodes = cls_data.gn[iel, :]
    Le_ = np.zeros((len(elem_glob_nodes), cls_data.n_verts))

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

    cls_data.Le.append(Le_[elem_local_node_nonzero_eqnId][:,mask])        
    cls_data.glob_node_eqnId.append(elem_glob_node_eqnId)
    cls_data.glob_node_nonzero_eqnId.append(elem_glob_node_nonzero_eqnId)
    cls_data.local_node_nonzero_eqnId.append(elem_local_node_nonzero_eqnId)
    cls_data.global_indices.append(elem_global_indices)
    cls_data.local_indices.append(elem_local_indices)

def dirichlet_bc(cls_data, sol_dir, dir_nodes, elem_glob_nodes):
    """
    Function: dirichlet_bc
    Overview: This function applies Dirichlet boundary conditions to a given element in the mesh.
    It identifies the nodes of the element that correspond to Dirichlet boundaries and assigns
    the associated temperature values to them.
    
    Inputs:
    - self: Refers to the data class that contains the mesh and finite element information.
    - sol_dir: An array containing the temperature values associated with Dirichlet boundary nodes.
    - dir_nodes: A list of global node numbers that correspond to Dirichlet boundary conditions.
    - elem_glob_nodes: A list of global node numbers associated with the current element.
    
    Outputs:
    - Returns an array 'z' containing the Dirichlet boundary condition values for the local degrees of freedom (DOFs) of the element.
    
    Example usage:
    z = obj.dirichlet_bc(sol_dir, dir_nodes, elem_glob_nodes)
    """

    # Initialize an array to store the Dirichlet boundary condition values for the local DOFs
    z = np.zeros(len(elem_glob_nodes)).astype(int)

    # Identify which nodes of the current element have Dirichlet boundary conditions
    mask = np.isin(elem_glob_nodes, dir_nodes)

    # Assign the Dirichlet boundary condition values to the local DOFs        
    for idx in np.where(mask)[0]:
        dir_index = np.searchsorted(dir_nodes, elem_glob_nodes[idx])
        z[idx] = sol_dir[dir_index]

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
    - Returns the initialized K, M, and rhs:
        - K: Stiffness matrix, represented as a sparse lil_matrix.
        - J: Jacobian matrix, represented as a sparse lil_matrix.
        - rhs: Right-hand side array, initialized to zeros.
    
    Example usage:
    K, M, rhs = init_global_systems(max_node_eqnId)
    """
    K = sparse.lil_matrix((max_node_eqnId, max_node_eqnId))
    M = sparse.lil_matrix((max_node_eqnId, max_node_eqnId))
    rhs = np.zeros(max_node_eqnId)
    return K, M, rhs

def compute_element_matrices(cls_data, iel,  t=None, eval_stiff = True, eval_forces =True, notconstant = True):
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
    Ke_ = None
    Me_ = None
    Ce_ = None
    qe_ = None

    if eval_stiff:
        Ke_,Me_,Ce_ = cls_data.element_KM_matrices_fn_x(iel, notconstant)
    
    if eval_forces:
        qe_ = cls_data.element_F_matrices_fn_x(iel, t)

        

    return Ke_,Me_,Ce_,qe_

def assemble_global_matrices(cls_data, iel, K, M, Ke_, Me_, Ce_,xi_iel):

    I_index, J_index = cls_data.data.global_indices[iel][0], cls_data.data.global_indices[iel][1]
    i_index, j_index = cls_data.data.local_indices[iel][0] , cls_data.data.local_indices[iel][1] 
    
    K[I_index, J_index] += xi_iel*Ke_[i_index, j_index]
    M[I_index, J_index] += xi_iel*Me_[i_index, j_index]
    # C[I_index, J_index] += Ce_[i_index, j_index] # we can avoid this for now. we can generate C later combining M and K.
    

    Ke_d_ = xi_iel*Ke_[i_index, j_index]
    Ce_d_ = xi_iel*Ce_[i_index, j_index]

    
    return K, M, Ke_d_, Ce_d_

def check_values(xi_):
    if xi_ is None:
        return False
    elif isinstance(xi_, (list, tuple, set, np.ndarray)):  # Check if xi_ is an iterable
        return any(value for value in xi_)
    else:
        return False

def global_KM_matrices(cls_data, node_eqnId, xi_=None):

    # Initialize global matrices and vectors
    K, M, _ = init_global_systems(max(node_eqnId))

    # Lists to store element matrices and vectors
    Ke_d = []
    Ce_d = []
    Ke = []
    
    if check_values(xi_):
        xi = xi_
        
    else:
        xi = np.ones(cls_data.data.n_cells)
            

    # Loop over all elements in the domain
    for iel in range(cls_data.data.n_cells):
        
        if xi[iel] == 0:
            continue
        
        # Compute element matrices for the current element
        Ke_, Me_, Ce_, _ = compute_element_matrices(cls_data, iel, eval_stiff = True, eval_forces=False, notconstant = True)

        # Assemble global matrices
        K, M, Ke_d_, Ce_d_ = assemble_global_matrices(cls_data, iel, K, M, Ke_, Me_, Ce_, xi[iel])

        # Append the element matrices and vectors to the lists
        Ke_d.append(Ke_d_)     
        Ke.append(Ke_)
        
        Ce_d.append(Ce_d_)
        
    
    cls_data.Ke_d = Ke_d
    cls_data.Ke = Ke

    cls_data.Ce_d = Ce_d

    return K,M

def assemble_global_forces_t(cls_data, iel, qe_, Ke_, rhs, xi_iel):

    elem_glob_nodes = cls_data.data.gn[iel, :]
    elem_glob_node_nonzero_eqnId = cls_data.data.glob_node_nonzero_eqnId[iel]
    elem_glob_node_eqnId = cls_data.data.glob_node_eqnId[iel]
    elem_local_node_nonzero_eqnId = cls_data.data.local_node_nonzero_eqnId[iel]

    # Check and handle Dirichlet boundary conditions
    if np.isin(0, elem_glob_node_eqnId):
        elem_dof_values = dirichlet_bc(cls_data, cls_data.sol_dir, cls_data.dir_nodes, elem_glob_nodes)
        fe = Ke_ @ elem_dof_values.reshape(-1, 1)
    else:
        fe = np.zeros((len(elem_glob_nodes), 1))

    fe = fe + 0.0*qe_
    
    # Compute the right-hand side for the element
    rhs_e_ = qe_[elem_local_node_nonzero_eqnId] - fe[elem_local_node_nonzero_eqnId]
    rhs[elem_glob_node_nonzero_eqnId-1] += xi_iel*rhs_e_

    return rhs, rhs_e_

def global_F_matrix_t(cls_data, node_eqnId,t, xi_=None):

    rhs_e = []
    _, _, rhs = init_global_systems(max(node_eqnId))
    
    rhs = rhs.reshape(-1, 1) + 0.0*t

    if check_values(xi_):
        xi = xi_
        
    else:
        xi = np.ones(cls_data.data.n_cells)

    iel_loop = 0            
    # Loop over all elements in the domain
    for iel in range(cls_data.data.n_cells):
        
        if xi[iel] == 0:
            continue

        _, _,_, qe_ = compute_element_matrices(cls_data, iel, t, eval_stiff = False, eval_forces=True)
        rhs, rhs_e_ = assemble_global_forces_t(cls_data, iel, qe_, cls_data.Ke[iel_loop], rhs, xi[iel])
        rhs_e.append(rhs_e_)

        iel_loop+=1
        
    cls_data.rhs = rhs
    return rhs_e, rhs

def convert_second_to_first_order(K, M, C, rhs, t):
    """
    Converts a second order system to a first order form.

    Parameters:
    K (numpy.ndarray): Stiffness matrix.
    M (numpy.ndarray): Mass matrix.
    C (numpy.ndarray): Damping matrix.
    rhs (numpy.ndarray, N_d x N_t): Force function.

    Returns:
    tuple: A tuple containing matrices A_sys, B_sys, C_sys, D_sys, and U.
    """

    # Create A_sys matrix
    M_inv = spalg.spsolve(M, sp.eye(M.shape[0]))  # Compute M inverse once
    MK_inv = spalg.spsolve(M, K).toarray()
    MC_inv = spalg.spsolve(M, C).toarray()
    
    A_sys = np.block([
        [np.zeros_like(M.toarray()), np.eye(M.shape[0])],
        [-MK_inv, -MC_inv]
    ])

    # Create B_sys matrix
    B_sys = np.zeros_like(A_sys)
    B_sys[M.shape[0]:, M.shape[0]:] = M_inv

    # Create C_sys2 matrix
    C_sys = np.eye(A_sys.shape[0])

    # Create D_sys matrix (assuming zero matrix for a purely dynamic system)
    D_sys = np.zeros((C_sys.shape[0], B_sys.shape[1]))

    # Solve for U
    U = np.zeros((B_sys.shape[0], len(t)))
    U[M.shape[0]:, :] = rhs

    return A_sys, B_sys, C_sys, D_sys, U

def condense_matrices(K, M, dofs_to_keep, dofs_to_condense):
    # Ensure input DOFs are numpy arrays
    dofs_to_keep = np.array(dofs_to_keep)
    dofs_to_condense = np.array(dofs_to_condense)

    # Verify if the indices are within the matrix dimensions
    if np.any(dofs_to_keep >= K.shape[0]) or np.any(dofs_to_condense >= K.shape[0]):
        raise ValueError("DOF indices exceed matrix dimensions")

    # Partitioning the matrices based on DOFs
    Kkk = K[np.ix_(dofs_to_keep, dofs_to_keep)]
    Kcc = K[np.ix_(dofs_to_condense, dofs_to_condense)]
    Kkc = K[np.ix_(dofs_to_keep, dofs_to_condense)]

    Mkk = M[np.ix_(dofs_to_keep, dofs_to_keep)]
    Mcc = M[np.ix_(dofs_to_condense, dofs_to_condense)]
    Mkc = M[np.ix_(dofs_to_keep, dofs_to_condense)]

    # Check if Kcc is singular
    if np.isclose(np.linalg.det(Kcc), 0):
        raise ValueError("Matrix Kcc is singular or nearly singular")

    # Condensation using the transpose of Kkc
    G_oc = -np.linalg.solve(Kcc, Kkc.T)
    TM = np.vstack([np.eye(G_oc.shape[1]), G_oc])  # Adjusting the size of the identity matrix

    # Combining the indices and condensing the matrices
    combined_index = np.concatenate((dofs_to_keep, dofs_to_condense))
    K_combined = K[np.ix_(combined_index, combined_index)]
    M_combined = M[np.ix_(combined_index, combined_index)]

    K_cond = TM.T.dot(K_combined).dot(TM)
    M_cond = TM.T.dot(M_combined).dot(TM)

    # Ensuring symmetry
    K_cond = (K_cond + K_cond.T) / 2
    M_cond = (M_cond + M_cond.T) / 2

    return K_cond, M_cond

def solve_fos_dynamics(cls_data, sol_init, cv=1e-2, cm=1e-4):
    
    # Handle boundary conditions and get node equation IDs    
    node_eqnId = cls_data.node_eqnId

    # Create a mask for nodes that do not have a Dirichlet boundary condition
    mask = node_eqnId != 0
    cls_data.mask = mask

    # Update initial temperature values for Dirichlet boundary nodes
    K, M = global_KM_matrices(cls_data, node_eqnId)
    # cond_dof = np.append(np.r_[0:200:2],199)
    # keep_dof = np.r_[1:199:2]

    # K_cond, M_cond = condense_matrices(K, M, keep_dof, cond_dof)
   
    C = cm*K + cv*M
    
    dt = cls_data.data.dt
    t = cls_data.data.t


    rhs_e, rhs = global_F_matrix_t(cls_data, node_eqnId, t)
    
    A_sys, B_sys, C_sys, D_sys, U = convert_second_to_first_order(K, M, C, rhs, t)


    ###
    #  special modification to C_sys, so that we can only look at the mid point of the beam. Disp and Vel.
    C_sys = np.zeros((1,C_sys.shape[0]))
    C_sys[0,99] = 1
    # C_sys[0,299] = 1

    D_sys = np.zeros((C_sys.shape[0],D_sys.shape[1]))
    ###


    # Create the state-space model
    full_sys = ctrl.ss(A_sys, B_sys, C_sys, D_sys)
        
    x0 = np.pad(sol_init[mask], (len(sol_init[mask]), 0), mode='constant', constant_values=0)
    t_out, _, x_out = ctrl.forced_response(full_sys, T=t, U=U, X0=x0, return_x=True)

    x_sol= np.zeros((len(sol_init),len(t)))      
    x_sol[mask,:] = x_out[:200,:]
    
    xd_sol= np.zeros((len(sol_init),len(t)))      
    xd_sol[mask,:] = x_out[200:,:]
    
    X = np.vstack([x_sol,xd_sol])
    
    
    return t_out, X, rhs_e, cls_data.Ke_d, cls_data.Ce_d, mask, U, full_sys


# def global_F_matrix(cls_data, node_eqnId,t=0):

#     rhs_e = []
#     _, _, rhs = init_global_systems(max(node_eqnId))

#     # Loop over all elements in the domain
#     for iel in range(cls_data.data.n_cells):

#         _, _, qe_ = compute_element_matrices(cls_data, iel, t, eval_stiff = False, eval_forces=True)
#         rhs, rhs_e_ = assemble_global_forces(cls_data, iel, qe_, cls_data.Ke[iel], rhs)
#         rhs_e.append(rhs_e_)
    
#     cls_data.rhs = rhs
#     return rhs_e, rhs


# def assemble_global_forces(cls_data, iel, qe_, Ke_, rhs, xi_iel):

#     elem_glob_nodes = cls_data.data.gn[iel, :]
#     elem_glob_node_nonzero_eqnId = cls_data.data.glob_node_nonzero_eqnId[iel]
#     elem_glob_node_eqnId = cls_data.data.glob_node_eqnId[iel]
#     elem_local_node_nonzero_eqnId = cls_data.data.local_node_nonzero_eqnId[iel]

#     # Check and handle Dirichlet boundary conditions
#     if np.isin(0, elem_glob_node_eqnId):
#         elem_dof_values = dirichlet_bc(cls_data, cls_data.sol_dir, cls_data.dir_nodes, elem_glob_nodes)
#         fe = Ke_ @ elem_dof_values.reshape(-1, 1)
#     else:
#         fe = np.zeros((len(elem_glob_nodes), 1))

#     # Compute the right-hand side for the element
#     rhs_e_ = qe_[elem_local_node_nonzero_eqnId] - fe[elem_local_node_nonzero_eqnId].flatten()
#     rhs[elem_glob_node_nonzero_eqnId-1] += xi_iel*rhs_e_

#     return rhs, rhs_e_