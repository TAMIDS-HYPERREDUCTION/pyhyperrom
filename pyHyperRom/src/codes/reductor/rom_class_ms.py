from src.codes.prob_classes.heat_conduction.base_class_heat_conduction import FOS_FEM
from src.codes.utils.fem_utils_HC import *
from src.codes.utils.rom_utils import *
from src.codes.basic import *
from scipy.optimize import fsolve

def weighted_matrix_assembly(rom_cls, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):
    """
    Assemble the global reduced stiffness matrix (K_r), reduced Jacobian (J_r), 
    mean contribution (K_r_mean), and reduced force vector (rhs_r) by weighting local 
    element contributions. This function is used for the full nonlinear model (without affine simplifications).

    Parameters:
        rom_cls: ROM class instance containing full-order data and assembly methods.
        mask: Boolean mask indicating free (non-Dirichlet) nodes.
        dir_nodes: Array of Dirichlet node indices.
        sol_dir: Prescribed Dirichlet values.
        T_red: Reduced solution vector.
        node_eqnId: Global equation IDs for the nodes.
        xi: Dictionary mapping element indices to their importance weights.
        V: Projection (reduced basis) matrix.

    Returns:
        K_r: Assembled global reduced stiffness matrix.
        J_r: Assembled global reduced Jacobian matrix.
        K_r_mean: Mean-field contribution to the reduced stiffness.
        rhs_r: Assembled global reduced force vector.
    """
    # Disable ECM mode.
    rom_cls.ecm = False

    # Initialize global reduced matrices and force vector.
    K_r, J_r = [np.zeros((V.shape[1], V.shape[1])) for _ in range(2)]
    rhs_r, K_r_mean = [np.zeros(V.shape[1]) for _ in range(2)]
    
    # Initialize full-order solution vector for free nodes.
    sol_prev = np.zeros(len(rom_cls.data.mask))
    if rom_cls.mean is not None:
        T_mean = rom_cls.mean.flatten()
        sol_prev[rom_cls.data.mask] = np.dot(V, T_red) + T_mean
    else:
        sol_prev[rom_cls.data.mask] = np.dot(V, T_red)
    # Set Dirichlet nodes to prescribed values.
    sol_prev[~rom_cls.data.mask] = rom_cls.data.T_dir

    # Loop over elements with nonzero importance.
    for iel in xi.keys():
        # Determine projection indices from the element’s connectivity (using maximum values of Le).
        col_indices = np.argmax(rom_cls.data.Le[iel], axis=1)
        
        # Retrieve element connectivity and global equation ID data.
        elem_glob_nodes = rom_cls.data.gn[iel, :]
        elem_glob_node_eqnId = rom_cls.data.glob_node_eqnId[iel]
        elem_glob_node_nonzero_eqnId = rom_cls.data.glob_node_nonzero_eqnId[iel]
        elem_local_node_nonzero_eqnId = rom_cls.data.local_node_nonzero_eqnId[iel]

        # Compute local element matrices: Ke_ (stiffness), Je_ (Jacobian), and qe_ (force vector).
        Ke_, Je_, qe_ = compute_element_matrices(rom_cls, sol_prev.flatten(), iel)

        # Retrieve the appropriate global indices for assembling the matrices.
        i_index, j_index = rom_cls.data.local_indices[iel][0], rom_cls.data.local_indices[iel][1]

        # Assemble the reduced matrices by weighting the local contributions with xi[iel].
        K_r += V[col_indices].T @ (Ke_[i_index, j_index] * xi[iel]) @ V[col_indices]
        J_r += V[col_indices].T @ (Je_[i_index, j_index] * xi[iel]) @ V[col_indices]

        # If a mean field is provided, accumulate its contribution.
        if rom_cls.mean is not None:
            K_r_mean += V[col_indices].T @ (Ke_[i_index, j_index] * xi[iel]) @ T_mean[col_indices]

        # Handle Dirichlet BC at the element level: if any global equation ID is zero.
        if np.isin(0, elem_glob_node_eqnId):
            elem_dof_values = dirichlet_bc(rom_cls, sol_dir, dir_nodes, elem_glob_nodes)
            fe = Ke_ @ elem_dof_values.reshape(-1, 1)
        else:
            fe = np.zeros((len(elem_glob_nodes), 1))

        # Compute the local reduced force vector by subtracting the Dirichlet effect.
        rhs_e_ = qe_[elem_local_node_nonzero_eqnId] - fe[elem_local_node_nonzero_eqnId].flatten()
        # Assemble the global reduced force vector.
        rhs_r += V[col_indices].T @ (xi[iel] * rhs_e_)

    return K_r, J_r, K_r_mean, rhs_r

def weighted_matrix_assembly_affine(rom_cls, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):
    """
    Assemble the global reduced matrices for the affine version of the nonlinear model.

    For the affine case, contributions are collected separately based on material and force indices.

    Parameters:
        rom_cls: ROM class instance with full-order data.
        mask: Boolean mask for free nodes.
        dir_nodes: Dirichlet node indices.
        sol_dir: Prescribed Dirichlet values.
        T_red: Reduced solution vector.
        node_eqnId: Global node equation IDs.
        xi: Dictionary mapping element indices to importance weights.
        V: Reduced basis matrix.

    Returns:
        K_r: List of reduced stiffness matrices for each material type.
        K_r_mean: List of mean-field contributions for each material type.
        rhs_qe_: List of reduced force vectors (without Dirichlet BC) for each source type.
        rhs_fe_: List of reduced force vectors (with Dirichlet BC) for each material type.
    """
    # Determine the number of material and force types.
    n_mat = int(np.max(rom_cls.data.cell2mat_layout) + 1)
    n_src = int(np.max(rom_cls.data.cell2src_layout) + 1)

    # Initialize arrays for each material/force type.
    K_r = [np.zeros((V.shape[1], V.shape[1])) for _ in range(n_mat)]
    rhs_qe_ = [np.zeros(V.shape[1]) for _ in range(n_src)]
    rhs_fe_ = [np.zeros(V.shape[1]) for _ in range(n_src)]
    K_r_mean = [np.zeros(V.shape[1]) for _ in range(n_src)]
    
    # Get the mean field if provided.
    if rom_cls.mean is not None:
        T_mean = rom_cls.mean.flatten()
    else:
        T_mean = 0.0

    # Loop over all elements.
    for iel in range(rom_cls.data.n_cells):
        if xi[iel] == 0:
            continue  # Skip elements with zero importance.

        # Compute the multi-dimensional index of the cell.
        cell_idx = tuple(e_n_2ij(rom_cls, iel))
        imat = rom_cls.data.cell2mat_layout[cell_idx].astype(int)
        isrc = rom_cls.data.cell2src_layout[cell_idx].astype(int)

        col_indices = np.argmax(rom_cls.data.Le[iel], axis=1)

        # Retrieve element connectivity and nonzero equation ID data.
        elem_glob_nodes = rom_cls.data.gn[iel, :]
        elem_glob_node_eqnId = rom_cls.data.glob_node_eqnId[iel]
        elem_glob_node_nonzero_eqnId = rom_cls.data.glob_node_nonzero_eqnId[iel]
        elem_local_node_nonzero_eqnId = rom_cls.data.local_node_nonzero_eqnId[iel]

        # Compute local element matrices and force vector in the affine setting.
        Ke_, _, qe_ = compute_element_matrices(rom_cls, np.zeros(len(rom_cls.data.mask)), iel, affine=True)

        # Get local indices for assembly.
        i_index, j_index = rom_cls.data.local_indices[iel][0], rom_cls.data.local_indices[iel][1]

        # Update the reduced stiffness matrix for the given material type.
        K_r[imat] += V[col_indices].T @ (Ke_[i_index, j_index] * xi[iel]) @ V[col_indices]

        # Add the mean field effect if available.
        if rom_cls.mean is not None:
            K_r_mean[imat] += V[col_indices].T @ (Ke_[i_index, j_index] * xi[iel]) @ T_mean[col_indices]

        # Handle Dirichlet BC: if the element contains any Dirichlet node.
        if np.isin(0, elem_glob_node_eqnId):
            elem_dof_values = dirichlet_bc(rom_cls, sol_dir, dir_nodes, elem_glob_nodes)
            fe = Ke_ @ elem_dof_values.reshape(-1, 1)
        else:
            fe = np.zeros((len(elem_glob_nodes), 1))

        # Update reduced force vectors for force (without Dirichlet BC) and for material (with Dirichlet BC).
        rhs_qe_[isrc] += V[col_indices].T @ (xi[iel] * qe_[elem_local_node_nonzero_eqnId])
        rhs_fe_[imat] += V[col_indices].T @ (xi[iel] * fe[elem_local_node_nonzero_eqnId].flatten())

    return K_r, K_r_mean, rhs_qe_, rhs_fe_

def e_n_2ij(self, iel, el=True):
    """
    Convert a global element index into multi-dimensional indices corresponding to its location 
    in the mesh grid. If 'el' is True, use the number of cells per dimension; if False, use the number of nodes.
    
    Parameters:
        iel: Global element index.
        el (bool): Flag indicating whether to use cell counts (True) or node counts (False).
    
    Returns:
        indices (list): List of indices for each dimension.
    """
    dim_ = self.dim_
    indices = []
    divisor = 1
    for d in range(dim_):
        size = self.ncells[d] if el else self.npts[d]
        idx = (iel // divisor) % size
        divisor *= size
        indices.append(idx)
    return indices

def weighted_matrix_assembly_deim(rom_cls, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):
    """
    Assemble the global reduced matrices and force vector using DEIM hyper-reduction.
    
    Parameters:
        rom_cls: ROM class instance containing full-order simulation data.
        mask: Boolean mask indicating free nodes.
        dir_nodes: Indices of Dirichlet nodes.
        sol_dir: Prescribed Dirichlet values.
        T_red: Reduced solution vector.
        node_eqnId: Global equation IDs.
        xi: Dictionary mapping element indices to importance weights.
        V: Reduced basis (projection) matrix.
    
    Returns:
        K: Global reduced stiffness matrix.
        J: Global reduced Jacobian matrix.
        sol_prev: Reconstructed full-order solution based on T_red.
        rhs: Global reduced force vector.
    """
    # Initialize global systems.
    K, J, rhs = init_global_systems(max(node_eqnId))
    sol_prev = np.zeros(len(rom_cls.data.mask))
    
    # Reconstruct full-order solution from reduced solution.
    T_mean = rom_cls.mean.flatten() if rom_cls.mean is not None else 0.0

    if rom_cls.mean is not None:
        sol_prev[rom_cls.data.mask] = np.dot(V, T_red) + T_mean
    else:
        sol_prev[rom_cls.data.mask] = np.dot(V, T_red)

    sol_prev[~rom_cls.data.mask] = rom_cls.data.T_dir

    # Loop over elements (only those with nonzero importance in xi).
    for iel in xi.keys():
        # Retrieve connectivity and equation data for the element.
        elem_glob_nodes = rom_cls.data.gn[iel, :]
        elem_glob_node_eqnId = rom_cls.data.glob_node_eqnId[iel]
        elem_glob_node_nonzero_eqnId = rom_cls.data.glob_node_nonzero_eqnId[iel]
        elem_local_node_nonzero_eqnId = rom_cls.data.local_node_nonzero_eqnId[iel]

        # Get global indices for assembly.
        I_index, J_index = rom_cls.data.global_indices[iel][0], rom_cls.data.global_indices[iel][1]
        i_index, j_index = rom_cls.data.local_indices[iel][0], rom_cls.data.local_indices[iel][1]

        # Compute local element matrices and force vector.
        Ke_, Je_, qe_ = compute_element_matrices(rom_cls, sol_prev.flatten(), iel)

        # Update global reduced matrices.
        K[I_index, J_index] += Ke_[i_index, j_index]
        J[I_index, J_index] += Je_[i_index, j_index]

        # Apply Dirichlet conditions if the element contains Dirichlet nodes.
        if np.isin(0, elem_glob_node_eqnId):
            elem_dof_values = dirichlet_bc(rom_cls, sol_dir, dir_nodes, elem_glob_nodes)
            fe = Ke_ @ elem_dof_values.reshape(-1, 1)
        else:
            fe = np.zeros((len(elem_glob_nodes), 1))

        # Compute local reduced force vector.
        rhs_e_ = qe_[elem_local_node_nonzero_eqnId] - fe[elem_local_node_nonzero_eqnId].flatten()
        # Assemble into the global force vector.
        rhs[elem_glob_node_nonzero_eqnId - 1] += rhs_e_

    return K, J, sol_prev, rhs

def weighted_matrix_assembly_ecm(rom_cls, mask, dir_nodes, sol_dir, T_red, node_eqnId, selected_elements, V):
    """
    Assemble the global reduced matrices and force vector using ECM hyper-reduction.

    Parameters:
        rom_cls: ROM class instance.
        mask: Boolean mask for free nodes.
        dir_nodes: Dirichlet node indices.
        sol_dir: Prescribed Dirichlet values.
        T_red: Reduced solution vector.
        node_eqnId: Global node equation IDs.
        selected_elements: Dictionary mapping selected element indices to their ECM weights.
        V: Reduced basis matrix.

    Returns:
        K_r: Global reduced stiffness matrix.
        J_r: Global reduced Jacobian matrix.
        K_r_mean: Mean-field contribution to the reduced stiffness.
        rhs_r: Global reduced force vector.
    """
    # Initialize global reduced matrices.
    K_r, J_r = [np.zeros((V.shape[1], V.shape[1])) for _ in range(2)]
    rhs_r, K_r_mean = [np.zeros(V.shape[1]) for _ in range(2)]
    
    sol_prev = np.zeros(len(rom_cls.data.mask))
    if rom_cls.mean is not None:
        T_mean = rom_cls.mean.flatten()
        sol_prev[rom_cls.data.mask] = np.dot(V, T_red) + T_mean
    else:
        sol_prev[rom_cls.data.mask] = np.dot(V, T_red)
    sol_prev[~rom_cls.data.mask] = rom_cls.data.T_dir

    # Initialize ECM-specific lists in the ROM class.
    rom_cls.Ke_gauss = []
    rom_cls.Je_gauss = []
    rom_cls.rhs_e_gauss = []

    # Loop over each selected element for ECM.
    for k, key in enumerate(selected_elements.keys()):
        iel = key
        w_ecm = selected_elements[iel]  # ECM weights for element iel.
        col_indices = np.argmax(rom_cls.data.Le[iel], axis=1)
        
        # Retrieve element connectivity and equation data.
        elem_glob_nodes = rom_cls.data.gn[iel, :]
        elem_glob_node_eqnId = rom_cls.data.glob_node_eqnId[iel]
        elem_glob_node_nonzero_eqnId = rom_cls.data.glob_node_nonzero_eqnId[iel]
        elem_local_node_nonzero_eqnId = rom_cls.data.local_node_nonzero_eqnId[iel]

        rom_cls.ecm = True

        # Compute local element matrices and force vectors.
        Ke_, _, _ = compute_element_matrices(rom_cls, sol_prev.flatten(), iel)
        Ke = 0
        Je = 0
        qe_ = 0

        # Sum contributions over Gauss points weighted by ECM weights.
        for gp in range(len(w_ecm)):
            Ke += w_ecm[gp] * rom_cls.Ke_gauss[k][gp]
            Je += w_ecm[gp] * rom_cls.Je_gauss[k][gp]
            qe_ += w_ecm[gp] * rom_cls.rhs_e_gauss[k][gp]

        # Retrieve local indices for assembly.
        i_index, j_index = rom_cls.data.local_indices[iel][0], rom_cls.data.local_indices[iel][1]

        # Assemble the global reduced matrices.
        K_r += V[col_indices].T @ (Ke[i_index, j_index]) @ V[col_indices]
        J_r += V[col_indices].T @ (Je[i_index, j_index]) @ V[col_indices]

        # Accumulate mean field contribution.
        if rom_cls.mean is not None:
            K_r_mean += V[col_indices].T @ (Ke[i_index, j_index]) @ T_mean[col_indices]

        # Apply Dirichlet BC if present.
        if np.isin(0, elem_glob_node_eqnId):
            elem_dof_values = dirichlet_bc(rom_cls, sol_dir, dir_nodes, elem_glob_nodes)
            fe = Ke @ elem_dof_values.reshape(-1, 1)
        else:
            fe = np.zeros((len(elem_glob_nodes), 1))

        # Compute local reduced force vector.
        rhs_e_ = qe_[elem_local_node_nonzero_eqnId] - fe[elem_local_node_nonzero_eqnId].flatten()
        rhs_r += V[col_indices].T @ rhs_e_

    return K_r, J_r, K_r_mean, rhs_r

def solve_reduced(rom_cls, T_init, xi, V, tol=1e-5, max_iter=300, op=False):
    """
    Solve the nonlinear reduced-order system using Newton-Raphson iteration.

    Parameters:
        rom_cls: Full-order simulation class instance with mesh and FE data.
        T_init: Initial guess for the reduced solution.
        xi: Element-wise importance weights.
        V: Reduced basis (projection) matrix.
        tol: Convergence tolerance for the residual (default is 1e-5).
        max_iter: Maximum number of iterations (default is 300).
        op: If True, print iteration details.

    Returns:
        T: Updated reduced solution after convergence or after max_iter iterations.
    """
    # Get global node equation IDs and construct a mask for free nodes.
    node_eqnId = rom_cls.node_eqnId
    mask = node_eqnId != 0

    # Initialize the reduced solution.
    T = np.copy(T_init)

    # Evaluate the initial reduced Jacobian and residual.
    Jac, res = rom_cls.eval_resJac_rom(mask, rom_cls.dir_nodes, rom_cls.sol_dir, T, node_eqnId, xi, V)
    norm_ = np.linalg.norm(res)

    if op:
        print('initial residual =', norm_, "\n")

    it = 0
    # Newton-Raphson iteration loop.
    while (it < max_iter) and (norm_ >= tol):
        # Solve for the increment delta.
        delta = np.linalg.solve(Jac, -res)
        # Update the reduced solution.
        T += delta
        # Re-evaluate the Jacobian and residual.
        Jac, res = rom_cls.eval_resJac_rom(mask, rom_cls.dir_nodes, rom_cls.sol_dir, T, node_eqnId, xi, V)
        norm_ = np.linalg.norm(res)

        if op:
            print("iter {}, NL residual = {}, max(delta) = {}".format(it, norm_, np.max(delta)))

        if norm_ < tol:
            if op:
                print('Convergence achieved.')
        elif it == max_iter - 1:
            print('\nWARNING: Nonlinear solution has not converged.')

        it += 1

    return T

def solve_reduced_fsolve(rom_cls, T_init, xi, V, tol=1e-5, max_iter=300):
    """
    Solve the nonlinear reduced-order system using the fsolve nonlinear solver.

    Parameters:
        rom_cls: Full-order simulation class instance.
        T_init: Initial guess for the reduced solution.
        xi: Element-wise importance weights.
        V: Reduced basis matrix.
        tol: Convergence tolerance (default is 1e-5).
        max_iter: Maximum iterations (default is 300).

    Returns:
        T_ans: Converged reduced solution.
    """
    from scipy.optimize import fsolve
    print("Using fsolve for reduced system.")

    node_eqnId = rom_cls.node_eqnId
    mask = node_eqnId != 0

    # Define the residual function for fsolve.
    res = lambda T_: rom_cls.eval_res_fsolve_rom(mask, rom_cls.dir_nodes, rom_cls.sol_dir, T_, node_eqnId, xi, V).flatten()

    T_ans = fsolve(res, T_init)

    return T_ans

class rom(FOS_FEM):
    def __init__(self, data, quad_degree, mean=None):
        """
        Constructor for the reduced-order model solver.
        
        Parameters:
            data: Full-order simulation data object (mesh, material properties, etc.).
            quad_degree: Quadrature degree for numerical integration.
            mean: Mean field used to reconstruct the full-order solution.
        """
        # Call the parent class constructor to initialize FOS_FEM.
        super().__init__(data, quad_degree)
        self.mean = mean  # Store the mean field

    def select_elements_and_weights(self, element_indices, weights):
        """
        Given arrays of element indices and corresponding weights, construct a dictionary
        mapping each element to its total weight.
        
        Parameters:
            element_indices: Array-like list of element indices.
            weights: Array-like list of weights corresponding to each element.
        
        Returns:
            element_to_gauss_weights: Dictionary where each key is an element index and the value
                                      is the accumulated weight.
        """
        element_to_gauss_weights = {}

        # Iterate over each provided index and its weight.
        for idx, weight in zip(element_indices, weights):
            # If idx is a single-element array, convert it to a scalar.
            if isinstance(idx, np.ndarray) and idx.size == 1:
                idx = idx.item()
            
            # Initialize the weight for the element if not already present.
            if idx not in element_to_gauss_weights:
                element_to_gauss_weights[idx] = 0
            
            # Accumulate the weight for the element.
            element_to_gauss_weights[idx] += weight

        return element_to_gauss_weights

    def solve_rom(self, T_init, V):
        """
        Solve the nonlinear reduced-order system.
        
        This function sets up the element weights (here, simply ones for all elements),
        then calls the nonlinear solver 'solve_reduced' with the current reduced initial guess.
        
        Parameters:
            T_init: Initial guess for the reduced temperature (or displacement) field.
            V: Projection matrix (reduced basis) used for the ROM.
        
        Returns:
            T: Updated reduced solution after convergence or reaching the maximum iterations.
        """
        # Build a dictionary with all elements assigned a weight of 1.
        all_elements = self.select_elements_and_weights(np.r_[0:self.data.n_cells:1], 
                                                          np.ones(self.data.n_cells, dtype=int))
        # Solve the nonlinear reduced system using the assembled weights.
        T = solve_reduced(self, T_init, all_elements, V)

        return T

    def eval_resJac_rom(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):
        """
        Assemble the reduced-order system's Jacobian and residual.
        
        This function calls the weighted matrix assembly routine to obtain the global reduced
        stiffness matrix (K_r), Jacobian (J_r), mean contribution (K_r_mean), and force vector (rhs_r).
        The residual is then computed as:
            res = (K_r @ T_red + K_r_mean - rhs_r)
        
        Parameters:
            mask: Boolean mask for free (non-Dirichlet) nodes.
            dir_nodes: Indices of Dirichlet nodes.
            sol_dir: Prescribed Dirichlet values.
            T_red: Current reduced solution.
            node_eqnId: Global node equation IDs.
            xi: Element-wise importance weights.
            V: Projection matrix.
        
        Returns:
            A tuple (Jacobian, residual) for the reduced nonlinear system.
        """
        # Assemble the global reduced matrices using the weighted matrix assembly routine.
        K_r, J_r, K_r_mean, rhs_r = weighted_matrix_assembly(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V)
        # Compute the residual vector.
        res = (K_r @ T_red + K_r_mean - rhs_r)
        
        # The total Jacobian for the system is the sum of K_r and J_r.
        return K_r + J_r, res

class rom_ecsw(FOS_FEM):
    """
    Class: FEM_solver_rom_ecsw
    Overview: Inherits from the base FEM class and implements reduced‐order modeling 
              with Element-based Coarse-Scale Weights (ECSW) for a nonlinear structural mechanics problem.
    Attributes:
        mean: Mean field used in reconstructing the full-order solution.
    """
    def __init__(self, data, quad_degree, mean=None):
        """
        Constructor to initialize the reduced-order FEM solver.
        
        Parameters:
            data: Mesh and finite element data.
            quad_degree: Degree for Gaussian quadrature integration.
            mean: (Optional) Mean field used for reconstructing the full-order solution.
        """
        super().__init__(data, quad_degree)
        self.mean = mean

    def solve_rom(self, T_init, xi, V):
        """
        Solve the nonlinear system for the reduced-order model.
        
        Parameters:
            T_init: Initial guess for the reduced temperature field.
            xi: Element-wise importance weights.
            V: Projection matrix.
        
        Returns:
            T: Updated reduced solution after convergence or reaching maximum iterations.
        """
        T = solve_reduced(self, T_init, xi, V)
        return T
    
    def eval_resJac_rom(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):
        """
        Assemble the reduced-order system's Jacobian and compute the residual.
        
        Parameters:
            mask: Boolean mask indicating free (non-Dirichlet) nodes.
            dir_nodes: Dirichlet node indices.
            sol_dir: Prescribed Dirichlet values.
            T_red: Current reduced solution.
            node_eqnId: Global node equation IDs.
            xi: Element-wise importance weights.
            V: Projection matrix.
        
        Returns:
            A tuple (Jacobian, residual) for the reduced nonlinear system.
        """
        K_r, J_r, K_r_mean, rhs_r = weighted_matrix_assembly(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V)
        res = (K_r @ T_red + K_r_mean - rhs_r)
        return K_r + J_r, res

class rom_ecm(FOS_FEM):
    """
    Class: FEM_solver_rom_ecsw
    Overview: Inherits from the base class for finite element method (FEM) simulations.
              This subclass focuses on reduced-order modeling with Element-based Coarse-Scale Weights (ECSW)
              for the nonlinear structural mechanics problem.
              
    Attributes:
        data: Mesh and finite element data.
        quad_degree: Degree for Gaussian quadrature integration.
        mean: Mean field used to reconstruct the full-order solution.
        ecm: Boolean flag set to True to indicate that ECM hyper-reduction is active.
    """
    def __init__(self, data, quad_degree, mean=None):
        """
        Constructor to initialize the reduced-order FEM solver with ECM.
        
        Parameters:
            data: Full-order simulation data.
            quad_degree: Degree for Gaussian quadrature.
            mean: (Optional) Mean field for reconstructing the full-order solution.
        """
        super().__init__(data, quad_degree)
        self.mean = mean
        self.ecm = True

    def solve_rom(self, T_init, xi, V):
        """
        Solve the nonlinear reduced-order system.
        
        Inputs:
            T_init: Initial guess for the reduced solution field.
            xi: Element-wise importance weights.
            V: Projection (reduced basis) matrix.
        
        Outputs:
            T: Updated reduced solution after convergence or reaching max iterations.
        """
        T = solve_reduced(self, T_init, xi, V)
        return T
    
    def eval_resJac_rom(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):
        """
        Assemble the reduced-order Jacobian and compute the residual.
        
        The function assembles the global reduced stiffness matrix, Jacobian, and mean-field contribution
        by calling the ECM-specific weighted assembly routine, and then computes the residual as:
        
            res = (K_r @ T_red + K_r_mean - rhs_r)
        
        Inputs:
            mask: Boolean mask for free (non-Dirichlet) nodes.
            dir_nodes: Dirichlet node indices.
            sol_dir: Prescribed Dirichlet values.
            T_red: Current reduced solution.
            node_eqnId: Global node equation IDs.
            xi: Element-wise importance weights.
            V: Projection matrix.
        
        Outputs:
            A tuple (Jacobian, residual) for the reduced nonlinear system.
        """
        K_r, J_r, K_r_mean, rhs_r = weighted_matrix_assembly_ecm(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V)
        res = (K_r @ T_red + K_r_mean - rhs_r)
        return K_r + J_r, res

class rom_deim(FOS_FEM):
    """
    Class: FEM_solver_rom_ecsw (with DEIM)
    Overview: This subclass of FOS_FEM implements a reduced-order model (ROM) solver
              using a DEIM-based hyper-reduction technique for a nonlinear structural
              mechanics problem. It computes a reduced solution by solving a nonlinear
              system with Newton-type iterations, projecting the full residual onto a 
              DEIM-selected subspace.
              
    Attributes:
        deim_cls: An instance of the DEIM class containing the DEIM projection matrix
                  (deim_mat) and a boolean array (bool_sampled) indicating the selected indices.
        mean: Mean field used to reconstruct the full-order solution from the reduced solution.
    """
    def __init__(self, data, deim_cls, quad_degree, mean=None):
        """
        Constructor to initialize the DEIM-based reduced-order solver.
        
        Parameters:
            data: Full-order simulation data (mesh, material properties, boundary conditions, etc.).
            deim_cls: DEIM class instance, providing the DEIM projection matrix and selection information.
            quad_degree: Degree of Gaussian quadrature for numerical integration.
            mean: (Optional) Mean field for reconstructing the full-order solution.
        """
        # Initialize the base FEM solver.
        super().__init__(data, quad_degree)
        self.deim_cls = deim_cls  # Store the DEIM instance.
        self.mean = mean          # Store the mean field.

    def solve_rom(self, T_init, xi, V):
        """
        Solve the reduced-order nonlinear system using a Newton-Raphson based solver.
        
        Parameters:
            T_init: Initial guess for the reduced solution (e.g., temperature or displacement field).
            xi: Element-wise importance weights (used during assembly).
            V: Projection (reduced basis) matrix.
        
        Returns:
            T: The converged reduced solution after applying the Newton-Raphson iterations.
        """
        T = solve_reduced(self, T_init, xi, V)
        # Alternatively, one could use the fsolve-based solver:
        # T = solve_reduced_fsolve(self, T_init, xi, V)
        return T

    def eval_resJac_rom(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):
        """
        Evaluate the Jacobian and the residual for the reduced system using DEIM projection.
        
        This function first assembles the global reduced stiffness matrix, Jacobian, and the mean
        contribution to the force vector by calling the weighted_matrix_assembly_deim routine.
        It then computes the full-order residual and projects it onto the DEIM subspace using the DEIM
        projection matrix.
        
        Parameters:
            mask: Boolean mask indicating free (non-Dirichlet) nodes.
            dir_nodes: Indices of Dirichlet nodes.
            sol_dir: Prescribed Dirichlet values.
            T_red: Current reduced solution.
            node_eqnId: Global node equation IDs.
            xi: Element-wise importance weights.
            V: Projection matrix.
        
        Returns:
            A tuple (Jacobian_projected, residual_projected) where:
                - Jacobian_projected: The Jacobian matrix (after DEIM projection) used for Newton updates.
                - residual_projected: The residual vector (projected onto the DEIM subspace).
        """
        # Assemble global matrices using DEIM-specific weighted assembly.
        K, J, sol_prev, rhs = weighted_matrix_assembly_deim(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V)
        # Compute the full-order residual over free nodes.
        residual = K @ sol_prev[mask] - rhs
        # Retrieve the DEIM projection matrix and the boolean mask of selected indices.
        M = self.deim_cls.deim_mat
        deim_mask = self.deim_cls.bool_sampled
        # Project the residual onto the DEIM subspace.
        res_projected = np.dot(M, residual[deim_mask])
        # Assemble the full Jacobian and project it onto the DEIM subspace.
        return M @ (K + J)[deim_mask] @ V, res_projected

    def eval_res_fsolve_rom(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V):
        """
        Evaluate the projected residual for use with an fsolve-based nonlinear solver.
        
        Parameters:
            mask: Boolean mask for free nodes.
            dir_nodes: Dirichlet node indices.
            sol_dir: Prescribed Dirichlet values.
            T_red: Current reduced solution.
            node_eqnId: Global node equation IDs.
            xi: Element-wise importance weights.
            V: Projection matrix.
        
        Returns:
            res_projected: The residual vector projected onto the DEIM subspace.
        """
        # Assemble the global reduced matrices and force vector.
        K, _, sol_prev, rhs = weighted_matrix_assembly_deim(self, mask, dir_nodes, sol_dir, T_red, node_eqnId, xi, V)
        # Compute the full-order residual.
        residual = K @ sol_prev[mask] - rhs
        # Get the DEIM projection matrix and selected indices.
        M = self.deim_cls.deim_mat
        deim_mask = self.deim_cls.bool_sampled
        # Project the residual and return.
        res_projected = np.dot(M, residual[deim_mask])
        return res_projected

class rom_affine(FOS_FEM):
    """
    Reduced-order model solver for the affine version of the nonlinear structural mechanics problem.
    
    This class assembles the global reduced stiffness matrix, the mean-field contribution, and the
    reduced force vectors (both without and with the effect of Dirichlet boundary conditions) using an affine formulation.
    """
    def __init__(self, data, quad_degree, mean=None):
        """
        Constructor to initialize the affine ROM solver.

        Parameters:
            data: Full-order simulation data (mesh, material properties, boundary conditions, etc.).
            quad_degree: Degree for Gaussian quadrature integration.
            mean: (Optional) Mean field used for reconstructing the full-order solution.
        """
        super().__init__(data, quad_degree)
        self.mean = mean

    def solve_rom(self, V):
        """
        Solve the nonlinear affine reduced-order system.

        This method assembles the reduced matrices by calling 'reduced_affine' with an initial reduced
        solution guess (a zero vector) and with element importance weights set to one.

        Parameters:
            V: Projection matrix (the reduced basis).

        Returns:
            A tuple (K_r, K_r_mean, rhs_qe_, rhs_fe_) where:
                - K_r is the global reduced stiffness matrix.
                - K_r_mean is the mean-field contribution.
                - rhs_qe_ is the reduced force vector computed without Dirichlet BC.
                - rhs_fe_ is the reduced force vector computed with the Dirichlet BC effect.
        """
        T_init = np.zeros(V.shape[1])
        K_r, K_r_mean, rhs_qe_, rhs_fe_ = self.reduced_affine(T_init, np.ones(self.data.n_cells), V)
        return K_r, K_r_mean, rhs_qe_, rhs_fe_

    def reduced_affine(self, T_red, xi, V):
        """
        Assemble the global reduced matrices for the affine formulation.

        This function computes the reduced stiffness matrix and force vectors by invoking the 
        'weighted_matrix_assembly_affine' routine.

        Parameters:
            T_red: The current reduced solution (initial guess).
            xi: Element-wise importance weights (typically a vector of ones).
            V: Projection matrix (the reduced basis).

        Returns:
            A tuple (K_r, K_r_mean, rhs_qe_, rhs_fe_) representing:
                - K_r: Global reduced stiffness matrix.
                - K_r_mean: Mean-field contribution matrix.
                - rhs_qe_: Reduced force vector computed without applying Dirichlet BC.
                - rhs_fe_: Reduced force vector corresponding to the Dirichlet BC effects.
        """
        node_eqnId = self.node_eqnId
        mask = node_eqnId != 0
        K_r, K_r_mean, rhs_qe_, rhs_fe_ = weighted_matrix_assembly_affine(
            self, mask, self.dir_nodes, self.sol_dir, T_red, node_eqnId, xi, V
        )
        return K_r, K_r_mean, rhs_qe_, rhs_fe_
