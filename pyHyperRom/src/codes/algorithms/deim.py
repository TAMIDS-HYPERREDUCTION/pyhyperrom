import numpy as np
from src.codes.utils.fem_utils_HC import *  # Import FEM utility functions (e.g., element matrix computations)
from ..utils.rom_utils import *             # Import ROM utility functions (e.g., snapshot handling, SVD selectors)

class deim:
    """
    Class to perform the Discrete Empirical Interpolation Method (DEIM) and its variants
    for reducing the dimension of nonlinear force terms in a reduced order model (ROM)
    within a finite element framework.
    """

    def __init__(self, d, F_nl, train_mask, param_list, V, sol_snapshots, mask, tol_f=1e-2, extra_modes=0, extra_samples=0, sopt=False):
        """
        Constructor to initialize DEIM parameters and input data.
        
        Parameters:
        -----------
        d : object
            An instance containing the FEM model data (e.g., node connectivity, equation IDs).
        F_nl : ndarray
            Snapshot matrix containing nonlinear force evaluations for various training samples.
        train_mask : array_like (boolean or index array)
            Mask to select the training snapshots from sol_snapshots.
        param_list : list
            List of parameter values corresponding to each training snapshot.
        V : ndarray
            Reduced basis matrix (projection basis for the full-order solution).
        sol_snapshots : ndarray
            Snapshot matrix of the full-order solutions.
        mask : array_like
            Boolean mask indicating which degrees-of-freedom (DOFs) are used.
        tol_f : float, optional
            Tolerance for selecting the number of singular value modes from the SVD (default is 1e-2).
        extra_modes : int, optional
            Additional modes to include beyond those selected by the tolerance criterion.
        extra_samples : int, optional
            Additional sampling points for oversampling in the S-optimal strategy.
        """
        self.cls = d                              # Store FEM model data (e.g., mesh, eqn IDs, etc.)
        self.tol_f = tol_f                        # Tolerance for singular value thresholding
        self.mask = mask                          # Mask for selecting DOFs from the full-order model
        self.sol_snapshots = sol_snapshots        # Full-order solution snapshots for ROM training
        self.V_m = V                              # Reduced basis matrix (could be further masked if needed)
        self.mu_list = param_list                 # List of parameter values for the training data

        self.F_nl = F_nl[train_mask]              # Select the nonlinear force snapshots based on the training mask

        self.extra_modes = extra_modes            # Number of extra modes to be added to the selected basis modes
        self.extra_samples = extra_samples        # Number of extra sampling points for oversampling
        self.sopt = sopt

    def select_elems(self):
        """
        Select the elements (or rows) for DEIM reduction based on the computed snapshot basis.
        This method:
          1. Performs SVD on the nonlinear force snapshots to determine dominant modes.
          2. Uses an S-optimal sampling strategy (or DEIM, if uncommented) to select rows.
          3. Maps the selected DOFs back to element indicators.
          4. Constructs the DEIM projection matrix.
        """
        # Determine the number of modes to retain using an SVD-based selector.
        n_f_sel, U_f = svd_mode_selector(self.F_nl, self.tol_f)
        n_f_sel += self.extra_modes  # Include any additional modes specified
        print(f"Selected modes:{n_f_sel}")
        
        # Truncate the full basis to only include the selected modes.
        U_fs = U_f[:, :n_f_sel]
        
        # Use the DEIM sampling strategy to select rows.
        # Alternatively use S-optimal sampling strategy.

        if self.sopt:

            f_basis_sampled, sampled_rows, self.bool_sampled = self.sopt_red(U_f, n_f_sel)    

        else:

            f_basis_sampled, sampled_rows, self.bool_sampled = self.deim_red(U_f, n_f_sel)


        # Retrieve the global equation IDs for the nodes corresponding to the provided mask.
        masked_eqnID = self.cls.node_eqnId[self.cls.mask]
        
        # Map the sampled row indices to the corresponding DOFs (degrees-of-freedom).
        deim_dof = [masked_eqnID[i] for i in sampled_rows]

        # Convert the selected DOFs to an element indicator vector.
        self.xi = self.deim_dof_to_elem(deim_dof)
        
        # Build the DEIM projection matrix using the reduced basis and the pseudo-inverse of the sampled basis.
        self.deim_mat = self.V_m.T @ U_fs @ np.linalg.pinv(f_basis_sampled)
        
        # Store additional attributes for potential later use.
        self.U_fs = U_fs
        self.n_f_sel = n_f_sel


    def deim_red(self, f_basis, num_f_basis_vectors_used):
        """
        Perform the standard Discrete Empirical Interpolation Method (DEIM) to select rows 
        that capture the dominant features of the nonlinear basis.
        
        Parameters:
        -----------
        f_basis : ndarray
            Basis matrix derived from the nonlinear force snapshots.
        num_f_basis_vectors_used : int
            Number of basis vectors (modes) to use for the reduction.
        
        Returns:
        --------
        f_basis_sampled : ndarray
            Matrix of the selected rows from the original basis.
        sampled_rows : list
            List of row indices that were selected.
        is_sampled : ndarray
            Boolean array indicating which rows of the original basis are selected.
        """
        # Determine the effective number of basis vectors to use (cannot exceed total available).
        num_basis_vectors = min(num_f_basis_vectors_used, f_basis.shape[1])
        basis_size = f_basis.shape[0]
        
        # Initialize a matrix to store the sampled rows (each row will contain a subset of the basis)
        f_basis_sampled = np.zeros((num_basis_vectors, num_basis_vectors))
        
        # List to store the indices of the rows selected by DEIM.
        sampled_rows = []
        
        # Boolean array to track which rows have been sampled.
        is_sampled = np.zeros(basis_size, dtype=bool)

        # Select the index corresponding to the maximum absolute value in the first basis vector.
        f_bv_max_global_row = np.argmax(np.abs(f_basis[:, 0]))
        sampled_rows.append(f_bv_max_global_row)
        is_sampled[f_bv_max_global_row] = True

        # Store the corresponding row from the first basis vector into the sampled matrix.
        f_basis_sampled[0, :] = f_basis[f_bv_max_global_row, :num_basis_vectors]

        # Iteratively select subsequent rows for each additional basis vector.
        for i in range(1, num_basis_vectors):
            # Solve for the interpolation coefficients that best approximate the i-th basis vector 
            # using the previously selected rows.
            c = np.linalg.solve(f_basis_sampled[:i, :i], f_basis_sampled[:i, i])
            
            # Compute the residual of the approximation for the i-th basis vector.
            r_val = np.abs(f_basis[:, i] - np.dot(f_basis[:, :i], c))
            
            # Select the row with the maximum residual as the next sampling index.
            f_bv_max_global_row = np.argmax(r_val)
            
            # Update the list of sampled rows and the corresponding boolean indicator.
            sampled_rows.append(f_bv_max_global_row)
            is_sampled[f_bv_max_global_row] = True

            # Update the sampled basis matrix with the newly selected row.
            f_basis_sampled[i, :] = f_basis[f_bv_max_global_row, :num_basis_vectors]

        # Return the sampled basis matrix, the list of selected row indices, and the boolean mask.
        return f_basis_sampled, sampled_rows, is_sampled

    def sopt_red(self, f_basis, num_f_basis_vectors_used):
        """
        Perform an S-optimal sampling reduction on the given basis. This method uses a QR
        factorization to extract an orthogonal basis and then iteratively selects the most
        representative rows (sampling points) based on a defined objective.
        
        Parameters:
        -----------
        f_basis : ndarray
            The original basis matrix (from the nonlinear force snapshots).
        num_f_basis_vectors_used : int
            The number of basis vectors (modes) to consider.
        
        Returns:
        --------
        f_sampled_row : ndarray
            The rows of the basis matrix corresponding to the selected sampling points.
        Z_indices : list
            List of indices (row numbers) that were selected.
        Z_boolean : ndarray
            Boolean vector indicating which rows have been selected.
        """
        # Determine the total number of sampling points to select (including any oversampling points)
        num_sampling_points = num_f_basis_vectors_used + self.extra_samples

        # Use only as many basis vectors as available.
        n_f = min(num_f_basis_vectors_used, f_basis.shape[1])
        f_basis = f_basis[:, :n_f]

        # Perform a QR factorization of the basis to obtain an orthogonal matrix Q.
        Q, R = np.linalg.qr(f_basis)
        Q_col_len, _ = Q.shape

        # Initialize selection: choose the index with the maximum absolute value in the first column of Q.
        i_star = np.argmax(np.abs(Q[:, 0]))
        Z_indices = [i_star]

        # Construct an initial selection matrix (Z_matrix) with a one-hot encoding for the selected index.
        Z_matrix = np.zeros((Q_col_len, 1))
        Z_matrix[i_star, 0] = 1
        E_ = np.eye(n_f)

        # Iteratively select additional sampling points based on an objective function.
        for j in range(1, n_f):
            # Construct E_j as the first j columns of the identity matrix.
            E_j = E_[:, :j]

            # Compute the matrix A = Z^T * Q * E_j (captures the contribution of already selected indices).
            A = Z_matrix.T @ Q @ E_j  # Shape: (number of selected indices, j)

            # Compute the projection of the (j+1)-th standard basis vector through Q, restricted to selected rows.
            e_j_plus1 = np.zeros(n_f)
            e_j_plus1[j] = 1
            c = Z_matrix.T @ Q @ e_j_plus1  # Shape: (number of selected indices,)

            # Inverse of A^T * A used for computing interpolation coefficients.
            AtA_inv = np.linalg.inv(A.T @ A)
            
            # Compute coefficients for reconstructing the (j+1)-th column.
            g = AtA_inv @ (A.T @ c)

            # Precompute the squared norms of the columns of A.
            norm_A_columns_squared = np.sum(A ** 2, axis=0)

            # Initialize variables to track the best candidate index.
            istar_max_obj = -np.inf
            i_star_candidate = -1

            # Evaluate all candidate indices that are not yet selected.
            candidates = [i for i in range(Q_col_len) if i not in Z_indices]
            for i in candidates:
                # Create a one-hot vector for the candidate index.
                e_i = np.zeros((Q_col_len, 1))
                e_i[i] = 1

                # Compute the projection of candidate row onto the current subspace.
                r_T = e_i.T @ Q @ E_j  # Shape: (1, j)
                r = r_T.T  # Transpose to shape (j, 1)
                b = AtA_inv @ r  # Coefficients from the projection

                # Extract the candidate's contribution in the (j+1)-th direction.
                gamma = Q[i, j]

                # Compute an intermediate term used to adjust the candidate's evaluation.
                cTA = c.T @ A
                gamma_rT = gamma * r_T
                numerator_alpha = cTA + gamma_rT
                denominator_alpha = 1 + r_T @ b
                correction_matrix = np.eye(j) - np.outer(b, r_T) / denominator_alpha
                g_plus_gamma_b = g + gamma * b.flatten()
                alpha = numerator_alpha.flatten() @ correction_matrix @ g_plus_gamma_b

                # Evaluate the candidate using a compound objective function that balances several terms.
                cTc = c.T @ c
                numerator1_istar = 1 + r_T @ b
                denominator1_istar = np.prod(norm_A_columns_squared + r ** 2)
                numerator2_istar = cTc + gamma ** 2 - alpha
                denominator2_istar = cTc + gamma ** 2

                istar_obj = (numerator1_istar / denominator1_istar) * (numerator2_istar / denominator2_istar)

                # Update the best candidate if the current candidate improves the objective.
                if istar_obj > istar_max_obj:
                    istar_max_obj = istar_obj
                    i_star_candidate = i

            # Append the best candidate index for the current iteration.
            Z_indices.append(i_star_candidate)

            # Update Z_matrix by adding a new column corresponding to the newly selected index.
            Z_new_column = np.zeros((Q_col_len, 1))
            Z_new_column[i_star_candidate, 0] = 1
            Z_matrix = np.hstack((Z_matrix, Z_new_column))

        # If more sampling points are desired than the number of basis vectors,
        # perform additional oversampling.
        if num_sampling_points > n_f:
            num_additional_samples = num_sampling_points - len(Z_indices)
            for _ in range(num_additional_samples):
                A = Z_matrix.T @ Q
                AtA_inv = np.linalg.pinv(A.T @ A)
                istar_max_obj = -np.inf
                i_star_candidate = -1

                # Consider candidate indices not already selected.
                candidates = [i for i in range(Q_col_len) if i not in Z_indices]
                for i in candidates:
                    r = Q[i, :]
                    norm_A_columns_squared = np.sum(A ** 2, axis=0)
                    numerator_istar = 1 + r.T @ AtA_inv @ r
                    denominator_istar = np.prod(norm_A_columns_squared + r ** 2)
                    istar_obj = numerator_istar / denominator_istar

                    # Update the best candidate if a higher objective is found.
                    if istar_obj > istar_max_obj:
                        istar_max_obj = istar_obj
                        i_star_candidate = i

                # Add the candidate from the oversampling process.
                Z_indices.append(i_star_candidate)
                Z_new_column = np.zeros((Q_col_len, 1))
                Z_new_column[i_star_candidate, 0] = 1
                Z_matrix = np.hstack((Z_matrix, Z_new_column))

        # Create a boolean vector that marks the selected indices.
        Z_boolean = np.zeros(Q_col_len, dtype=bool)
        Z_boolean[Z_indices] = True

        # Extract the rows of the original basis that correspond to the selected indices.
        f_sampled_row = f_basis[Z_boolean, :]
        print(f"{Z_indices=}")
        return f_sampled_row, Z_indices, Z_boolean

    def deim_dof_to_elem(self, deim_dof):
        """
        Map the selected degrees-of-freedom (DOFs) to an element indicator vector.
        Each element (or cell) is flagged if any of its nonzero node equation IDs 
        appear in the DEIM-selected DOFs.
        
        Parameters:
        -----------
        deim_dof : list
            List of DOF indices selected by the DEIM or S-optimal procedure.
        
        Returns:
        --------
        x : ndarray
            A binary vector indicating for each element whether it contains any
            of the selected DOFs (1 if yes, 0 if no).
        """
        # Retrieve the list of nonzero equation IDs for each global node associated with elements.
        glob_node_nonzero_eqnId = self.cls.glob_node_nonzero_eqnId

        # Initialize an array to mark elements; default is 0 (not selected).
        x = np.zeros(len(glob_node_nonzero_eqnId))

        # Loop over each element and mark it if any of its node equation IDs are in the selected DOFs.
        for iel in range(len(glob_node_nonzero_eqnId)):
            # Check if any equation ID in the current element is among the DEIM-selected DOFs.
            bool_array = np.isin(glob_node_nonzero_eqnId[iel], deim_dof)
            if np.any(bool_array):                
                x[iel] = 1  # Mark the element as selected

        return x