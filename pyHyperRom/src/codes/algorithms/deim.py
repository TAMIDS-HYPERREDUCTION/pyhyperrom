import numpy as np
from src.codes.utils.fem_utils_HC import *
from ..utils.rom_utils import *

class deim:

    def __init__(self,d,F_nl,train_mask,param_list,V,sol_snapshots,mask,tol_f=1e-2, extra_modes = 0, extra_samples = 0):

        self.cls=d
        self.tol_f=tol_f
        self.mask=mask
        self.sol_snapshots=sol_snapshots
        self.V_m=V#[mask]
        self.mu_list = param_list

        self.F_nl=F_nl[train_mask]

        self.extra_modes = extra_modes
        self.extra_samples = extra_samples

    def select_elems(self):

        # Build nonlinear force snapshots

        n_f_sel, U_f = svd_mode_selector(self.F_nl, self.tol_f)
        n_f_sel+=self.extra_modes
        print(f"Selected modes:{n_f_sel}")
        U_fs = U_f[:,:n_f_sel]
        f_basis_sampled, sampled_rows, self.bool_sampled = self.sopt_red(U_f, n_f_sel) #self.deim_red(U_f, n_f_sel) 
        masked_eqnID = self.cls.node_eqnId[self.cls.mask]
        deim_dof = [masked_eqnID[i] for i in sampled_rows]

        self.xi = self.deim_dof_to_elem(deim_dof)
        self.deim_mat = self.V_m.T @ U_fs @ np.linalg.pinv(f_basis_sampled)
        self.U_fs = U_fs
        self.n_f_sel = n_f_sel

    def deim_red(self,f_basis, num_f_basis_vectors_used):

        """
        Perform Discrete Empirical Interpolation Method (DEIM) to reduce the
        dimension of the right-hand side function.
        
        Parameters:
        -----------
        f_basis : ndarray
            Basis matrix (full-order).
        num_f_basis_vectors_used : int
            Number of basis vectors to use.
        
        Returns:
        --------
        f_basis_sampled : ndarray
            Sampled basis vectors.
        sampled_rows : list
            Indices of the rows that are sampled.
        is_sampled : ndarray
            Boolean array indicating which rows are sampled.
        
        Example Usage:
        --------------
        >>> f_basis = np.random.rand(100, 10)
        >>> num_f_basis_vectors_used = 5
        >>> result, sampled_rows, is_sampled = deim_red(f_basis, num_f_basis_vectors_used)
        """
        
        # Initialize variables
        num_basis_vectors = min(num_f_basis_vectors_used, f_basis.shape[1])
        basis_size = f_basis.shape[0]
        f_basis_sampled = np.zeros((num_basis_vectors, num_basis_vectors))
        
        # List to store the indices of the sampled rows
        sampled_rows = []
        
        # Boolean array to indicate which rows are sampled
        is_sampled = np.zeros(basis_size, dtype=bool)

        # Find the row index of the maximum value for the first basis vector
        f_bv_max_global_row = np.argmax(np.abs(f_basis[:, 0]))
        sampled_rows.append(f_bv_max_global_row)
        is_sampled[f_bv_max_global_row] = True

        # Store the sampled row of the first basis vector
        f_basis_sampled[0, :] = f_basis[f_bv_max_global_row, :num_basis_vectors]


        # Loop to find the other sampled rows
        for i in range(1, num_basis_vectors):
            # Solve for the coefficients c
            c = np.linalg.solve(f_basis_sampled[:i, :i], f_basis_sampled[:i, i])
            
            # Compute the residual and find its maximum value's index
            r_val = np.abs(f_basis[:, i] - np.dot(f_basis[:, :i], c))
            f_bv_max_global_row = np.argmax(r_val)
            
            # Store the sampled row index and update the boolean array
            sampled_rows.append(f_bv_max_global_row)
            is_sampled[f_bv_max_global_row] = True

            # Update the sampled basis matrix
            f_basis_sampled[i, :] = f_basis[f_bv_max_global_row, :num_basis_vectors]


        return f_basis_sampled, sampled_rows, is_sampled



    def sopt_red(self,f_basis, num_f_basis_vectors_used):

        # Step 1: QR Factorization - Obtain orthogonal basis Q from f_basis
                # Initialize variables

        num_sampling_points = num_f_basis_vectors_used+self.extra_samples

        n_f = min(num_f_basis_vectors_used, f_basis.shape[1])

        f_basis = f_basis[:,:n_f]

        Q, R = np.linalg.qr(f_basis)
        Q_col_len, _ = Q.shape

        # Initialization - Select the initial index with the maximum value in the first column of Q
        i_star = np.argmax(np.abs(Q[:, 0]))
        Z_indices = [i_star]

        # Construct initial Z_matrix
        Z_matrix = np.zeros((Q_col_len, 1))
        Z_matrix[i_star, 0] = 1
        E_ = np.eye(n_f)

        # Iteratively select features
        for j in range(1, n_f):
            # Construct E_j matrix (first j standard basis vectors)
            E_j = E_[:, :j]

            # Construct A = Z^T * Q * E_j
            A = Z_matrix.T @ Q @ E_j  # Shape (len(Z_indices), j)

            # Construct c = Z^T * Q * e_{j+1}
            e_j_plus1 = np.zeros(n_f)
            e_j_plus1[j] = 1
            c = Z_matrix.T @ Q @ e_j_plus1  # Shape (len(Z_indices),)

            # Compute g
            AtA_inv = np.linalg.inv(A.T @ A)  # Inverse of A^T * A
            g = AtA_inv @ (A.T @ c)

            # Precompute norms of columns of A squared
            norm_A_columns_squared = np.sum(A ** 2, axis=0)  # Shape (j,)

            # Initialize variables to keep track of the maximum objective
            istar_max_obj = -np.inf
            i_star_candidate = -1

            # Iterate over candidate indices (not in Z_indices)
            candidates = [i for i in range(Q_col_len) if i not in Z_indices]

            for i in candidates:
                # Construct e_i vector
                e_i = np.zeros((Q_col_len, 1))
                e_i[i] = 1

                # Compute r = e_i^T * Q * E_j
                r_T = e_i.T @ Q @ E_j  # Shape (1, j)
                r = r_T.T  # Transpose to match the shape (j, 1)
                b = AtA_inv @ r  # Shape (j,)

                # Compute gamma = e_i^T * Q * e_{j+1}
                gamma = Q[i, j]

                # Compute alpha
                cTA = c.T @ A
                gamma_rT = gamma * r_T
                numerator_alpha = cTA + gamma_rT
                denominator_alpha = 1 + r_T @ b
                correction_matrix = np.eye(j) - np.outer(b, r_T) / denominator_alpha
                g_plus_gamma_b = g + gamma * b.flatten()
                alpha = numerator_alpha.flatten() @ correction_matrix @ g_plus_gamma_b

                # Calculate objective to find the best candidate
                cTc = c.T @ c
                numerator1_istar = 1 + r_T @ b
                denominator1_istar = np.prod(norm_A_columns_squared + r ** 2)
                numerator2_istar = cTc + gamma ** 2 - alpha
                denominator2_istar = cTc + gamma ** 2

                istar_obj = (numerator1_istar / denominator1_istar) * (numerator2_istar / denominator2_istar)

                if istar_obj > istar_max_obj:
                    istar_max_obj = istar_obj
                    i_star_candidate = i

            # Update Z_indices with the index that maximizes the objective
            Z_indices.append(i_star_candidate)

            # Update Z_matrix by adding a new column for the new index
            Z_new_column = np.zeros((Q_col_len, 1))
            Z_new_column[i_star_candidate, 0] = 1
            Z_matrix = np.hstack((Z_matrix, Z_new_column))

        # Perform oversampling if needed

        if num_sampling_points > n_f:
            num_additional_samples = num_sampling_points - len(Z_indices)
            for _ in range(num_additional_samples):
                A = Z_matrix.T @ Q
                AtA_inv = np.linalg.pinv(A.T @ A)
                istar_max_obj = -np.inf
                i_star_candidate = -1

                # Iterate over candidate indices for oversampling
                candidates = [i for i in range(Q_col_len) if i not in Z_indices]
                for i in candidates:
                    r = Q[i, :]
                    norm_A_columns_squared = np.sum(A ** 2, axis=0)
                    numerator_istar = 1 + r.T @ AtA_inv @ r
                    denominator_istar = np.prod(norm_A_columns_squared + r ** 2)
                    istar_obj = numerator_istar / denominator_istar

                    # Update the best candidate if the current objective is better
                    if istar_obj > istar_max_obj:
                        istar_max_obj = istar_obj
                        i_star_candidate = i

                # Update Z_indices and Z_matrix with the selected candidate
                Z_indices.append(i_star_candidate)
                Z_new_column = np.zeros((Q_col_len, 1))
                Z_new_column[i_star_candidate, 0] = 1
                Z_matrix = np.hstack((Z_matrix, Z_new_column))


        # Create a boolean vector indicating selected indices
        Z_boolean = np.zeros(Q_col_len, dtype=bool)
        Z_boolean[Z_indices] = True

        f_sampled_row = f_basis[Z_boolean, :]
        print(f"{Z_indices=}")
        return f_sampled_row, Z_indices, Z_boolean



    def deim_dof_to_elem(self,deim_dof):

        glob_node_nonzero_eqnId = self.cls.glob_node_nonzero_eqnId

        x = np.zeros(len(glob_node_nonzero_eqnId))

        for iel in range(len(glob_node_nonzero_eqnId)):

            bool_array = np.isin(glob_node_nonzero_eqnId[iel], deim_dof)

            if np.any(bool_array):                
                x[iel] = 1

        return x

    # def build_nonlinear_term_(self,T,mask):

    #     sol1 = np.copy(T)
    #     sol = sol1[mask]

    #     K, J, rhs = init_global_systems(max(self.cls.node_eqnId))

    #     for iel in range(self.cls.ncells[0]):

    #         # Compute element matrices for the current element
    #         Ke_, Je_, qe_ = compute_element_matrices(self.cls, sol1, iel)

    #         # Assemble global matrices
    #         K, J, _ = assemble_global_matrices(self.cls, iel, K, J, Ke_, Je_)

    #         rhs, _ = assemble_global_forces(self.cls, iel, qe_, Ke_, rhs)

    #     # F_nl = K.dot(sol) #- rhs

    #     return  rhs, K.dot(sol)