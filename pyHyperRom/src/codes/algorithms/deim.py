import numpy as np
from src.codes.utils.fem_utils import *
from ..utils.rom_utils import *

class deim:

    def __init__(self,d,F_nl,train_mask,param_list,V,sol_snapshots,mask,tol_f=1e-2, extra_modes = 0):

        self.cls=d
        self.tol_f=tol_f
        self.mask=mask
        self.sol_snapshots=sol_snapshots
        self.V_m=V[mask]
        self.mu_list = param_list
        self.F_nl=F_nl[train_mask]
        self.extra_modes = extra_modes

    def select_elems(self):

        # Build nonlinear force snapshots

        n_f_sel, U_f = svd_mode_selector(self.F_nl, self.tol_f)
        n_f_sel+=self.extra_modes
        print(f"Selected modes:{n_f_sel}")
        U_fs = U_f[:,:n_f_sel]
        f_basis_sampled, sampled_rows, self.bool_sampled = self.deim_red(U_f, n_f_sel)
        masked_eqnID = self.cls.node_eqnId[self.cls.mask]
        deim_dof = [masked_eqnID[i] for i in sampled_rows]

        self.xi = self.deim_dof_to_elem(deim_dof)
        self.deim_mat = self.V_m.T @ U_fs @ np.linalg.inv(f_basis_sampled)
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