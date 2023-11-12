import numpy as np
from src.codes.utils.fem_utils import *

class deim:

    def __init__(self,fos_cls,V,sol_snapshots,mask,tol_f=1e-2):

        self.cls=fos_cls
        self.tol_f=tol_f
        self.mask=mask
        self.sol_snapshots=sol_snapshots
        self.V_m=V[mask]


    def select_elems(self):

        # Build nonlinear force snapshots
        F_nl = []

        for snapshot in self.sol_snapshots:

            F_nl.append(self.build_nonlinear_term_(snapshot, self.mask))

        n_f_sel, U_f = self.force_mode_selector(F_nl, self.tol_f)

        U_fs = U_f[:,:n_f_sel]

        f_basis_sampled, sampled_rows, self.bool_sampled = self.deim_red(U_f, n_f_sel)

        masked_eqnID = self.cls.data.node_eqnId[self.cls.data.mask]

        deim_dof = [masked_eqnID[i] for i in sampled_rows]

        self.xi = self.deim_dof_to_elem(deim_dof)

        self.deim_mat = self.V_m.T @ U_fs @ np.linalg.inv(f_basis_sampled)

        self.U_fs = U_fs


    def force_mode_selector(self,F_nl, tol_f):
        
        NLF = np.asarray(F_nl)
        U_f, S_f, _ = np.linalg.svd(NLF.T, full_matrices=False)
        S_f_cumsum = np.cumsum(S_f)/np.sum(S_f)

        n_f_sel = np.where((1.0-S_f_cumsum)<tol_f)[0]
        n_f_sel = n_f_sel[0] if n_f_sel[0]!=0 else 1

        plt.figure(figsize = (6,4))
        plt.semilogy(1.0-S_f_cumsum,'s-')
        plt.axhline(y=tol_f, color="black", linestyle="--")

        plt.show()

        self.n_f_sel = n_f_sel

        return n_f_sel, U_f


    def build_nonlinear_term_(self,T,mask):

        sol = np.copy(T)
        sol = sol[mask]

        K, J, rhs = init_global_systems(max(self.cls.data.node_eqnId))

        for iel in range(self.cls.ncells[0]):

            # Compute element matrices for the current element
            Ke_, Je_, qe_ = compute_element_matrices(self.cls, T, iel)

            # Assemble global matrices
            K, J, _ = assemble_global_matrices(self.cls, iel, K, J, Ke_, Je_)

            # rhs, _ = assemble_global_forces(self, iel, qe_, rhs)

        # F_nl = K.dot(sol) #- rhs

        return  K.dot(sol)


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

        glob_node_nonzero_eqnId = self.cls.data.glob_node_nonzero_eqnId

        x = np.zeros(len(glob_node_nonzero_eqnId))

        for iel in range(len(glob_node_nonzero_eqnId)):

            bool_array = np.isin(glob_node_nonzero_eqnId[iel], deim_dof)

            if np.any(bool_array):                
                x[iel] = 1

        return x