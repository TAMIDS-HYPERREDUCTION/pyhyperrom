from src.codes.basic import *
from src.codes.algorithms.nnls_scipy import nnls as nnls_sp
from scipy.optimize import nnls
import numpy as np
import pylibROM.linalg as linalg
import pylibROM.utils as utils
from mpi4py import MPI

def ecsw_red_ms(d, V_sel, Le, K_mus, q_mus, n_sel, N_snap, mask, NL_solutions, NL_solutions_mean, tol=None):
    """
    Executes Enhanced Compressed Snapshot Weighting (ECSW) reduction on nonlinear Finite Element Method (FEM) problems.
    
    Parameters:
    - d: Object with mesh and FEM details.
    - V_sel: Basis vectors selected for reduction.
    - Le: Connectivity matrix for elements and nodes.
    - K_mus: Stiffness matrices for each snapshot.
    - q_mus: Source terms for each snapshot.
    - n_sel: Count of selected basis vectors.
    - N_snap: Total number of snapshots.
    - mask: Boolean array for nodes exempt from Dirichlet boundary conditions.
    - NL_solutions: Nonlinear solutions for all snapshots, mean adjusted.
    - NL_solutions_mean: Mean of the nonlinear solutions.
    - tol: Solver tolerance (optional).
    
    Returns:
    - x: Solution vector for the least squares problem.
    - residual: Residual of the solution.
    """
    
    ncells = d.n_cells  # Number of cells in the mesh
    C = np.zeros((n_sel * N_snap, int(ncells)))  # Initialize C matrix

    P_sel = V_sel @ V_sel.T  # Projection matrix

    for i in range(N_snap):
        projected_sol_mask = np.dot(P_sel, NL_solutions[i]) + NL_solutions_mean

        for j in range(ncells):
            col_indices = np.argmax(Le[j], axis=1)
            K_mus_ij = K_mus[i][j]
            q_mus_ij = np.array(q_mus[i][j])
            Ce = np.dot(np.transpose(V_sel[col_indices]), (np.dot(K_mus_ij, projected_sol_mask[col_indices]) - q_mus_ij))
            C[i * n_sel: (i + 1) * n_sel, j] = Ce

    d_vec = C @ np.ones((ncells, 1))
    norm_d_vec = np.linalg.norm(d_vec)
    print(f"norm of rhs: {norm_d_vec}")

    if tol is None:
        x, residual = nnls(C, d_vec.flatten(), maxiter=1e6)
    else:
        comm = MPI.COMM_WORLD
        rank = comm.rank
        comm.Barrier()
        
        local_dim = utils.split_dimension(C.shape[1], comm)
        nnls_llnl = linalg.NNLSSolver(const_tol=tol)
        x = linalg.Vector(np.zeros(local_dim), True, True)
        rhs_lb, rhs_ub = (d_vec.flatten() - 10 * tol), (d_vec.flatten() + 10 * tol)
        At_lr = linalg.Matrix(C.T.copy(), False, True)
        At_lr.distribute(local_dim)
        lb_lr, ub_lr = linalg.Vector(rhs_lb.copy(), False, True), linalg.Vector(rhs_ub.copy(), False, True)
        nnls_llnl.normalize_constraints(At_lr, lb_lr, ub_lr)
        nnls_llnl.solve_parallel_with_scalapack(At_lr, lb_lr, ub_lr, x)

        x_local = np.array(x).flatten()
        if rank == 0:
            x_global = np.empty(C.shape[1], dtype=x_local.dtype)
        else:
            x_global = None

        comm.Gather(x_local, x_global, root=0)

        if rank == 0:
            residual = np.linalg.norm(np.dot(C, x_global) - d_vec.flatten())
            MPI.Finalize()
            return x_global, residual / norm_d_vec, rank
        else:
            MPI.Finalize()
            return None, None, None
