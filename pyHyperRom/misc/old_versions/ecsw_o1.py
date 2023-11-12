# from ..basic import *

# def ecsw_red_old(d, V_sel, Le, K_mus, q_mus, P_sel, tol, n_sel,N_snap,mask,NL_solutions):

#     ncells = d.n_cells
#     C = np.zeros((n_sel*N_snap,ncells))

#     for i in range(N_snap):
#         for j in range(ncells):
#             Ce = np.transpose(V_sel[mask,:])@np.transpose(Le[j])@K_mus[i][j]@Le[j]@P_sel@NL_solutions[i][mask].reshape(-1,1) - np.transpose(V_sel[mask,:])@np.transpose(Le[j])@np.array(q_mus[i][j]).reshape(-1,1)
#             C[i*n_sel:(i+1)*n_sel,j] = Ce.flatten()
            
#     d_vec = C@np.ones((d.n_cells,1))

#     x = fe.fnnls(C, d_vec.flatten(), tolerance=tol)       #, max_iterations=20)
#     residual = np.linalg.norm(d_vec.flatten() - np.dot(C, x))/np.linalg.norm(d_vec.flatten())

#     return x, residual
