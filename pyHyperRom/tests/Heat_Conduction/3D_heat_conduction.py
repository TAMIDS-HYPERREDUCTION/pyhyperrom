from Heat_Conduction import *

n_ref= [3,3,3]
L = [10.,12.,14.]

mat_layout = np.zeros((4,4,4),dtype=int)
src_layout = np.zeros((4,4,4),dtype=int)

# src_layout[0,0,0] = 1
# src_layout[1,1,1] = 1
# src_layout[2,2,2] = 1
# src_layout[3,3,3] = 1
fdict = {}

cond_list = []
cond_list.append( lambda T,mu: mu*T + 100. + 0.*T )
fdict["cond"] = cond_list

dcond_list = []
dcond_list.append( lambda T,mu: mu + 0. + 0.*T )
fdict["dcond"] = dcond_list

qext_list = []
qext_list.append( lambda T,mu: 0.0+100.0 + 0.*T )
fdict["qext"] = qext_list

# qext_list = []
# qext_list.append(lambda T,mu: mu + 0.*T)
# qext_list.append(lambda T,mu: 2.0*mu + 0.*T)
# fdict["qext"] = qext_list
bc = {}
bc['x_min']={'type':'dirichlet','value':0.}
bc['x_max']={'type':'dirichlet','value':0.}
bc['y_min']={'type':'dirichlet','value':0.}
bc['y_max']={'type':'dirichlet','value':0.}
bc['z_min']={'type':'dirichlet','value':0.}
bc['z_max']={'type':'dirichlet','value':0.}
params = np.r_[1.:4.0:0.01]
quad_deg = 3
N_snap = 15 # Training Snapshots
NL_solutions = []
param_list = []
K_mus = []
q_mus = []
pb_dim = 3
k_const= 4.0
Rom_const = 2.0
n_sel = 4
tol = 1e-9
heat_conduction_workflow(N_snap, params, bc, mat_layout, src_layout, fdict, n_ref, L, pb_dim, quad_deg, k_const, Rom_const, n_sel, tol, True, True, True)
NLS, NL_solutions, param_list, K_mus, q_mus, d, mask = simulate_FOS(N_snap, params, bc, mat_layout, src_layout, fdict, n_ref, L, pb_dim, quad_deg,k_const)
V_sel, xi= ECSW_heatconduction(NLS, N_snap, NL_solutions, n_sel, pb_dim, d, mask, K_mus, q_mus, tol, plot=True)
Rom_simulation(params, param_list, Rom_const, pb_dim, V_sel, xi, bc, mat_layout, src_layout, fdict, n_ref, L, quad_deg, FEM_solver_rom_ecsw, solve_fos)
