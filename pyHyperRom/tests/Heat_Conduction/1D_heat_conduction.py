from Heat_Conduction import *

n_ref = np.array([400, 100], dtype=int)
w = np.array([ 0.4, 0.1])
L = np.sum(w)

# Create arrays of zeros and ones
zeros_array = np.zeros((1, n_ref[0]))
ones_array = np.ones((1, n_ref[1]))

# Concatenate along the second axis (axis=1)
mat_layout = np.concatenate((zeros_array, ones_array), axis=1)
src_layout = np.concatenate((zeros_array, ones_array), axis=1)

fdict = {}

tune = 1

cond_list = []
cond_list.append(lambda T,mu: 1.05*mu*tune + 2150/(T-73.15))
cond_list.append(lambda T,mu: mu*tune*7.51 + 2.09e-2*T - 1.45e-5*T**2 + 7.67e-9*T**3)
fdict["cond"] = cond_list

dcond_list = []
dcond_list.append(lambda T,mu: -2150/(T-73.15)**2 )
dcond_list.append(lambda T,mu: 2.09e-2 - 2*1.45e-5*T + 3*7.67e-9*T**2)
fdict["dcond"] = dcond_list

qext_list = []
qext_list.append( lambda T,mu: 35000.0 + 0.*T)
qext_list.append( lambda T,mu: 0.0 + 0.*T)
fdict["qext"] = qext_list

bc = {}
bc['x_min']={'type':'refl','value':np.nan}
bc['x_max']={'type':'dirichlet','value':273.15+300.}   

print(mat_layout.flatten().shape[0])

N_snap = 15
params = np.r_[1.:4.0:0.01]
quad_deg = 3
k_const= 273.15
n_sel = 4
tol = 1e-8
Rom_const= 273.15
pb_dim =1
heat_conduction_workflow(N_snap, params, bc, mat_layout, src_layout, fdict, n_ref, L, pb_dim, quad_deg, k_const, Rom_const, n_sel, tol, True, True, True)
NLS, NL_solutions, param_list, K_mus, q_mus, d, mask = simulate_FOS(N_snap, params, bc, mat_layout, src_layout, fdict, n_ref, L, pb_dim, quad_deg,k_const)
V_sel, xi= ECSW_heatconduction(NLS, N_snap, NL_solutions, n_sel, 1, d, mask, K_mus, q_mus, tol, plot=True)
Rom_simulation(params, param_list, Rom_const, 1, V_sel, xi, bc, mat_layout, src_layout, fdict, n_ref, L, quad_deg, FEM_solver_rom_ecsw, solve_fos)
