from utils import *
from FEM_solver_rom import FEM_solver_rom
from FEM_solver import FEM_solver 

class mydata:
    
    def __init__(self, n_ref, zone_widths, mu):
        """
        Initialize the mesh data for a given domain with specified refinement and zone widths.

        Parameters:
        - n_ref: List of refinement levels for each zone
        - zone_widths: List of widths for each zone in the domain
        - mu: Parameter value
        """

        # Check if the input lists have the same length
        if len(zone_widths) != len(n_ref):
            raise ValueError("dx and nref should have the same length")
        
        self.n_zones = len(zone_widths)

        # Compute the width of each cell in the mesh
        dx = np.repeat(zone_widths / n_ref, n_ref)

        # Compute the x-coordinates of the nodes in the mesh
        x = np.zeros(len(dx) + 1)
        for i in range(len(dx)):
            x[i + 1] = x[i] + dx[i]

        # Define which material each cell belongs to
        cell2mat = np.repeat(np.arange(len(zone_widths)), n_ref)

        # Store mesh information
        self.n_cells = len(dx)
        self.n_nodes = self.n_cells + 1
        
        # Create nodal connectivity for the continuous Finite Element Method (cFEM)
        self.connectivity()
        
        # Store additional data
        self.dx = np.copy(dx)
        self.x = np.copy(x)
        self.cell2mat = np.copy(cell2mat)
        
        # Store parameter value
        self.mu = mu
        
    def connectivity(self):
        """
        Define nodal connectivity for each cell in the mesh.
        """

        # Initialize the connectivity array
        self.gn = np.zeros((self.n_cells, 2), dtype=int)

        # Loop over all cells to define their nodal connectivity
        for i in range(self.n_cells):
            # For each cell, define the left and right nodes
            self.gn[i, 0] = i
            self.gn[i, 1] = i + 1


# Generate Snapshots from multiple runs with varying  𝜇
random.seed(25)
params = np.r_[1.:10.0:0.01]
NL_solutions = []
param_list = []
quad_deg = 5
N_snap = 15 # Training Snapshots

K_mus = []
q_mus = []

width   = np.array([ 0.4, 0.1])
n_ref = np.array([400, 100], dtype=int)
cond_arr = []

## Conductivity array as function of T and parameter

# data T in Kelvin. conductivity in W/m/K
# conductivity functions per zone

tune = 1 # Parameter to tune nonlinearity

cond_arr = []

cond_arr.append( [lambda T,mu: 1.05*mu*tune + 2150/(T-73.15),lambda T,mu: -2150/(T-73.15)**2]  )
cond_arr.append( [lambda T,mu: mu*tune*7.51 + 2.09e-2*T - 1.45e-5*T**2 + 7.67e-9*T**3,\
                  lambda T,mu: 2.09e-2 - 2*1.45e-5*T + 3*7.67e-9*T**2])
    
# cond_arr.append( [lambda T,mu: (1+mu) + tune*T, lambda T,mu: tune] ) #[k, dk/dT]
# cond_arr.append( [lambda T,mu: 10*abs(np.sin(mu)) + 0.*T, lambda T,mu: 0.0*T] )
cond_arr = np.asarray(cond_arr)

    
qext_arr = []
qext_arr.append( lambda T,mu: 35000.0 + 0.*T)
qext_arr.append( lambda T,mu: 0.0 + 0.*T)
qext_arr = np.asarray(qext_arr)

## Boundary Conditions (in degree Kelvin):
dummy_param = 1
dp = mydata(n_ref, width, dummy_param)

bc = {}

bc_sym={'type':'refl','value':np.nan}
bc_dir={'type':'dirichlet','value':273.15+300.}

bc['xmin']=bc_sym
bc['xmax']=bc_dir

T_init = np.zeros(dp.n_nodes) + 273.15

if bc['xmin']['type']=='dirichlet':
    T_init[0] = bc['xmin']['value']
if bc['xmax']['type']=='dirichlet':
    T_init[-1] = bc['xmax']['value']

for i in range(N_snap):
    
    param = random.choice(params) # Choose from parameter list
    param_list.append(param)
    d = mydata(n_ref, width,param)
    solver = FEM_solver(d, quad_deg)
    
    # solve
    if i==0:
        NL_solution_p, Le, Ke, rhs_e,_,_ = solver.solve_system(cond_arr, qext_arr, bc, T_init)
        NL_solution_p = NL_solution_p
    else:
        NL_solution_p,_, Ke, rhs_e, mask, T_dir = solver.solve_system(cond_arr, qext_arr, bc, T_init) # Note mask and T_dir are taken out
        NL_solution_p = NL_solution_p 
    NL_solutions.append(NL_solution_p.flatten())
    K_mus.append(Ke)
    q_mus.append(rhs_e)

plt.figure()
for _,val in enumerate(NL_solutions):
    plt.plot(d.x,val)
    plt.grid(True)  
plt.show()

NLS = np.asarray(NL_solutions)
np.shape(NLS)


# ECSW Hyper-reduction
#SVD
n_sel = 8
U, S, Vt = np.linalg.svd(np.transpose(NLS), full_matrices=False)
V_sel = U[:, :n_sel]

plt.figure(figsize = (6,4))
plt.semilogy(S,'s-')
plt.show()

plt.figure(figsize=(6,4))

for i in range(n_sel):
    plt.plot(d.x,V_sel[:,i])
plt.grid(True)
plt.show()

P_sel = V_sel[mask,:]@np.transpose(V_sel[mask,:])

tic_h_setup_a = time.time()

ncells = d.n_cells
C = np.zeros((n_sel*N_snap,ncells))

for i in range(N_snap):
    for j in range(ncells):
        Ce = np.transpose(V_sel[mask,:])@np.transpose(Le[j])@K_mus[i][j]@Le[j]@P_sel@NL_solutions[i][mask].reshape(-1,1) - np.transpose(V_sel[mask,:])@np.transpose(Le[j])@np.array(q_mus[i][j]).reshape(-1,1)
        C[i*n_sel:(i+1)*n_sel,j] = Ce.flatten()
        
d_vec = C@np.ones((d.n_cells,1))
toc_h_setup_a = time.time()

tic_h_setup_b = time.time()

x = fe.fnnls(C, d_vec.flatten(), tolerance=1e-11)#, max_iterations=20) 

toc_h_setup_b = time.time()

residual = np.linalg.norm(d_vec.flatten() - np.dot(C, x))/np.linalg.norm(d_vec.flatten())
print(residual)
plt.stem(x)
plt.show()
print(f"Fraction of total elements active in the ROM: {len(x[x>0])*100/len(x)}%")


# Solving a ROM for an unknown  𝜇
params_rm = params[~np.isin(params,param_list)]
param_rom = random.choice(params_rm)

d_fos = mydata(n_ref, width,param_rom)
solver = FEM_solver(d_fos, quad_deg)
    
d_rom = mydata(n_ref, width, param_rom)
solver_rom = FEM_solver_rom(d_rom, quad_deg)

T_init_fos = np.zeros(d_rom.n_nodes) + 273.15
T_init_rom = np.transpose(V_sel)@T_init_fos


tic_fos = time.time()
NL_solution_p_fos_test, _, _, _,_,_ = solver.solve_system(cond_arr, qext_arr, bc, T_init_fos);
toc_fos = time.time()
tic_rom_woh = time.time()
NL_solution_p_reduced_woh = solver_rom.solve_reduced_system(cond_arr, qext_arr, bc, T_init_rom,np.ones_like(x),V_sel);
toc_rom_woh = time.time()
tic_rom = time.time()
NL_solution_p_reduced = solver_rom.solve_reduced_system(cond_arr, qext_arr, bc, T_init_rom,x,V_sel);
toc_rom = time.time()
sol_red = V_sel@NL_solution_p_reduced.reshape(-1,1)#+pca.mean_.reshape(-1,1)

plt.plot(d.x, sol_red)
plt.plot(d.x,NL_solution_p_fos_test,'k.')
plt.title(f"RMS_error is {np.linalg.norm(sol_red-NL_solution_p_fos_test.reshape(-1,1))*100/np.linalg.norm(NL_solution_p_fos_test.reshape(-1,1))} %")
plt.grid(True)
plt.show()
print(f"ROM Error without hyperreduction is {np.linalg.norm(V_sel@NL_solution_p_reduced_woh.reshape(-1,1)-NL_solution_p_fos_test.reshape(-1,1))*100/np.linalg.norm(NL_solution_p_fos_test.reshape(-1,1))} %")

## Speedups
fos_sim_time = toc_fos - tic_fos
rom_sim_time_woh = toc_rom_woh - tic_rom_woh

rom_sim_time = toc_rom - tic_rom
h_total_setup_time = (toc_h_setup_b+toc_h_setup_a) - (tic_h_setup_b+tic_h_setup_a) #this is one time

print(f"speedup without hyperreduction:{fos_sim_time/rom_sim_time_woh}")
print(f"speedup with hyperreduction:{fos_sim_time/(rom_sim_time)}")






