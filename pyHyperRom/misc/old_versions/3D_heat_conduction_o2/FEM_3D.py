# %%
# Restart the kernel
import os
current_dir ='../../'
os.chdir(current_dir)

# %%
from src.codes.basic import *
from src.codes.utils import *
from src.codes.base_classes import Base_class_fem_heat_conduction
from src.codes.reductor.rom_class import FEM_solver_rom_ecsw
from src.codes.algorithms.ecsw import ecsw_red

# %% [markdown]
# ### class for data (geometry, material property, mesh)

# %%
nref= [2,2,2]
L = [10.,12.,14.]

mat_layout = np.zeros((4,4,4),dtype=int)
src_layout = np.zeros((4,4,4),dtype=int)

src_layout[0,0,0] = 1
src_layout[1,1,1] = 1
src_layout[2,2,2] = 1
src_layout[3,3,3] = 1

# %%
fdict = {}

cond_list = []
cond_list.append( lambda T,mu:  1 + mu*T )
fdict["cond"] = cond_list

dcond_list = []
dcond_list.append( lambda T,mu: mu + 0.*T )
fdict["dcond"] = dcond_list

qext_list = []
qext_list.append(lambda T,mu: mu + 0.*T)
qext_list.append(lambda T,mu: 2.0*mu + 0.*T)
fdict["qext"] = qext_list

# %%
bc = {}
bc['x_min']={'type':'dirichlet','value':1.}
bc['x_max']={'type':'dirichlet','value':0.}
bc['y_min']={'type':'dirichlet','value':0.}
bc['y_max']={'type':'dirichlet','value':0.}
bc['z_min']={'type':'dirichlet','value':0.}
bc['z_max']={'type':'dirichlet','value':0.}

# %%
class probdata:
    
    def __init__(self, bc, cond_layout, qext_layout, fdict, nref, L, mu, pb_dim=3):
        
        self.dim_ = pb_dim
        # refine the mesh and update material and source layouts
        repeats = np.asarray(nref, dtype=int)
        self.cell2mat_layout = self.repeat_array(mat_layout,repeats)
        self.cell2src_layout = self.repeat_array(src_layout,repeats)
        
        ## change this mapping if needed.
        
        self.fdict = fdict
        
        # mesh data
        # cells
        self.ncells = [None] * pb_dim
        self.npts = [None] * pb_dim
        self.deltas = [None] * pb_dim
        self.xi=[]
        for i in range(pb_dim):
            self.ncells[i] = self.cell2mat_layout.shape[i]
            self.npts[i] = self.ncells[i]+1
            self.xi.append(np.linspace(0,L[i],self.npts[i]))
            self.deltas[i] = L[i]/self.ncells[i]
    
        self.n_verts = np.prod(np.array(self.npts))
                
        # Create nodal connectivity for the continuous Finite Element Method (cFEM)
        self.connectivity()
                
        # Store parameter value
        self.mu = mu
        
        # Store the dirichlet nodes if any
        handle_boundary_conditions(self, bc)
        
        # Determining the global equation numbers based on dirichlet nodes and storing in class
        get_glob_node_equation_id(self, self.dir_nodes)

        # Get global node numbers and equation IDs for the current element
        self.glob_node_eqnId = []
        self.glob_node_nonzero_eqnId = []
        self.local_node_nonzero_eqnId = []
        self.Le = []
        self.global_indices = []
        self.local_indices = []

        for i in range(self.n_cells):
            get_element_global_nodes_and_nonzero_eqnId(self, i, self.node_eqnId)

    
              
    def repeat_array(self, arr,repeats):
        for dim,n in enumerate(repeats):
            arr = np.repeat(arr,n,axis=dim)
        return arr     
    
    
    def connectivity(self):
        """
        Define nodal connectivity for each cell in the mesh.
        """

        # Initialize the connectivity array
        self.n_cells = np.prod(np.array(self.ncells))
        self.gn = np.zeros((self.n_cells,2**self.dim_),dtype=int)

        # # compute node ID from (i,j) cell identifiers
        # def node(*args):
        #     index = 0
        #     multiplier = 1
        #     for i, n in enumerate(args):
        #         index += n * multiplier
        #         if i < len(self.npts) - 1:
        #             multiplier *= self.npts[i]
        #     return index
        
        node = lambda i,j,k: i+j*self.npts[0]+k*self.npts[0]*self.npts[1]
        # Loop over all cells to define their nodal connectivity
        iel = 0
        for k in range(self.ncells[2]):
            for j in range(self.ncells[1]):
                for i in range(self.ncells[0]):
                    # counter-clockwise
                    self.gn[iel,0] = node(i  ,j  ,k  )
                    self.gn[iel,1] = node(i+1,j  ,k  )
                    self.gn[iel,2] = node(i+1,j+1,k  )
                    self.gn[iel,3] = node(i  ,j+1,k  )
                    self.gn[iel,4] = node(i  ,j  ,k+1)
                    self.gn[iel,5] = node(i+1,j  ,k+1)
                    self.gn[iel,6] = node(i+1,j+1,k+1)
                    self.gn[iel,7] = node(i  ,j+1,k+1)
                    iel += 1
  

# %% [markdown]
# ### Simulate FOS

# %%
random.seed(25)
params = np.r_[1.:4.0:0.01]
quad_deg = 3
N_snap = 15 # Training Snapshots
NL_solutions = []
param_list = []
K_mus = []
q_mus = []

# %%
for i in range(N_snap):
    param = random.choice(params) # Choose from parameter list
    param_list.append(param)
    
    if i==0:
        d = probdata(bc, mat_layout, src_layout, fdict, nref, L, param, pb_dim=3)
        FOS = Base_class_fem_heat_conduction(d,quad_deg)
    else:
        FOS.mu = param
    T_init = np.zeros(d.n_verts) + 2.0
    NL_solution_p, Ke, rhs_e, mask = solve_fos(FOS, T_init)
    NL_solutions.append(NL_solution_p.flatten())
    K_mus.append(Ke)
    q_mus.append(rhs_e)
    plot3D(d.xi[0], d.xi[1], d.xi[2], NL_solution_p, hmap=True)

# %%
NLS = np.asarray(NL_solutions)
np.shape(NLS)

# %% [markdown]
# 
# ### ECSW Hyper-reduction
# #### Step 1: Perform SVD on the snapshots (calculate $\mathbb{V}(=\mathbb{W}$)):

# %%
n_sel = 6
U, S, Vt = np.linalg.svd(np.transpose(NLS), full_matrices=False)
V_sel = U[:, :n_sel]
P_sel = V_sel[mask,:]@np.transpose(V_sel[mask,:])

# %%
np.shape(P_sel)

# %%
plt.figure(figsize = (6,4))
plt.semilogy(S,'s-')
plt.show()

# %%
for i in range(n_sel):
    plot3D(d.xi[0],d.xi[1],d.xi[2],V_sel[:,i],hmap=True)

# %% [markdown]
#  
# #### ECSW

# %%
tic_h_setup_b = time.time()
tol = 1e-15
xi, residual = ecsw_red(d, V_sel, d.Le, K_mus, q_mus, P_sel, tol, n_sel, N_snap, mask,NL_solutions)
toc_h_setup_b = time.time()

# %%
print(f"this is the residual from fnnls: {residual}")

# %%
colors = ['red' if value > 0 else 'blue' for value in xi]
sizes = [15 if value > 0 else 1 for value in xi]

# %%
plot3D(np.arange(d.ncells[0]),np.arange(d.ncells[1]),np.arange(d.ncells[2]),xi, sz = sizes, clr = colors, save_file=False)

# %%
print(f"Fraction of total elements active in the ROM: {len(xi[xi>0])*100/len(xi)}%")

# %% [markdown]
# ### ROM Simulation

# %%
# Choose unknown parameter

params_rm = params[~np.isin(params,param_list)]
param_rom = random.choice(params_rm)

# %%
# Define the data-class

d_test = probdata(bc, mat_layout, src_layout, fdict, nref, L, param_rom, pb_dim=3)
FOS_test = Base_class_fem_heat_conduction(d_test,quad_deg)
ROM = FEM_solver_rom_ecsw(d_test, quad_deg)

# %%
# Initial guess

T_init_fos = np.zeros(FOS_test.n_nodes) + 2.
T_init_rom = np.transpose(V_sel)@T_init_fos # crucial to ensure the initial guess is contained in the reduced subspace

# %%
# Time taken to perform a FO simulation with the current parameter value

tic_fos = time.time()
NL_solution_p_fos_test, _, _, _, = solve_fos(FOS_test,T_init_fos)
toc_fos = time.time()

# %%
# Time taken to simulate a ROM without hyper-reduction

tic_rom_woh = time.time()
NL_solution_p_reduced_woh = ROM.solve_rom(T_init_rom,np.ones_like(xi),V_sel)
toc_rom_woh = time.time()

# %% [markdown]
# 

# %%
# Time taken to simulate a ROM *with* hyper-reduction

tic_rom = time.time()
NL_solution_p_reduced = ROM.solve_rom(T_init_rom,xi,V_sel)
toc_rom = time.time()

# %%
sol_red = V_sel@NL_solution_p_reduced.reshape(-1,1)  #+pca.mean_.reshape(-1,1)
plot3D(d_test.xi[0], d_test.xi[1], d_test.xi[2], sol_red,hmap=True)
print(f"RMS_error is {np.linalg.norm(sol_red-NL_solution_p_fos_test.reshape(-1,1))*100/np.linalg.norm(NL_solution_p_fos_test.reshape(-1,1))} %")

# %%
plot3D(d_test.xi[0], d_test.xi[1], d_test.xi[2], NL_solution_p_fos_test,hmap=True)

print(f"\n\nROM Error without hyperreduction is {np.linalg.norm(V_sel@NL_solution_p_reduced_woh.reshape(-1,1)-NL_solution_p_fos_test.reshape(-1,1))*100/np.linalg.norm(NL_solution_p_fos_test.reshape(-1,1))} %")

# %% [markdown]
#  
# ### Speedups

# %%
fos_sim_time = toc_fos - tic_fos
rom_sim_time_woh = toc_rom_woh - tic_rom_woh
rom_sim_time = toc_rom - tic_rom

# %%
print(f"speedup without hyperreduction:{fos_sim_time/rom_sim_time_woh}")
print(f"speedup with hyperreduction:{fos_sim_time/(rom_sim_time)}")
# h_total_setup_time = (toc_h_setup_b+toc_h_setup_a) - (tic_h_setup_b+tic_h_setup_a) #this is one time


