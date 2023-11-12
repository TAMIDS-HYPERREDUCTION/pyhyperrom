# %%
import os
current_dir = "D:\D\ONEDRIVE\OneDrive - Texas A&M University\TAMU_MATERIALS\POSTDOC\HYPERREDUCTION\SUPARNO\Hyperreduction_tamids\pyHyperRom"
os.chdir(current_dir)
# print(os.getcwd())

# %%
from src.codes.basic import *
from src.codes.utils import *
from src.codes.base_classes import Base_class_fem_heat_conduction
from src.codes.reductor.rom_class import FEM_solver_rom_ecsw
from src.codes.algorithms.ecsw import ecsw_red


# %% 
# ### class for data (geometry, material property, mesh)

# %%
nref= [20,2]
L = [20,20]
mat_layout = np.zeros((5,2),dtype=int)
src_layout = np.zeros((5,2),dtype=int)

# %%
fdict = {}

cond_list = []
cond_list.append( lambda T,mu: mu*T + 100. + 0.*T )
fdict["cond"] = cond_list

dcond_list = []
dcond_list.append( lambda T,mu: mu + 0. + 0.*T )
fdict["dcond"] = dcond_list

qext_list = []
qext_list.append( lambda T,mu: 0.0+1.0 + 0.*T )
fdict["qext"] = qext_list

# %%
bc = {}
bc['x_min']={'type':'dirichlet','value':10.0}
bc['x_max']={'type':'dirichlet','value':7.0}
bc['y_min']={'type':'refl','value':np.nan}
bc['y_max']={'type':'refl','value':np.nan}

# %%
class probdata:
    
    def __init__(self, cond_layout, qext_layout, fdict, nref, L, mu, pb_dim=2):

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

        # compute node ID from (i,j) cell identifiers
        def node(*args):
            index = 0
            multiplier = 1
            for i, n in enumerate(args):
                index += n * multiplier
                if i < len(self.npts) - 1:
                    multiplier *= self.npts[i]
            return index
        
#        node = lambda i,j: i+j*self.npts_x

        # Loop over all cells to define their nodal connectivity
        iel = 0
        for j in range(self.ncells[1]):
            for i in range(self.ncells[0]):
                # counter-clockwise
                self.gn[iel,0] = node(i  ,j  )
                self.gn[iel,1] = node(i+1,j  )
                self.gn[iel,2] = node(i+1,j+1)
                self.gn[iel,3] = node(i  ,j+1)
                iel += 1

# %%
class FEM_solver_fos(Base_class_fem_heat_conduction):

    def __init__(self,d,quad_deg):
        super().__init__(d,quad_deg)
        
    def solve_fos(self,params,bc):
            
        T_init = np.zeros(self.d.n_verts) + 2
        NL_solution_p, Le, Ke, rhs_e,mask, T_dir = solve_system(self, bc, T_init)

        return NL_solution_p, Le, Ke, rhs_e,mask,T_dir
  

#%%
#### Simulate FOS

# %%
random.seed(25)
params = np.r_[1.:4.0:0.01]
quad_deg = 3
N_snap = 30 # Training Snapshots
NL_solutions = []
param_list = []
K_mus = []
q_mus = []
# %%
for _ in range(N_snap):
    param = random.choice(params) # Choose from parameter list
    param_list.append(param)
    d = probdata(mat_layout, src_layout, fdict, nref, L, param, pb_dim=2)
    T_init = np.zeros(d.n_verts) + 2.0
    FOS = FEM_solver_fos(d,quad_deg)
    NL_solution_p, Le, Ke, rhs_e, mask,T_dir = solve_fos(FOS,bc, T_init, tol=1e-5, max_iter=300)
    NL_solutions.append(NL_solution_p.flatten())
    K_mus.append(Ke)
    q_mus.append(rhs_e)

#%%
# ### Plot Snapshots

# %%
XX,YY = np.meshgrid(d.xi[0],d.xi[1])
sx = 4
sy = sx * (np.max(d.xi[1])-np.min(d.xi[1]))/ (np.max(d.xi[0])-np.min(d.xi[0]))
plt.figure()
for _,val in enumerate(NL_solutions):
    TT = np.reshape(val,(d.npts[1],d.npts[0]))
    # Plot the surface.
    fig, ax = plt.subplots(figsize=(2*sx, 2*sy),subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(XX.T, YY.T, TT.T, cmap=cm.coolwarm,linewidth=1)
    ax.scatter(XX.T, YY.T, TT.T)

    #, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(np.min(val),np.max(val))
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
# rotate as needed
ax.view_init(30,-60)
ax.set_box_aspect(aspect=(1, sy/sx, 1))
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# %%
NLS = np.asarray(NL_solutions)
np.shape(NLS)

# %% 
# ### ECSW Hyper-reduction

# %% 
# #### Step 1: Perform SVD on the snapshots (calculate $\mathbb{V}(=\mathbb{W}$)):

# %%
n_sel = 8
U, S, Vt = np.linalg.svd(np.transpose(NLS), full_matrices=False)
V_sel = U[:, :n_sel]

# %%
plt.figure(figsize = (6,4))
plt.semilogy(S,'s-')
plt.show()

# %%
plt.figure(figsize=(6,4))
for i in range(n_sel):
    TT = np.reshape(V_sel[:,i],(d.npts[1],d.npts[0]))
    fig, ax = plt.subplots(figsize=(2*sx, 2*sy),subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(XX.T, YY.T, TT.T, cmap=cm.coolwarm,linewidth=1)
plt.grid(True)
plt.show()

# %%
P_sel = V_sel[mask,:]@np.transpose(V_sel[mask,:])

# %% 
# #### ECSW

# %%
tic_h_setup_b = time.time()

tol = 1e-15
x, residual = ecsw_red(d, V_sel, Le, K_mus, q_mus, P_sel, tol, n_sel, N_snap, mask,NL_solutions)

toc_h_setup_b = time.time()

# %%
print(f"this is the residual from fnnls: {residual}")

# %%
XX_n,YY_n = np.meshgrid(np.arange(d.ncells[0]),np.arange(d.ncells[1]))

fig, ax = plt.subplots()
xTT = np.reshape(x,(d.ncells[0],d.ncells[1]))

colors = ['red' if value > 0 else 'blue' for value in (xTT.T).flatten()]
size_ = [15 if value > 0 else 1 for value in (xTT.T).flatten()]

ax.scatter((XX_n.T).flatten(), (YY_n.T).flatten(), c=colors, s=size_)

plt.show()

# %%
print(f"Fraction of total elements active in the ROM: {len(x[x>0])*100/len(x)}%")

# %% 
# ### ROM Simulation

# %%
params_rm = params[~np.isin(params,param_list)]
param_rom = random.choice(params_rm)

# %%
d_test = probdata(mat_layout, src_layout, fdict, nref, L, param_rom, pb_dim=2)

# %%
FOS_test = FEM_solver_fos(d_test,quad_deg)
ROM = FEM_solver_rom_ecsw(d_test, quad_deg)

# %%
T_init_fos = np.zeros(FOS_test.n_nodes) + 2.
T_init_rom = np.transpose(V_sel)@T_init_fos

# %%
tic_fos = time.time()
NL_solution_p_fos_test, _, _, _,_,_ = solve_fos(FOS, bc, T_init_fos)
toc_fos = time.time()

# %%
tic_rom_woh = time.time()
NL_solution_p_reduced_woh = ROM.solve_rom(param_rom,bc, T_init_rom,np.ones_like(x),V_sel)
toc_rom_woh = time.time()

# %%
tic_rom = time.time()
NL_solution_p_reduced = ROM.solve_rom(param_rom,bc, T_init_rom,x,V_sel)
toc_rom = time.time()

# %%
sol_red = V_sel@NL_solution_p_reduced.reshape(-1,1)#+pca.mean_.reshape(-1,1)

# %%
plt.figure()
fig, ax = plt.subplots(figsize=(2*sx, 2*sy),subplot_kw={"projection": "3d"})
TT_fos = np.reshape(NL_solution_p_fos_test,(d.npts[1],d.npts[0]))
TT_rom = np.reshape(sol_red,(d.npts[1],d.npts[0]))
ax.scatter(XX.T, YY.T, TT_fos.T)
surf = ax.plot_surface(XX.T, YY.T, TT_rom.T, cmap=cm.coolwarm,linewidth=1)
ax.set_zlim(np.min(sol_red),np.max(sol_red))
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
# rotate as needed
ax.view_init(30,-60)
ax.set_box_aspect(aspect=(1, sy/sx, 1))
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.title(f"RMS_error is {np.linalg.norm(sol_red-NL_solution_p_fos_test.reshape(-1,1))*100/np.linalg.norm(NL_solution_p_fos_test.reshape(-1,1))} %")
plt.show()
print(f"ROM Error without hyperreduction is {np.linalg.norm(V_sel@NL_solution_p_reduced_woh.reshape(-1,1)-NL_solution_p_fos_test.reshape(-1,1))*100/np.linalg.norm(NL_solution_p_fos_test.reshape(-1,1))} %")

# %% 
# #### Speedups

# %%
fos_sim_time = toc_fos - tic_fos
rom_sim_time_woh = toc_rom_woh - tic_rom_woh

# %%
rom_sim_time = toc_rom - tic_rom
# h_total_setup_time = (toc_h_setup_b+toc_h_setup_a) - (tic_h_setup_b+tic_h_setup_a) #this is one time

# %%
print(f"speedup without hyperreduction:{fos_sim_time/rom_sim_time_woh}")
print(f"speedup with hyperreduction:{fos_sim_time/(rom_sim_time)}")


