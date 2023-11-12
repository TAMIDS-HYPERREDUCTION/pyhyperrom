import sys
import os
dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(dir)
from src.codes.basic import *
from src.codes.utils import *
from src.codes.base_classes import Base_class_fem_heat_conduction
import src.codes.reductor.rom_class as rom_class
from src.codes.algorithms.ecsw import ecsw_red
from importlib import reload


class probdata:

    def __init__(self, bc, mat_layout, src_layout, fdict, nref, L, mu, pb_dim):

        self.dim_ = pb_dim
        # refine the mesh and update material and source layouts

        if(pb_dim==1):
            
            self.cell2mat_layout = mat_layout.flatten()
            self.cell2src_layout = src_layout.flatten()

        else:

            repeats = np.asarray(nref, dtype=int)
            self.cell2mat_layout = self.repeat_array(mat_layout, repeats)
            self.cell2src_layout = self.repeat_array(src_layout, repeats)
	    
        ## change this mapping if needed.
        self.fdict = fdict

	    # mesh data cells
        self.ncells = [None] * pb_dim
        self.npts = [None] * pb_dim
        self.deltas = [None] * pb_dim
        self.xi = []

        for i in range(pb_dim):
            self.ncells[i] = self.cell2mat_layout.shape[i]
            self.npts[i] = self.ncells[i] + 1
            if pb_dim == 1:
                self.xi.append(np.linspace(0,L,self.npts[i]))
                self.deltas[i] = L/self.ncells[i]
            else:
                self.xi.append(np.linspace(0, L[i], self.npts[i]))
                self.deltas[i] = L[i] / self.ncells[i]

        if pb_dim == 1:
        	self.n_verts = self.npts[0]
        else:
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

                # Create a mask for nodes that do not have a Dirichlet boundary condition
        mask = self.node_eqnId != 0
        self.mask = mask

    def repeat_array(self, arr, repeats):
        for dim, n in enumerate(repeats):
            arr = np.repeat(arr, n, axis=dim)
        return arr

    def connectivity(self):
        """
        Define nodal connectivity for each cell in the mesh.
        """
        if self.dim_ == 1:
            self.connectivity_1d()
        elif self.dim_ == 2:
            self.connectivity_2d()
        elif self.dim_ == 3:
            self.connectivity_3d()
        else:
            raise ValueError("Unsupported dimension")

    def connectivity_1d(self):
        # Initialize the connectivity array for 1D
        self.n_cells = self.ncells[0]
        self.gn = np.zeros((self.n_cells, 2**self.dim_), dtype=int)
	    # Loop over all cells to define their nodal connectivity
        for iel in range(self.n_cells):
	    # For each cell, define the left and right nodes	
            self.gn[iel, 0] = iel
            self.gn[iel, 1] = iel + 1

    def connectivity_2d(self):
        # Define connectivity for 2D
        self.n_cells = np.prod(np.array(self.ncells))
        self.gn = np.zeros((self.n_cells, 2**self.dim_), dtype=int)

        node = lambda i, j: i + j * self.npts[0]

        iel = 0
        for j in range(self.ncells[1]):
            for i in range(self.ncells[0]):
                self.gn[iel, 0] = node(i, j)
                self.gn[iel, 1] = node(i + 1, j)
                self.gn[iel, 2] = node(i + 1, j + 1)
                self.gn[iel, 3] = node(i, j + 1)
                iel += 1

    def connectivity_3d(self):
        # Initialize the connectivity array for 3D
        self.n_cells = np.prod(np.array(self.ncells))
        self.gn = np.zeros((self.n_cells, 2**self.dim_), dtype=int)

        node = lambda i, j, k: i + j * self.npts[0] + k * self.npts[0] * self.npts[1]
        # Loop over all cells to define their nodal connectivity
        iel = 0
        for k in range(self.ncells[2]):
            for j in range(self.ncells[1]):
                for i in range(self.ncells[0]):
                     # counter-clockwise
                    self.gn[iel, 0] = node(i, j, k)
                    self.gn[iel, 1] = node(i + 1, j, k)
                    self.gn[iel, 2] = node(i + 1, j + 1, k)
                    self.gn[iel, 3] = node(i, j + 1, k)
                    self.gn[iel, 4] = node(i, j, k + 1)
                    self.gn[iel, 5] = node(i + 1, j, k + 1)
                    self.gn[iel, 6] = node(i + 1, j + 1, k + 1)
                    self.gn[iel, 7] = node(i, j + 1, k + 1)
                    iel += 1

def simulate_FOS(N_snap, params, bc, mat_layout, src_layout, fdict, n_ref, L, pb_dim, quad_deg,k_const):
    random.seed(25)
    NL_solutions = []
    param_list = []
    K_mus = []
    q_mus = []

    for i in range(N_snap):
        print(f"\n\n\n Snapshot {i} \n\n\n")
        param = random.choice(params)
        param_list.append(param)

        if i == 0:
            d = probdata(bc, mat_layout, src_layout, fdict, n_ref, L, param, pb_dim)
            FOS = Base_class_fem_heat_conduction(d, quad_deg)
        else:
            FOS.mu = param

        T_init = np.zeros(d.n_verts) + k_const
        NL_solution_p, Ke, rhs_e, mask = solve_fos(FOS, T_init)
        NL_solutions.append(NL_solution_p.flatten())
        K_mus.append(Ke)
        q_mus.append(rhs_e)

        if pb_dim == 1:
            plot1D(d.xi[0], NL_solution_p)
        elif pb_dim == 2:
            plot2D(d.xi[0], d.xi[1], NL_solution_p)
        elif pb_dim == 3:
            plot3D(d.xi[0], d.xi[1], d.xi[2], NL_solution_p, hmap=True)

    NLS = np.asarray(NL_solutions)
    return NLS, NL_solutions, param_list, K_mus, q_mus ,d, mask 

def ECSW_heatconduction(NLS, N_snap, NL_solutions, n_sel, pb_dim, d, mask, K_mus, q_mus, tol, plot=True):
    U, S, Vt = np.linalg.svd(np.transpose(NLS), full_matrices=False)
    V_sel = U[:, :n_sel]
    P_sel = V_sel[mask, :] @ np.transpose(V_sel[mask, :])
    
    if plot:
        plt.figure(figsize=(6, 4))
        plt.semilogy(S, 's-')
        plt.show()

        for i in range(n_sel):
            if pb_dim == 1:
                fig, ax = plt.subplots()
                plot1D(d.xi[0], V_sel[:, i], ax=ax)
                plt.show()
            elif pb_dim == 2:
                plot2D(d.xi[0], d.xi[1], V_sel[:, i])
            else:
                plot3D(d.xi[0], d.xi[1], d.xi[2], V_sel[:, i], hmap=True)

    tic_h_setup_b = time.time()
    xi, residual = ecsw_red(d, V_sel, d.Le, K_mus, q_mus, n_sel, N_snap, mask, NL_solutions, tol)
    toc_h_setup_b = time.time()
    print(f"this is the residual from fnnls: {residual}")

    colors = ['red' if value > 0 else 'blue' for value in xi]
    sizes = [15 if value > 0 else 1 for value in xi]

    if pb_dim == 1:
        plot1D(np.arange(d.ncells[0]), np.zeros_like(xi), scattr=True, clr=colors, sz=sizes)
    elif pb_dim == 2:
        plot2D(np.arange(d.ncells[0]), np.arange(d.ncells[1]), xi, scattr=True, clr=colors, sz=sizes)
    else:
        plot3D(np.arange(d.ncells[0]), np.arange(d.ncells[1]), np.arange(d.ncells[2]), xi, sz=sizes, clr=colors, save_file=False)

    print(f"Fraction of total elements active in the ROM: {len(xi[xi > 0]) * 100 / len(xi)}%")
    return V_sel, xi

def Rom_simulation(params, param_list, Rom_const, pb_dim, V_sel, xi, bc, mat_layout, src_layout, fdict, n_ref, L, quad_deg, FEM_solver_rom_ecsw, solve_fos):
    params_rm = params[~np.isin(params, param_list)]
    param_rom = random.choice(params_rm)

    # Define the data-class
    d_test = probdata(bc, mat_layout, src_layout, fdict, n_ref, L, param_rom, pb_dim)
    FOS_test = Base_class_fem_heat_conduction(d_test, quad_deg)
    ROM = FEM_solver_rom_ecsw(d_test, quad_deg)

    # Initial guess
    T_init_fos = np.zeros(FOS_test.n_nodes) + Rom_const
    T_init_rom = np.transpose(V_sel) @ T_init_fos  # Ensure the initial guess is contained in the reduced subspace

    # Time taken to perform a FO simulation with the current parameter value
    tic_fos = time.time()
    NL_solution_p_fos_test, _, _, _ = solve_fos(FOS_test, T_init_fos)
    toc_fos = time.time()

    # Time taken to simulate a ROM without hyper-reduction
    tic_rom_woh = time.time()
    NL_solution_p_reduced_woh = ROM.solve_rom(T_init_rom, np.ones_like(xi), V_sel)
    toc_rom_woh = time.time()

    # Time taken to simulate a ROM *with* hyper-reduction
    tic_rom = time.time()
    NL_solution_p_reduced = ROM.solve_rom(T_init_rom, xi, V_sel)
    toc_rom = time.time()
    sol_red = V_sel @ NL_solution_p_reduced.reshape(-1, 1)

    if pb_dim == 1:
        fig, ax = plt.subplots()
        plot1D(d_test.xi[0], sol_red, ax=ax)
        plot1D(d_test.xi[0], NL_solution_p_fos_test, ax=ax, scattr=True, clr='k', sz=10)
        plt.show()
    elif pb_dim == 2:
        plot2D(d_test.xi[0], d_test.xi[1], sol_red)
    else:
        plot3D(d_test.xi[0], d_test.xi[1], d_test.xi[2], sol_red, hmap=True)

    rms_error = np.linalg.norm(sol_red - NL_solution_p_fos_test.reshape(-1, 1)) * 100 / np.linalg.norm(NL_solution_p_fos_test.reshape(-1, 1))
    print(f"RMS_error is {rms_error} %")
    if pb_dim == 2:
        plot2D(d_test.xi[0], d_test.xi[1], NL_solution_p_fos_test)
    elif pb_dim == 3:
        plot3D(d_test.xi[0], d_test.xi[1], d_test.xi[2], NL_solution_p_fos_test, hmap=True)


    rom_error_woh = np.linalg.norm(V_sel @ NL_solution_p_reduced_woh.reshape(-1, 1) - NL_solution_p_fos_test.reshape(-1, 1)) * 100 / np.linalg.norm(NL_solution_p_fos_test.reshape(-1, 1))
    print(f"\nROM Error without hyperreduction is {rom_error_woh} %")
    fos_sim_time = toc_fos - tic_fos
    rom_sim_time_woh = toc_rom_woh - tic_rom_woh
    rom_sim_time = toc_rom - tic_rom
    speedup_woh = fos_sim_time / rom_sim_time_woh
    speedup = fos_sim_time / rom_sim_time

    print(f"Speedup without hyperreduction: {speedup_woh}")
    print(f"Speedup with hyperreduction: {speedup}")
    # h_total_setup_time = (toc_h_setup_b+toc_h_setup_a) - (tic_h_setup_b+tic_h_setup_a) #this is one time

def heat_conduction_workflow(N_snap, params, bc, mat_layout, src_layout, fdict, n_ref, L, pb_dim, quad_deg, k_const, Rom_const, n_sel, tol,sim_FOS,ECSW_heatcond,Rom_sim):
    # Simulate FOS 
    if sim_FOS == True:
        NLS, NL_solutions, param_list, K_mus, q_mus, d, mask = simulate_FOS(N_snap, params, bc, mat_layout, src_layout, fdict, n_ref, L, pb_dim, quad_deg, k_const)
    
    # Apply hyper-reduction
    if ECSW_heatcond == True:
        V_sel, xi = ECSW_heatconduction(NLS, N_snap, NL_solutions, n_sel, pb_dim, d, mask, K_mus, q_mus, tol, plot=True)
    
    # Perform ROM simulation
    if Rom_sim == True:
        Rom_simulation(params, param_list, Rom_const, pb_dim, V_sel, xi, bc, mat_layout, src_layout, fdict, n_ref, L, quad_deg, FEM_solver_rom_ecsw, solve_fos)
