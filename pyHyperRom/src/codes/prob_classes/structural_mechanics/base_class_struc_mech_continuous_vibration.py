from platform import node
from ...utils.fem_utils_StrucMech import *
from ...basic import *
import time


class probdata:

    def __init__(self, bc, mat_layout, src_layout, fdict, nref, L, dof_p_node, node_p_elem, cv, cm, dt, t):
        """
        Initializes the problem data.

        Args:
            bc: Boundary condition data.
            mat_layout: Material layout matrix.
            src_layout: Source layout matrix.1
            fdict: Dictionary of functions.
            nref: Not used, can be removed or repurposed.
            L (float): Length of the domain.
            dof_p_node (int): Degrees of freedom per node.
            node_p_elem (int): Nodes per element.
            cv, cm: Material properties (specific heat and mass).
            dt (float): Time step.
            t (float): Current time.
        """
        self.dim_ = 1  # Assuming a 1-dimensional problem
        self.dof_p_node = dof_p_node
        self.node_p_elem = node_p_elem
        self.elem_dof = dof_p_node * node_p_elem
        self.cv = cv
        self.cm = cm
        self.dt = dt
        self.t = t

        if self.dim_ > 2:
            raise ValueError("Dimension should be < 2")

        self.cell2mat_layout = mat_layout
        self.cell2src_layout = src_layout
        self.fdict = fdict

        # Setting up mesh parameters
        self.ncells = [mat_layout.shape[0]]
        self.n_cells = mat_layout.shape[0]
        self.npts = [self.ncells[0] + 1]
        self.deltas = [L / self.ncells[0]]
        self.xi = [np.linspace(0, L, self.npts[0])]

        self.n_verts = self.npts[0] * dof_p_node

        # Create nodal connectivity for the continuous Finite Element Method (cFEM)
        self.connectivity()

        # Store the dirichlet nodes if any
        handle_boundary_conditions(self, bc)

        # Determining the global equation numbers based on dirichlet nodes and storing in class
        get_glob_node_equation_id(self, self.dir_nodes)

        self.prepare_element_data()

        # Create a mask for nodes that do not have a Dirichlet boundary condition

        mask = self.node_eqnId != 0
        self.mask = mask

    def connectivity(self):
        """Determines the connectivity of nodes based on the problem dimension."""
        if self.dim_ == 1:
            self.connectivity_1d()
        else:
            raise ValueError("Unsupported dimension")

    def connectivity_1d(self):
        """
        Defines the nodal connectivity for each cell in a 1D mesh.
        """
        self.gn = np.zeros((self.ncells[0], self.elem_dof), dtype=int)
        self.gnodes = np.zeros((self.ncells[0], self.node_p_elem), dtype=int)

        for iel in range(self.ncells[0]):
            dof_start = self.dof_p_node * iel
            dofs = np.arange(dof_start, dof_start + self.elem_dof)
            
            self.gn[iel] = dofs 
            self.gnodes[iel] = dofs[::self.dof_p_node] // self.dof_p_node

    def prepare_element_data(self):
        """Prepares data for each element in the mesh."""

        # Get global node numbers and equation IDs for the current element
        self.glob_node_eqnId = []
        self.glob_node_nonzero_eqnId = []
        self.local_node_nonzero_eqnId = []
        self.Le = []
        self.global_indices = []
        self.local_indices = []

        for i in range(self.n_cells):
            get_element_global_nodes_and_nonzero_eqnId(self, i, self.node_eqnId)

class FOS_FEM:

    def __init__(self, data, quad_degree, tau, ep, T):
        """
        Initialize the class with given data and quadrature degree.
        """
        self.data = data
        self.tau = tau
        self.ep = ep
        self.T = T

        self.dim_ = data.dim_
        self.ncells = data.ncells.copy()
        self.npts = data.npts.copy()
        self.deltas = data.deltas.copy()
        self.n_nodes = data.n_verts

        self.sol_dir = data.sol_dir
        self.dir_nodes = data.dir_nodes
        self.node_eqnId = data.node_eqnId

        self.basis()
        self.basis_q(quad_degree)

    def basis(self):
        """
        Compute the continuous Finite Element (cFEM) basis functions.
        """
        x = sp.symbols('x')
        shape_functions = [1/4*((1-x)**2)*(2+x), 1/4*((1-x)**2)*(1+x), 
                           1/4*((1+x)**2)*(2-x), 1/4*((1+x)**2)*(x-1)]
        
        self.b = [sp.lambdify(x, bf, 'numpy') for bf in shape_functions]
        self.dbdxi = [sp.lambdify(x, sp.diff(bf, x), 'numpy') for bf in shape_functions]
        self.d2bdx2i = [sp.lambdify(x, sp.diff(bf, x, x), 'numpy') for bf in shape_functions]

    def basis_q(self, quad_degree):
        """
        Compute the basis functions and their derivatives at the quadrature points.
        """
        xq, wq = np.polynomial.legendre.leggauss(quad_degree)
        self.xq = xq
        nq = len(xq)
        
        self.bq = np.array([[fi(xi) for fi in self.b] for xi in xq])
        self.dbdxiq = np.array([[dfi(xi) for dfi in self.dbdxi] for xi in xq])
        self.d2bdx2iq = np.array([[d2fi(xi) for d2fi in self.d2bdx2i] for xi in xq])
        self.w = wq

    def element_KM_matrices_fn_x(self, iel, notconstant = False):
        """
        Compute the element matrices and vectors.
        """
        dx = self.data.deltas[0]
        mod_jac = dx/2
        elem_glob_nodes = self.data.gn[iel, :]

        imat = self.data.cell2mat_layout[iel].astype(int)
        E, I, rho, A = self.data.fdict["E"][imat], self.data.fdict["I"][imat], self.data.fdict["rho"][imat], self.data.fdict["A"][imat]

        n = len(elem_glob_nodes)
        Ke_, Me_ = np.zeros((n, n)), np.zeros((n, n))

        if notconstant:
            w_factor = np.array([1, dx/2, 1, dx/2])
            bq = w_factor * self.bq
            self.bq_ = bq
            d2bdx2iq = w_factor * self.d2bdx2iq

            E_xq, I_xq, rho_xq, A_xq = E(self.xq), I(self.xq), rho(self.xq), A(self.xq)
            Jac_inv = (dx/2)**(-4)

            for i in range(n):
                for j in range(n):
                    K_temp = Jac_inv * mod_jac * np.dot(self.w * d2bdx2iq[:,i], E_xq * I_xq * d2bdx2iq[:, j])
                    M_temp = mod_jac * np.dot(self.w * bq[:, i], rho_xq * A_xq * bq[:, j])
                    Ke_[i, j] += K_temp
                    Me_[i, j] += M_temp
        else:
            # Precomputed stiffness and mass matrices for constant properties
            Ke_ = ((E * I / dx**3) * np.array([[12, 6*dx, -12, 6*dx],
                                              [6*dx, 4*dx**2, -6*dx, 2*dx**2],
                                              [-12, -6*dx, 12, -6*dx],
                                              [6*dx, 2*dx**2, -6*dx, 4*dx**2]]))

            Me_ = ((dx * rho * A / 420) * np.array([[156, 22*dx, 54, -13*dx],
                                                    [22*dx, 4*dx**2, 13*dx, -3*dx**2],
                                                    [54, 13*dx, 156, -22*dx],
                                                    [-13*dx, -3*dx**2, -22*dx, 4*dx**2]]))

        Ce_ = self.data.cm * Ke_ + self.data.cv * Me_
        return Ke_, Me_, Ce_

    def element_F_matrices_fn_x(self, iel, t=None):
        """
        Compute the element source vector for a given temperature field.
        """
        isrc = self.data.cell2src_layout[iel].astype(int)
        fext = self.data.fdict["fext"][isrc]
        mod_jac = self.deltas[0]/2

        elem_glob_dofs = self.data.gn[iel, :]
        elem_glob_nodes = self.data.gnodes[iel, :]
        
        n = len(elem_glob_dofs)
        qe_ = np.zeros((n, len(t)))

        x_iel = self.data.xi[0][elem_glob_nodes]
       
        fext_q = fext(self.xq, x_iel, t, self.tau, self.ep, self.T)

        for i in range(n):
            if i % 2 == 0:
                qe_[i, :] += mod_jac * np.dot(self.w * self.bq_[:, i], fext_q)
                
        return qe_

    def residual_func(self,i,j,p_sol_d,p_sol_v,data):

        # Extract relevant stiffness matrices and source terms for the current snapshot and cell
                # Project the solution onto the selected basis
        
        K_mus = data['K_mus']
        q_mus = data['q_mus']
        C_mus = data['C_mus']

        K_mus_ij = K_mus[0][j]
        C_mus_ij = C_mus[0][j]
        q_mus_ij = np.array(q_mus[0][j][:,i])
        
        res = np.dot(K_mus_ij, p_sol_d) + np.dot(C_mus_ij, p_sol_v) - q_mus_ij

        return res
    
    def residual_func_p(self,i,j,k,p_sol_d,p_sol_v,data,train_mask_t):

        # Extract relevant stiffness matrices and source terms for the current snapshot and cell
                # Project the solution onto the selected basis
        # print(f"{k=},{j=},{i=}")

        K_mus = data['K_mus']
        q_mus = data['q_mus']
        C_mus = data['C_mus']

        K_mus_ij = K_mus[k][j]
        C_mus_ij = C_mus[k][j]
        q_mus_ij_masked = np.array(q_mus[k][j][:,train_mask_t])
        q_mus_ij = q_mus_ij_masked[:,i]
        res = np.dot(K_mus_ij, p_sol_d) + np.dot(C_mus_ij, p_sol_v) - q_mus_ij

        return res
    
class StructuralDynamicsSimulationData:

    def __init__(self, n_ref, L, T, params, dt, t, ep=0.02, quad_deg=3, num_snapshots=1, pb_dim=1, cv=1e-2, cm=1e-4):
        """
        Initialize the simulation data class.
        """
        # Initialize layout instance
        from examples.structural_mechanics.Transverse.continuous_vibrations.oneD_beam.SystemProperties import SystemProperties

        self.properties = SystemProperties(n_ref, cv, cm)
        self.cv, self.cm = cv, cm
        self.n_ref, self.L, self.T = n_ref, L, T
        self.quad_deg, self.num_snapshots = quad_deg, num_snapshots
        self.mat_layout, self.src_layout = self.properties.create_layouts()
        self.fdict = self.properties.define_properties()
        self.bc = self.properties.define_boundary_conditions()
        self.params, self.ep, self.dt, self.t = params, ep, dt, t

        self.NL_solutions, self.param_list = [], []
        self.fos_time, self.K_mus, self.C_mus, self.q_mus = [], [], [], []
        self.dof_p_node, self.node_p_elem = 2, 2

    def run_simulation(self):
        """
        Run the simulation for the given number of snapshots.
        """
        random.seed(25)

        for i in range(self.num_snapshots):
            print(f"Snap {i}")
            tau = self.params[i]
            self.param_list.append(tau)

            if i == 0:
                d = probdata(self.bc, self.mat_layout, self.src_layout, self.fdict, self.n_ref, self.L, self.dof_p_node, self.node_p_elem, self.cv, self.cm, self.dt, self.t)                
                self.FOS = FOS_FEM(d, self.quad_deg, tau, self.ep, self.T)
            else:
                self.FOS.tau = tau

            sol_init = np.zeros(d.n_verts)
            tic_fos = time.time()
            t_out, x_out, rhs_e, Ke_d, Ce_d, mask, U, fom = solve_fos_dynamics(self.FOS, sol_init, self.cv, self.cm)
            toc_fos = time.time()

            self.fos_time.append(toc_fos - tic_fos)
            self.NL_solutions.append(x_out)
            self.q_mus.append(rhs_e)
            self.K_mus.append(Ke_d)
            self.C_mus.append(Ce_d)
            self.fom = fom

        self.t = t_out

class ROM_simulation_non_p:
    
    def __init__(self, f_cls, V_sel, xi=None, deim=None, sol_init_guess=0, N_rom_snap=1):

        """
        Initialize the Reduced Order Model (ROM) simulation class.
        """
        self.f_cls = f_cls
        self.V_sel = V_sel
        self.deim = deim
        self.xi = xi
        self.quad_deg = f_cls.quad_deg
        self.NL_solutions_rom = []
        self.d = f_cls.FOS.data
        self.t = f_cls.FOS.data.t
        self.N_rom_snap = N_rom_snap
        self.sol_init_guess = sol_init_guess
        self.speed_up = []

    def run_simulation(self):
        """
        Run the ROM simulation.
        """
        import src.codes.reductor.rom_class_StrucMech as rom_class

        N_dir = self.V_sel.shape[0]
        N_full = 2 * self.d.n_verts

        sol_init_fos = np.zeros(N_dir)
        sol_init_rom = np.append(np.dot(self.V_sel.T, sol_init_fos), np.dot(self.V_sel.T, sol_init_fos))

        for i in range(self.N_rom_snap):
            tic_rom = time.time()
            ROM = rom_class.rom(self.f_cls, self.quad_deg, self.f_cls.params[0], self.f_cls.ep, self.f_cls.T, self.f_cls.cv, self.f_cls.cm, self.xi)
            NL_solution_p_reduced = ROM.solve_rom(sol_init_rom, self.V_sel)
            toc_rom = time.time()

            rom_sim_time = toc_rom - tic_rom

            shape_0 = NL_solution_p_reduced.shape[0]
            sol_rom_d = np.zeros((int(N_full / 2), len(self.t)))
            sol_rom_v =  np.zeros_like(sol_rom_d)
            sol_rom_d[self.d.mask] = np.dot(self.V_sel, NL_solution_p_reduced[:int(shape_0 / 2), :])
            sol_rom_v[self.d.mask] = np.dot(self.V_sel, NL_solution_p_reduced[int(shape_0 / 2):, :])
            
            self.NL_solutions_rom.append([sol_rom_d, sol_rom_v])
            self.speed_up.append(self.f_cls.fos_time[i]/rom_sim_time)

class ROM_simulation_p:
    
    def __init__(self, f_cls, V_sel, xi=None, deim=None, sol_init_guess=0, N_rom_snap=1):

        """
        Initialize the Reduced Order Model (ROM) simulation class.
        """
        self.f_cls = f_cls
        self.V_sel = V_sel
        self.deim = deim
        self.xi = xi
        self.quad_deg = f_cls.quad_deg
        self.NL_solutions_rom = []
        self.d = f_cls.FOS.data
        self.t = f_cls.FOS.data.t
        self.N_rom_snap = N_rom_snap
        self.sol_init_guess = sol_init_guess
        self.speed_up = []

    def run_simulation(self):
        """
        Run the ROM simulation.
        """
        import src.codes.reductor.rom_class_StrucMech as rom_class

        N_dir = self.V_sel.shape[0]
        N_full = 2 * self.d.n_verts

        sol_init_fos = np.zeros(N_dir)
        sol_init_rom = np.append(np.dot(self.V_sel.T, sol_init_fos), np.dot(self.V_sel.T, sol_init_fos))

        for i in range(self.N_rom_snap):
            tic_rom = time.time()
            ROM = rom_class.rom(self.f_cls, self.quad_deg, self.f_cls.params[i], self.f_cls.ep, self.f_cls.T, self.f_cls.cv, self.f_cls.cm, self.xi)
            NL_solution_p_reduced = ROM.solve_rom(sol_init_rom, self.V_sel)
            toc_rom = time.time()

            rom_sim_time = toc_rom - tic_rom

            shape_0 = NL_solution_p_reduced.shape[0]
            sol_rom_d = np.zeros((int(N_full / 2), len(self.t)))
            sol_rom_v =  np.zeros_like(sol_rom_d)
            sol_rom_d[self.d.mask] = np.dot(self.V_sel, NL_solution_p_reduced[:int(shape_0 / 2), :])
            sol_rom_v[self.d.mask] = np.dot(self.V_sel, NL_solution_p_reduced[int(shape_0 / 2):, :])
            
            self.NL_solutions_rom.append([sol_rom_d, sol_rom_v])
            self.speed_up.append(self.f_cls.fos_time[i]/rom_sim_time)

class ROM_simulation_p_UQ:
    
    def __init__(self, f_cls, V_sel, param_list, xi=None, deim=None, sol_init_guess=0, N_rom_snap=1, ax = None):

        """
        Initialize the Reduced Order Model (ROM) simulation class.
        """
        self.f_cls = f_cls
        self.V_sel = V_sel
        self.deim = deim
        self.xi = xi
        self.quad_deg = f_cls.quad_deg
        self.NL_solutions_rom_UQ_1period_0p62 = []
        self.NL_solutions_rom_UQ_1p = [] # We are only taking the last 1 Period
        self.d = f_cls.FOS.data
        self.t = f_cls.FOS.data.t
        self.N_rom_snap = N_rom_snap
        self.sol_init_guess = sol_init_guess
        self.speed_up = []
        self.param_list=param_list
        self.ax = ax

    def run_simulation(self):
        """
        Run the ROM simulation.
        """
        import src.codes.reductor.rom_class_StrucMech as rom_class

        N_dir = self.V_sel.shape[0]
        N_full = 2 * self.d.n_verts

        sol_init_fos = np.zeros(N_dir)
        sol_init_rom = np.append(np.dot(self.V_sel.T, sol_init_fos), np.dot(self.V_sel.T, sol_init_fos))

        for i in range(self.N_rom_snap):
            # tic_rom = time.time()
            ROM = rom_class.rom(self.f_cls, self.quad_deg, self.param_list[i], self.f_cls.ep, self.f_cls.T, self.f_cls.cv, self.f_cls.cm, self.xi)
            NL_solution_p_reduced = ROM.solve_rom(sol_init_rom, self.V_sel)
            # toc_rom = time.time()

            # rom_sim_time = toc_rom - tic_rom

            shape_0 = NL_solution_p_reduced.shape[0]
            sol_rom_d = np.zeros((int(N_full / 2), len(self.t)))
            sol_rom_v =  np.zeros_like(sol_rom_d)
            sol_rom_d[self.d.mask] = np.dot(self.V_sel, NL_solution_p_reduced[:int(shape_0 / 2), :])
            sol_rom_v[self.d.mask] = np.dot(self.V_sel, NL_solution_p_reduced[int(shape_0 / 2):, :])

            if self.ax is not None:
                self.ax[0].plot(self.t[-int(self.f_cls.T/(self.t[1]-self.t[0])+1):], sol_rom_d[124,-int(self.f_cls.T/(self.t[1]-self.t[0])+1):], color='grey',linewidth=0.05)
                self.ax[0].set_xlabel('$t$')
                self.ax[0].set_ylabel('$w(0.25,t)$')

                self.ax[1].plot(self.t[-int(self.f_cls.T/(self.t[1]-self.t[0])+1):], sol_rom_v[124,-int(self.f_cls.T/(self.t[1]-self.t[0])+1):], color='grey',linewidth=0.05)
                self.ax[1].set_xlabel('$t$')
                self.ax[1].set_ylabel('$\dot{w}(0.25,t)$a')

                self.NL_solutions_rom_UQ_1period_0p62.append([sol_rom_d[124,-int(self.f_cls.T/(self.t[1]-self.t[0])+1):], sol_rom_v[124,-int(self.f_cls.T/(self.t[1]-self.t[0])+1):]])
                self.NL_solutions_rom_UQ_1p.append([sol_rom_d[::2,-int(self.f_cls.T/(self.t[1]-self.t[0])+1):],sol_rom_v[::2,-int(self.f_cls.T/(self.t[1]-self.t[0])+1):]])
            else:
                self.NL_solutions_rom_UQ_1period_0p62.append([sol_rom_d[124,-int(self.f_cls.T/(self.t[1]-self.t[0])+1):], sol_rom_v[124,-int(self.f_cls.T/(self.t[1]-self.t[0])+1):]])
                self.NL_solutions_rom_UQ_1p.append([sol_rom_d[::2,-int(self.f_cls.T/(self.t[1]-self.t[0])+1):],sol_rom_v[::2,-int(self.f_cls.T/(self.t[1]-self.t[0])+1):]])

            # self.speed_up.append(self.f_cls.fos_time[i]/rom_sim_time)
