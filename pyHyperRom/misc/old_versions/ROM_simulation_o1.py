import numpy as np
import random
import time
import matplotlib.pyplot as plt
from src.codes.utils.plot_utils import OneDPlot as plot
from codes.prob_classes.heat_conduction.base_class_heat_conduction import probdata, FOS_FEM


class ROM_simulation:
    
    def __init__(self, layout, quad_deg, param_list, V_sel, deim, xi):
        self.layout = layout
        self.quad_deg = quad_deg
        self.param_list = param_list
        self.V_sel = V_sel
        self.deim = deim
        self.xi = xi
        self.params = np.arange(1., 4.0, 0.01)
        
    def select_new_parameter(self):
        params_rm = self.params[~np.isin(self.params, self.param_list)]
        return random.choice(params_rm)

    def setup_simulation(self, param_rom):
        d_test = probdata(self.layout.bc, self.layout.mat_layout, self.layout.src_layout, self.layout.fdict, self.layout.n_ref, self.layout.L, param_rom, pb_dim=1)
        FOS_test = FOS_FEM(d_test, self.quad_deg)
        ROM_h = rom_class.rom_deim(d_test, self.deim, self.quad_deg)
        ROM = rom_class.rom(d_test, self.quad_deg)
        return d_test, FOS_test, ROM_h, ROM

    def run_simulation(self):
        param_rom = self.select_new_parameter()
        d_test, FOS_test, ROM_h, ROM = self.setup_simulation(param_rom)
        
        # Initial guess for temperature
        T_init_fos = np.zeros(FOS_test.n_nodes) + 273.15
        T_init_rom = np.transpose(self.V_sel) @ T_init_fos  # Initial guess in the reduced subspace
        
        # Run FOS simulation
        tic_fos = time.time()
        NL_solution_fos_test = solve_fos(FOS_test, T_init_fos).flatten()
        toc_fos = time.time()
        
        # Run ROM simulation without hyper-reduction
        tic_rom_woh = time.time()
        NL_solution_reduced_woh = ROM.solve_rom(T_init_rom, self.V_sel)
        sol_red_woh = np.dot(self.V_sel, NL_solution_reduced_woh)  # Full-scale
        toc_rom_woh = time.time()
        
        # Run ROM simulation with hyper-reduction
        tic_rom = time.time()
        NL_solution_reduced = ROM_h.solve_rom(T_init_rom, self.xi, self.V_sel)
        sol_red = np.dot(self.V_sel, NL_solution_reduced)  # Full-scale
        toc_rom = time.time()
        
        # Plot results
        self.plot_results(d_test.xi[0], sol_red, NL_solution_fos_test)
        
        # Print statistics
        self.print_statistics(NL_solution_fos_test, sol_red, sol_red_woh, toc_fos, tic_fos, toc_rom_woh, tic_rom_woh, toc_rom, tic_rom)

    def plot_results(self, x_coords, sol_rom, sol_fos):
        fig, ax = plt.subplots()
        plot(x_coords, sol_rom, ax=ax).line_()
        plot(x_coords, sol_fos, ax=ax, clr='k', sz=20).scatter_()
        plt.show()

    def print_statistics(self, sol_fos, sol_rom, sol_rom_woh, toc_fos, tic_fos, toc_rom_woh, tic_rom_woh, toc_rom, tic_rom):
        error_rom = np.linalg.norm(sol_rom - sol_fos) * 100 / np.linalg.norm(sol_fos)
        error_rom_woh = np.linalg.norm(sol_rom_woh - sol_fos) * 100 / np.linalg.norm(sol_fos)
        fos_sim_time = toc_fos - tic_fos
        rom_sim_time_woh = toc_rom_woh - tic_rom_woh
        rom_sim_time = toc_rom - tic_rom
        
        print(f"ROM error with hyperreduction: {error_rom} %")
        print(f"ROM error without hyperreduction: {error_rom_woh} %")
        print(f"speedup without hyperreduction: {fos_sim_time / rom_sim_time_woh}")
        print(f"speedup with hyperreduction: {fos_sim_time / rom_sim_time}")

# Usage example assuming layout, V_sel, deim, and xi are defined elsewhere
simulation_runner = SimulationRunner(layout, quad_deg, param_list, V_sel, deim, xi)
simulation_runner.run_simulation()
