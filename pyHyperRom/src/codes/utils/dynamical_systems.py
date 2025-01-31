# Import necessary libraries and re-define the function for FFT-based PSD calculation
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
# import control as ctl

"""
The H2 and H-infinity norms are pivotal in evaluating control systems, offering insights into both average performance and robustness under varying conditions. Calculating these norms provides valuable quantitative measures for system analysis and design.

**Finding the H2 Norm:**
The H2 norm is interpreted as a measure of the total energy output of a system in response to a unit impulse input. In practical terms, it's akin to the root-mean-square (RMS) value of the system's impulse response. Computationally, the H2 norm of a linear time-invariant (LTI) system can be determined by integrating the square of the system's impulse response over time. For state-space models, the H2 norm is more efficiently calculated using the solution to a specific type of matrix equation known as the Lyapunov equation. This equation relates the system's state matrices (A, B, C, and D) to its energy content. In the context of control systems, software packages like MATLAB or Python's `control` library can compute the H2 norm directly from the state-space representation, abstracting away the underlying mathematical complexity.

**Implications of the H2 Norm:**
In practical applications, the H2 norm is especially relevant in systems where the minimization of energy or power is crucial. For instance, in electrical engineering, it can guide the design of energy-efficient circuits or control strategies. In mechanical systems, such as vibration control or noise reduction, a lower H2 norm indicates a system that dissipates less energy, which is desirable. It’s also a key metric in signal processing for designing optimal filters that minimize energy amplification in the presence of noise.

**Finding the H-infinity Norm:**
The H-infinity norm of a system measures its worst-case gain across all frequencies, essentially quantifying the system's maximum response to any given input. Computationally, it's the maximum singular value of the system's transfer function matrix. For a state-space system, this involves computing the peak value of the system's frequency response. This computation can be quite complex, as it entails finding the maximum over all frequencies of the largest singular value of the frequency response matrix. Tools like MATLAB and Python's `control` library provide functions to calculate the H-infinity norm directly from the system's state-space matrices.

**Implications of the H-infinity Norm:**
The H-infinity norm is particularly crucial in robust control design, where it's used to ensure stability and performance under worst-case disturbances or uncertainties. For example, in aerospace engineering, it helps in designing control systems that can withstand a wide range of operating conditions and external disturbances. In automotive engineering, control systems with low H-infinity norms ensure that vehicles respond predictably under various driving conditions, enhancing safety and reliability.

By applying both the H2 and H-infinity norms, engineers and system designers can achieve a balanced design that optimizes both average and worst-case performance, leading to systems that are not only efficient under normal conditions but also resilient and stable under challenging or unpredictable scenarios.
"""

def compare_models(FOS_sol, ROM_sol, t, fc=None):
# def compare_models(fom, rom, FOS_sol, ROM_sol, t, fc=None):
    # Time Response Analysis
    err_disp, err_vel = time_freq_comparison(FOS_sol, ROM_sol, t, fc)

    # Eigenvalue Analysis
    # eigenvalue_analysis(fom, rom)
    return err_disp, err_vel


def time_freq_comparison(FOS_sol, ROM_sol, t, fc=None):

    # displacements
    FOS_disp = FOS_sol[:FOS_sol.shape[0] // 2, :]
    FOS_disp = FOS_disp[::2,:] 
    ROM_disp = ROM_sol[0][::2,:]
    err_disp = np.linalg.norm(ROM_disp - FOS_disp) * 100 / np.linalg.norm(FOS_disp)
    print(f"Relative error in displacement: {err_disp}")

    # velocities
    FOS_vel = FOS_sol[FOS_sol.shape[0] // 2:, :]
    FOS_vel = FOS_vel[::2,:]
    ROM_vel = ROM_sol[1][::2,:]
    err_vel = np.linalg.norm(ROM_vel - FOS_vel) * 100 / np.linalg.norm(FOS_vel)
    print(f"Relative error in velocity: {err_vel}")

    # Beam-midpoint
    y_fom = FOS_disp[FOS_disp.shape[0] // 2, :]
    _, y_fom_psd = pypsd(t, y_fom,fc)
    
    y_rom = ROM_disp[ROM_disp.shape[0] // 2, :]
    fd, y_rom_psd = pypsd(t, y_rom,fc)

    yv_fom = FOS_vel[FOS_vel.shape[0] // 2, :]
    _, yv_fom_psd = pypsd(t, yv_fom,fc)

    yv_rom = ROM_vel[ROM_vel.shape[0] // 2, :]
    fv, yv_rom_psd = pypsd(t, yv_rom,fc)


    # plots_t

    fig = plt.figure(figsize=(16, 4))
    ax = fig.subplot_mosaic("""AB""")

    ax['A'].plot(t[-3000:], y_fom[-3000:], label='FOM_disp')
    ax['A'].plot(t[-3000:], y_rom[-3000:],'--', label='ROM_disp')
    ax['A'].set_xlabel('$t$')
    ax['A'].set_ylabel('$w(0.5,t)$')
    ax['A'].legend()
    plt.autoscale(tight=True)


    ax['B'].plot(t[-3000:], yv_fom[-3000:], label='FOM_vel')
    ax['B'].plot(t[-3000:], yv_rom[-3000:],'--', label='ROM_vel')
    ax['B'].set_xlabel('$t$')
    ax['B'].set_ylabel('$\dot{w}(0.5,t)$')
    ax['B'].legend()
    plt.autoscale(tight=True)
    plt.show()
        
    # plots_f
    fig = plt.figure(figsize=(16, 4))
    ax = fig.subplot_mosaic("""AB""")

    ax['A'].plot(fd, y_fom_psd, label='FOM_disp')
    ax['A'].plot(fd, y_rom_psd, 'k--', label='ROM_disp')
    ax['A'].set_xlabel('$F$')
    ax['A'].set_ylabel('$P_d(dB)$')
    ax['A'].legend()
    plt.autoscale(tight=True)


    ax['B'].plot(fv, yv_fom_psd, label='FOM_vel')
    ax['B'].plot(fv, yv_rom_psd, 'k--', label='ROM_vel')  # Assuming yv_rom_psd is intended here
    ax['B'].set_xlabel('$F$')
    ax['B'].set_ylabel('$P_v(dB)$')
    ax['B'].legend()
    plt.autoscale(tight=True)
    plt.show()

    return err_disp, err_vel
    

# def calculate_error_norms(fom, rom):

#         # Define the state-space matrices for the FOM
#     A_fom = fom.A 
#     B_fom = fom.B 
#     C_fom = fom.C  
#     D_fom = fom.D 

#     # Define the state-space matrices for the ROM
#     A_rom = rom.A
#     B_rom = rom.B
#     C_rom = rom.C 
#     D_rom = rom.D 


#     # Construct the augmented system for the error dynamics
#     A_err = np.block([
#         [A_fom, np.zeros((A_fom.shape[0], A_rom.shape[1]))],
#         [np.zeros((A_rom.shape[0], A_fom.shape[1])), A_rom]
#     ])
#     B_err = np.vstack([B_fom, B_rom])
#     C_err = np.hstack([C_fom, -C_rom])
#     D_err = np.zeros((C_err.shape[0],B_err.shape[1]))

#     # Create the error system as a state-space object
#     error_system = ctl.ss(A_err, B_err, C_err, D_err)

#     # Frequency range for the Bode plot
#     omega = np.logspace(-2, 2, 1000)

#     # Calculate the magnitude and phase of the error system
#     mag, phase, omega = ctl.bode(error_system, omega, Plot=False)

#     # Plot the magnitude
#     plt.figure()
#     plt.semilogx(omega, 20 * np.log10(mag))
#     plt.xlabel('Frequency [rad/s]')
#     plt.ylabel('Magnitude [dB]')
#     plt.title('Magnitude plot of the error system')
#     plt.grid(which='both', linestyle='--', linewidth=0.5)
#     plt.show()

#     # error_system = ctl.ss(fom.A - rom.A, fom.B - rom.B, fom.C - rom.C, fom.D - rom.D)
#     # h2_norm = ctl.h2norm(error_system)
#     # hinfinity_norm = ctl.hinfnorm(error_system)
#     # print(f"H2 Norm of the error system: {h2_norm}")
#     # print(f"H-Infinity Norm of the error system: {hinfinity_norm}")


# def eigenvalue_analysis(fom, rom):
#     eigenvalues_fom = np.linalg.eigvals(fom.A)
#     eigenvalues_rom = np.linalg.eigvals(rom.A)
    
#     # Plotting
#     plt.figure(figsize=(8, 6))
#     plt.scatter(eigenvalues_rom.real, eigenvalues_rom.imag, marker='x', label='ROM')
#     plt.scatter(eigenvalues_fom.real, eigenvalues_fom.imag, marker='o', label='FOM')
#     plt.title('Eigenvalue Comparison of FOM and ROM')
#     plt.xlabel('Real Part')
#     plt.ylabel('Imaginary Part')
#     plt.axhline(y=0, color='grey', linestyle='--')  # X-axis
#     plt.axvline(x=0, color='grey', linestyle='--')  # Y-axis
#     plt.grid(True)
#     plt.legend()
#     plt.show()


def pypsd(t, ts, fd=None):
    """
    Estimate Power Spectral Density (PSD) using FFT, without using Welch's method.
    
    Parameters:
    t : array_like
        Time vector.
    ts : array_like
        Time series.
    fd : float, optional
        Frequency divider.
        
    Returns:
    F : array_like
        Frequency vector.
    Pxx : array_like
        Power spectral density.
    """
    if ts.shape[0] != 1:
        ts = np.transpose(ts)
    
    ts = ts - np.mean(ts)
    ts = ts * hann(len(ts))
    
    dt = t[1] - t[0]
    fs = 1 / dt
    n = len(ts)
   
    if fd is not None:
        m = int(np.floor(0.5 * fs / fd))        
        lenC = int(np.floor(len(ts) / m))
        Px = []
        
        for loop in range(m):
            
            tmpvar = ts[loop::m]
            tmpvar = tmpvar[:lenC]
            Px_loop = np.abs(np.fft.fft(tmpvar)) ** 2 / len(tmpvar)
            F = np.fft.fftfreq(len(tmpvar), dt * m)
            F_,Px_ = combine_pos_neg_freq(F, Px_loop)
            Px.append(Px_)
        
        if m > 1:
            Pxx_ = np.mean(Px, axis=0)
        else:
            Pxx_ = Px

    else:

        F = np.fft.fftfreq(n,dt)
        Pxx = np.abs(np.fft.fft(ts)) ** 2 / n
        F_,Pxx_ = combine_pos_neg_freq(F, Pxx)
    
    return F_, 10 * np.log10(Pxx_)

# Function to combine positive and negative frequency components of the FFT
def combine_pos_neg_freq(F, Pxx):
    
    F_positive = F[F >= 0]
    F_negative = F[F < 0]
    Pxx_negative = Pxx[len(F_positive):][::-1]
    Pxx_positive = Pxx[:len(F_positive)]
    
    if len(F)%2!=0:
        Pxx_negative = np.append(Pxx_positive[0],Pxx_negative)
    
    Pxx_combined = Pxx_positive + Pxx_negative
    
    return F_positive, Pxx_combined



# def fom_data_processor_stacked(t, T, solution_snapshot_orig):

#     # Extract and compute time parameters
#     fs = 1 / (t[1] - t[0])
#     nt = len(t)
#     ndata = solution_snapshot_orig.shape[0]
#     nset = round(ndata / nt)


#     # Remove transient and center on IC
#     start_indices = np.arange(0, nset * nt, nt)
#     end_indices = start_indices + nt
#     solution_snapshot = np.concatenate([solution_snapshot_orig[start:end, :] for start, end in zip(start_indices, end_indices)], axis=0)
#     ic_values = solution_snapshot[start_indices, :]
#     solution_snapshot_f1 = solution_snapshot - np.repeat(ic_values, nt, axis=0)
    
#     # Remove time-periods
#     len_1p = int(np.ceil(T * fs)+1)
#     # m = 2

#     # solution_snapshot_f2 = np.concatenate([solution_snapshot_f1[start + m*len1p : start + (m+1)*len_1p, :] for start in start_indices], axis=0)
#     solution_snapshot_f2 = np.concatenate([solution_snapshot_f1[end-len_1p : end, :] for end in end_indices], axis=0)

#     # Recalculating Parameters
#     # t = t[int(fs):int(fs) + len_1p] - t[int(fs)]
#     t = t[-len_1p:]
#     t-= t[0]
#     nt = len(t)
#     tstop = t[-1]

        
#     return solution_snapshot_f2, nt, fs, tstop, ndata, nset, t


def fom_data_processor_stacked_q(t, T, solution_snapshot_orig,q_mus):

    # Extract and compute time parameters
    t=np.copy(t)
    fs = 1 / (t[1] - t[0])
    nt = len(t)
    ndata = solution_snapshot_orig.shape[0]
    nset = round(ndata / nt)


    # Remove transient and center on IC
    start_indices = np.arange(0, nset * nt, nt)
    end_indices = start_indices + nt
    solution_snapshot = np.concatenate([solution_snapshot_orig[start:end, :] for start, end in zip(start_indices, end_indices)], axis=0)
    ic_values = solution_snapshot[start_indices, :]
    solution_snapshot_f1 = solution_snapshot - np.repeat(ic_values, nt, axis=0)
    
    # Remove time-periods
    len_1p = int(np.ceil(T * fs)+1)
    # m = 2

    # solution_snapshot_f2 = np.concatenate([solution_snapshot_f1[start + m*len1p : start + (m+1)*len_1p, :] for start in start_indices], axis=0)
    solution_snapshot_f2 = np.concatenate([solution_snapshot_f1[end-len_1p : end, :] for end in end_indices], axis=0)

    ### process q_mus 
    lp_1 = len(q_mus)
    lp_2 = len(q_mus[0])
    # lp_2 = len(q_mus[0][0].shape[])
    q_mus_truncated = [[] for _ in range(lp_1)]

    for i in range(lp_1):
        for j in range(lp_2):
            q_mus_truncated[i].append(q_mus[i][j][:,-len_1p:])



    ###

    # Recalculating Parameters
    # t = t[int(fs):int(fs) + len_1p] - t[int(fs)]
    t = t[-len_1p:]
    t-= t[0]
    nt = len(t)
    tstop = t[-1]

        
    return solution_snapshot_f2, nt, fs, tstop, ndata, nset, t, q_mus_truncated

