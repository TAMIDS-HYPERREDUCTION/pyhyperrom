# Import necessary libraries and re-define the function for FFT-based PSD calculation
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann

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
    
    return F_, 10 * np.log10(Pxx_)[0]

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



# # Generate the sine wave with a frequency of pi and a sampling frequency of 40 Hz
# t_long_40 = np.linspace(0, 1000, int(1000 * 20))  # 100 seconds, 40 samples/second
# y_long_40 = np.sin(2*np.pi *2.33* t_long_40)  # Sine wave with a frequency of pi

# # Test the new function with the previous sine wave with a frequency of pi, fs=40, and fd=5
# F_fft_no_welch, Pxx_fft_no_welch = psd_su4_fft_no_welch(t_long_40, y_long_40, 5)

# # Plotting
# plt.figure(figsize=(14, 6))

# # Frequency-domain plot with combined positive and negative frequencies
# plt.subplot(1, 1, 1)
# plt.plot(F_fft_no_welch, Pxx_fft_no_welch)
# plt.title('Power Spectral Density (PSD) with fd=4, fs=40Hz (Combined Positive and Negative Frequencies)')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power/Frequency (dB)')
# plt.xlim([min(F_fft_no_welch), max(F_fft_no_welch)])  # Set maximum x-limit to the maximum frequency value
# plt.tight_layout()
# plt.show()