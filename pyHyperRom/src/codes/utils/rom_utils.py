from ..basic import *
#from .plot_utils_style import *
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, LogLocator
import scipy.stats.qmc as qmc
from scipy.stats.qmc import Sobol
from scipy.stats.qmc import LatinHypercube
import numpy as np
from pyDOE import lhs

def svd_mode_selector(data, tolerance=1e-3, modes=False):
    """
    Selects the number of singular value decomposition (SVD) modes based on a tolerance.
    
    Parameters:
    - data: The input data for SVD.
    - tolerance: The threshold for cumulative energy content in the SVD spectrum.
    - modes: If True, prints the number of selected modes.
    
    Returns:
    - The number of selected modes and the matrix of SVD left singular vectors.
    """

    # Convert input data to a NumPy array and compute SVD
    data_array = np.asarray(data)
    U, singular_values, _ = np.linalg.svd(data_array.T, full_matrices=False)
    singular_values_cumsum = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    singular_values_cumsum_tol = 1.0-singular_values_cumsum
    singular_values_cumsum_tol[singular_values_cumsum_tol <np.finfo(float).eps] = np.finfo(float).eps

    # Determine the number of modes where the cumulative sum meets the tolerance
    selected_indices = np.where(singular_values_cumsum_tol < tolerance)[0]
    num_selected_modes = selected_indices[0] + 1 if selected_indices.size > 0 else 1
    

    # Plot the cumulative sum of singular values
    max_ticker = 7
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(np.arange(1,len(singular_values)+1), singular_values_cumsum_tol, 's-', color='orange')
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))


    ax.axhline(y=tolerance, color="black", linestyle="--")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=max_ticker))
    ax.autoscale(tight=True)
    ax.margins(x=0.02,y=0.02)
    #plt.autoscale(tight=rue)
    
    # Print the number of modes if requested
    if modes:
        print(f"Number of modes selected: {num_selected_modes}")
    
    return num_selected_modes, U


def train_test_split(N_snap, N_sel=None,train_percentage = 0.8):

    # Generate a random permutation of indices from 0 to data_size - 1
    indices = np.random.permutation(N_snap)
    
    if N_sel is not None:
        indices = np.random.choice(indices, N_sel, replace=False)


    # Calculate the number of samples in the training set
    train_set_size = int(N_snap * train_percentage)

    # Initialize boolean masks
    train_mask = np.zeros(N_snap, dtype=bool)
    test_mask = np.zeros(N_snap, dtype=bool)

    # Set the first train_set_size indices to True for the training mask
    train_mask[indices[:train_set_size]] = True

    # Set the remaining indices to True for the testing mask
    test_mask[indices[train_set_size:]] = True
    
    return train_mask, test_mask


def latin_hypercube_train_test_split(N_snap, train_percentage=0.8):
    # Generate Latin Hypercube Sampling indices
    LHS_indices = lhs(N_snap, samples=N_snap, criterion='maximin')
    LHS_indices = np.argsort(LHS_indices[:, 0])  # Convert to indices

    # Calculate the number of samples in the training set
    train_set_size = int(N_snap * train_percentage)

    # Initialize boolean masks
    train_mask = np.zeros(N_snap, dtype=bool)
    test_mask = np.zeros(N_snap, dtype=bool)

    # Set masks according to LHS indices
    train_mask[LHS_indices[:train_set_size]] = True
    test_mask[LHS_indices[train_set_size:]] = True

    return train_mask, test_mask


def sobol_train_test_split(N_snap, train_percentage=0.8):
    # Calculate the nearest power of two
    m = int(np.ceil(np.log2(N_snap)))
    sobol_gen = Sobol(d=1)

    # Generate more points if needed and trim to N_snap
    sobol_indices = sobol_gen.random_base2(m=m)
    sobol_indices = sobol_indices.flatten()[:N_snap]  # Trim if longer than N_snap
    sobol_indices = np.argsort(sobol_indices)         # Convert to sorted indices

    # Calculate the number of samples in the training set
    train_set_size = int(N_snap * train_percentage)

    # Initialize boolean masks
    train_mask = np.zeros(N_snap, dtype=bool)
    test_mask = np.zeros(N_snap, dtype=bool)

    # Set masks according to Sobol indices
    train_mask[sobol_indices[:train_set_size]] = True
    test_mask[sobol_indices[train_set_size:]] = True

    return train_mask, test_mask



def generate_sobol(dimensions, num_points, bounds):
    """
    Generates a Sobol sequence.
    
    Parameters:
    dimensions (int): Number of dimensions in the Sobol sequence.
    num_points (int): Number of points in the sequence.
    bounds (list of tuples): A list of tuples containing the lower and upper bounds for each dimension.
    
    Returns:
    np.array: A numpy array containing the Sobol sequence scaled to the provided bounds.
    """
    sobol = Sobol(d=dimensions)
    samples = sobol.random_base2(m=int(np.log2(num_points)))
    scaled_samples = np.empty_like(samples)
    
    for i in range(dimensions):
        lower, upper = bounds[i]
        scaled_samples[:, i] = samples[:, i] * (upper - lower) + lower
        
    return scaled_samples

    # # Example usage
    # dim = 2  # Number of dimensions
    # points = 8  # Number of points in the sequence, must be a power of 2
    # bounds = [(0, 1), (0, 5)]  # Bounds for each dimension

    # sobol_sequence = generate_sobol(dim, points, bounds)
    # sobol_sequence


def generate_lhs(dimensions, num_points, bounds):
    """
    Generates a Latin Hypercube Sampling (LHS).
    
    Parameters:
    dimensions (int): Number of dimensions in the sample.
    num_points (int): Number of points in the sample.
    bounds (list of tuples): Each tuple contains the lower and upper bounds for each dimension.
    
    Returns:
    np.array: A numpy array containing the LHS points scaled to the provided bounds.
    """
    lhs = LatinHypercube(d=dimensions)
    samples = lhs.random(n=num_points)
    scaled_samples = np.empty_like(samples)
    
    for i in range(dimensions):
        lower, upper = bounds[i]
        scaled_samples[:, i] = samples[:, i] * (upper - lower) + lower
        
    return scaled_samples

    # Example usage
    # dim = 2  # Number of dimensions
    # points = 10  # Number of points in the sample
    # bounds = [(0, 1), (0, 5)]  # Bounds for each dimension

    # lhs_samples = generate_lhs(dim, points, bounds)
    # print(lhs_samples)


def generate_gaussian_samples(dimensions, num_points, bounds):
    """
    Generates Gaussian distributed samples for each dimension based on calculated means and standard deviations from bounds,
    without clipping them to the specified bounds.
    
    Parameters:
    dimensions (int): Number of dimensions.
    num_points (int): Number of points to generate.
    bounds (list of tuples): Bounds for each dimension in the form (min, max), from which means and standard deviations are calculated.
    
    Returns:
    np.array: A numpy array containing the Gaussian distributed points.
    """
    samples = np.zeros((num_points, dimensions))
    means = []
    std_devs = []
    
    for lower, upper in bounds:
        mean = (upper + lower) / 2
        std_dev = (upper - lower) / 5  # One-third of half the range
        means.append(mean)
        std_devs.append(std_dev)
    
    for i in range(dimensions):
        samples[:, i] = np.random.normal(loc=means[i], scale=std_devs[i], size=num_points)
        
    return samples

    # # Example usage
    # dim = 2  # Number of dimensions
    # points = 10  # Number of points to generate
    # bounds = [(-2, 2), (2.5, 3.5)]  # Bounds for each dimension

    # gaussian_samples_no_clip = generate_gaussian_samples_no_clip(dim, points, bounds)
    # gaussian_samples_no_clip
