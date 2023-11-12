from ..basic import *
from .plot_utils_style import *
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, LogLocator

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
    singular_values_cumsum = np.cumsum(singular_values) / np.sum(singular_values)
    singular_values_cumsum_tol = 1.0-singular_values_cumsum
    singular_values_cumsum_tol[singular_values_cumsum_tol<np.finfo(float).eps] = np.finfo(float).eps

    # Determine the number of modes where the cumulative sum meets the tolerance
    selected_indices = np.where(singular_values_cumsum_tol< tolerance)[0]
    num_selected_modes = selected_indices[0] + 1 if selected_indices.size > 0 else 1
    
    # Plot the cumulative sum of singular values
    max_ticker = 7
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(np.arange(1,len(singular_values)+1), singular_values_cumsum_tol, 's-', color='orange')
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))

    ax.axhline(y=tolerance, color="black", linestyle="--")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=max_ticker))

    plt.autoscale(enable=False, tight=True)
    plt.show()
    
    # Print the number of modes if requested
    if modes:
        print(f"Number of modes selected: {num_selected_modes}")
    
    return num_selected_modes, U


def train_test_split(N_snap, train_percentage = 0.8):

    # Split percentage
    train_percentage = 0.8

    # Generate a random permutation of indices from 0 to data_size - 1
    indices = np.random.permutation(N_snap)

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