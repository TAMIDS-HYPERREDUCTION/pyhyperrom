"""
This module imports essential libraries commonly used in pyHyperRom.
By using ``from pyHyperRom.basic import *``, users can access frequently utilized libraries.
"""

# Import standard Python libraries for system interaction and time handling
import os  # Provides functions for interacting with the operating system
import sys  # Provides system-specific parameters and functions
import time  # Provides time-related functions
import random  # Provides functions for generating random numbers
from importlib import reload  # Allows reloading previously imported modules

# Modify system path to include the directory of this script for local imports
desired_path = os.path.join(os.path.dirname(__file__))  # Determine the directory of the current file
sys.path.append(desired_path)  # Append the directory to the system path

# Import numerical and symbolic computation libraries
import numpy as np  # Provides support for large, multi-dimensional arrays and matrices
import sympy as sp  # Provides symbolic mathematics capabilities
from scipy import sparse  # Provides support for sparse matrices
from scipy.sparse import linalg  # Provides sparse matrix linear algebra utilities
from itertools import product  # Generates Cartesian product of input iterables

# Import libraries for data visualization
import matplotlib  # Provides comprehensive 2D and limited 3D plotting functionality
import matplotlib.pyplot as plt  # Provides a MATLAB-like interface for plotting
from matplotlib import cm  # Provides color maps for visualizations
from matplotlib.ticker import LinearLocator  # Helps control tick locations for plots
from matplotlib.ticker import MaxNLocator  # Allows specifying the maximum number of ticks
from mpl_toolkits.mplot3d import Axes3D  # Provides tools for 3D plotting

# Apply a custom plotting style for publication-quality figures
plt.style.use(os.path.join(desired_path, 'utils/plot_files/publication.mplstyle'))

# Import custom animation modules for 1D and 2D data visualization
from utils.plot_files.animate_1D import AnimatedPlot  # Handles 1D animated plotting
from utils.plot_files.animate_2D import AnimatedContourPlot  # Handles 2D contour animations

# Optional imports (uncomment if needed)
# from pylab import plot  # Provides MATLAB-like plotting functions
# import fnnlsEigen as fe # Implements non-negative least squares (NNLS) using Eigen
# import decompyle3  # Tool for decompiling Python bytecode
