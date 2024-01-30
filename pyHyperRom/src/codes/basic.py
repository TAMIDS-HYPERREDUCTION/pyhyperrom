"""
This module imports some commonly used librarires.
You can use ``from pyHyperRom.basic import *``
to use the most frequently used libraries.
"""
# Standard Library Imports
import os
import sys
import time
import random
from importlib import reload

# System Path Modification for Local Imports
desired_path = os.path.join(os.path.dirname(__file__))
sys.path.append(desired_path)

# Numerical Libraries
import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import linalg
from itertools import product

# Plotting Libraries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

# Custom Style Application for Plots
plt.style.use(os.path.join(desired_path,'utils/plot_files/publication.mplstyle'))

# Local Module Imports
from utils.plot_files.animate_1D import AnimatedPlot
from utils.plot_files.animate_2D import AnimatedContourPlot

# Optional Imports (Uncomment if needed)
# from pylab import plot
# import fnnlsEigen as fe # nnls
# import decompyle3
