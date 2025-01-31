# Restart the kernel
import os
import sys

# Adjust the file path as necessary to navigate to the desired directory
desired_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
os.chdir(desired_path)

# Append the source code directory to the system path
sys.path.append(desired_path)

# Import basic functionalities
from src.codes.basic import *


class SystemProperties:

    def __init__(self, n_ref):
        self.n_ref = n_ref # cells
        self.L = 420 #inches
        self.expon = 0.26-1

    def create_layouts(self):
        # Create arrays of zeros and ones
        mat_layout = np.zeros(self.n_ref)
        src_layout = np.zeros(self.n_ref)
        
        return mat_layout, src_layout

    def define_boundary_conditions(self):
        bc = {
            'x_min': {'type': 'dirichlet', 'value': 0},
            'x_max': {'type': 'refl', 'value': 0}
        }
        return bc


    def forcing_fn(self, xq_, x_elem, tau, A, rho):
        
        a = x_elem[0]
        b = x_elem[1]
        
        xq = 0.5 * (b - a) * xq_ + 0.5 * (b + a)

        # Initialize fx to zero for all elements
        fx = np.zeros_like(xq)
        # rho = prop['rho']
        # A = prop['A']

        # Apply the calculation where the condition is true

        ## Training
        # fx = 1000*tau + rho(xq)* A(xq) * (self.L-xq)

        ## UQ
        fx = 1000 + tau + rho(xq)* A(xq) * (self.L-xq)

        return fx

    
    def define_properties(self):

        tune = 1
        fdict = {}
        
        # Define conductivity properties using lambda functions
        E = [
            lambda x, mu: mu + 77e3 +0.0*x  #lb/in^2
        ]
        fdict["E"] = E

        rho = [
            lambda x: 0.28907 +0.0*x #lb/in^3
        ]
        fdict["rho"] = rho

        A = [
            lambda x: 10.0 +0.0*x
        ]
        fdict["A"] = A

        fext = [
            lambda x, x_elem, tau, A, rho: self.forcing_fn(x, x_elem, tau, A,rho)
        ]
        fdict["fext"] = fext
        
        return fdict