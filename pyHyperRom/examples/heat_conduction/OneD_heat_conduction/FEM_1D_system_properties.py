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

    def __init__(self, n_ref, params = np.arange(1., 4.0, 0.01)):
        self.n_ref = n_ref
        self.params = params

    def create_layouts(self):
        # Create arrays of zeros and ones
        zeros_array = np.zeros((1, self.n_ref[0]))
        ones_array = np.ones((1, self.n_ref[1]))

        # Concatenate along the second axis (axis=1)
        mat_layout = np.concatenate((zeros_array, ones_array), axis=1)
        src_layout = np.concatenate((zeros_array, ones_array), axis=1)
        
        return mat_layout, src_layout

    def define_properties(self):
        tune = 1
        fdict = {}
        
        # Define conductivity properties using lambda functions
        cond_list = [
            lambda T, mu: 1.05 * mu * tune + 2150 / (T - 73.15),
            lambda T, mu: mu * tune * 7.51 + 2.09e-2 * T - 1.45e-5 * T**2 + 7.67e-9 * T**3
        ]
        fdict["cond"] = cond_list

        # Define the derivative of conductivity properties
        dcond_list = [
            lambda T, mu: -2150 / (T - 73.15)**2,
            lambda T, mu: 2.09e-2 - 2 * 1.45e-5 * T + 3 * 7.67e-9 * T**2
        ]
        fdict["dcond"] = dcond_list

        # External heat source properties
        qext_list = [
            lambda T, mu: 35000.0 +0.0*T,
            lambda T, mu: 0.0*T
        ]
        fdict["qext"] = qext_list

        # Derivative of the external heat source properties
        dqext_list = [
            lambda T, mu: 0.0*T,
            lambda T, mu: 0.0*T
        ]
        fdict["dqext"] = dqext_list
        
        return fdict

    def define_boundary_conditions(self):
        bc = {
            'x_min': {'type': 'refl', 'value': np.nan},
            'x_max': {'type': 'dirichlet', 'value': 273.15 + 300}
        }
        return bc