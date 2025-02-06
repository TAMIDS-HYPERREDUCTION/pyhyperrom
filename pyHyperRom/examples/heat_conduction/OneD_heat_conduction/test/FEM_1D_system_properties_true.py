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
        # cond_list_lin = [
        #     lambda T, mu: 1.05 * mu * tune+0.0*T,
        #     lambda T, mu: mu * tune * 7.51+0.0*T
        # ]
        # fdict["cond_lin"] = cond_list_lin

        # # Define conductivity properties using lambda functions
        # dcond_list_lin = [
        #     lambda T, mu: 0.0*T,
        #     lambda T, mu: 0.0*T
        # ]
        # fdict["dcond_lin"] = dcond_list_lin

        # # Define conductivity properties using lambda functions
        # qext_list = [
        #     lambda T, mu: 1.05 * mu * tune + 150.0*T, #+ 2150 / (T - 73.15),
        #     lambda T, mu: mu * tune * 7.51 + 0.0*T + 2.09e-2 * T #- 1.45e-5 * T**2 + 7.67e-9 * T**3
        # ]
        # fdict["qext"] = qext_list

        # # Define the derivative of conductivity properties
        # dqext_list = [
        #     lambda T, mu: 150.+0.0*T, #-2150 / (T - 73.15)**2,
        #     lambda T, mu: 0.0*T + 2.09e-2 #- 2 * 1.45e-5 * T + 3 * 7.67e-9 * T**2
        # ]
        # fdict["dqext"] = dqext_list


        # External heat source properties

        # ## Linear Heat conduction:
        
        # qext_list = [
        #     lambda T, beta: beta + 35000.0 + 0.0*T, #for deim used  100*beta on both
        #     lambda T, beta: beta + 5000.0 + 0.0*T
        # ]
        
        # fdict["qext"] = qext_list

        # # Derivative of the external heat source properties
        
        # dqext_list = [
        #     lambda T, beta: 0.0*T,
        #     lambda T, beta: 0.0*T
        # ]
        # fdict["dqext"] = dqext_list
        

        # # Define conductivity properties using lambda functions
        
        # cond_list = [
        #     lambda T, mu: 2 + mu + 14 + 0.0*T, #2150 / (T - 73.15),
        #     lambda T, mu: 2 + mu + 30 + 0.0*T#2.09e-2 * T - 1.45e-5 * T**2 + 7.67e-9 * T**3
        # ]
        # fdict["cond"] = cond_list

        # # Define the derivative of conductivity properties
        
        # dcond_list = [
        #     lambda T, mu: 0.0*T,#-2150 / (T - 73.15)**2,
        #     lambda T, mu: 0.0*T #2.09e-2 - 2 * 1.45e-5 * T + 3 * 7.67e-9 * T**2
        # ]
        # fdict["dcond"] = dcond_list
       
        
        # return fdict
    

        ## Nonlinear HC

        qext_list = [
            lambda T, beta: beta + 35000.0 + 0.0*T, #for deim used  100*beta on both
            lambda T, beta: beta + 5000.0 + 0.0*T
        ]
        
        fdict["qext"] = qext_list

        # Derivative of the external heat source properties
        
        dqext_list = [
            lambda T, beta: 0.0*T,
            lambda T, beta: 0.0*T
        ]
        fdict["dqext"] = dqext_list
        

        # Define conductivity properties using lambda functions
        
        cond_list = [
            lambda T, mu:  2+mu + 2150 / (T - 73.15),
            lambda T, mu:  2+mu + 2.09e-2 * T - 1.45e-5 * T**2 + 7.67e-9 * T**3
        ]
        fdict["cond"] = cond_list

        # Define the derivative of conductivity properties
        
        dcond_list = [
            lambda T, mu: -2150 / (T - 73.15)**2,
            lambda T, mu:  2.09e-2 -  2 *1.45e-5 * T + 3 * 7.67e-9 * T**2
        ]
        fdict["dcond"] = dcond_list
       
        
        return fdict


    def define_boundary_conditions(self):
        bc = {
            'x_min': {'type': 'refl', 'value': np.nan},
            'x_max': {'type': 'dirichlet', 'value': 273.15 + 300}
        }
        return bc