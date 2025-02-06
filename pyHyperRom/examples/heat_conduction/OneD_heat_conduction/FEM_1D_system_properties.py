import os
import sys

# Determine the desired directory path by moving three levels up from the current file's directory.
desired_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
# Change the current working directory to the desired path.
os.chdir(desired_path)

# Append the desired path to the system path so that modules can be imported from this directory.
sys.path.append(desired_path)

# Import basic functionalities from the project's source code.
from src.codes.basic import *

class SystemProperties:
    """
    A class to define system properties, including layout creation, material property definitions,
    and boundary condition specifications for finite element analysis.
    """

    def __init__(self, n_ref, params=np.arange(1., 4.0, 0.01)):
        # Store the reference dimensions or counts provided by n_ref.
        self.n_ref = n_ref
        # Store an array of parameter values, defaulting to values from 1.0 to 4.0 with a step of 0.01.
        self.params = params

    def create_layouts(self):
        """
        Create layout arrays for materials and sources by concatenating arrays of zeros and ones.
        
        Returns:
            tuple: A pair of arrays (mat_layout, src_layout) representing the material and source layouts.
        """
        # Create a 1 x n_ref[0] array filled with zeros.
        zeros_array = np.zeros((1, self.n_ref[0]))
        # Create a 1 x n_ref[1] array filled with ones.
        ones_array = np.ones((1, self.n_ref[1]))
        
        # Concatenate the zero and one arrays along the columns to form the material layout.
        mat_layout = np.concatenate((zeros_array, ones_array), axis=1)
        # Similarly, form the source layout by concatenating the zero and one arrays.
        src_layout = np.concatenate((zeros_array, ones_array), axis=1)
        
        # Return both layouts.
        return mat_layout, src_layout

    def define_properties(self):
        """
        Define material and source property functions for a nonlinear heat conduction problem.
        The properties include external heat source functions, their derivatives, conductivity functions,
        and the derivatives of the conductivity functions.

        Returns:
            dict: A dictionary with keys "qext", "dqext", "cond", and "dcond" mapping to lists of lambda functions.
        """
        # Initialize an empty dictionary to store property functions.
        fdict = {}
        
        ################################################################
        # The following block of code is commented out and represents an alternative
        # definition for linear heat conduction properties. 
        #
        # qext_list = [
        #     lambda T, beta: beta + 35000.0 + 0.0*T,
        #     lambda T, beta: 10*beta + 5000.0 + 0.0*T
        # ]
        # fdict["qext"] = qext_list
        
        # dqext_list = [
        #     lambda T, beta: 0.0*T,
        #     lambda T, beta: 0.0*T
        # ]
        # fdict["dqext"] = dqext_list
        
        # cond_list = [
        #     lambda T, mu: mu + 16 + 0.0*T,
        #     lambda T, mu: mu + 30 + 0.0*T 
        # ]
        # fdict["cond"] = cond_list
        
        # dcond_list = [
        #     lambda T, mu: 0.0*T,
        #     lambda T, mu: 0.0*T
        # ]
        # fdict["dcond"] = dcond_list
        
        # return fdict
        ################################################################

        ## Nonlinear Heat Conduction (HC) Property Definitions

        # Define external heat source functions (qext) as lambda functions.
        # The first function adds a temperature-dependent term (T/10) to beta and a constant.
        # The second function is a linear function of beta with a fixed constant.
        qext_list = [
            lambda T, beta: beta + 35000.0 + T/10,
            lambda T, beta: 10 * beta + 5000.0 + 0.0 * T
        ]
        # Store the external heat source functions in the dictionary.
        fdict["qext"] = qext_list

        # Define the derivatives of the external heat source functions (dqext).
        # The first derivative returns a constant 1/10 (ignoring T except for scaling).
        # The second derivative returns zero for all T.
        dqext_list = [
            lambda T, beta: 0.0 * T + 1/10,
            lambda T, beta: 0.0 * T
        ]
        # Store the derivatives of the external heat source functions.
        fdict["dqext"] = dqext_list
        
        # Define conductivity functions (cond) using lambda functions.
        # The first function includes an inverse dependency on (T - 73.15) and an additive term with mu.
        # The second function is given as a cubic polynomial in T with coefficients and an additive mu.
        cond_list = [
            lambda T, mu: 16 + mu + 2150 / (T - 73.15),
            lambda T, mu: 30 + mu + 2.09e-2 * T - 1.45e-5 * T**2 + 7.67e-9 * T**3
        ]
        # Store the conductivity functions.
        fdict["cond"] = cond_list

        # Define the derivatives of the conductivity functions (dcond).
        # The first derivative is computed analytically for the inverse dependency in the first function.
        # The second derivative is the derivative of the cubic polynomial in T.
        dcond_list = [
            lambda T, mu: -2150 / (T - 73.15)**2,
            lambda T, mu: 2.09e-2 - 2 * 1.45e-5 * T + 3 * 7.67e-9 * T**2
        ]
        # Store the derivatives of the conductivity functions.
        fdict["dcond"] = dcond_list
       
        # Return the dictionary containing all defined property functions.
        return fdict


    def define_boundary_conditions(self):
        """
        Define the boundary conditions for the system.

        Returns:
            dict: A dictionary specifying the type and value of boundary conditions at the domain boundaries.
        """
        # Create a dictionary to hold boundary conditions.
        bc = {
            'x_min': {'type': 'refl', 'value': np.nan},  # Reflective boundary condition at the minimum x-boundary.
            'x_max': {'type': 'dirichlet', 'value': 273.15 + 300}  # Dirichlet boundary condition at the maximum x-boundary with a specified temperature.
        }
        # Return the boundary conditions.
        return bc
