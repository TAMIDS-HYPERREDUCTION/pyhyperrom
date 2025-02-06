# Restart the kernel
import os  # Provides functions for interacting with the operating system.
import sys  # Provides access to system-specific parameters and functions.

# Adjust the file path to navigate three directories up from the current file's directory.
desired_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')

# Change the current working directory to the desired path.
os.chdir(desired_path)

# Append the source directory to sys.path so that modules containing numerical routines and structural mechanics utilities can be imported.
sys.path.append(desired_path)


# Import basic functionalities (such as numerical solvers or helper routines) required for the structural analysis.
from src.codes.basic import *



# Define the SystemProperties class to encapsulate various parameters and methods used in the structural mechanics simulation.
class SystemProperties:


    # Constructor: Initializes the system properties for the structural mechanics problem.
    # n_ref: Number of reference cells (e.g., finite elements) used to discretize the structure.
    def __init__(self, n_ref):
        self.n_ref = n_ref          # Number of finite elements or reference cells in the discretization.
        self.L = 420                # Total length of the structure (e.g., a beam) in inches.
        self.expon = 0.26 - 1       # An exponential parameter, possibly used in scaling laws or material nonlinearity.


    # Method to create layout arrays representing the material and load distribution across the structure.
    # In structural mechanics, these arrays can be used to model spatial variations in material properties and applied loads.
    def create_layouts(self):
        # Create an array of zeros representing the material layout (e.g., stiffness or density distribution).
        mat_layout = np.zeros(self.n_ref)
        # Create an array of zeros representing the source layout (e.g., locations of applied forces).
        src_layout = np.zeros(self.n_ref)
        
        # Return the material and source layouts as a tuple.
        return mat_layout, src_layout


    # Method to define the boundary conditions for the structural model.
    # Boundary conditions are crucial in structural mechanics as they simulate supports and constraints.
    def define_boundary_conditions(self):
        # Define boundary conditions in a dictionary:
        # 'x_min': Represents the left end of the structure with a Dirichlet condition (e.g., a fixed or clamped end) at 0 displacement.
        # 'x_max': Represents the right end with a reflective condition (which might model symmetry or a free end under certain assumptions) at 0.
        bc = {
            'x_min': {'type': 'dirichlet', 'value': 0},
            'x_max': {'type': 'refl', 'value': 0}
        }
        return bc


    # Method to compute the forcing function, representing external loads acting on the structure.
    # xq_: Normalized quadrature points used for numerical integration in finite element methods.
    # x_elem: The endpoints of a finite element, defining its local domain along the structure.
    # tau: A parameter that could represent an additional load or stress adjustment.
    # A: Lambda function for the cross-sectional area, affecting the structural stiffness.
    # rho: Lambda function for the material density, affecting inertia and load distribution.
    def forcing_fn(self, xq_, x_elem, tau, A, rho):
        # Extract the endpoints of the finite element.
        a = x_elem[0]
        b = x_elem[1]
        
        # Map normalized quadrature points (from the reference element) to the physical coordinates of the element.
        xq = 0.5 * (b - a) * xq_ + 0.5 * (b + a)

        # Initialize the force array with zeros, corresponding to each quadrature point.
        fx = np.zeros_like(xq)
        
        # The commented formulation below (Training) shows an alternative approach:
        # fx = 1000 * tau + rho(xq) * A(xq) * (self.L - xq)
        
        # For uncertainty quantification (UQ) in the structural response, the active formulation is:
        fx = 1000 + tau + rho(xq) * A(xq) * (self.L - xq)

        # Return the computed force distribution along the element.
        return fx


    # Method to define and return key material and structural properties for the simulation.
    # These properties serve as input parameters in the structural mechanics analysis.
    def define_properties(self):
        # The 'tune' variable is initialized for potential future calibration (currently unused).
        tune = 1
        # Create a dictionary to store the material and load properties.
        fdict = {}
        
        # Define the elastic modulus 'E' as a list containing a lambda function.
        # In structural mechanics, E represents the stiffness of the material (lb/in^2).
        E = [
            lambda x, mu: mu + 77e3 + 0.0 * x  # Computes the elastic modulus using a base offset (77e3) and a variable parameter mu.
        ]
        fdict["E"] = E

        # Define the material density 'rho' as a list containing a lambda function.
        # The density (lb/in^3) is critical for assessing inertial effects and load distribution.
        rho = [
            lambda x: 0.28907 + 0.0 * x  # Returns a constant density value for the entire structure.
        ]
        fdict["rho"] = rho

        # Define the cross-sectional area 'A' as a list containing a lambda function.
        # The cross-sectional area is a key factor in determining bending stiffness and load-bearing capacity.
        A = [
            lambda x: 10.0 + 0.0 * x  # Returns a constant cross-sectional area across the structure.
        ]
        fdict["A"] = A

        # Define the external force function 'fext' as a list containing a lambda function.
        # This function wraps around forcing_fn to compute the load distribution based on the element's properties.
        fext = [
            lambda x, x_elem, tau, A, rho: self.forcing_fn(x, x_elem, tau, A, rho)
        ]
        fdict["fext"] = fext
        
        # Return the complete dictionary of defined structural properties.
        return fdict