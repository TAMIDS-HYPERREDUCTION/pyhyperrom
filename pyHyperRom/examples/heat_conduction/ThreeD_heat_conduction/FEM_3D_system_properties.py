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
                
        mat_layout = np.zeros((4,4,4),dtype=int)
        src_layout = np.zeros((4,4,4),dtype=int)
        
        return mat_layout, src_layout

    def define_properties(self):
        
        fdict = {}

        cond_list = []
        cond_list.append( lambda T,mu: 10*np.exp(mu)*T + 100. + 0.*T )
        fdict["cond"] = cond_list

        dcond_list = []
        dcond_list.append( lambda T,mu: 10*np.exp(mu) + 0. + 0.*T )
        fdict["dcond"] = dcond_list

        qext_list = []
        qext_list.append( lambda T,mu: 0.0+100.0*np.exp(mu)+ 0.*T )
        fdict["qext"] = qext_list

        dqext_list = []
        dqext_list.append( lambda T,mu: 0.0 + 0.*T)
        fdict["dqext"] = dqext_list
        
        return fdict

    def define_boundary_conditions(self):
        bc = {}
        bc['x_min']={'type':'dirichlet','value':0.}
        bc['x_max']={'type':'dirichlet','value':0.}
        bc['y_min']={'type':'dirichlet','value':0.}
        bc['y_max']={'type':'dirichlet','value':0.}
        bc['z_min']={'type':'dirichlet','value':0.}
        bc['z_max']={'type':'dirichlet','value':0.}
        return bc