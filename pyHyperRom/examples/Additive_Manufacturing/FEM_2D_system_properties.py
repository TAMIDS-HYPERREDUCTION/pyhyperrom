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

    def __init__(self, n_ref, params = 7.85e-9):
        self.n_ref = n_ref
        self.params = params
        self.q = 60.0/params

    def create_layouts(self):
        
        self.mat_layout = np.zeros((1,1),dtype=int)
        self.src_layout = np.zeros((1,1),dtype=int)
        
        return self.mat_layout, self.src_layout

    def define_boundary_conditions(self):
        bc = {}
        # bc['x_min']={'type':'dirichlet','value':298.0}
        # bc['x_max']={'type':'dirichlet','value':298.0}
        # bc['y_min']={'type':'dirichlet','value':298.0}
        # bc['y_max']={'type':'dirichlet','value':298.0}
        
        bc['x_min']={'type':'refl','value':np.nan}
        bc['x_max']={'type':'refl','value':np.nan}
        bc['y_min']={'type':'refl','value':np.nan}
        bc['y_max']={'type':'refl','value':np.nan}
        
        return bc

    def define_properties(self):
        
        fdict = {}
        
        cond_list = []
        # cond_list.append( lambda T: 28. + 0.*T )
        cond_list.append(16.0)
        fdict["cond"] = cond_list
        
        # dcond_list = []
        # dcond_list.append( lambda T: 0. + 0.*T )
        # fdict["dcond"] = dcond_list

        rho = []
        rho.append(7800)
        fdict["rho"] = rho

        C_v = []
        C_v.append(500.0)
        fdict["C_v"] = C_v

        fext = []
        
        fext.append(lambda c_elem, t, A_t, fr, deltas, coords: self.forcing_fn(c_elem, t, A_t, fr, deltas,coords ))
        fdict["fext"] = fext
               
        return fdict


    def forcing_fn(self,c_elem, t, A_t, fr, deltas, coords):
        

        elem_force = np.zeros(len(t))

        Torch_rad = 0.5*1e-4

        xc = (10 + np.ceil(Torch_rad/deltas[0]))*deltas[0] + fr*t
        yc = coords[1][int(len(coords[1])//2)] + 0.0*xc

        x_elem = c_elem[0].reshape(-1,1)
        y_elem = c_elem[1].reshape(-1,1)

        r_elem = np.array(( (x_elem-xc)**2 + (y_elem-yc)**2 )**0.5)

        mask_ = r_elem < Torch_rad
        mask = np.sum(mask_,axis=0).astype(bool)

        if mask.any()==True:
            elem_force[mask] = self.q
            
            
        # say the welding time is smaller than simulation time. 
        t_welding = 0.003
        mask_t = t>t_welding
        
        elem_force[mask_t] = 0.0
        

        return elem_force