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

    def __init__(self, n_ref, cv=1e-2, cm = 1e-4):
        self.n_ref = n_ref # cells
        self.cv = cv
        self.cm = cm


    def create_layouts(self):
        # Create arrays of zeros and ones
        mat_layout = np.zeros(self.n_ref)
        src_layout = np.zeros(self.n_ref)
        
        return mat_layout, src_layout


    def define_boundary_conditions(self):
        bc = {
            'x_min': {'type': 'dirichlet', 'value': 0},
            'x_min_theta': {'type': 'refl', 'value': 0},
            'x_max': {'type': 'dirichlet', 'value': 0},
            'x_max_theta': {'type': 'refl', 'value': 0}
        }
        return bc


    def forcing_fn_x(self, xq_, x_elem, epsilon, s=2**(-0.5)):
        
        a = x_elem[0]
        b = x_elem[1]
        
        
        xq = 0.5 * (b - a) * xq_ + 0.5 * (b + a)
        # s = 2**(-0.5)  # at the mid point
    
        # Vectorized condition
        condition = (s - epsilon / 2 <= xq) & (xq <= s + epsilon / 2)
    
        # Initialize fx to zero for all elements
        fx = np.zeros_like(xq)
    
        # Apply the calculation where the condition is true
        fx[condition] = (2 / epsilon) * np.cos(np.pi / epsilon * (xq[condition] - s)) ** 2
    
        return fx.reshape(-1,1)

    
    def forcing_fn_t(self, t, tau, T, K=90):
        ft = 0#1 / T
    
        def dk(k, tau, T):
            if T == k * tau:
                return (-1)**k / T
            else:
                numerator = 2 * (T**3) * np.cos(np.pi * k) * np.sin(np.pi * k * tau / T)
                denominator = T * (np.pi * k * tau * T**2 - np.pi * tau**3 * k**3)
                return numerator / denominator
    
        for k in range(1, K + 1):
            ft += dk(k, tau, T) * np.cos(2 * np.pi * k * t / T)
    
        return ft


    def forcing_fn(self,xq, x_elem, t, ep, tau, T):

        if len(tau)>1:
            tau_p = tau[0]
            s = 2**(-0.5) + tau[1]
        else:
            tau_p = tau
            s = 2**(-0.5)

        fx = self.forcing_fn_x(xq, x_elem, ep, s=s)
        ft = self.forcing_fn_t(t,tau_p,T)

        return fx*ft


    # def forcing_fn(self,xq, x_elem, t, ep, tau, T):
        
    #     for i in range(len(x_elem)):
    #         if x_elem[i] == 0.5:
    #             fx = 100
    #         else:
    #             fx = 0
                
    #     ft = np.sin(t)#self.forcing_fn_t(t,tau,T)

    #     return fx*ft

    
    def define_properties(self):
        tune = 1
        fdict = {}
        
        # Define conductivity properties using lambda functions
        E = [
            lambda x: 1e4/1e4 +0.0*x
        ]
        fdict["E"] = E

        I = [
            lambda x: 1e-4/1e-4 +0.0*x
        ]
        fdict["I"] = I

        rho = [
            lambda x: 2710.0/2710. +0.0*x
        ]
        fdict["rho"] = rho

        A = [
            lambda x: 1e-2/1e-2 +0.0*x
        ]
        fdict["A"] = A

        fext = [
            lambda x, x_elem, t, tau, ep, T: self.forcing_fn(x, x_elem, t, ep, tau, T)
        ]
        fdict["fext"] = fext
        
        return fdict