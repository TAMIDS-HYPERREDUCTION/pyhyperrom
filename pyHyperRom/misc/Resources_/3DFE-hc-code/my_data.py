# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:29:30 2023

@author: jean.ragusa
"""
import numpy as np


class mydata:
    
    def __init__(self, cond_layout, siga_layout , qext_layout, nref, L, pb_dim=3):
        # cond_layout: 3D layout of the conductivity (material zones)
        # qext_layout: 3D layout of the heat source (material zones)
        #     Note the two layouts must be of compatible shape
        # nref: a list or array of length 2 containing the number of mesh refinement in x and y
        #     Note that mesh refinement is uniform in this code
        # L: a list or array of legnth 2 containing the Lx and Ly dimensions
        
        # sanity checks
        if pb_dim != 3:
            raise ValueError('Only pb_dim=3 for now')
        if cond_layout.ndim !=3:
            raise ValueError('cond_layout: only dim=3 for now')
        if siga_layout.ndim !=3:
            raise ValueError('siga_layout: only dim=3 for now')
        if qext_layout.ndim !=3:
            raise ValueError('qext_layout: only dim=3 for now')
        if cond_layout.shape != siga_layout.shape:
            raise ValueError('cond_layout and siga_layout of different shapes')
        if cond_layout.shape != qext_layout.shape:
            raise ValueError('cond_layout and qext_layout of different shapes')
        if len(nref) !=3:
            raise ValueError('nref: only len=2 for now')
        if len(L) !=3:
            raise ValueError('L: only len=2 for now')
        
        # refine the mesh and update material and source layouts
        repeats = np.asarray(nref, dtype=int)
        self.cond = self.repeat_array(cond_layout,repeats)
        self.siga = self.repeat_array(siga_layout,repeats)
        self.qext = self.repeat_array(qext_layout,repeats)

        # mesh data
        # cells
        self.ncells_x, self.ncells_y, self.ncells_z = self.cond.shape
        self.n_cells = self.ncells_x * self.ncells_y * self.ncells_z
        # vertices
        self.npts_x = self.ncells_x + 1
        self.npts_y = self.ncells_y + 1
        self.npts_z = self.ncells_z + 1
        self.n_verts = self.npts_x * self.npts_y  * self.npts_z
        # coordinates
        self.x = np.linspace(0,L[0],self.npts_x)
        self.y = np.linspace(0,L[1],self.npts_y)
        self.z = np.linspace(0,L[2],self.npts_z)
        
        self.dx = L[0] / self.ncells_x
        self.dy = L[1] / self.ncells_y
        self.dz = L[2] / self.ncells_z
        
        # nodal connectivity for cFEM
        self.connectivity()

            
    def repeat_array(self, arr,repeats):
        for dim,n in enumerate(repeats):
            arr = np.repeat(arr,n,axis=dim)
        return arr        
    
    
    def connectivity(self):
        self.gn = np.zeros((self.n_cells,8),dtype=int)

        # compute node ID from (i,j,k) cell identifiers
        node = lambda i,j,k: i+j*self.npts_x+k*self.npts_x*self.npts_y

        iel = 0
        for k in range(self.ncells_z):
            for j in range(self.ncells_y):
                for i in range(self.ncells_x):
                    # counter-clockwise
                    self.gn[iel,0] = node(i  ,j  ,k  )
                    self.gn[iel,1] = node(i+1,j  ,k  )
                    self.gn[iel,2] = node(i+1,j+1,k  )
                    self.gn[iel,3] = node(i  ,j+1,k  )
                    self.gn[iel,4] = node(i  ,j  ,k+1)
                    self.gn[iel,5] = node(i+1,j  ,k+1)
                    self.gn[iel,6] = node(i+1,j+1,k+1)
                    self.gn[iel,7] = node(i  ,j+1,k+1)

                    iel += 1    