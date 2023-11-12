# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:30:18 2023

@author: jean.ragusa
"""
import numpy as np

from scipy import sparse
from scipy.sparse import linalg


class FEM_solver:
    def __init__(self, data):
        
        # store the data received
        self.data = data
        
        # duplicates, shortcuts
        self.ncells_x = data.ncells_x
        self.ncells_y = data.ncells_y
        self.ncells_z = data.ncells_z

        self.npts_x = data.npts_x
        self.npts_y = data.npts_y
        self.npts_z = data.npts_z
        
        self.n_nodes = data.n_verts
        self.n_cells = data.n_cells
        
        self.dx = data.dx
        self.dy = data.dy
        self.dz = data.dz

        # compute the cFEM basis functions
        self.basis()
        # compute the elemental matrices
        self.compute_elemental_matrices()

        
    def elem2ijk(self,iel,verbose=False):
        # from element ID (iel) get the (i,j) cell integers
        # iel = i + j*nx + k*nx*ny
        k = int(iel/self.ncells_x/self.ncells_y)
        j = int( (iel-k*self.ncells_x*self.ncells_y ) / self.ncells_x)
        i = iel - j*self.ncells_x - k*self.ncells_x*self.ncells_y
        if verbose:
            print("i=",i,", j=",j,", k=",k)
        return i,j,k
    
    
    def node2ijk(self,ivert,verbose=False):
        # from node ID (ivert) get the (i,j) node integers
        # ivert = i + j*npts_x + k *npts_x*npts_y
        k = int(ivert/self.npts_x/self.npts_y)
        j = int( (ivert-k*self.npts_x*self.npts_y) / self.npts_x)
        i = ivert - j*self.npts_x - k*self.npts_x*self.npts_y
        if verbose:
            print("i=",i,", j=",j,", k=",k)
        return i,j,k
    
    
    def basis(self):
        # list of basis functions in [-1,1]^dim in counter-clockwise ordering
        self.b = []
        self.b.append(lambda u,v,w: (1-u)*(1-v)*(1-w)/8 )
        self.b.append(lambda u,v,w: (1+u)*(1-v)*(1-w)/8 )
        self.b.append(lambda u,v,w: (1+u)*(1+v)*(1-w)/8 )
        self.b.append(lambda u,v,w: (1-u)*(1+v)*(1-w)/8 )
        self.b.append(lambda u,v,w: (1-u)*(1-v)*(1+w)/8 )
        self.b.append(lambda u,v,w: (1+u)*(1-v)*(1+w)/8 )
        self.b.append(lambda u,v,w: (1+u)*(1+v)*(1+w)/8 )
        self.b.append(lambda u,v,w: (1-u)*(1+v)*(1+w)/8 )
        # their derivatives with respect to u
        self.dbdx = []
        self.dbdx.append(lambda u,v,w: -(1-v)*(1-w)/8 )
        self.dbdx.append(lambda u,v,w:  (1-v)*(1-w)/8 )
        self.dbdx.append(lambda u,v,w:  (1+v)*(1-w)/8 )
        self.dbdx.append(lambda u,v,w: -(1+v)*(1-w)/8 )
        self.dbdx.append(lambda u,v,w: -(1-v)*(1+w)/8 )
        self.dbdx.append(lambda u,v,w:  (1-v)*(1+w)/8 )
        self.dbdx.append(lambda u,v,w:  (1+v)*(1+w)/8 )
        self.dbdx.append(lambda u,v,w: -(1+v)*(1+w)/8 )
        # their derivatives with respect to v
        self.dbdy = []
        self.dbdy.append(lambda u,v,w: -(1-u)*(1-w)/8 )
        self.dbdy.append(lambda u,v,w: -(1+u)*(1-w)/8 )
        self.dbdy.append(lambda u,v,w:  (1+u)*(1-w)/8 )
        self.dbdy.append(lambda u,v,w:  (1-u)*(1-w)/8 )
        self.dbdy.append(lambda u,v,w: -(1-u)*(1+w)/8 )
        self.dbdy.append(lambda u,v,w: -(1+u)*(1+w)/8 )
        self.dbdy.append(lambda u,v,w:  (1+u)*(1+w)/8 )
        self.dbdy.append(lambda u,v,w:  (1-u)*(1+w)/8 )
        # their derivatives with respect to w
        self.dbdz = []
        self.dbdz.append(lambda u,v,w: -(1-u)*(1-v)/8 )
        self.dbdz.append(lambda u,v,w: -(1+u)*(1-v)/8 )
        self.dbdz.append(lambda u,v,w: -(1+u)*(1+v)/8 )
        self.dbdz.append(lambda u,v,w: -(1-u)*(1+v)/8 )
        self.dbdz.append(lambda u,v,w:  (1-u)*(1-v)/8 )
        self.dbdz.append(lambda u,v,w:  (1+u)*(1-v)/8 )
        self.dbdz.append(lambda u,v,w:  (1+u)*(1+v)/8 )
        self.dbdz.append(lambda u,v,w:  (1-u)*(1+v)/8 )
        
        
    def compute_elemental_matrices(self,verbose=False):
        # select spatial quadrature
        degree = 3
        [x_,w_] = np.polynomial.legendre.leggauss(degree)

        # matrices
        local_dofs = len(self.b)
        self.Q   = np.zeros(local_dofs)
        self.M   = np.zeros((local_dofs,local_dofs))
        self.Kxx = np.zeros((local_dofs,local_dofs))
        self.Kyy = np.zeros((local_dofs,local_dofs))
        self.Kzz = np.zeros((local_dofs,local_dofs))

        for i,fi in enumerate(self.b):
            for (uq,wuq) in zip(x_,w_):
                for (vq,wvq) in zip(x_,w_):
                    for (wq,wwq) in zip(x_,w_):
                        self.Q[i]   += wuq*wvq**wwq*fi(uq,vq,wq)

        for i,(fi,fxi,fyi,fzi) in enumerate(zip(self.b,self.dbdx,self.dbdy,self.dbdz)):
            for j,(fj,fxj,fyj,fzj) in enumerate(zip(self.b,self.dbdx,self.dbdy,self.dbdz)):
                for (uq,wuq) in zip(x_,w_):
                    for (vq,wvq) in zip(x_,w_):
                        for (wq,wwq) in zip(x_,w_):
                                self.M[i,j]   += wuq*wvq*wwq*fi (uq,vq,wq)*fj (uq,vq,wq)
                                self.Kxx[i,j] += wuq*wvq*wwq*fxi(uq,vq,wq)*fxj(uq,vq,wq)
                                self.Kyy[i,j] += wuq*wvq*wwq*fyi(uq,vq,wq)*fyj(uq,vq,wq)
                                self.Kzz[i,j] += wuq*wvq*wwq*fzi(uq,vq,wq)*fzj(uq,vq,wq)

        if verbose:
            print(self.Q)
            print(self.M)
            print(self.Kxx)
            print(self.Kyy)
            print(self.Kzz)
            

    def assemble_system(self, bc):
        # assemble the cFEM system
        
        # sanity check
        if len(bc) !=6:
            raise ValueError('bc dictionary must have 6 keys')
                
        # stiffness matrix
        K = sparse.lil_matrix((self.n_nodes, self.n_nodes))
        # rhs
        rhs = np.zeros(self.n_nodes)

        # loop over elements
        for iel in range(self.n_cells):
            icell,jcell,kcell = self.elem2ijk(iel)
            mat_prop = self.data.cond[icell,jcell,kcell]
            mat_siga = self.data.siga[icell,jcell,kcell]
            src_prop = self.data.qext[icell,jcell,kcell]
            vol = self.dx*self.dy*self.dz/8.

            # rhs[self.data.gn[iel,:]] += (src_prop * vol) * self.Q
                
            for i,ind_i in enumerate(self.data.gn[iel,:]):
                rhs[ind_i] += (src_prop * vol) * self.Q[i]
                for j,ind_j in enumerate(self.data.gn[iel,:]):
                    K[ind_i,ind_j] += mat_prop*(  self.dy*self.dz/self.dx*self.Kxx[i,j] \
                                                + self.dz*self.dx/self.dy*self.Kyy[i,j] \
                                                + self.dx*self.dy/self.dz*self.Kzz[i,j] )\
                        + mat_siga * vol * self.M[i,j]


        # apply bc
        node = lambda i,j,k: i + j*self.npts_x + k*self.npts_x*self.npts_y
        
        dir_nodes = []
        T_dir = []
        if bc['zmin']['type'] != 'refl':
            for i in range(self.npts_x):
                for j in range(self.npts_y):
                    dir_nodes.append( node(i,j,0) )
                    T_dir.append(bc['zmin']['value'])
        if bc['zmax']['type'] != 'refl':
            for i in range(self.npts_x):
                for j in range(self.npts_y):
                    dir_nodes.append( node(i,j,self.npts_z-1) )
                    T_dir.append(bc['zmax']['value'])
        if bc['ymin']['type'] != 'refl':
            for i in range(self.npts_x):
                for k in range(self.npts_z):
                    dir_nodes.append( node(i,0,k) )
                    T_dir.append(bc['ymin']['value'])
        if bc['ymax']['type'] != 'refl':
            for i in range(self.npts_x):
                for k in range(self.npts_z):
                    dir_nodes.append( node(i,self.npts_y-1,k) )
                    T_dir.append(bc['ymax']['value'])
        if bc['xmin']['type'] != 'refl':
            for j in range(self.npts_y):
                for k in range(self.npts_z):
                    dir_nodes.append( node(0,j,k) )
                    T_dir.append(bc['xmin']['value'])
        if bc['xmax']['type'] != 'refl':
            for j in range(self.npts_y):
                for k in range(self.npts_z):
                    dir_nodes.append( node(self.npts_x-1,j,k) )
                    T_dir.append(bc['xmax']['value'])
        dir_nodes = np.asarray(dir_nodes)
        # remove duplicate 8 corners
        # dir_nodes = np.unique(dir_nodes)

        for i,inode in enumerate(dir_nodes):
            K[inode,:]     = 0.
            K[inode,inode] = 1.
            rhs[inode]     = T_dir[i]

        return K, rhs

    
    def solve_system(self, A, b):
        solution = linalg.spsolve(A.tocsc(), b)

        return solution   