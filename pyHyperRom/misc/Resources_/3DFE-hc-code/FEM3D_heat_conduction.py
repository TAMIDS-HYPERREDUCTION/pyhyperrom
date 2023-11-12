# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:30:57 2023

@author: jean.ragusa
"""
#%% import python modules
import numpy as np
import time as time
import matplotlib.pyplot as plt
plt.close('all')

#%% import code specific modules
from my_data import mydata
from my_solver import FEM_solver

t_init = time.time()
#%%
# data
# set siga to 0 to do heat conduction
siga = .3*np.ones((4,4,4))
cond = 1*np.ones((4,4,4))
# cond[1,0] =1.
# cond[2,2] =1000.
# cond[0,1] =1.

qext = np.zeros_like(cond) +0.0
# qext[0,0] = 100.
# qext[3,0] = 50.
qext[0,0,0] = 2
qext[1,1,1] = 2
qext[2,2,2] = 2
qext[3,3,3] = 2

#%% create data and solver objects
d = mydata(cond, siga, qext, [7,8,9], [10.,12.,14.])
solver = FEM_solver(d)

bc = {}
bc['xmin']={'type':'dirichlet','value':0.}
bc['xmax']={'type':'dirichlet','value':0.}
bc['ymin']={'type':'dirichlet','value':0.}
bc['ymax']={'type':'dirichlet','value':0.}
bc['zmin']={'type':'dirichlet','value':0.}
bc['zmax']={'type':'dirichlet','value':0.}

#%% solve
t0 = time.time()
A,b = solver.assemble_system(bc)
t1 = time.time()
solution = solver.solve_system(A, b)
t2 = time.time()
print("Assembly time = ", t1-t0)
print("Solution time = ", t2-t1)

TT = np.reshape(solution,(d.npts_z,d.npts_y,d.npts_x)).T
print("reshaped solution array: ",TT.shape)
# print(d.x.shape,d.y.shape,d.z.shape)

# some debugging to make sure shape is right
# plt.figure()
# i,j,k=TT.shape
# i = int((i-1)/2)
# j = int((j-1)/2)
# k = int((k-1)/2)
# plt.plot(d.x, TT[:,j,k],'-o',label='x')
# # plt.figure()
# plt.plot(d.y, TT[i,:,k],'-+',label='y')
# # plt.figure()
# plt.plot(d.z, TT[i,j,:],'-s',label='z')
# plt.legend()

#%% plot sparsity pattern
plot_sparsity = False
if plot_sparsity:
    plt.figure()
    plt.spy(A)

#%% export to vtk
export_to_vtk = True
if export_to_vtk:
    from pyevtk.hl import gridToVTK
    gridToVTK("./structured", d.x, d.y, d.z, \
              cellData = {"cond" : d.cond, "qext" : d.qext, "siga" : d.siga},\
              pointData = {"temp" : TT})

#%% plot in browser using plotly
plot_browser = False
if plot_browser == True:
    import plotly.graph_objects as go
    import plotly.io as pio
    # pio.renderers
    pio.renderers.default = "browser"
    # material layout fig
    fig1 = go.Figure()
    min_ = np.min(TT)
    max_ = np.max(TT)
    color_scale='hot'
    my_title=''
    
    # select the x,y,z plane coordinates for plotting
    plane_list = []
    # plane_list.append( [np.mean(d.x)] )
    plane_list.append( [2,4,6,8] )
    plane_list.append( [] )
    plane_list.append( [2,5,8] )
    
    
    
    # xx and yy have the same shapes
    xx, yy = np.meshgrid(d.x, d.y)
    # print('shape xx and yy =',xx.shape,yy.shape)
    # print('extend =',np.min(xx),np.max(xx),np.min(yy),np.max(yy))
    for z_ in plane_list[2]:
        idx = (np.abs(d.z - z_)).argmin()
        # print('index max =',idx)
        print('vert z=',d.z[idx])
        # create constant plane of the same shape as xx and yy
        zz = z_*np.ones(xx.shape)
        myslice = TT[:,:,idx].T
        # print(myslice.shape)
        
        fig1.add_trace(go.Surface(x = xx, y = yy, z = zz, colorscale = color_scale,\
                                  cmin = min_, cmax = max_, \
                                  surfacecolor = myslice))
    
    # xx and yy have the same shapes
    xx, zz = np.meshgrid(d.x, d.z)
    # print('shape xx and zz =',xx.shape,zz.shape)
    # print('extend =',np.min(xx),np.max(xx),np.min(zz),np.max(zz))
    for y_ in plane_list[1]:
        idx = (np.abs(d.y - y_)).argmin()
        # print('index max =',idx)
        print('vert y=',d.y[idx])
        # create constant plane of the same shape as xx and yy
        yy = y_*np.ones(xx.shape)
        myslice = TT[:,idx,:].T
        # print(myslice.shape)
        
        fig1.add_trace(go.Surface(x = xx, y = yy, z = zz, colorscale = color_scale,\
                                  cmin = min_, cmax = max_, \
                                  surfacecolor = myslice))
    
    # xx and yy have the same shapes
    yy, zz = np.meshgrid(d.y, d.z)
    # print('shape yy and zz =',yy.shape,zz.shape)
    # print('extend =',np.min(yy),np.max(yy),np.min(zz),np.max(zz))
    for x_ in plane_list[0]:
        idx = (np.abs(d.x - x_)).argmin()
        # print('index max =',idx)
        print('vert x=',d.x[idx])
        # create constant plane of the same shape as xx and yy
        xx = x_*np.ones(yy.shape)
        myslice = TT[idx,:,:].T
        # print(myslice.shape)
        
        fig1.add_trace(go.Surface(x = xx, y = yy, z = zz, colorscale = color_scale,\
                                  cmin = min_, cmax = max_, \
                                  surfacecolor = myslice))
    
    
    fig1.update_layout(title=my_title,
                       scene = dict(
                           xaxis_title='X ',
                           yaxis_title='Y ',
                           zaxis_title='Z '),
    # width=700,
    margin=dict(l=65, r=50, b=65, t=90))
    fig1.show()

t_final = time.time()
print("Total time = ", t_final-t_init)

"""
# loop over dimensions
for dim0 in range(3):
    dim1,dim2 = dim0+1,dim0+2
    if dim1>2:
        dim1 -=3
    if dim2>2:
        dim2 -=3
    print('')
    print('dims=',dim0,dim1,dim2)
    
    # xx and yy have the same shapes
    xx, yy = np.meshgrid(vert[dim0], vert[dim1])
    print('shape xx and yy =',xx.shape,yy.shape)
    print('extend =',np.min(xx),np.max(xx),np.min(yy),np.max(yy))
    for z_ in plane_list[dim2]:
        idx = (np.abs(vert[dim2] - z_)).argmin()
        print('index max dim2=',idx)
        print('vert=',vert[dim2][idx])
        # create constant plane of the same shape as xx and yy
        zz = z_*np.ones(xx.shape)
        if dim2==2:
            myslice = TT[:,:,idx]
        if dim2==1:
            myslice = TT[:,idx,:].T
        if dim2==0:
            myslice = TT[idx,:,:]
        print(myslice.shape)
        # print(np.rollaxis(TT,-dim1)[:,:,idx].shape)
        # print(np.rollaxis(TT,-dim2)[:,:,idx].shape)
        
        fig1.add_trace(go.Surface(x = xx, y = yy, z = zz, colorscale = color_scale,\
                                  cmin = min_, cmax = max_, \
                                  surfacecolor = myslice))

fig1.update_layout(title=my_title,
                   scene = dict(
                       xaxis_title='X ',
                       yaxis_title='Y ',
                       zaxis_title='Z '),
# width=700,
margin=dict(l=65, r=50, b=65, t=90))
fig1.show()
"""