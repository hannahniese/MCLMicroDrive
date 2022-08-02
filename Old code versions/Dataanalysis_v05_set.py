# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:15:40 2022

@author: Hannah Niese
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import constants
import os



### Importing E-field data
#masterpath  =   r"C:\Users\Hannah Niese\Documents\GitHub\MCLMicroDrive\22_06_20_50x"
masterpath  =  r"C:\Users\Congreve Optics\Desktop\Hannah\MCLMicroDrive\22_07_06_20x"
file        =   '22-07-06_14-46-50_data'
ftype       =   '.txt'
datafile    =   masterpath + '\\' + file + ftype
outpath     =   masterpath + '\\analysis'
if os.path.isdir(outpath) == False:
    os.mkdir(outpath)
    
data    = np.loadtxt(datafile, delimiter=',',  skiprows=1)


# import sideprofile
side_file   =   '22-07-06_14-46-50_data'
ftype       =   '.txt'
side_datafile    =   masterpath + '\\' + side_file + ftype


side_data    = np.loadtxt(side_datafile, delimiter=',',  skiprows=1)

#%% normal dataset

x = data[:,4]
y = data[:,5]
z = data[:,6]
I = data[:,7]*1000 # convert to mV

# calculate range of Intensitiy for normalization
delta_I     =   I.max()-I.min()

# first dataset
x1 = data[0:6480,4]
y1 = data[0:6480,5]
z1 = data[0:6480,6]
I1 = (data[0:6480,7]*1000 - I.min())/delta_I # convert to mV

# second dataset
x2 = data[6480:12960,4]
y2 = data[6480:12960,5]
z2 = data[6480:12960,6]
I2 = (data[6480:12960,7]*1000 - I.min())/delta_I # convert to mV

# third dataset
x3 = data[12960:19441,4]
y3 = data[12960:19441,5]
z3 = data[12960:19441,6]
I3 = (data[12960:19441,7]*1000 - I.min())/delta_I # convert to mV

# side dataset
x_side = side_data[:,4]
y_side = side_data[:,5]
z_side = side_data[:,6]
I_side = (side_data[:,7]*1000 - I.min())/delta_I # convert to mV
I_side_limit = I_side

#for i in range(len(I_side_limit)):
#    if I_side[i] < 0.5:
#        I_side_limit[i] = 0
#    elif 0.55 > I_side[i] > 0.5:
#        I_side_limit[i] = 1
#    else:
#        I_side_limit[i] = 0


#%% create subplot

fig, (ax1, ax2, ax3)     =   plt.subplots(1,3, figsize=(10.5,3), sharey=True, frameon=False)    
a = ax1.scatter(y1, -x1, c=I1, cmap='viridis', marker="s", s=3, vmin=0, vmax=1)
b = ax2.scatter(y2, -x2, c=I2, cmap='viridis', marker="s", s=3, vmin=0, vmax=1)
c = ax3.scatter(y3, -x3, c=I3, cmap='viridis', marker="s", s=3, vmin=0, vmax=1)
plt.colorbar(b, ax=fig)

ax1.axis('equal')
ax2.axis('equal')
ax3.axis('equal')



ax3.xlabel('y-coordinate [mm]')
ax3.ylabel('x-coordinate [mm]')

#%% plotting data as heatmap xy plane

plt.scatter(y_side, -x_side, c=I_side, cmap='viridis', marker="s", s=8, vmin=0, vmax=1)
plt.axis('equal')
plt.xlabel('y-coordinate [mm]')
plt.ylabel('x-coordinate [mm]')
plt.colorbar(label='Intensity')

os.chdir(outpath)
plt.savefig('%s_2D_norm3.png' % file, dpi=600)

#%% plotting data as heatmap xz plane

plt.scatter(y_side, -z_side, c=I_side, cmap='viridis', marker="s", s=28, vmin=0, vmax=1)
plt.axis('equal')
plt.xlabel('y-coordinate [mm]')
plt.ylabel('z-coordinate [mm]')
plt.colorbar(label='Intensity')

os.chdir(outpath)
plt.savefig('%s_2D_norm_side.png' % file, dpi=600)


#%% plot multiple measurements xy plane

ax = plt.axes(projection='3d')
ax.scatter(x,y,z, c=I, cmap='viridis', s=7, linewidth=0.1)
ax.set_xlabel('x-coordinate [mm]')
ax.set_ylabel('y-coordinate [mm]')
ax.set_zlabel('z-coordinate [mm]')

#%% plotting data as heatmap yz plane

plt.scatter(y, z, c=I, cmap='viridis', marker="s", s=8, vmin=-9.5, vmax=-2.5)
plt.axis('equal')
plt.xlabel('y-coordinate [mm]')
plt.ylabel('z-coordinate [mm]')
plt.colorbar()

os.chdir(outpath)
plt.savefig('%s_2D_zplane.png' % file, dpi=600)


#%% plotting trajectories
dim=len(data)


plt.plot(data[:,1], data[:,2], color='green')
plt.scatter(data[:,4], data[:,5], s=7, color='blue')

plt.axis('equal')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('Acquired datapoints: %d' % dim) 

os.chdir(outpath)
plt.savefig('%s_points_1.png' % file, dpi=600)

#%% plotting 3d image at real coordinates
plt.close()


ax = plt.axes(projection='3d')
ax.scatter(x,y,I, c=I, cmap='viridis', linewidth=0.3, vmin=-9.5, vmax=-2)
ax.set_xlabel('x-coordinate [mm]')
ax.set_ylabel('y-coordinate [mm]')
ax.set_zlabel('Intensity [mV]')
plt.show()

os.chdir(outpath)
plt.savefig('%s_values_1.png' % file, dpi=600)