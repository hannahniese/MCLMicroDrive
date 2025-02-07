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
masterpath  =   r"C:\Users\Hannah Niese\Documents\GitHub\MCLMicroDrive"
file        =   '220608172447_data'
ftype       =   '.txt'
datafile    =   masterpath + '\\' + file + ftype
outpath     =   masterpath + '\\analysis'
if os.path.isdir(outpath) == False:
    os.mkdir(outpath)
    
    
data    = np.loadtxt(datafile, delimiter=',',  skiprows=3)


x = data[:,4]
y = data[:,5]
z = data[:,6]
I = data[:,7]*1000 # convert to mV

#%% plotting trajectories
dim=len(data)


plt.plot(data[:,1], data[:,2], color='green')
plt.scatter(data[:,4], data[:,5], s=7, color='blue')

plt.axis('equal')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('Acquired datapoints: %d' % dim) 

os.chdir(masterpath)
plt.savefig('%s_points.png' % file, dpi=600)



#%% plotting 3d image at real coordinates
plt.close()


ax = plt.axes(projection='3d')
ax.scatter(x,y,I, c=I, cmap='viridis', linewidth=0.3)
ax.set_xlabel('x-coordinate [mm]')
ax.set_ylabel('y-coordinate [mm]')
ax.set_zlabel('Intensity [mV]')
plt.show()

os.chdir(masterpath)
plt.savefig('%s_values.png' % file, dpi=600)

#%% plotting data as heatmap

plt.scatter(x,y, c=z, cmap='viridis', s=7)
plt.axis('equal')
plt.xlabel('x-coordinate [mm]')
plt.ylabel('y-coordinate [mm]')
plt.cbar()

os.chdir(outpath)
plt.savefig('%s_2D.png' % file, dpi=600)


#%% plot multiple measurements

ax = plt.axes(projection='3d')
ax.scatter(x,y,z, c=I, cmap='viridis', s=7, linewidth=0.1)
ax.set_xlabel('x-coordinate [mm]')
ax.set_ylabel('y-coordinate [mm]')
ax.set_zlabel('z-coordinate [mm]')

