# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:15:40 2022

@author: Hannah Niese
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import os


### Importing E-field data
masterpath  =   r"C:\Users\Hannah Niese\Documents\GitHub\MCLMicroDrive"
file        =   '220606141302_data'
ftype       =   '.txt'
datafile    =   masterpath + '\\' + file + ftype
outpath     =   masterpath + '\\analysis'
if os.path.isdir(outpath) == False:
    os.mkdir(outpath)
    
    
data    = np.loadtxt(datafile, delimiter=',',  skiprows=1)

#%% plotting trajectories
dim=len(data)

plt.plot(data[:,1], data[:,2], color='green')
plt.scatter(data[:,4], data[:,5], s=7, color='blue')

plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('Acquired datapoints: %d' % dim) 

os.chdir(masterpath)
plt.savefig('%s_points.png' % file, dpi=600)



#%% plotting 3d image at real coordinates
plt.close()

x = data[:,4]
y = data[:,5]
z = data[:,7]*1000 # convert to mV

ax = plt.axes(projection='3d')
ax.scatter(x,y,z, c=z, cmap='viridis', linewidth=0.5)
ax.set_xlabel('x-coordinate')
ax.set_ylabel('y-coordinate')
ax.set_zlabel('Intensity [mV]')
plt.show()

os.chdir(masterpath)
plt.savefig('%s_values.png' % file, dpi=600)

#%% plotting data as heatmap


plt.imshow(z)