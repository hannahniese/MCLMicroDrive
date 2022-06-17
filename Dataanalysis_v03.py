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


#%% import multiple plots
# close all open plot windows
plt.close('all')

Directory = 'C:\Users\Hannah Niese\Documents\GitHub\MCLMicroDrive'
#Directory = 'C:\Users\Congreve Optics\Documents\GitHub\MCLMicroDrive'
Files = listdir(Directory)

Files = [(Directory + scanfile) for scanfile in Files if '220608' in scanfile and 'data' in scanfile]
Nf = len(Files)


def plotplane(datafile):
    
    data = ImportData(DataFile)                      # measured data
    
    x = data[:,4]                                    # assign data
    y = data[:,5]
    z = data[:,6]
    I = data[:,7]*1000 # convert to mV
    
    plt.scatter(x,y, c=z, cmap='viridis', s=7)
    plt.axis('equal')
    plt.xlabel('x-coordinate [mm]')
    plt.ylabel('y-coordinate [mm]')
    plt.cbar()
    
    os.chdir(outpath)
    plt.savefig('%s_2D.png' % file, dpi=600)



for i in range(0,Nf):
    Res1 = plotplane(Files[i])




#%% manual stuff

### Importing E-field data
masterpath  =   r"C:\Users\Hannah Niese\Documents\GitHub\MCLMicroDrive\22_06_16_10x"
#masterpath  =  r"C:\Users\Congreve Optics\Documents\GitHub\MCLMicroDrive"
file        =   '220616150754_data'
ftype       =   '.txt'
datafile    =   masterpath + '\\' + file + ftype
outpath     =   masterpath + '\\analysis'
if os.path.isdir(outpath) == False:
    os.mkdir(outpath)
    
    
data    = np.loadtxt(datafile, delimiter=',',  skiprows=1)


#%% normal dataset

x = data[:,4]
y = data[:,5]
z = data[:,6]
I = data[:,7]*1000 # convert to mV

#%%

x = data[0:6480,4]
y = data[0:6480,5]
z = data[0:6480,6]
I = data[0:6480,7]*1000 # convert to mV

#%%
x = data[6480:12960,4]
y = data[6480:12960,5]
z = data[6480:12960,6]
I = data[6480:12960,7]*1000 # convert to mV

#%%
x = data[12960:19441,4]
y = data[12960:19441,5]
z = data[12960:19441,6]
I = data[12960:19441,7]*1000 # convert to mV

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

#%% plotting data as heatmap

plt.scatter(x, y, c=I, cmap='viridis', marker="s", s=8, vmin=-9.5, vmax=-2.5)
plt.axis('equal')
plt.xlabel('x-coordinate [mm]')
plt.ylabel('y-coordinate [mm]')
plt.colorbar()

os.chdir(outpath)
plt.savefig('%s_2D_focus3.png' % file, dpi=600)


#%% plot multiple measurements

ax = plt.axes(projection='3d')
ax.scatter(x,y,z, c=I, cmap='viridis', s=7, linewidth=0.1)
ax.set_xlabel('x-coordinate [mm]')
ax.set_ylabel('y-coordinate [mm]')
ax.set_zlabel('z-coordinate [mm]')

#%% plotting data as heatmap xz

plt.scatter(y, -z, c=I, cmap='viridis', marker="s", s=30, vmin=-9.5, vmax=-2.5)
plt.axis('equal')
plt.xlabel('x-coordinate [mm]')
plt.ylabel('z-coordinate [mm]')
plt.colorbar()

os.chdir(outpath)
plt.savefig('%s_2D_test.png' % file, dpi=600)
