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
masterpath  =   r"C:\Users\Hannah Niese\Documents\GitHub\MCLMicroDrive\22_07_06_20x"
#masterpath  =  r"C:\Users\Congreve Optics\Desktop\Hannah\MCLMicroDrive\22_07_07_20x"
file        =   '22-07-06_16-50-45_data'
ftype       =   '.txt'
datafile    =   masterpath + '\\' + file + ftype
outpath     =   masterpath + '\\analysis'
if os.path.isdir(outpath) == False:
    os.mkdir(outpath)
    
data    = np.loadtxt(datafile, delimiter=',',  skiprows=1)


# import sideprofile
side_file   =   '22-07-06_21-03-25_data'
ftype       =   '.txt'
side_datafile    =   masterpath + '\\' + side_file + ftype


side_data    = np.loadtxt(side_datafile, delimiter=',',  skiprows=1)

#%% normal dataset

# side dataset for normalization
I_side = side_data[:,7]*1000

# calculate range of Intensitiy for normalization
delta_I     =   I_side.max()-I_side.min()

x = data[:,1]
y = data[:,2]
z = data[:,3]
I = data[:,7]*1000 # convert to mV


# normalized datasets
I_norm = (I - I.min())/delta_I # convert to mV


# side dataset
x_side = side_data[:,4]
y_side = side_data[:,5]
z_side = side_data[:,6]
I_side_norm = (I_side - I.min())/delta_I # convert to mV
I_side_limit = I_side

#for i in range(len(I_side_limit)):
#    if I_side[i] < 0.5:
#        I_side_limit[i] = 0
#    elif 0.55 > I_side[i] > 0.5:
#        I_side_limit[i] = 1
#    else:
#        I_side_limit[i] = 0



#%% plotting data as heatmap xy plane

plt.scatter(x, y, c=I_norm, cmap='viridis', marker="s", s=7, vmin=0, vmax=1)
plt.axis('equal')
plt.xlabel('x-coordinate [mm]')
plt.ylabel('y-coordinate [mm]')
plt.colorbar(label='Intensity')

os.chdir(outpath)
plt.savefig('%s_2D_norm.png' % file, dpi=600)

#%% plotting data as heatmap xz plane

plt.scatter(y_side, z_side, c=I_side_norm, cmap='viridis', marker="s", s=6, vmin=0, vmax=1)
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

#%% Meshgrid trial
import scipy as sc

def griddata(x, y, z, binsize=0.0015, retbin=True, retloc=True):
    """
    Place unevenly spaced 2D data on a grid by 2D binning (nearest
    neighbor interpolation).
    
    Parameters
    ----------
    x : ndarray (1D)
        The idependent data x-axis of the grid.
    y : ndarray (1D)
        The idependent data y-axis of the grid.
    z : ndarray (1D)
        The dependent data in the form z = f(x,y).
    binsize : scalar, optional
        The full width and height of each bin on the grid.  If each
        bin is a cube, then this is the x and y dimension.  This is
        the step in both directions, x and y. Defaults to 0.01.
    retbin : boolean, optional
        Function returns `bins` variable (see below for description)
        if set to True.  Defaults to True.
    retloc : boolean, optional
        Function returns `wherebins` variable (see below for description)
        if set to True.  Defaults to True.
   
    Returns
    -------
    grid : ndarray (2D)
        The evenly gridded data.  The value of each cell is the median
        value of the contents of the bin.
    bins : ndarray (2D)
        A grid the same shape as `grid`, except the value of each cell
        is the number of points in that bin.  Returns only if
        `retbin` is set to True.
    wherebin : list (2D)
        A 2D list the same shape as `grid` and `bins` where each cell
        contains the indicies of `z` which contain the values stored
        in the particular bin.

    Revisions
    ---------
    2010-07-11  ccampo  Initial version
    """
    # get extrema values.
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # make coordinate arrays.
    xi      = np.arange(xmin, xmax+binsize, binsize)
    yi      = np.arange(ymin, ymax+binsize, binsize)
    xi, yi  = np.meshgrid(xi,yi)

    # make the grid.
    grid           = np.zeros(xi.shape, dtype=x.dtype)
    nrow, ncol = grid.shape
    if retbin: bins = np.copy(grid)

    # create list in same shape as grid to store indices
    if retloc:
        wherebin = np.copy(grid)
        wherebin = wherebin.tolist()

    # fill in the grid.
    for row in range(nrow):
        for col in range(ncol):
            xc = xi[row, col]    # x coordinate.
            yc = yi[row, col]    # y coordinate.

            # find the position that xc and yc correspond to.
            posx = np.abs(x - xc)
            posy = np.abs(y - yc)
            ibin = np.logical_and(posx < binsize/2., posy < binsize/2.)
            ind  = np.where(ibin == True)[0]

            # fill the bin.
            bin = z[ibin]
            if retloc: wherebin[row][col] = ind
            if retbin: bins[row, col] = bin.size
            if bin.size != 0:
                binval         = np.median(bin)
                grid[row, col] = binval
            else:
                grid[row, col] = np.nan   # fill empty bins with nans.

    # return the grid
    if retbin:
        if retloc:
            return grid, bins, wherebin, xi, yi
        else:
            return grid, bins
    else:
        if retloc:
            return grid, wherebin
        else:
            return grid


grid, bins, wherebin, xi, yi = griddata(x, y, I, binsize=0.0015, retbin=True, retloc=True)

interpolatedgrid = sc.interpolate.interp2d(xi, yi, grid)


plt.imshow(interpolatedgrid)
