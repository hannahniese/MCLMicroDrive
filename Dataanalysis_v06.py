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
import scipy as sc
import matplotlib.cm as cm
import matplotlib.tri as tri



### Importing E-field data
masterpath  =   r"C:\Users\Hannah Niese\Documents\GitHub\MCLMicroDrive\22_07_07_20x"
#masterpath  =  r"C:\Users\Congreve Optics\Desktop\Hannah\MCLMicroDrive\22_07_07_20x"
file        =   '22-07-07_10-14-38_data'
ftype       =   '.txt'
datafile    =   masterpath + '\\' + file + ftype
outpath     =   masterpath + '\\analysis'
if os.path.isdir(outpath) == False:
    os.mkdir(outpath)
    
data    = np.loadtxt(datafile, delimiter=',',  skiprows=1)


# import sideprofile
side_file   =   '22-07-07_10-27-23_data'
ftype       =   '.txt'
side_datafile    =   masterpath + '\\' + side_file + ftype


side_data    = np.loadtxt(side_datafile, delimiter=',',  skiprows=1)

#%% normal dataset

# side dataset for normalization
I_side = side_data[:,7]*1000

# calculate range of Intensitiy for normalization
delta_I     =   I_side.max()-I_side.min()

x = data[:,4]
y = data[:,5]
z = data[:,6]
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

plotgriddata(x, y, I_norm, profile=0, binsize=0.0027)

plotgriddata(y_side, z_side, I_side_norm, profile = 1, binsize=0.0027)

#%% try out interpolation onto grid



def gridinterpolation(x,y,I_norm):
    
    ngridx = ngridy = 0.0001     # Grid resolution
    z = I_norm
    
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    
    # -----------------------
    # Interpolation on a grid
    # -----------------------
    # A contour plot of irregularly spaced data coordinates
    # via interpolation on a grid.
    
    # Create grid values first.
    xi = np.arange(x.min(), x.max(), ngridx)
    yi = np.arange(y.min(), y.max(), ngridy)
    extent = (x.min(), x.max(), y.min(), y.max()) # extent of the plot
    
    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    #triang = tri.Triangulation(x, y)
    #interpolator = tri.LinearTriInterpolator(triang, z)
    #Xi, Yi = np.meshgrid(xi, yi)
    #zi = interpolator(Xi, Yi)
    
    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    from scipy.interpolate import griddata
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
    
    ax2.imshow(zi, extent=extent, origin='upper')
    ax1.contour(xi, yi, zi)
    cntr1 = ax1.contourf(xi, yi, zi, levels=7, cmap="RdBu_r", aspect='equal')
    
    fig.colorbar(cntr1, ax=ax1)
    ax1.plot(x, y, 'ko', ms=3)
    ax1.set()
    ax1.set_title('grid and contour (%d points, %d grid points)' %
                  (ngridx * ngridy))
    
    # ----------
    # Contour lines
    # ----------
    # Directly supply the unordered, irregularly spaced coordinates
    # to tricontour.
    
    #ax2.tricontour(x, y, z)
    #cntr2 = ax2.tricontourf(x, y, z, levels=7, cmap="RdBu_r")
    #
    #fig.colorbar(cntr2, ax=ax2)
    #ax2.plot(x, y, 'ko', ms=3)
    #ax2.set()
    #ax2.set_title('tricontour')
    
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    
    return

gridinterpolation(y_side, z_side, I_side_norm)

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


def griddata(x, y, z, binsize, retbin=True, retloc=True):
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
    2022-07-11  nieseh  Adaptation
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


def plotgriddata(x, y, I_norm, profile, binsize):
    '''    
    Function plots the interpolated (bilinear) image of the measurement area, 
    with the input binsize (should correspond to stepsize of measurement plus error)
    
    x:       first coordinate (x for x-y datasets, y for y-z datasets)
    y:       second coordinate (y for x-y datasets, z for y-z datasets)
    I_norm:  Intensity information, usually normalized to the largest value in y-z dataset
    binsize: Binning for the grid
    
    '''
    grid, bins, wherebin, xi, yi = griddata(x, y, I_norm, binsize, retbin=True, retloc=True)
    
    if profile == 0:
        Axis1 = 'X-values'
        Axis2 = 'Y-values'
        save = file
    else:
        Axis1 = 'X-values'
        Axis2 = 'Y-values'
        save = side_file + '_side'
        
    
    # minimum values for colorbar. filter our nans which are in the grid
    zmin    = 0 #grid[np.where(np.isnan(grid) == False)].min()
    zmax    = 1 #grid[np.where(np.isnan(grid) == False)].max()
    
    
    
    # colorbar stuff
    palette = 'viridis'#plt.matplotlib.colors.LinearSegmentedColormap('viridis',plt.cm.datad['viridis'],2048)
    #palette.set_under(alpha=0.0)
    
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    fig.suptitle(file)
    
    # plot the results.  first plot is x, y vs z, where z is a filled level plot.
    extent = (x.min(), x.max(), y.min(), y.max()) # extent of the plot
    axs[0].plot(1, 2, 1)
    pic = axs[0].imshow(grid, extent=extent, cmap=palette, origin='lower', vmin=zmin, vmax=zmax, aspect='equal', interpolation='nearest')
    axs[0].set_xlabel(Axis1)
    axs[0].set_ylabel(Axis2)
    axs[0].set_title('Normalized Intensity')
    plt.colorbar(pic, ax = axs[0])
    
    # now show the number of points in each bin.  since the independent data are
    # Gaussian distributed, we expect a 2D Gaussian.
    axs[1].plot(1, 2, 2)
    bins = axs[1].imshow(bins, extent=extent, cmap=palette, origin='lower', vmin=0, vmax=bins.max(), aspect='equal', interpolation='none')
    axs[1].set_xlabel(Axis1)
    axs[1].set_ylabel(Axis2)
    axs[1].set_title('No. of Pts Per Bin')
    plt.colorbar(bins, ax = axs[1])
    
    axs[2].plot(1, 2, 3)
    cont = axs[2].contour(xi, yi, grid)
    axs[2].clabel(cont, inline=True, fontsize=10)
    axs[2].set_title('Contour lines')
    
    os.chdir(outpath)
    plt.savefig('%s_binning.png' % save, dpi=600)

    return

#plt.imshow(grid)

#plotgriddata(x, y, I_norm, binsize=0.0015, 0)
