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
from scipy.interpolate import griddata


### Importing E-field data
masterpath  =   r"C:\Users\Hannah Niese\Documents\GitHub\MCLMicroDrive\22_07_18_20x_chess"
#masterpath  =  r"C:\Users\Congreve Optics\Desktop\Hannah\MCLMicroDrive\22_07_07_20x"
file        =   '22-07-18_15-01-49_data'
ftype       =   '.txt'
datafile    =   masterpath + '\\' + file + ftype
outpath     =   masterpath + '\\analysis'
if os.path.isdir(outpath) == False:
    os.mkdir(outpath)
    
data    = np.loadtxt(datafile, delimiter=',',  skiprows=1)


# import sideprofile
side_file   =   '22-07-18_15-38-47_data'
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
I_norm = (I - I.min())/delta_I**2 # convert to mV


# side dataset
x_side = side_data[:,4]
y_side = side_data[:,5]
z_side = side_data[:,6]
I_side_norm = (I_side - I.min())/delta_I**2 # convert to mV
I_side_limit = I_side

I_squared_max = I_side_norm.max()

#for i in range(len(I_side_limit)):
#    if I_side[i] < 0.5:
#        I_side_limit[i] = 0
#    elif 0.55 > I_side[i] > 0.5:
#        I_side_limit[i] = 1
#    else:
#        I_side_limit[i] = 0
#%%
binsize=0.0027
profile=0
plotgriddata(x, y, I_norm, profile, binsize, I_squared_max) 

profile=1
plotgriddata(y_side, z_side, I_side_norm, profile, binsize, I_squared_max)

#%% try out interpolation onto grid

def pointsongrid(x, y, I_norm):
    # input: takes coordinates and intensities at measured locations
    # output: coordinates and intensities on grid (x, y, Intensity)
    ngridx = ngridy = 0.0001     # Grid resolution
    z = I_norm
    
    # Create grid values
    xi = np.arange(x.min(), x.max(), ngridx)
    yi = np.arange(y.min(), y.max(), ngridy)
    extent = (x.min(), x.max(), y.min(), y.max()) # extent of the plot
    
    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).  
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
    
    return xi, yi, zi


def gridinterpolation(x,y,I_norm):
    
    ngridx = ngridy = 0.0001     # Grid resolution
    z = I_norm
    
    fig, (ax1) = plt.subplots(ncols=1, figsize=(5,4))
    
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
    
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
    
    ax1.imshow(zi, extent=extent, vmin=0, vmax=1, origin='lower')
    ax1.contour(xi, yi, zi, aspect='equal')
    cntr1 = ax1.contourf(xi, yi, zi, levels=10, cmap="RdBu_r", aspect='equal')
    
    ax2.contour(xi, yi, zi, aspect='equal')
    cntr1 = ax2.contourf(xi, yi, zi, levels=7, cmap="RdBu_r", aspect='equal')
    ax2.clabel(cntr1, inline=True, fontsize=10)
    
    fig.colorbar(cntr1, ax=ax2)
    ax2.plot(x, y, 'ko', ms=3)
    ax2.set()
    ax2.set_title('grid and contour (%d points, %d grid points)' %
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
    
    return zi, xi, yi

gridinterpolation(y_side, z_side, I_side_norm)
gridinterpolation(x, y, I_norm)

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


def gridondata(x, y, z, binsize, retbin=True, retloc=True):
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


def plotgriddata(x, y, I_norm, profile, binsize, I_side_max):
    '''    
    Function plots the interpolated (bilinear) image of the measurement area, 
    with the input binsize (should correspond to stepsize of measurement plus error)
    
    x:       first coordinate (x for x-y datasets, y for y-z datasets)
    y:       second coordinate (y for x-y datasets, z for y-z datasets)
    I_norm:  Intensity information, usually normalized to the largest value in y-z dataset
    binsize: Binning for the grid
    
    '''
    grid, bins, wherebin, xi, yi = gridondata(x, y, I_norm, binsize, retbin=True, retloc=True)
    
    
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
    zmax    = I_squared_max #grid[np.where(np.isnan(grid) == False)].max()
    
    
    
    # colorbar stuff
    palette = 'viridis'#plt.matplotlib.colors.LinearSegmentedColormap('viridis',plt.cm.datad['viridis'],2048)
    #palette.set_under(alpha=0.0)
    
    fig, axs = plt.subplots(1,2, figsize=(12,4))
    fig.suptitle(save)
    
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
    
#    axs[2].plot(1, 2, 3)
#    cont = axs[2].contour(xi, yi, grid, extent=extent, oritin='lower')
#    axs[2].clabel(cont, inline=True, fontsize=10)
#    axs[2].set_title('Contour lines')
    
    os.chdir(outpath)
    plt.savefig('%s_binning.png' % save, dpi=600)

    return

#plt.imshow(grid)

#plotgriddata(x, y, I_norm, binsize=0.0015, 0)

#%% fit
    
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x_grid, y_grid, Int = pointsongrid(x, y, I_norm)
# The two-dimensional domain of the fit.
#xmin, xmax, nx = x_grid.min(), x_grid.max(), 100
#ymin, ymax, ny = y_grid.min(), y_grid.max(), 100
#x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x_grid, y_grid)
Z = np.nan_to_num(Int)

# Plot the 3D figure of the fitted function and the residuals.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='plasma')
ax.set_zlim(0,np.max(Z)+0.2)
plt.show()

def get_basis(x, y, max_order=4):
    """Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc."""
    basis = []
    for i in range(max_order+1):
        for j in range(max_order - i +1):
            basis.append(x**j * y**i)
    return basis

# We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
x, y = X.ravel(), Y.ravel()
# Maximum order of polynomial term in the basis.
max_order = 8
basis = get_basis(x, y, max_order)
# Linear, least-squares fit.
A = np.vstack(basis).T
b = Z.ravel()
c, r, rank, s = np.linalg.lstsq(A, b, rcond=None)

print('Fitted parameters:')
print(c)

# Calculate the fitted surface from the coefficients, c.
fit = np.sum(c[:, None, None] * np.array(get_basis(X, Y, max_order))
                .reshape(len(basis), *X.shape), axis=0)

rms = np.sqrt(np.mean((Z - fit)**2))
print('RMS residual =', rms)

# Plot the 3D figure of the fitted function and the residuals.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, fit, cmap='viridis')
cset = ax.contourf(X, Y, Z-fit, zdir='z', offset=-4, cmap='viridis')
ax.set_zlim(-4,np.max(fit))
plt.show()

# Plot the test data as a 2D image and the fit as overlaid contours.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(Z, origin='lower', cmap='viridis',
          extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(X, Y, fit, colors='w')
plt.show()


#%%

# Our function to fit is going to be a sum of two-dimensional Gaussians
def gaussian(x, y, x0, y0, xalpha, yalpha, A):
    return A * np.exp( -((x-x0)/xalpha)**2 -((y-y0)/yalpha)**2)

# This is the callable that is passed to curve_fit. M is a (2,N) array
# where N is the total number of data points in Z, which will be ravelled
# to one dimension.
def _gaussian(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//5):
       arr += gaussian(x, y, *args[i*5:i*5+5])
    return arr

# Initial guesses to the fit parameters.
guess_prms = [(0, 0, 1, 1, 2),
              (-1.5, 5, 5, 1, 3),
              (-1, -1, 1.5, 1.5, 3),
              (-1, -1, 1.5, 1.5, 6.5)
             ]
# Flatten the initial guess parameter list.
p0 = [p for prms in guess_prms for p in prms]

# We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
xdata = np.vstack((X.ravel(), Y.ravel()))
# Do the fit, using our custom _gaussian function which understands our
# flattened (ravelled) ordering of the data points.
popt, pcov = curve_fit(_gaussian, xdata, Z.ravel(), p0)
fit = np.zeros(Z.shape)
for i in range(len(popt)//5):
    fit += gaussian(X, Y, *popt[i*5:i*5+5])
print('Fitted parameters:')
print(popt)

rms = np.sqrt(np.mean((Z - fit)**2))
print('RMS residual =', rms)

# Plot the 3D figure of the fitted function and the residuals.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, fit, cmap='plasma')
cset = ax.contourf(X, Y, Z-fit, zdir='z', offset=-4, cmap='plasma')
ax.set_zlim(np.min(fit),np.max(fit))
plt.show()

# Plot the test data as a 2D image and the fit as overlaid contours.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(Z, origin='bottom', cmap='plasma',
          extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(X, Y, fit, colors='w')
plt.show()