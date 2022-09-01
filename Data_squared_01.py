# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:15:40 2022

This script is used to add up the two squared images of two different checkerboard images. 

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
from scipy.optimize import curve_fit
import math


### Importing E-field data
masterpath   =   r"C:\Users\Hannah Niese\Documents\GitHub\MCLMicroDrive\Measurements\22_08_19_20x_newchessintsquared"
#masterpath  =  r"C:\Users\Congreve Optics\Desktop\Hannah\MCLMicroDrive\22_07_07_20x"
file1        =   '22-08-19_17-47-18_data'
ftype        =   '.txt'
datafile1    =   masterpath + '\\' + file1 + ftype
outpath      =   masterpath + '\\analysis'
if os.path.isdir(outpath) == False:
    os.mkdir(outpath)
    
data1        = np.loadtxt(datafile1, delimiter=',',  skiprows=1)


# import sideprofile
file2             =   '22-08-19_18-13-42_data'
ftype             =   '.txt'
datafile2         =    masterpath + '\\' + file2 + ftype


data2        = np.loadtxt(datafile2, delimiter=',',  skiprows=1)


#%% positive numbers dataset

# first dataset
x1      = data1[:,4]
y1      = data1[:,5]
z1      = data1[:,6]
I1      = data1[:,7]*1000 # convert to mV

# second dataset
x2      = data2[:,4]
y2      = data2[:,5]
z2      = data2[:,6]
I2      = data2[:,7]*1000 # convert to mV

# calculate range of Intensitiy for normalization
delta_I1     =   I1.max()-I1.min()
delta_I2     =   I2.max()-I2.min()
delta        =   max(delta_I1, delta_I2)
floor        =   min(I1.min(), I2.min())
# positive numbers datasets
I1_pos = (I1 - floor + 1)*10
I2_pos = (I2 - floor + 1)*10

# square both datasets
I1_squared      = np.square(I1_pos)
I2_squared      = np.square(I2_pos)
I_max           = (delta*10)**2

#%%

# yz plots
ymin = 1 
ymax = 2004

z = 1100


def pointsongrid(x, y, I):
    # input: takes coordinates and intensities at measured locations
    # output: coordinates and intensities on grid (x, y, Intensity)
    gridresolution = 0.0001     # Grid resolution
        
    # Create grid values
    xi = np.arange(x.min(), x.max(), gridresolution)
    yi = np.arange(y.min(), y.max(), gridresolution)
    
    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).  
    Ii = griddata((x, y), I, (xi[None, :], yi[:, None]), method='linear')
    
    return xi, yi, Ii, gridresolution


#%% plots measurement data on array

y1i, z1i, I1, gridres = pointsongrid(y1, z1, I1_squared)
y2i, z2i, I2, gridres = pointsongrid(y2, z2, I2_squared)

if 
I1 = np.delete(I1,0,0)  # manually make them the same dimension
#I2 = np.delete(I2,0,1)
Added = np.add(I1, I2)

plt.imshow(Added, cmap='viridis', extent=(y1i.min(), y1i.max(), z1i.min(), z1i.max()), origin='lower')
plt.axis('equal')
plt.xlabel('y-coordinate [mm]')
plt.ylabel('z-coordinate [mm]')
plt.colorbar(label='Intensity')

os.chdir(outpath)
#plt.savefig('%s_interpolated_norm.png' % file, dpi=600)

#%% plots measurement data on array

y1i, z1i, I1, gridres = pointsongrid(y1, z1, I1_squared)
y2i, z2i, I2, gridres = pointsongrid(y2, z2, I2_squared)
I2 = np.delete(I2,0,0)  # manually make them the same dimension
I2 = np.delete(I2,0,1)
Added = np.add(I1, I2)

plt.imshow(I2, cmap='viridis', extent=(y2i.min(), y2i.max(), z2i.min(), z2i.max()), origin='lower', vmin=1, vmax=I_max*1.3)
plt.axis('equal')
plt.xlabel('y-coordinate [mm]')
plt.ylabel('z-coordinate [mm]')
plt.colorbar(label='Intensity')

os.chdir(outpath)
#plt.savefig('%s_interpolated_norm.png' % file, dpi=600)

#%% create a function to model and create data
def func(x, a, x0, sigma):
	return a*np.exp(-(x-x0)**2/(2*sigma**2))

def coordfunc(a, x0, sigma, value):
    return x0 - 2*sigma * np.sqrt( - np.log(value/a))

# select range of data to be fitted
yg = yi[ymin:ymax]
Ig = Added[z][ymin:ymax]


# Plot out the current state of the data and model
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(yg, Ig, c='k', label='Function')
ax.scatter(yg, Ig)

# Executing curve_fit
popt, pcov  =   curve_fit(func, yg, Ig)
FWHM        =   np.abs(2*np.sqrt(2*np.log(2))*popt[2])
peak        =   func(popt[1], popt[0], popt[1], popt[2])
oesq        =   1/np.e**2 * np.abs(peak) 
coord       =   coordfunc(popt[0], popt[1], popt[2], oesq) 
w           =   popt[1] - coord

#popt returns the best fit values for parameters of the given model (func)
print (popt, FWHM)

Im = func(yg, popt[0], popt[1], popt[2])
ax.plot(yg, Im, c='r', label='Best fit')
ax.plot(popt[1], peak, 'o', color='orange')
ax.plot(coord, oesq, 'o', color='green')
ax.legend()
#fig.savefig('model_fit.png')

#%% plot several crosscuts through yz image

z = 0
for z in range(1080 , 2000, z + 100):
    # select range of data to be fitted
    yg = yi[ymin:ymax]
    Ig = Added[z][ymin:ymax]
    plt.plot(yg, Ig, label=z)

plt.legend()
    
#%% or

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = yi
y = np.arange(100)
X,Y = np.meshgrid(x,y)
Z = np.zeros((len(y),len(x)))


z = 0
for i in range(len(y)):
    Z[i] = Added[z]
    z += 20

ax.plot_surface(X, Y, Z, rstride=1, cmap=cm.coolwarm, cstride=1000, shade=True, lw=.5)

#ax.set_zlim(0, 5)
#ax.set_xlim(-51, 51)
#ax.set_zlabel("Intensity")
#ax.view_init(20,-120)
plt.show()

#%% gaussian fits for all z values

yi, zi, Ii, gridres = pointsongrid(y_side, z_side, I_side_squared)

FWHM = np.zeros(len(zi))
w = np.zeros(len(zi))
peak = np.zeros(len(zi))
oesq = np.zeros(len(zi))
yg = yi[ymin:ymax]
e = np.e

for z in range(270 , 1250):
    Ig          =   Ii[z][ymin:ymax]
    popt, pcov  =   curve_fit(func, yg, Ig)                             # fit
    FWHM[z]     =   np.abs(2*np.sqrt(2*np.log(2))*popt[2])
    peak[z]     =   func(popt[1], popt[0], popt[1], popt[2])
    oesq[z]     =   1/(e**2) * peak[z] 
    coord       =   coordfunc(popt[0], popt[1], popt[2], oesq[z]) 
    w[z]        =   abs(popt[1]-coord)
    

plt.plot(w, 'o')

wfit = w[270 : 1250]
zfit = zi[270 : 1250]
#%% Fit 
def waist(w0, z):
    return w0 * np.sqrt(1 + ((0.000625*z)/(np.pi*w0**2))**2)

# Plot out the current state of the data and model
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(zfit, wfit, c='k', label='Waist')
ax.scatter(zfit, wfit)

# Executing curve_fit
popt, pcov  =   curve_fit(waist, zfit, wfit, bounds=(-1.97, -0.102))                                 # fit
Im = waist(popt[0], zfit)
ax.plot(zfit, Im, c='r', label='Best fit')

#%%
def gridinterpolation(x,y,I):
    
    ngridx = ngridy = 0.0001     # Grid resolution
    z = I
    
    #fig, (ax1) = plt.subplots(ncols=1, figsize=(5,4))
    
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
    
#    ax1.imshow(zi, extent=extent, vmin=I.min(), vmax=I.max(), origin='lower')
#    ax1.contour(xi, yi, zi, aspect='equal')
#    cntr1 = ax1.contourf(xi, yi, zi, levels=6, cmap="RdBu_r", aspect='equal')
#    ax1.cbar()
#    
#    ax2.contour(xi, yi, zi, aspect='equal')
#    cntr1 = ax2.contourf(xi, yi, zi, levels=7, cmap="RdBu_r", aspect='equal')
#    ax2.clabel(cntr1, inline=True, fontsize=10)
#    
#    fig.colorbar(cntr1, ax=ax2)
#    ax2.plot(x, y, 'ko', ms=3)
#    ax2.set()
#    ax2.set_title('grid and contour (%d points, %d grid points)' %
#                  (ngridx * ngridy))
#    
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
    
#    plt.subplots_adjust(hspace=0.5)
#    plt.show()
    
    return xi, yi, zi

gridinterpolation(y_side, z_side, I_side)
gridinterpolation(x, y, I)

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


x_grid, y_grid, Int = pointsongrid(x, y, I)
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