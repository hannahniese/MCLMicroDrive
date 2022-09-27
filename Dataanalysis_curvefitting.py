# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:15:40 2022

This script analzyes the measurement data of intensity measurements with a scanned
photodiode. 
The data is exported as .txt file in the Multimeter script. 

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
import scipy.integrate as integrate
from scipy.integrate import quad
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show


### Import Measurement Data

# Two measurements of the same 

#masterpath  =   r"C:\Users\Hannah Niese\Documents\GitHub\MCLMicroDrive\Measurements\22_07_07_20x"
masterpath  =  r"C:\Users\Hannah Niese\Documents\GitHub\MCLMicroDrive\Measurements\22_07_25_20x_intsquared"
file        =   '22-07-25_16-28-16_data'
ftype       =   '.txt'
datafile    =   masterpath + '\\' + file + ftype
outpath     =   masterpath + '\\analysis'
if os.path.isdir(outpath) == False:
    os.mkdir(outpath)
    
data    = np.loadtxt(datafile, delimiter=',',  skiprows=1)


# import sideprofile
side_file   =   '22-07-25_14-47-33_data'
ftype       =   '.txt'
side_datafile    =   masterpath + '\\' + side_file + ftype


side_data    = np.loadtxt(side_datafile, delimiter=',',  skiprows=1)


# initialize function for grid interpolation
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

#gaussian
def func(x, a, x0, sigma):
	 return a*np.exp(-(x-x0)**2/(4*sigma**2))
 
def func3D(x, y, a, x0, y0, sigma):
	 return a*np.exp(-(x-x0)**2/(4*sigma**2)+(y-y0)**2/(4*sigma**2))

def coordfunc(a, x0, sigma, value):
    return x0 - 2*sigma * np.sqrt(np.log(a/value))

def FWHMtoSigma(FWHM):
    return FWHM / (2 * np.sqrt(2 * np.ln(2)))

def SigmatoFWHM(sigma):
    return sigma * 2 * np.sqrt(2 * np.ln(2))

# Voigt 
def Voigt(x, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1):
    return (ampG1*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cenG1)**2)/((2*sigmaG1)**2)))) +\
              ((ampL1*widL1**2/((x-cenL1)**2+widL1**2)) )


#%% normal dataset

# side dataset for normalization
I_side = side_data[:,7]*1000

# calculate range of Intensitiy for normalization
delta_I     =   np.abs(I_side.max()-I_side.min())

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

I_squared_max = I_side_norm.max()

#%% positive numbers dataset

# side dataset for normalization
I_side = side_data[:,7]*1000

# calculate range of Intensitiy for normalization
delta_I     =   I_side.max()-I_side.min()

x = data[:,4]
y = data[:,5]
z = data[:,6]
I = data[:,7]*1000 # convert to mV


# normalized datasets
I_pos = I - I_side.min() # convert to mV


# side dataset
x_side = side_data[:,4]
y_side = side_data[:,5]
z_side = side_data[:,6]
I_side_pos = I_side - I_side.min() # convert to mV


I_squared       = np.square(I_pos)
I_side_squared  = np.square(I_side_pos)
#%%


#%%

yi, zi, Ii, gridres = pointsongrid(y_side, z_side, I_side_pos)
#yi, zi, Ii, gridres = pointsongrid(y_side, z_side, I_side_norm)

plt.imshow(Ii, cmap='viridis', extent=(yi.min(), yi.max(), zi.min(), zi.max()), origin='lower', vmin=0, vmax=2)
plt.axis('equal')
plt.xlabel('x-coordinate [mm]')
plt.ylabel('y-coordinate [mm]')
plt.colorbar(label='Intensity')

os.chdir(outpath)
#plt.savefig('%s_interpolated_norm.png' % file, dpi=600)

#%%

# yz plots
ymin = 200 
ymax = 1900

z = 300


# fit gaussian through line plots
yi, zi, Ii, gridres = pointsongrid(y_side, z_side, I_side_pos)

#%% create a function to model and create data


# select range of data to be fitted
yg = yi[ymin:ymax]
Ig = Ii[z][ymin:ymax]


# Plot out the current state of the data and model
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(yg, Ig, c='k', label='Function')
ax.scatter(yg, Ig)

# Executing curve_fit
popt, pcov  =   curve_fit(func, yg, Ig, p0=[0.3, -1.77, -0.004])
sigma       =   abs(popt[2])
coordSigma  =   coordfunc(popt[0], popt[1], popt[2], sigma)
FWHM        =   np.abs(2.355*popt[2])
coordFWHM   =   popt[1] + FWHM/2
vFWHM       =   func(coordFWHM, popt[0], popt[1], popt[2])   
peak        =   func(popt[1], popt[0], popt[1], popt[2]) 
oesq        =   1/np.e**2 * np.abs(peak) 
coord       =   coordfunc(popt[0], popt[1], popt[2], oesq) 
w           =   coord


#popt returns the best fit values for parameters of the given model (func)
print (popt, FWHM)

Im = func(yg, popt[0], popt[1], popt[2])

area        =   quad(func, -np.inf, np.inf, args=(popt[0], popt[1], popt[2]))

ax.plot(yg, Im, c='r', label='Best fit')
ax.plot(popt[1], peak, 'o', color='orange')
ax.plot(coordSigma, sigma, 'o', color='green')
ax.legend()
#fig.savefig('model_fit.png')


#%% gaussian fits for all z values

# range, adjust manually depending on data
r_start =    300
r_stop  =    1650

yi, zi, Ii, gridres = pointsongrid(y_side, z_side, I_side_pos)

FWHM         =    np.zeros(len(zi))
vFWHM        =    np.zeros(len(zi))
coordFWHM    =    np.zeros(len(zi))
w            =    np.zeros(len(zi))
x0           =    np.zeros(len(zi))
a            =    np.zeros(len(zi))
peak         =    np.zeros(len(zi))
oesq         =    np.zeros(len(zi))
sigma        =    np.zeros(len(zi))
coordsig     =    np.zeros(len(zi))
area         =    np.zeros(len(zi))
adjarea      =    np.zeros(len(zi))
corrfact     =    np.zeros(len(zi))
adjmaxima    =    np.zeros(len(zi))

yg = yi[ymin:ymax]
e = np.e

for z in range(r_start , r_stop):
    Ig              =   Ii[z][ymin:ymax]
    popt, pcov      =   curve_fit(func, yg, Ig, p0=[0.3, -1.77, -0.004])                             # fit
    FWHM[z]         =   np.abs(2*np.sqrt(2*np.log(2))*popt[2])
    a[z]            =   popt[0]
    x0[z]           =   popt[1]
    coordFWHM[z]    =   popt[1] + FWHM[z]/2
    vFWHM[z]        =   func(coordFWHM[z], popt[0], popt[1], popt[2]) 
    peak[z]         =   func(popt[1], popt[0], popt[1], popt[2])
    oesq[z]         =   1/(e**2) * peak[z] 
    sigma[z]        =   abs(popt[2])
    coordsig[z]     =   coordfunc(popt[0], popt[1], popt[2], sigma[z]) 
    coord           =   coordfunc(popt[0], popt[1], popt[2], oesq[z]) 
    w[z]            =   abs(popt[1]-coord)
    area[z]         =   quad(func, -5, 0, args=(popt[0], popt[1], popt[2]))[0]        # calculate the area underneath the fitted curve
    corrfact[z]     =   1/area[z]                                                     # calculate a factor that normalizes the area to the value of 1
    adjmaxima[z]    =   np.multiply(corrfact[z], a[z])                                      # 'normalize' the maxima by multiplying with the correction factor
    adjarea[z]      =   quad(func, -5, 0, args=(adjmaxima[z], popt[1], popt[2]))[0] 

wfit = sigma[r_start : r_stop]              # crop data to avoid overfitting
afit = adjmaxima[r_start : r_stop]          # crop data to avoid overfitting
zfit = zi[r_start : r_stop]                 # crop data to avoid overfitting
   
plt.figure(1)
plt.plot(sigma, zi, 'o')
plt.title('Sigma')
plt.ylabel('z-coordinate')
plt.xlabel('a.u.')

plt.figure(2)
plt.plot(a, zi, 'o', label='Maximum')
plt.title('Maximum')
plt.ylabel('z-coordinate')
plt.xlabel('a.u.')


plt.figure(3)
plt.plot(area, zi, 'o', label='Area')
plt.title('Area')
plt.ylabel('z-coordinate')
plt.xlabel('a.u.')
#plt.legend()

plt.figure(4)
plt.plot(adjmaxima, zi, 'o', label='Adjusted Maxima')
plt.title('Adjusted Maxima')
plt.ylabel('z-coordinate')
plt.xlabel('a.u.')

plt.figure(5)
plt.plot(adjarea, zi, 'o', label='Adjusted Area')
plt.title('Adjusted Area')
plt.ylabel('z-coordinate')
plt.xlabel('a.u.')

#%% Fit Hyperbola to FWHM data
def waist(z, a, b, c, d):
    return b * np.sqrt(1 + (z-c)**2/a**2) + d

# Plot out the current state of the data and model
fig     =   plt.figure()
ax      =   fig.add_subplot(111)
#ax.plot(zfit, wfit, c='k', label='Waist')
ax.scatter(wfit, zfit)


# Executing curve_fit
popt_sig, pcov_sig  =   curve_fit(waist, zfit, wfit, p0=[-0.05, 0.01, -0.6, 0.008])                      # fit,  initial guess p0
Im          =   waist(zfit, popt_sig[0], popt_sig[1], popt_sig[2], popt_sig[3])
ax.plot(Im, zfit, c='r', label='Best fit')

a_sig = popt_sig[0]
b_sig = popt_sig[1]
c_sig = popt_sig[2]
d_sig = popt_sig[3]

#%% Fit Gaussian to peak function

# Plot out the current state of the data and model
fig     =   plt.figure()
ax      =   fig.add_subplot(111)
#ax.plot(zfit, wfit, c='k', label='Waist')
ax.scatter(afit, zfit)

## Executing curve_fit Voigt 
#popt, pcov  =   curve_fit(Voigt, zfit, afit)                      # fit,  initial guess p0
#amp         =   Voigt(zfit, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
#ax.plot(zfit, amp, c='r', label='Best fit')
#ax.set_title('Voigt')

# Executing curve_fit Gaussian
popt_peak, pcov_peak  =   curve_fit(func, zfit, afit)                      # fit,  initial guess p0
amp         =   func(zfit, popt_peak[0], popt_peak[1], popt_peak[2])

amp_peak = popt_peak[0]
mid_peak = popt_peak[1]
sig_peak = popt_peak[2]

#residues gaussian
#err_func = np.sqrt(np.diag(pcov))



ax.plot(amp, zfit, c='r', label='Best fit')
ax.set_title('Gauss')



#%% plotting all fits together

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13,3), gridspec_kw={'width_ratios': [2, 1, 1]}, ncols=3)

overview = ax1.imshow(Ii, cmap='viridis', extent=(yi.min(), yi.max(), zi.min(), zi.max()), origin='lower', vmin=0, vmax=1)
ax1.axis('equal')
ax1.set_xlabel('x-coordinate [mm]')
ax1.set_ylabel('z-coordinate [mm]')
ax1.set_title('Measured data')
ax1.set_aspect('equal', 'box')
colorbar(overview, ax=ax1)


ax2.scatter(wfit, zfit, s=2)
ax2.plot(Im, zfit, c='r', label='Best fit')
ax2.set_title('Sigma values of gaussian fit')
ax2.legend()

ax3.scatter(afit, zfit, s=2)
ax3.plot(amp, zfit, c='r', label='Best fit')
ax3.set_title('Maximum of gaussian fit')
ax3.legend()

#%% fit those two together to recreate a function

def fit01(x, a, b, c):
    y =  a * (-x+c)**2 + b
    return y

# Fit amplitude along z
fig     =   plt.figure()
ax      =   fig.add_subplot(111)
ax.scatter(zfit, afit)
popt_peak, pcov_peak  =   curve_fit(fit01, zfit, afit, p0=[0.5, 55, -0.6])                      # fit,  initial guess p0
amp         =   fit01(zfit, popt_peak[0], popt_peak[1], popt_peak[2])
ax.plot(zfit, amp, c='r', label='Best fit')
ax.set_title('smoothmaxima quadratic')

#%%

def pixel(x, x0, z):
    
    pixelfunc = func(x, func(z, amp_peak, mid_peak, sig_peak), x0, waist(z, a_sig, b_sig, c_sig, d_sig)) # gaussiannormal(x, x0, sigma)
    
    return pixelfunc

xmin = -1.6 #-1.875
xmax = -1.4  #-1.67
ymin = -0.1
ymax = 0.1
zmin = -0.7
zmax = -0.5
res  = 0.0005

x = np.arange(xmin,xmax,res)
z = np.arange(zmin,zmax,res)
x,z = meshgrid(x,z)

H = pixel(x, -1.463, z)
I = pixel(x, -1.469, z)
J = pixel(x, -1.475, z)
K = pixel(x, -1.481, z)

L = pixel(x, -1.5, z)
M = pixel(x, -1.506, z)

N = pixel(x, -1.525, z)

added = H + I + J + K + L + M + N 
squared = np.square(added)

raster1 = np.square(H + J + L + N)
raster2 = np.square(I + K + M)
raster  = raster1 + raster2

vmin = np.min(squared)
vmax = np.max(squared)

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(8,3), ncols=3)

add = ax1.imshow(added, cmap='viridis', extent=[xmin, xmax, zmin, zmax,], origin = 'lower', vmin=vmin, vmax=np.max(added), interpolation = 'none')
ax1.set_title('Normal')
#ax1.contour(add, origin='lower', colors=['white'])

one = ax2.imshow(squared, cmap='viridis', extent=[xmin, xmax, zmin, zmax,], origin = 'lower', vmin=vmin, vmax=vmax, interpolation = 'none')
ax2.set_title('Squared')
ax2.axes.yaxis.set_ticklabels([])

two = ax3.imshow(raster, cmap='viridis', extent=[xmin, xmax, zmin, zmax,], origin = 'lower', vmin=vmin, vmax=np.max(raster), interpolation = 'none')
ax3.set_title('Raster Squared')
ax3.axes.yaxis.set_ticklabels([])

#fig.colorbar(add, ax=ax3)

#%% single image
fig, (ax1) = plt.subplots(ncols=1)

add = ax1.imshow(added, cmap='viridis', extent=[xmin, xmax, zmin, zmax,], origin = 'lower', vmin=vmin, vmax=vmax, interpolation = 'none')
ax1.set_title('1, 2, 4 pixels')
#ax1.contour(add, origin='lower', colors=['w


#%% 2D gaussian functions for pixels

def pixel3D(x, y, x0, y0, z):
    
    pixel3Dfunc = func3D(x, y, func(z, amp_peak, mid_peak, sig_peak), x0, y0, waist(z, a_sig, b_sig, c_sig, d_sig))
        
    return pixel3Dfunc

x = np.arange(xmin,xmax,res)
x = x[2:]
y = np.arange(ymin,ymax,res)
z = np.arange(zmin,zmax,res)
xgrid, ygrid, zgrid = np.mgrid[xmin:xmax:res, ymin:ymax:res, zmin:zmax:res]

H3d = pixel3D(xgrid, ygrid, -1.463, 0, zgrid)

plt.scatter(xgrid, ygrid, zgrid, c=H3d)

fig, (ax1) = plt.subplots(ncols=1)

add = ax1.imshow(added, cmap='viridis', extent=[xmin, xmax, zmin, zmax,], origin = 'lower', vmin=vmin, vmax=vmax, interpolation = 'none')
ax1.set_title('1, 2, 4 pixels')


#%% OLD STUFF


#%% find maxima by keeping the integrated area constant, referencing the z layer with the highest maximum

amax = afit.max()
sigmin = wfit.min()
smoothmaxima = np.zeros(len(zi))

def gaussian(x, mu, sig):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def amplitudefunction(x, x0, sigma, sigmin):
    function = np.exp(-(x-x0)**2/(4*sigma**2)+(x-x0)**2/(4*sigmin**2))
    return function

def equalarea(x0, sigma, sigmin, amax):
    
    amplitude = amax * quad(amplitudefunction, -5, 0, args=(x0, sigma, sigmin))[0]
    return amplitude
    
for z in range(r_start , r_stop):
    smoothmaxima[z]  =  a[z] * equalarea(x0[z], sigma[z], sigmin, amax)
    
for z in range(r_start , r_stop):
    smoothmaxima[z] = gaussiannormal(x, x0[z], sigma[z]).max()


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


