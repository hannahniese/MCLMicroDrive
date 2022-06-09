# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:28:44 2022

@author: niese
"""

import pyvisa as visa
import time
import csv
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

# Generate the name of the data file based on the start of the script
dataTimeString      =   datetime.datetime.now().strftime("%y%m%d%H%M%S")
fileNameString      =   dataTimeString + "_data.txt"
figureNameString    =   dataTimeString + "_figure.png"
figureName3D        =   dataTimeString + "_3Dfigure.png"

# initialize arrays
measuredValue       =   0.00000

# initialize multimeter
rm      =   visa.ResourceManager()
v34401A =   rm.open_resource('GPIB0::22::INSTR')

# open file to write data
file    =   open(fileNameString, 'w')

# Test for connection by asking for identification 
ReturnValue = v34401A.query('*IDN?')

# initialize stage 
motor = MicroDrive()
counter = 0

# write vital parameters of the measurement into the file
file.write('Voltage = 3.3, Current = 14.4, Pinhole = 15um \n')





#%% iterate through points in a spiral starting from the highest intensity

def measure(xcoord, ycoord, zcoord, c):
    # acquisition control
    #temp_values = v34401A.query_ascii_values(':MEASure:VOLTage:DC? %s,%s' % ('MIN', 'MAX'))
    temp_values = v34401A.query_ascii_values(':MEASure:VOLTage:DC? %s,%s' % ('MIN', 'MAX'))  #DC voltage measurements using the MIN V range with MAX mV resolution. Then make and read one measurement
    measuredValue = temp_values[0]
    print('Voltage:  {0}'.format(str(measuredValue)))     # Commenting out prints increases acquistion speed
    position = motor.getPosition()
    xpos = position[0]
    ypos = position[1]
    zpos = position[2]
    print(str(position) + ', ' + str(measuredValue))      # print progress
    acquisitionArray.append(t)                            # Store our data in a local array
    coord_x.append(xcoord)                                     # Store our data in a local array
    coord_y.append(ycoord)                                     # Store our data in a local array
    dataArray.append(float(measuredValue))           # 
    file.write(str(c) + ', ' + str(xcoord) + ', ' + str(ycoord) + ', ' + str(zcoord) + ', ' + str(xpos) + ', ' + str(ypos) + ', ' + str(zpos) + ', ' '{0}\n'.format(str(measuredValue)))
    
    return measuredValue

xstep = 0.01
ystep = 0.01
xsign = 1
ysign = -1
xcoord = 2.25
ycoord = -2.9
zcoord = -6.9
c = 0

#motor.getposition()
acq=70

motor.move(xcoord, ycoord, zcoord)
measure(xcoord, ycoord, zcoord, c)

while c < acq:
    c += 1
    print(c)
    for i in range(c):
        xcoord = xcoord + xstep*xsign
        motor.move(xcoord, ycoord, zcoord)
        measure(xcoord, ycoord, zcoord, c)
    for i in range(c):
        ycoord = ycoord + ystep*ysign
        motor.move(xcoord, ycoord, zcoord)
        measure(xcoord, ycoord, zcoord, c)
    xsign = xsign * -1
    ysign = ysign * -1    
v34401A.close()    # Close our connection to the instrument
rm.close()
    
#%% plotting trajectories

data    = np.loadtxt(fileNameString, delimiter=',', skiprows=1)

dim=len(data)


plt.plot(data[:,1], data[:,2], color='green')
plt.scatter(data[:,4], data[:,5], s=7, color='blue')

plt.axis('equal')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('Acquired datapoints: %d' % dim) 

#os.chdir(masterpath)
plt.savefig('%s_points.png' % dataTimeString, dpi=600)



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

#os.chdir(masterpath)
plt.savefig('%s_3D.png' % dataTimeString, dpi=600)



#%% old stuff

#%% iterate through points on a rectangular square

xstep = 0.01
ystep = 0.01
xsign = 1
ysign = -1
xcoord = 4
ycoord = 3
zcoord = -2
c = 0
    
#motor.getposition()
acq=50
    
motor.move(xcoord, ycoord, zcoord)
measure(xcoord, ycoord, zcoord, c)

while t < acq:
    
        t += 1
        print(t)
        for i in range(c):
            xcoord = xcoord + xstep*t
            motor.move(xcoord, ycoord, zcoord)
            measure(xcoord, ycoord, zcoord, c)
        for i in range(c):
            ycoord = ycoord + ystep*ysign
            motor.move(xcoord, ycoord, zcoord)
            measure(xcoord, ycoord, zcoord, c)
        xsign = xsign * -1
        ysign = ysign * -1 
    
    
    t += 1
    xmem = xmem + x
    while p < rows:
        # stage control
        p += 1
        ymem = ymem + y
        y = -0.05
        z = 0
        print(x,y,z)
        motor.move(0,y,z)
        counter += 1
        
        measure()

    x = 0.05
    motor.move(x,-rows*y,0)
    ymem = 0
    y = 0
    p = 0
v34401A.close()    # Close our connection to the instrument
rm.close()
#%%


# Generate quick plot
plt.xlabel('x-coordinate')
plt.ylabel('Voltage')
plt.title('HP 34401A Data')
plt.plot(coord_x,dataArray)
plt.savefig(figureNameString)
plt.show()

#%% plot 2D heatmap

array = np.zeros((400,4))
array[:,0] = coord_x
array[:,1] = coord_y
#array[:,2] = coord_z
array[:,3] = dataArray

# importing mplot3d toolkits, numpy and matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
 
fig = plt.figure()
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d')
 
# defining all 3 axes
z = dataArray
x = coord_x
y = coord_y
 
# plotting
ax.scatter(x, y, z, c=z, cmap='viridis')
ax.set_title('3D line plot geeks for geeks')
plt.savefig(figureName3D)
plt.show()

    