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
import numpy as np

# Generate the name of the data file based on the start of the script
dataTimeString = datetime.datetime.now().strftime("%y%m%d%H%M%S")
fileNameString = dataTimeString + "_data.txt"
figureNameString = dataTimeString + "_figure.png"
figureName3D = dataTimeString + "_3Dfigure.png"

# initialize arrays
measuredValue = 0.00000
acquisitionArray    = []
dataArray           = []
coord_x             = []
coord_y             = []
coord_z             = []

# initialize multimeter
rm = visa.ResourceManager()
v34401A = rm.open_resource('GPIB0::22::INSTR')
# open file to write data
file = open(fileNameString, 'w')


# Test for connection by asking for identification 
ReturnValue = v34401A.query('*IDN?')

# scan through spatial locations
motor = MicroDrive()
acq     = 15
rows    = 15
t       = 0
p       = 0
xmem    = 0
ymem    = 0
counter = 0
x = 0
y = 0

file.write('No of points in x' + str(acq) + 'no of points in y' + str(rows) + '\n')




#%% iterate through points on a rectangular square
while t < acq:
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
        
        # acquisition control
        temp_values = v34401A.query_ascii_values(':MEASure:VOLTage:DC? %s,%s' % ('MIN', 'MAX'))
        measuredValue = temp_values[0]
        print('Voltage:  {0}'.format(str(measuredValue)))   # Commenting out prints increases acquistion speed
        print('p = ' + str(p) + ', t = ' + str(t) + ', acq ' + str(counter) + '/' + str(acq*rows))
        print('x = ' + str(xmem) + ', y = ' + str(ymem))              # print progress
        acquisitionArray.append(t)        # Store our data in a local array
        coord_x.append(t)                 # Store our data in a local array
        coord_y.append(p)                 # Store our data in a local array
        dataArray.append(float(measuredValue))           # 
        file.write(str(xmem) + ', ' + str(ymem) + ', ' + str(z) + ', ' + '{0}\n'.format(str(measuredValue)))
    x = 0.05
    motor.move(x,-rows*y,0)
    ymem = 0
    y = 0
    p = 0
v34401A.close()    # Close our connection to the instrument
rm.close()

#%% iterate through points in a spiral starting from the highest intensity

def measure():
    # acquisition control
    temp_values = v34401A.query_ascii_values(':MEASure:VOLTage:DC? %s,%s' % ('MIN', 'MAX'))
    measuredValue = temp_values[0]
    print('Voltage:  {0}'.format(str(measuredValue)))     # Commenting out prints increases acquistion speed
    position = getposition()
    print(str(position) + ', ' + str(measuredValue))   # print progress
    acquisitionArray.append(t)                            # Store our data in a local array
    coord_x.append(t)                                     # Store our data in a local array
    coord_y.append(p)                                     # Store our data in a local array
    dataArray.append(float(measuredValue))           # 
    file.write(str(position) + '{0}\n'.format(str(measuredValue)))
    
    return measuredValue

xstep = 0.1
ystep = 0.1
xsign = 1
ysign = -1
xcoord = 0
ycoord = 0
c = 0

#motor.getposition()
acq=7

#measure()

while c < acq:
    c += 1
    print(c)
    for i in range(c):
#        motor.move(xstep*xsign,0,0)
#        measure()
#        print(xstep*xsign, 0, 0)
        xcoord = xcoord + xstep*xsign
        print(xcoord, ycoord)
    for i in range(c):
#        motor.move(0,ystep*ysign,0)
#        measure
#        print(0, ystep*ysign, 0)
        ycoord = ycoord + ystep*ysign
        print(xcoord, ycoord)
    xsign = xsign * -1
    ysign = ysign * -1    

    



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

    