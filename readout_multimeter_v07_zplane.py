# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 18:56:07 2022

@author: Hannah Niese
"""



import pyvisa as visa
import time
import csv
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

##### INITIALIZE READOUT OF THE MULTIMETER



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
# run the stage class before running this
motor = MicroDrive()
counter = 0

# write vital parameters of the measurement into the file
file.write('Voltage = 3.9, Current = 20.8, Pinhole = 15um \n')



#%% iterate through points in a spiral starting from the highest intensity

def getPosition(self):
        """
        This function takes approximately 10ms.
        """
        
        e1 = ctypes.pointer(ctypes.c_double())
        e2 = ctypes.pointer(ctypes.c_double())
        e3 = ctypes.pointer(ctypes.c_double())
        e4 = ctypes.pointer(ctypes.c_double())
        errorNumber = self.mcl.MCL_MDReadEncoders(e1, e2, e3, e4, self.handle)
        if errorNumber != 0:
            print('Error reading the encoders: ' + self.errorDictionary[errorNumber])
        #print('Encoders: ' + str(np.round(e1.contents.value,4)) + ', ' + str(np.round(e2.contents.value,5)) + ', ' + str(np.round(e3.contents.value,5)))
        position_temp = [e1.contents.value, e2.contents.value, e3.contents.value]
        del e1
        del e2
        del e3
        del e4
        return position_temp

def measure(xcoord, ycoord, zcoord, c):
    # acquisition control
    #temp_values = v34401A.query_ascii_values(':MEASure:VOLTage:DC? %s,%s' % ('MIN', 'MAX'))
    temp_values = v34401A.query_ascii_values(':MEASure:VOLTage:DC? %s,%s' % ('MIN', 'MAX'))  #DC voltage measurements using the MIN V range with MAX mV resolution. Then make and read one measurement
    measuredValue = temp_values[0]
    #print('Voltage:  {0}'.format(str(measuredValue)))     # Commenting out prints increases acquistion speed
    position = motor.getPosition()
    xpos = position[0]
    ypos = position[1]
    zpos = position[2]
    #print(str(position) + ', ' + str(measuredValue))      # print progress
    file.write(str(c) + ', ' + str(xcoord) + ', ' + str(ycoord) + ', ' + str(zcoord) + ', ' + str(xpos) + ', ' + str(ypos) + ', ' + str(zpos) + ', ' '{0}\n'.format(str(measuredValue)))
    
    return measuredValue

#%% xy plane

xstep = 0.003
ystep = 0.003
zstep = 0.1
xsign = 1
ysign = -1
xc = 9.72
yc = -3.9

xcoord = 9.72
ycoord = -3.9
zcoord = 3.34
c = 0
d = 0

#motor.getposition()
acq     =   100
planes  =   1

motor.moveControlled(xcoord, ycoord, zcoord, velocity = 3, errorMove = 0.001)
measure(xcoord, ycoord, zcoord, c)

while d < planes:
    d += 1
    while c < acq:
        c += 1
        print(c)
        for i in range(c):
            xcoord = xcoord + xstep*xsign
            motor.moveControlled(xcoord, ycoord, zcoord, velocity = 3, errorMove = 0.001)
            #motor.move(xcoord, ycoord, zcoord)
            measure(xcoord, ycoord, zcoord, c)
        for i in range(c):
            ycoord = ycoord + ystep*ysign
            motor.moveControlled(xcoord, ycoord, zcoord, velocity = 3, errorMove = 0.001)
            #motor.move(xcoord, ycoord, zcoord)
            measure(xcoord, ycoord, zcoord, c)
        xsign = xsign * -1
        ysign = ysign * -1    
    zcoord = zcoord + d*zstep
    xcoord = xc
    ycoord = yc
    c = 0
    measure(xcoord, ycoord, zcoord, c)
v34401A.close()    # Close our connection to the instrument
rm.close()


data    = np.loadtxt(fileNameString, delimiter=',', skiprows=1)

#%% plotting trajectories


dim=len(data)


plt.plot(data[:,1], data[:,2], color='green')
plt.scatter(data[:,4], data[:,5], s=7, color='blue')

plt.axis('equal')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('Acquired datapoints: %d' % dim) 

#os.chdir(masterpath)
plt.savefig('%s_points.png' % dataTimeString, dpi=600)



# plotting 2d heatmap at real coordinates
plt.close()

x = data[:,4]
y = data[:,5]
z = data[:,6]
I = data[:,7]*1000 # convert to mV


plt.scatter(x,y, c=I, cmap='viridis', s=7)
plt.axis('equal')
plt.xlabel('x-coordinate [mm]')
plt.ylabel('y-coordinate [mm]')
plt.colorbar()

#os.chdir(outpath)
plt.savefig('%s_2D.png' % dataTimeString, dpi=600)


#%% plotting 3d image


ax = plt.axes(projection='3d')
ax.scatter(x,y,I, c=I, cmap='viridis', linewidth=0.5)
ax.set_xlabel('x-coordinate')
ax.set_ylabel('y-coordinate')
ax.set_zlabel('Intensity [mV]')
plt.show()

#os.chdir(masterpath)
plt.savefig('%s_3D.png' % dataTimeString, dpi=600)


#%% XZ XZ XZ XZ XZXZXZXZXZXZXZXZXZXZ 

ystep = 0.05
zstep = 0.05

ysign = 1
zsign = -1


xcoord = 9.7
ycoord = -3.89
zcoord = 3.24
c = 0
d = 0

#motor.getposition()
acq     =   40
planes  =   1

motor.moveControlled(xcoord, ycoord, zcoord, velocity = 3, errorMove = 0.001)
measure(xcoord, ycoord, zcoord, c)


while c < acq:
    c += 1
    print(c)
    for i in range(c):
        ycoord = ycoord + ystep*ysign
        motor.moveControlled(xcoord, ycoord, zcoord, velocity = 3, errorMove = 0.001)
        #motor.move(xcoord, ycoord, zcoord)
        measure(xcoord, ycoord, zcoord, c)
    for i in range(c):
        zcoord = zcoord + zstep*zsign
        motor.moveControlled(xcoord, ycoord, zcoord, velocity = 3, errorMove = 0.001)
        #motor.move(xcoord, ycoord, zcoord)
        measure(xcoord, ycoord, zcoord, c)
    ysign = ysign * -1
    zsign = zsign * -1    

v34401A.close()    # Close our connection to the instrument
rm.close()


data    = np.loadtxt(fileNameString, delimiter=',', skiprows=1)

    
#%% plotting trajectories


dim=len(data)


plt.plot(data[:,2], data[:,3], color='green')
plt.scatter(data[:,5], data[:,6], s=7, color='blue')

plt.axis('equal')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('Acquired datapoints: %d' % dim) 

#os.chdir(masterpath)
plt.savefig('%s_points.png' % dataTimeString, dpi=600)



#%% plotting 2d heatmap at real coordinates
plt.close()

x = data[:,4]
y = data[:,5]
z = data[:,6]
I = data[:,7]*1000 # convert to mV


plt.scatter(y,z, c=I, cmap='viridis', s=7)
plt.axis('equal')
plt.xlabel('y-coordinate [mm]')
plt.ylabel('z-coordinate [mm]')
plt.colorbar()

#os.chdir(outpath)
plt.savefig('%s_2D.png' % dataTimeString, dpi=600)


#%% plotting 3d image


ax = plt.axes(projection='3d')
ax.scatter(x,z,I, c=I, cmap='viridis', linewidth=0.5)
ax.set_xlabel('x-coordinate')
ax.set_ylabel('y-coordinate')
ax.set_zlabel('Intensity [mV]')
plt.show()

#os.chdir(masterpath)
plt.savefig('%s_3D.png' % dataTimeString, dpi=600)
