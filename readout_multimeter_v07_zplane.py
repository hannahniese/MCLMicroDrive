# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 18:56:07 2022

@author: Hannah Niese

This script controls the x-y-z stage and reads out the values from the photodiode
from the multimeter. The data is then re-arranged and plotted to get a first
overview of the quality. 

Anything labelled with #VARIABLE(S) is a variable that the use can change, other things should not be changed (unless you want to improve the code)
"""

# reference the stage before first running the script (only once!)

motor = MicroDrive()
motor.centerHomePosition()

#%% set up all devices and the measurement file
## if this section gives an error message, try unplugging the Multimeter and diode, restart and re-run the section

# import the necessary modules
import pyvisa as visa
import time
import csv
import os
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

# %matplotlib qt
#os.chdir('C:\Users\Congreve Optics\Desktop\Hannah\MCLMicroDrive\22_07_06_20x')

# Generate the name of the data file based on the start of the script
dataTimeString      =   datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
fileNameString      =   dataTimeString + "_data.txt"
figureNameString    =   dataTimeString + "_figure.png"
figureName3D        =   dataTimeString + "_3Dfigure.png"

# set initial value to 0
measuredValue       =   0.00000

##### INITIALIZE MULTIMETER #####
rm                  =   visa.ResourceManager()
v34401A             =   rm.open_resource('GPIB0::22::INSTR')

# open file to write data
file                =   open(fileNameString, 'w')

# Test for connection by asking for identification 
ReturnValue         =   v34401A.query('*IDN?')

##### INITIALIZE STAGE ##### 
# run the stage class before running this
motor               =   MicroDrive()
counter             =    0

# write vital parameters of the measurement into the first line of the file
file.write('Voltage = 4.3, Current = 20.7, Pinhole = 15um, Objective = 20x, NA = 0.42, blue LED \n')

# measurement function
def measure(xcoord, ycoord, zcoord, c):
    # this function measures the photodiode voltage at the input coordinates, writes it to the file and returns the measured value
    #temp_values    =   v34401A.query_ascii_values(':MEASure:VOLTage:DC? %s,%s' % ('MIN', 'MAX'))
    temp_values     =   v34401A.query_ascii_values(':MEASure:VOLTage:DC? %s,%s' % ('MIN', 'MAX'))  #DC voltage measurements using the MIN V range with MAX mV resolution. Then make and read one measurement
    measuredValue   =   temp_values[0]
    #print('Voltage:  {0}'.format(str(measuredValue)))     # Commenting out prints increases acquistion speed
    position        =   motor.getPosition()
    xpos            =   position[0]
    ypos            =   position[1]
    zpos            =   position[2]
    #print(str(position) + ', ' + str(measuredValue))      # print progress
    file.write(str(c) + ', ' + str(xcoord) + ', ' + str(ycoord) + ', ' + str(zcoord) + ', ' + str(xpos) + ', ' + str(ypos) + ', ' + str(zpos) + ', ' '{0}\n'.format(str(measuredValue)))
    
    return measuredValue

# quick measurement function
def quickmeasure(xcoord, ycoord, zcoord):
    # this function measures the photodiode voltage at the input coordinates and prints the voltage 
    temp_values     =   v34401A.query_ascii_values(':MEASure:VOLTage:DC? %s,%s' % ('MIN', 'MAX'))  #DC voltage measurements using the MIN V range with MAX mV resolution. Then make and read one measurement
    measuredValue   =   temp_values[0]
    print('Voltage:  {0}'.format(str(measuredValue)))     # Commenting out prints increases acquistion speed
    
    return measuredValue

#%% input initial coordinates and step size
    
# VARIABLES
    
# define step sizes and error for the stage movements
xstep   =    0.001         # in mm
ystep   =    xstep
zstep   =    0.1          # distance between z planes in mm 
error   =    0.0007        # error in mm

# define starting location
xcoord  =    10.353    #
ycoord  =    1.915
zcoord  =    0.46   #        # -0.415 z planes will have positive addition 

motor.moveControlled(xcoord, ycoord, zcoord, velocity = 3, errorMove = error) # use these two functions to update location and check if there is signal
quickmeasure(xcoord, ycoord, zcoord)

# save initial coordinates so we can return to them for measurements on the additional plane (do not change)
xc      =    xcoord
yc      =    ycoord
zc      =    zcoord



#%% Run several measurements in the x-y plane
# initialize signs and counters for spiral motion
xsign = 1
ysign = -1
c = 0
d = 0

# define the number of acquisitions and the number of planes that should be sampled
acq     =   25           # VARIABLE  estimate: 30 for simple measurement, 60 for adjusting, 120 for high quality measurement
planes  =   1            # VARIABLE

motor.moveControlled(xcoord, ycoord, zcoord, velocity = 3, errorMove = error)
measure(xcoord, ycoord, zcoord, c)

while d < planes:
    d += 1
    while c < acq:
        c += 1
        print(c)
        for i in range(c):
            xcoord = xcoord + xstep*xsign
            motor.moveControlled(xcoord, ycoord, zcoord, velocity = 3, errorMove = error)
            #motor.move(xcoord, ycoord, zcoord)
            measure(xcoord, ycoord, zcoord, c)
        for i in range(c):
            ycoord = ycoord + ystep*ysign
            motor.moveControlled(xcoord, ycoord, zcoord, velocity = 3, errorMove = error)
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

#%% X-Y measurements plotting

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


plt.scatter(x,y, c=I, cmap='viridis', s=4)
plt.axis('equal')
plt.xlabel('x-coordinate [mm]')
plt.ylabel('y-coordinate [mm]')
plt.colorbar()

#os.chdir(outpath)
plt.savefig('%s_2D.png' % dataTimeString, dpi=600)


# plotting 3d image
plt.close()

ax = plt.axes(projection='3d')
ax.scatter(x,y,I, c=I, cmap='viridis', linewidth=0.5)
ax.set_xlabel('x-coordinate')
ax.set_ylabel('y-coordinate')
ax.set_zlabel('Intensity [mV]')
plt.show()

#os.chdir(masterpath)
plt.savefig('%s_3D.png' % dataTimeString, dpi=600)
#plt.close()

#%% X-Z measurements 

ystep   =   0.002
zstep   =   ystep
acq     =   60         # number of sides of acquisitions


ysign = 1
zsign = -1


xcoord = xc
ycoord = yc
zcoord = zc
c = 0
d = 0

# move to the center coordinate
motor.moveControlled(xcoord, ycoord, zcoord, velocity = 3, errorMove = error)
measure(xcoord, ycoord, zcoord, c)

# move in spiral and acquire data
while c < acq:
    c += 1
    print(c)
    for i in range(c):
        ycoord = ycoord + ystep*ysign
        motor.moveControlled(xcoord, ycoord, zcoord, velocity = 3, errorMove = error)
        #motor.move(xcoord, ycoord, zcoord)
        measure(xcoord, ycoord, zcoord, c)
    for i in range(c):
        zcoord = zcoord + zstep*zsign
        motor.moveControlled(xcoord, ycoord, zcoord, velocity = 3, errorMove = error)
        #motor.move(xcoord, ycoord, zcoord)
        measure(xcoord, ycoord, zcoord, c)
    ysign = ysign * -1
    zsign = zsign * -1    

v34401A.close()    # Close our connection to the instrument
rm.close()

# load data into python
data    = np.loadtxt(fileNameString, delimiter=',', skiprows=1)

    
# X-Z measurements plotting


dim=len(data)


plt.plot(data[:,2], data[:,3], color='green')
plt.scatter(data[:,5], data[:,6], s=7, color='blue')

plt.axis('equal')
plt.xlabel('y-coordinate')
plt.ylabel('z-coordinate')
plt.title('Acquired datapoints: %d' % dim) 

#os.chdir(masterpath)
plt.savefig('%s_side_points.png' % dataTimeString, dpi=600)



# plotting 2d heatmap at real coordinates
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
plt.savefig('%s_side_2D.png' % dataTimeString, dpi=600)


#%% plotting 3d image


ax = plt.axes(projection='3d')
ax.scatter(x,z,I, c=I, cmap='viridis', linewidth=0.5)
ax.set_xlabel('x-coordinate')
ax.set_ylabel('y-coordinate')
ax.set_zlabel('Intensity [mV]')
plt.show()

#os.chdir(masterpath)
plt.savefig('%s_side_3D.png' % dataTimeString, dpi=600)


#%% key controller

position        =   motor.getPosition()

step            =   0.2
zstep           =   0.05

import keyboard


def mywait():
    keyboard.read_key()

while True:
    if keyboard.read_key() == "right":
        motor.move(position[0], position[1] - step, position[2])
        position        =   motor.getPosition()
    if keyboard.read_key() == "left":
        motor.move(position[0], position[1] + step, position[2])
        position        =   motor.getPosition()
    if keyboard.read_key() == "up":
        motor.move(position[0] - step, position[1], position[2])
        position        =   motor.getPosition()
    if keyboard.read_key() == "down":
        motor.move(position[0] + step, position[1], position[2])
        position        =   motor.getPosition()
    if keyboard.read_key() == "m":                                  # m 
        motor.move(position[0], position[1], position[2] - zstep)
        position        =   motor.getPosition()
    if keyboard.read_key() == "n":
        motor.move(position[0], position[1], position[2] + zstep)
        position        =   motor.getPosition()


#%%

position        =   motor.getPosition()

step            =   0.1

import sys, pygame
pygame.init()

run = True

while run:
    for event in pygame.event.get():
        if event.type == pygame.K_RIGHT:
            motor.move(position[0], position[1] - step, position[2])
            position        =   motor.getPosition()
        if event.type == pygame.K_LEFT:
            motor.move(position[0], position[1] + step, position[2])
            position        =   motor.getPosition()
        if event.type == pygame.K_UP:
            motor.move(position[0] - step, position[1], position[2])
            position        =   motor.getPosition() 
        if event.type == pygame.K_DOWN:
            motor.move(position[0] + step, position[1], positi on[2])
            position        =   motor.getPosition()
        if event.type == pygame.K_SPACE:
            run = False
pygame.quit()
   
# while True:

        
    
# while True:

        
    
# while True:

       
#keyboard.on_press_key("down", motor.move(position[0] + step, position[1], position[2]))


