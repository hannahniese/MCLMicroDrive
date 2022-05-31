# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:28:44 2022

@author: Congreve Optics
"""

import pyvisa as visa
import time
import csv
import datetime
import matplotlib.pyplot as plt

# Generate the name of the data file based on the start of the script
dataTimeString = datetime.datetime.now().strftime("%y%m%d%H%M%S")
fileNameString = dataTimeString + "_data.txt"
figureNameString = dataTimeString + "_figure.png"

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
acq     = 150
t       = 0
while t < acq:
    # stage control
    t += 1
    x = -0.1
    y = 0
    z = 0
    print(x,y,z)
    motor.move(x,y,z)
    
    # acquisition control
    temp_values = v34401A.query_ascii_values(':MEASure:VOLTage:DC? %s,%s' % ('MIN', 'MAX'))
    measuredValue = temp_values[0]
    print('Voltage:  {0}'.format(str(measuredValue))) # Commenting out prints increases acquistion speed
    acquisitionArray.append(t)        # Store our data in a local array
    dataArray.append(float(measuredValue))           # 
    file.write(str(x) + ', ' + str(y) + ', ' + str(z) + ', ' + '{0}\n'.format(str(measuredValue)))
v34401A.close()    # Close our connection to the instrument
rm.close()


# Generate quick plot
plt.xlabel('x-coordinate')
plt.ylabel('Voltage')
plt.title('HP 34401A Data')
plt.plot(acquisitionArray,dataArray)
plt.savefig(figureNameString)
plt.show()

    