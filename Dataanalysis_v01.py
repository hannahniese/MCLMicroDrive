# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:15:40 2022

@author: Hannah Niese
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import os


### Importing E-field data
masterpath  =   r"C:\Users\Hannah Niese\Documents\GitHub\MCLMicroDrive"
file        =   '220531172415_data'
ftype       =   '.txt'
datafile    =   masterpath + '\\' + file + ftype
outpath     =   masterpath + '\\analysis'
if os.path.isdir(outpath) == False:
    os.mkdir(outpath)
    
    
data    = np.loadtxt(datafile, delimiter=',',  skiprows=1)

#%% plotting

plt.plot(data[:,0], data[:,3])


os.chdir(masterpath)
plt.savefig('%s.png' % file, dpi=600)

#%% image 2D

dim = len(data)
matrix = np.zeros([dim, dim])

X=abs(data[:,0].transpose()*-10)
Y=data[:,1].transpose()*-10
I=data[:,3].transpose()

matrix[X, Y] = I