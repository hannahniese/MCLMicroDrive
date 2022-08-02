# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:06:43 2022

Programme controls stage manually by using the arrow keys (x and y, as in microscope camera)
and z (keys m and n). 


@author: Hannah Niese
"""

# First run the MicroDrive programme and start a handle. Center the stage
motor = MicroDrive()
motor.centerHomePosition()

#%% key controller

position        =   motor.getPosition()

step            =   1
zstep           =   1

import keyboard


run = True

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
    if keyboard.read_key() == "m":                                  # m move in z direction 
        motor.move(position[0], position[1], position[2] - zstep)
        position        =   motor.getPosition()
    if keyboard.read_key() == "n":                                  # m move in opposite z direction 
        motor.move(position[0], position[1], position[2] + zstep)
        position        =   motor.getPosition()
    if keyboard.read_key() == "space":                              # space bar to end navigation 
        run = False
        break
