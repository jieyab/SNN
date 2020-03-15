#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:28:49 2017

@author: panin
"""

from brian2 import *
import numpy as np
import vrep
import matplotlib.pyplot as plt
from aux_function import *
from CD_net import *


def getSensor(clientID):
    #set sensor
    err_code,sensor1 = vrep.simxGetObjectHandle(clientID,"KJunior_proxSensor1", vrep.simx_opmode_blocking)
    err_code,sensor2 = vrep.simxGetObjectHandle(clientID,"KJunior_proxSensor2", vrep.simx_opmode_blocking)
    err_code,sensor3 = vrep.simxGetObjectHandle(clientID,"KJunior_proxSensor3", vrep.simx_opmode_blocking)
    err_code,sensor4 = vrep.simxGetObjectHandle(clientID,"KJunior_proxSensor4", vrep.simx_opmode_blocking)
    err_code,sensor5 = vrep.simxGetObjectHandle(clientID,"KJunior_proxSensor5", vrep.simx_opmode_blocking)
    err_code,sensor6 = vrep.simxGetObjectHandle(clientID,"KJunior_proxSensor6", vrep.simx_opmode_blocking)
    err_code,sensor7 = vrep.simxGetObjectHandle(clientID,"KJunior_proxSensor7", vrep.simx_opmode_blocking)
    err_code,sensor8 = vrep.simxGetObjectHandle(clientID,"KJunior_proxSensor8", vrep.simx_opmode_blocking)
    print("sensor name",sensor1,sensor2,sensor3,sensor4,sensor5,sensor6,sensor7,sensor8)
    return sensor1,sensor2,sensor3,sensor4,sensor5,sensor6,sensor7,sensor8

def getDetectedpoint(sensor1,sensor2,sensor3,sensor4,sensor5,sensor6,sensor7,sensor8,clientID):
    err_code,detectionState1,detectedPoint1,detectedObjectHandle1,detectedSurfaceNormalVector1=vrep.simxReadProximitySensor(clientID,sensor1,vrep.simx_opmode_streaming)
    err_code,detectionState2,detectedPoint2,detectedObjectHandle2,detectedSurfaceNormalVector2=vrep.simxReadProximitySensor(clientID,sensor2,vrep.simx_opmode_streaming)
    err_code,detectionState3,detectedPoint3,detectedObjectHandle3,detectedSurfaceNormalVector3=vrep.simxReadProximitySensor(clientID,sensor3,vrep.simx_opmode_streaming)
    err_code,detectionState4,detectedPoint4,detectedObjectHandle4,detectedSurfaceNormalVector4=vrep.simxReadProximitySensor(clientID,sensor4,vrep.simx_opmode_streaming)
    err_code,detectionState5,detectedPoint5,detectedObjectHandle5,detectedSurfaceNormalVector5=vrep.simxReadProximitySensor(clientID,sensor5,vrep.simx_opmode_streaming)
    err_code,detectionState6,detectedPoint6,detectedObjectHandle6,detectedSurfaceNormalVector6=vrep.simxReadProximitySensor(clientID,sensor6,vrep.simx_opmode_streaming)
    err_code,detectionState7,detectedPoint7,detectedObjectHandle7,detectedSurfaceNormalVector7=vrep.simxReadProximitySensor(clientID,sensor7,vrep.simx_opmode_streaming)
    err_code,detectionState8,detectedPoint8,detectedObjectHandle8,detectedSurfaceNormalVector8=vrep.simxReadProximitySensor(clientID,sensor8,vrep.simx_opmode_streaming)
    #print("get detection point values",err_code,detectionState1,detectedPoint1,detectedObjectHandle1,detectedSurfaceNormalVector1)
    return detectedPoint1,detectedPoint2,detectedPoint3,detectedPoint4,detectedPoint5,detectedPoint6,detectedPoint7,detectedPoint8


    
    
def getSensorDistance(detectedPoint1,detectedPoint2,detectedPoint3,detectedPoint4,detectedPoint5,detectedPoint6,detectedPoint7,detectedPoint8):
    sensor_val1 = np.linalg.norm(detectedPoint1)
    sensor_val2 = np.linalg.norm(detectedPoint2)
    sensor_val3 = np.linalg.norm(detectedPoint3)
    sensor_val4 = np.linalg.norm(detectedPoint4)
    sensor_val5 = np.linalg.norm(detectedPoint5)
    sensor_val6 = np.linalg.norm(detectedPoint6)
    sensor_val7 = np.linalg.norm(detectedPoint7)
    sensor_val8 = np.linalg.norm(detectedPoint8)
    return sensor_val1,sensor_val2,sensor_val3,sensor_val4,sensor_val5,sensor_val6,sensor_val7,sensor_val8
    
#Controlling 

#right turn random from 0~60 degree
def TurnRight(r_steer,l_steer,delta_t):
    #random 60 degree
    #r = 0.04975
    #robot radius
    r = 0.0586
    zeta = np.random.uniform(5,30)
    r_steer = 0
    l_steer = 2 * zeta * r / delta_t
    print('zeta is:', zeta, ' left speed is:', l_steer)

    return r_steer, l_steer, zeta
    
    
    
#left turn random from 0-60 degree    
def TurnLeft(r_steer,l_steer,delta_t):
    #random 60 degree
    #r = 0.04975
    #robot radius
    r = 0.0586
    zeta = np.random.uniform(5,30)
    r_steer = 2 * zeta * r / delta_t
    l_steer = 0
    print('zeta is:', zeta, ' right speed is:', r_steer)
    return r_steer, l_steer, zeta

    
def feedbackController(sensor_val1,sensor_val2,sensor_val8):
        #feedback controller if 2,3,4 activate --> feedback; else go straight
    if sensor_val2 < 0.07 and sensor_val2>0.02:
        r_steer, l_steer, zeta = TurnRight()
    elif sensor_val1 < 0.15 and sensor_val1>0.02:
        r_steer, l_steer, zeta = TurnLeft()
    elif sensor_val8 < 0.07 and sensor_val8>0.02:
        r_steer, l_steer, zeta = TurnLeft()
    else:
         l_steer = 1.5
         r_steer = 1.5
         zeta = 0
    return r_steer, l_steer, zeta
    
def getHeadingdirection(Angles):
    #transformation to normal coordinate
    #heading angle in scalar
    heading_dir1=(np.array(Angles)[0]*180/3.14)
    if heading_dir1>0:
        k=1
        c=0
    else:
        k=-1
        c=360
    #real head direction
    heading_dir2=((np.array(Angles)[1]*180/3.14)+90)*k+c
    return heading_dir2

def encounterLandmarkSelfRotate(clientID, l_motor_handle,r_motor_handle):
    # agent rotate itself for 360 degrees while encountering a landmark
    count = 0
    while count < 83:
        print("rotating now yooooooooo")
        l_steer = 0.6
        r_steer = -0.6
        count += 1
        err_code = vrep.simxSetJointTargetVelocity(clientID, l_motor_handle, l_steer, vrep.simx_opmode_streaming)
        err_code = vrep.simxSetJointTargetVelocity(clientID, r_motor_handle, r_steer, vrep.simx_opmode_streaming)

    
    