#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 02:28:59 2017

@author: panin
"""

import pylab
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt



prefs.codegen.target = "numpy"

#detect collision when distance from sensor is less than 0.5 
def collision(*args):
    col=False
    for sensor in args:
        col = col or (sensor<0.5 and sensor>0.001)
    return col

def gaussian_spike(n,mean,Am,sigma):
    spike_train=np.zeros(n)
    for i in range(n):
        if mean>=n/2:
            if Am*np.exp(-((i-mean)**2)/(2*sigma*sigma))>=Am*np.exp(-((i+n-mean)**2)/(2*sigma*sigma)):

                spike_train[i]=Am*np.exp(-((i-mean)**2)/(2*sigma*sigma))

            else:
               spike_train[i]= Am*np.exp(-((i+n-mean)**2)/(2*sigma*sigma))
        else:
            if Am*np.exp(-((i-mean)**2)/(2*sigma*sigma))>=Am*np.exp(-((i-n-mean)**2)/(2*sigma*sigma)):
                spike_train[i]=Am*np.exp(-((i-mean)**2)/(2*sigma*sigma))
            else:
               spike_train[i]= Am*np.exp(-((i-n-mean)**2)/(2*sigma*sigma))
    return spike_train

#WTA
def WTA(args):
    winner=list()
    for i,j in enumerate(args): 
        if j==max(args):
            winner.extend([i])
    return winner


    #noise filter for the sensor; set distance from sensor =0 / 70 is randomly chosen number
#inverse because we want to feed the value such that the closer the sensor the higher the value it is
def inv_filter(sensor):
    if 1/sensor>70:
        sensor=0
    else:
        sensor=1/sensor
    return sensor  
    
#find nearest neuron of head
def nearest_neuron_head(angle, N_HD):
    angle_per_neuron = int(360/N_HD)
    full_neuron=int(angle/angle_per_neuron)
    reminder = angle%angle_per_neuron
    if reminder >= angle_per_neuron/2:
        return full_neuron
    else:
        if full_neuron == 0:
            return full_neuron
        else:
            return full_neuron-1

#find nearest neuron of speed
def nearest_neuron_speed(angle, N_speed):
    angle_per_neuron = int(30/N_speed)
    full_neuron=int(angle/angle_per_neuron)
    reminder = angle%angle_per_neuron
    if reminder >= angle_per_neuron/2:
        return full_neuron
    else:
        if full_neuron == 0:
            return full_neuron
        else:
            return full_neuron-1
            
            
#find nearest neuron of x-axis
def nearest_neuron_x_axis(x_axis, N_x_axis, x_scale):
    distance_per_neuron = x_scale/N_x_axis
    full_neuron=round((x_axis+x_scale/2)/distance_per_neuron)
    return full_neuron
    
#find nearest neuron of y-axis
def nearest_neuron_y_axis(y_axis, N_y_axis, y_scale):
    distance_per_neuron = y_scale/N_y_axis
    full_neuron=round((y_axis+y_scale/2)/distance_per_neuron)
    return full_neuron
        
    
#Plot comparitive result
def plot_result(spikemon_CD, spikemon_HD, angle,N_CD,N_HD,all_time):
    #multiple plot
    fig = plt.figure()
    fig.suptitle('Encoded Collision direction, Heading direction and True Heading direction', fontsize=14, fontweight='bold')
    fig.set_size_inches(9.5, 7.5)
    ax1 = fig.add_subplot(311)
    ax1.plot(spikemon_CD.t/ms,spikemon_CD.i, '.k')
    pylab.ylim([0,N_CD])
    #pylab.xlim([0,all_time[-1]])
    xlabel('Time (s)')
    ylabel('Neuron index')

    ax2 = fig.add_subplot(312)
    ax2.plot(spikemon_HD.t/ms,spikemon_HD.i, '.k')
    pylab.ylim([0,N_HD])
 #   pylab.xlim([0,1000])
    xlabel('Time (s)')
    ylabel('Neuron index')

    ax3 = fig.add_subplot(313)
    ax3.plot(all_time,angle,'.k')
    pylab.ylim([0,360])
    xlabel('Time (s)')
    ylabel('angle')
#savefig('HD_pic/good_mix3.pdf')
    plt.show()

def encoder(index,N_x_axis,N_y_axis,scale):
    x=np.array([])
    y=np.array([])
    for element in index:
        x=np.append(x,int(element/N_y_axis))
        y=np.append(y,int(element%N_x_axis))
        
    reds = plt.get_cmap("Reds")
    fig = plt.figure()
    plt.scatter(y, x, c=scale, cmap=reds, s=35)
    fig.set_size_inches(9.5, 7.5)
    pylab.xlim([0,N_x_axis])
    pylab.ylim([0,N_y_axis])
    tick_params(axis='x', labelsize=22)
    tick_params(axis='y', labelsize=22)
    xlabel('Neuron_x_axis',fontsize=22)
    ylabel('Neuron_y_axis',fontsize=22)
    return x,y

#encoder with the same color
def encoder_true(index,N_x_axis,N_y_axis):
    x=np.array([])
    y=np.array([])
    for element in index:
        x=np.append(x,int(element/N_y_axis))
        y=np.append(y,int(element%N_x_axis))
    fig = plt.figure()
    plt.scatter(x, y)
    fig.set_size_inches(9.5, 7.5)
    pylab.xlim([0,N_x_axis])
    pylab.ylim([0,N_y_axis])
    
    tick_params(axis='x', labelsize=25)
    tick_params(axis='y', labelsize=25)
    xlabel('Neuron_x_axis',fontsize=25)
    ylabel('Neuron_y_axis',fontsize=25)
    return x,y   

#encoder for the collision
def encoder_collision(index,N_x_axis,N_y_axis):
    x=np.array([])
    y=np.array([])
    for element in index:
        x=np.append(x,int(element/N_y_axis))
        y=np.append(y,int(element%N_x_axis))
    return x,y       


#print obstacle distances
def print_collision(sensor_val1,sensor_val2,sensor_val3,sensor_val4,sensor_val5,sensor_val6,sensor_val7,sensor_val8):
    if sensor_val1>0.00001 and sensor_val1<50:
        print ('sensor1: ',sensor_val1)
    if sensor_val2>0.00001 and sensor_val2<50:
        print ('sensor2: ',sensor_val2)
    if sensor_val3>0.00001 and sensor_val3<50:
        print ('sensor3: ',sensor_val3)
    if sensor_val4>0.00001 and sensor_val4<50:
        print ('sensor4: ',sensor_val4)
    if sensor_val5>0.00001 and sensor_val5<50:
        print ('sensor5: ',sensor_val5)
    if sensor_val6>0.00001 and sensor_val6<50:
        print ('sensor6: ',sensor_val6)
    if sensor_val7>0.00001 and sensor_val7<50:
        print ('sensor7: ',sensor_val7)
    if sensor_val8>0.00001 and sensor_val8<50:
        print ('sensor8: ',sensor_val8)

#creating vision sensor values
def generate_vision_val(pixelimage,N_VD):
    if len(pixelimage)>1:
        print("encounter landmark")
        sensor_val_vision=np.array([np.repeat(np.random.uniform(1,15),N_VD),np.zeros(N_VD)])
    else:
        sensor_val_vision =np.array([np.zeros(N_VD),np.repeat(np.random.uniform(1,3), N_VD)])
    return sensor_val_vision

#encoding plastic synapse
def plastic_encoder_CD_PI(w_plastic,N_x_axis,N_y_axis):
    collision_index = np.array([])
    syn_index = np.array([])
    w_plas= np.array([])
    for i in range(np.shape(w_plastic)[0]):
        if w_plastic[i] != 0:
            collision_index = np.append(collision_index,int(i/(N_x_axis*N_y_axis)))
            syn_index = np.append(syn_index,i)
            w_plas = np.append(w_plas,w_plastic[i])
    return syn_index, w_plas,collision_index
    
#plastic encoder for positin neuron
def plastic_encoder_PI_CD(N_CD,w_plastic,N_x_axis,N_y_axis):
    collision_index = np.array([])
    syn_index = np.array([])
    w_plas= np.array([])
    for i in range(np.shape(w_plastic)[0]):
        if w_plastic[i] != 0:
            collision_index = np.append(collision_index,int(i%(N_CD)))
            syn_index = np.append(syn_index,i)
            w_plas = np.append(w_plas,w_plastic[i])
    return syn_index, w_plas, collision_index
#plastic encoder for positin neuron
def fusi_encoder_PI_CD(w_plastic,N_x_axis,N_y_axis):
    syn_index = np.array([])
    w_plas= np.array([])
    for i in range(np.shape(w_plastic)[0]):
        if w_plastic[i] != 0:
            syn_index = np.append(syn_index,i)
            w_plas = np.append(w_plas,w_plastic[i])
    return syn_index, w_plas


    
#draw weight map
def draw_weight_map(N_CD,N_x_axis,N_y_axis,collision_index,w_plas,syn_index):
    weight_matrix = numpy.zeros((N_CD,N_x_axis,N_y_axis))
    for index in range(np.shape(syn_index)[0]):
        x_axis_col = int((syn_index[index] / N_CD) % N_y_axis)
        y_axis_col = int((syn_index[index]  /N_CD)  / N_y_axis)
        weight_matrix[int(collision_index[index])][x_axis_col][y_axis_col] = w_plas[index]
    all_weight = np.sum(weight_matrix,axis=0)
    plt.imshow(np.rot90(all_weight, 1))
    return all_weight
    
#draw weight map
def draw_fusi_weight_map(N_x_axis,N_y_axis,w_plas,syn_index):
    weight_matrix = numpy.zeros((N_x_axis,N_y_axis))
    for index in range(np.shape(syn_index)[0]):
        x_axis_col = int((syn_index[index]) % N_y_axis)
        y_axis_col = int((syn_index[index])  / N_y_axis)
        weight_matrix[x_axis_col][y_axis_col] = w_plas[index]
    plt.imshow(np.rot90(weight_matrix, 1))
    return weight_matrix  

