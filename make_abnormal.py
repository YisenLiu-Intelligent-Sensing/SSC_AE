"""
 @file   make_abnormal.py
 @brief  Script for generating abnormal sample
 @author Yisen Liu
 Copyright (C) 2021 Institute of Intelligent Manufacturing, Guangdong Academy of Sciences, All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import random

import matplotlib.pyplot as plt
import numpy as np

import common as com

########################################################################
param = com.yaml_load()

# make abnormal data
def interpolate(data,max_value=0.3,min_value=0.05):
    #load data of other objects
    files_list = os.listdir(param["other_object_data_directory"])
    data_other = np.empty((0,181))
    for f in files_list:
        file_path = os.path.join(param["other_object_data_directory"],f)
        data_new = np.load(file_path)
        data_other = np.append(data_other,data_new,axis=0)
        plt.subplots(1, 1)
        plt.plot(data_new[0,:],color='blue')
        file_path = os.path.join(param["pic_directory"],f.replace('npy','jpg'))
        plt.savefig(file_path)

    np.random.shuffle(data_other)

    #making abnormal data
    data_size = 2000
    data_output = np.zeros((data_size,data.shape[1]))
    for i in range (2000):
        random.seed(i)
        a = np.random.uniform(min_value, max_value)
        b = np.random.randint(0, data.shape[0])
        c = np.random.randint(0, data_other.shape[0])
        noise = a * data_other[c]
        data_output[i] = (1-a) * data[b] + noise

    return data_output

# data augmentation
def data_aug(data):
    
    data_size = 2000
    data_output = np.zeros((data_size,data.shape[1]))
    for i in range (2000):
        random.seed( )
        a = np.random.uniform(0,1)
        random.seed( )
        b = np.random.randint(0,data.shape[0])
        random.seed( )
        c = np.random.randint(0,data.shape[0])
        data_1 = a*data[b,:]
        data_2 = (1.0-a)*data[c,:]
        data_output[i,:] = data_1 + data_2

    return data_output
