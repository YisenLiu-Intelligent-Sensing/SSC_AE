"""
 @file   SSC_AE_model.py
 @brief  Script for tensorflow models
 @author Yisen Liu
 Copyright (C) 2021 Institute of Intelligent Manufacturing, Guangdong Academy of Sciences. All right reserved.
"""

########################################################################
# import python-library
########################################################################
import tensorflow as tf
import tflearn

########################################################################
# AE model
########################################################################
def AE(x,reuse):
    with tf.variable_scope("AE",reuse=tf.AUTO_REUSE):

        h = tflearn.fully_connected(x,32, activation='relu')
        h = tflearn.fully_connected(h,16)
        code=h
        h = tflearn.fully_connected(h,32, activation='relu')
        h = tflearn.fully_connected(h,181,activation='sigmoid')

    return h,code
#########################################################################


# self-supervised classifier model
########################################################################
def discriminator(x,reuse):
    with tf.variable_scope("Disc",reuse=tf.AUTO_REUSE):

        x=tf.reshape(x,(-1,16,1))
        c=tflearn.batch_normalization(x)
        c=tflearn.fully_connected(c,16,activation='tanh')
        c=tflearn.fully_connected(c,16,activation='tanh')
        c=tflearn.fully_connected(c,2,activation='softmax')
        
    return c
#########################################################################
    
