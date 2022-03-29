"""
 @file   train.py
 @brief  Script for training
 @author Yisen Liu
 Copyright (C) 2021 Institute of Intelligent Manufacturing, Guangdong Academy of Sciences. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob

import common as com
import make_abnormal
import numpy as np
import tensorflow as tf
import tflearn

import SSC_AE_model


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################


def load_data(seed):

  healthy_paths = glob.glob(os.path.join(param["data_directory"],"healthy*.npy"))
  
  normal_data_mean = []
  for p in healthy_paths:

    normal_data_mean.append(np.load(p))

  normal_data_mean = np.concatenate(normal_data_mean,axis=0)

  #split train and test 

  np.random.seed(seed)
  np.random.shuffle(normal_data_mean)
  normal_train_data = normal_data_mean[0:normal_data_mean.shape[0]//2]

  normal_train_data = make_abnormal.data_aug(normal_train_data)

  #normalization
  com.normalize_data(normal_train_data)
  
  # make labels for normal samples
  y_normal=np.zeros((normal_train_data.shape[0],2))
  y_normal[:,0] = 1

  return normal_train_data,y_normal


def make_abnormal_data(data,max_value=0.3,min_value=0.05):
  #generation for abnormal data
  abnormal_data = make_abnormal.interpolate(data,max_value=0.3,min_value=0.05)

  # make labels for abnormal samples
  y_abnormal = np.zeros((abnormal_data.shape[0],2))
  y_abnormal[:,1] = 1

  # normalization
  com.normalize_data(abnormal_data)

  return abnormal_data,y_abnormal

########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
  #set GPU
  os.environ['CUDA_VISIBLE_DEVICES'] = '2'

  # make output directory
  os.makedirs(param["model_directory"], exist_ok=True)

  # load base_directory list
  dirs = com.select_dirs(param=param)

  for itr in range (0,10):

    # set path
    sample_type = 'strawberry'
    model_file_path = "{model}/model_SSC_AE_{sample_type}_{itr}itr.model".format(model=param["model_directory"],
                                                                    sample_type=sample_type,itr=itr)
    history_img = "{model}/history_{sample_type}_{itr}itr.png".format(model=param["model_directory"],
                                                              sample_type=sample_type,itr=itr)

    with tf.Graph().as_default():

      # generate dataset
      print("============== DATASET_GENERATOR ==============")
      normal_train_data,y_normal = load_data(itr)
      abnormal_train_data,y_abnormal = make_abnormal_data(normal_train_data)

      # Input tensor define
      normal_input_tensor = tf.placeholder(tf.float32, shape=[None, normal_train_data.shape[1]],name='normal_input_tensor')
      abnormal_input_tensor = tf.placeholder(tf.float32, shape=[None, normal_train_data.shape[1]],name='abnormal_input_tensor')
      Discriminator_normal_label_tensor = tf.placeholder(tf.float32, shape=[None,2], name='Discriminator_normal_label_tensor')
      Discriminator_abnormal_label_tensor = tf.placeholder(tf.float32, shape=[None,2], name='Discriminator_abnormal_label_tensor')
      
      # Build AE
      rebuilt_normal_data,code_normal_data = SSC_AE_model.AE(normal_input_tensor,reuse=tf.AUTO_REUSE)
      rebuilt_abnormal_data,code_abnormal_data = SSC_AE_model.AE(abnormal_input_tensor,reuse=True)

      # Build self-supervised discriminator
      dis_pred_normal = SSC_AE_model.discriminator(SSC_AE_model.AE(normal_input_tensor,reuse=True)[1],reuse=True)
      dis_pred_abnormal = SSC_AE_model.discriminator(SSC_AE_model.AE(abnormal_input_tensor,reuse=True)[1],reuse=True)

      #label clip
      dis_pred_normal = tf.clip_by_value(dis_pred_normal, 1e-5,(1. - 1e-5))
      dis_pred_abnormal = tf.clip_by_value(dis_pred_abnormal, 1e-5,(1. - 1e-5))

      #define optimizer
      Optimizer = tf.train.AdamOptimizer(param["fit"]["learning_rate"])
      disc_Optimizer = tf.train.AdamOptimizer(param["fit"]["learning_rate"])

      #define loss
      loss_AE_normal = tf.reduce_mean(tf.square(normal_input_tensor-rebuilt_normal_data))
      loss_AE_abnormal = tf.reduce_mean(tf.square(abnormal_input_tensor-rebuilt_abnormal_data))
      loss_discriminator_normal = tflearn.categorical_crossentropy (dis_pred_normal,Discriminator_normal_label_tensor)
      loss_discriminator_abnormal = tflearn.categorical_crossentropy (dis_pred_abnormal,Discriminator_abnormal_label_tensor)
      loss_discriminator = 0.5*loss_discriminator_normal + 0.5*loss_discriminator_abnormal
      loss = 0.0001*loss_AE_normal + 0.5*loss_discriminator

      #define backward
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        gradients = Optimizer.compute_gradients(loss)
        capped_gradients, fixed_global_norm = tf.contrib.opt.clip_gradients_by_global_norm(gradients,clip_norm=5.0)
        apply_gradients = Optimizer.apply_gradients(capped_gradients)
    
        disc_gradients = disc_Optimizer.compute_gradients(loss_discriminator)
        disc_capped_gradients, disc_fixed_global_norm = tf.contrib.opt.clip_gradients_by_global_norm(disc_gradients,clip_norm=5.0)
        disc_apply_gradients = disc_Optimizer.apply_gradients(disc_capped_gradients)

      #define train step
      def train_step(sess, normal_train_data, abnormal_train_data, y_normal,y_abnormal):

        feed_dict = {
          normal_input_tensor: normal_train_data, 
          abnormal_input_tensor: abnormal_train_data, 
          Discriminator_normal_label_tensor: y_normal,
          Discriminator_abnormal_label_tensor: y_abnormal,
        }

        loss_value, loss_AE_value, loss_disc_normal_value,loss_disc_abnormal_value, _ = sess.run([loss,loss_AE_normal, loss_discriminator_normal,loss_discriminator_abnormal, apply_gradients], feed_dict=feed_dict)

        return loss_value, loss_AE_value, loss_disc_normal_value,loss_disc_abnormal_value
      

      def pre_train_step(sess, normal_train_data, abnormal_train_data, y_normal,y_abnormal):

        feed_dict= {normal_input_tensor: normal_train_data, abnormal_input_tensor:abnormal_train_data, Discriminator_normal_label_tensor:y_normal,Discriminator_abnormal_label_tensor:y_abnormal}
        loss_AE_normal_value,loss_disc_normal_value,loss_disc_abnormal_value, _ = sess.run([loss_AE_normal,loss_discriminator_normal,loss_discriminator_abnormal, disc_apply_gradients], feed_dict=feed_dict)

        return loss_AE_normal_value,loss_disc_normal_value,loss_disc_abnormal_value


      # get batch samples for abnormal data
      def get_batch_abnormal(data,seed):

        batch_size = 300
        np.random.seed(seed)
        np.random.shuffle(data)

        data_output = data[:batch_size]
        label_output = np.zeros((data_output.shape[0],2))
        label_output[:,1] = 1

        return data_output,label_output

      # get batch samples for normal data
      def get_batch_normal(data,seed):
        batch_size = 300
        np.random.seed(seed)
        np.random.shuffle(data)

        data_output = data[:batch_size]
        label_output = np.zeros((data_output.shape[0],2))
        label_output[:,0] = 1

        return data_output,label_output
      
      # train model
      print("============== MODEL TRAINING ==============")
      with tf.Session() as sess:

        model_saver = tf.train.Saver()
        # initialize
        sess.run(tf.global_variables_initializer())

        check_interval = 10

        #training
        for i in range(2000):

          abnormal_train_data_batch,y_abnormal_batch=get_batch_abnormal(abnormal_train_data,i)
          normal_train_data_batch,y_normal_batch=get_batch_normal(normal_train_data,i)

          loss_AE_normal_value,loss_disc_normal_value, loss_disc_abnormal_value = pre_train_step(sess, normal_train_data=normal_train_data_batch, abnormal_train_data=abnormal_train_data_batch, y_normal=y_normal_batch,y_abnormal=y_abnormal_batch)

          if i % check_interval == 0:

            print('pre-train''iteration:',i,'loss_AE_normal',loss_AE_normal_value,'loss_Disc_normal',loss_disc_normal_value,'loss_Disc_abnormal',loss_disc_abnormal_value)

        # hard sample
        # abnormal_train_data, y_abnormal = make_abnormal_data(normal_train_data, max=0.1, min=0.03)
        
        for i in range(2000):

          abnormal_train_data_batch,y_abnormal_batch=get_batch_abnormal(abnormal_train_data,i)
          normal_train_data_batch,y_normal_batch=get_batch_normal(normal_train_data,i)
          _, loss_AE_normal_value, loss_disc_normal_value, loss_disc_abnormal_value = train_step(sess, normal_train_data=normal_train_data_batch, abnormal_train_data=abnormal_train_data_batch, y_normal=y_normal_batch,y_abnormal=y_abnormal_batch)
          
          if i % check_interval == 0:
            
            print('train''iteration:',i,'loss_AE_normal',loss_AE_normal_value,'loss_Disc_normal',loss_disc_normal_value,'loss_Disc_abnormal',loss_disc_abnormal_value)

            
        #save model
        model_saver.save(sess, model_file_path)

  print("============== END TRAINING ==============")
