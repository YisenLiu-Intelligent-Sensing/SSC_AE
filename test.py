"""
 @file   test.py
 @brief  Script for test
 @author Yisen Liu
 Copyright (C) 2021 Institute of Intelligent Manufacturing, Guangdong Academy of Sciences. All right reserved.
"""

import csv
import glob
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn import metrics

import common as com
import SSC_AE_model

# load parameter.yaml
########################################################################
param = com.yaml_load()
#######################################################################

########################################################################
#save csv file
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


#load normal train data
def load_normal_train_data(seed):
  healthy_paths = glob.glob(os.path.join(param["data_directory"],"healthy*.npy"))
  
  normal_data_mean = []
  for p in healthy_paths:

    normal_data_mean.append(np.load(p))

  normal_data_mean = np.concatenate(normal_data_mean,axis=0)

  #split train and test 

  np.random.seed(seed)
  np.random.shuffle(normal_data_mean)
  normal_train_data = normal_data_mean[0:normal_data_mean.shape[0]//2]

  #normalization
  com.normalize_data(normal_train_data)

  return normal_train_data


#load normal test data
def load_normal_test_data(seed):

  healthy_paths = glob.glob(os.path.join(param["data_directory"],"healthy*.npy"))
  
  normal_data_mean = []
  for p in healthy_paths:

    normal_data_mean.append(np.load(p))

  normal_data_mean = np.concatenate(normal_data_mean,axis=0)

  #split train and test 

  np.random.seed(seed)
  np.random.shuffle(normal_data_mean)
  normal_test_data = normal_data_mean[normal_data_mean.shape[0]//2:]

  #normalization
  com.normalize_data(normal_test_data)

  # define label
  y_true_normal = np.zeros((normal_test_data.shape[0]))

  return normal_test_data, y_true_normal


# load_abnormal_test_data
def load_abnormal_test_data():

    data_file = os.path.join(param["data_directory"],'bruise_mean_2.npy')
    abnormal_data_mean_1 = np.load(data_file)
    print('bruise:',abnormal_data_mean_1.shape[0])

    data_file = os.path.join(param["data_directory"],'decay_mean_2.npy')
    abnormal_data_mean_2 = np.load(data_file)
    print('decay:',abnormal_data_mean_2.shape[0])

    data_file = os.path.join(param["data_directory"],'contamination_mean_2.npy')
    abnormal_data_mean_3 = np.load(data_file)
    print('contaminated:',abnormal_data_mean_3.shape[0])

    abnormal_test_data = np.concatenate([abnormal_data_mean_1,abnormal_data_mean_2,abnormal_data_mean_3],axis=0)
    print('abnormal:',abnormal_test_data.shape)

    #define label
    y_true_abnormal = np.ones((abnormal_test_data.shape[0]))
    
    #normalization
    com.normalize_data(abnormal_test_data)
    com.normalize_data(abnormal_data_mean_1)
    com.normalize_data(abnormal_data_mean_2)
    com.normalize_data(abnormal_data_mean_3)
    
    return abnormal_test_data, y_true_abnormal,abnormal_data_mean_1,abnormal_data_mean_2,abnormal_data_mean_3,abnormal_data_mean_1.shape[0],abnormal_data_mean_2.shape[0],abnormal_data_mean_3.shape[0]


# define cosine_similarity
def cosine_similarity(x1, x2): 
    if x1.ndim == 1:                                      
        x1 = x1[np.newaxis]
    if x2.ndim == 1:
        x2 = x2[np.newaxis]

    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    cosine_sim = np.dot(x1, x2.T)/(x1_norm*x2_norm+1e-10)
    return cosine_sim

########################################################################
# main test.py
########################################################################
if __name__ == "__main__":

  #set GPU
  os.environ['CUDA_VISIBLE_DEVICES'] = '2'

  # make output result directory
  os.makedirs(param["result_directory"], exist_ok=True)

  auc_total = np.zeros((10))
  prec_total = np.zeros((10))
  recall_total = np.zeros((10))
  f1_total = np.zeros((10))
  ds_total = np.zeros((10))
  acc_normal_total = np.zeros((10))
  acc_bruise_total = np.zeros((10))
  acc_decay_total = np.zeros((10))
  acc_contaminated_total = np.zeros((10))

  # initialize lines in csv for statistical result
  csv_lines = []

  # results by type
  csv_lines.append(["AUC", "F1 score","ACC_normal","ACC_bruise","ACC_decay","ACC_contaminated"])

  for itr in range (0,10):

    # setup anomaly score file path
    sample_type='strawberry'
    anomaly_score_csv = "{result}/anomaly_score_{sample_type}_{itr}itr.csv".format(result=param["result_directory"],
                                                                                sample_type=sample_type,itr=itr)
    anomaly_score_list = []
    # setup decision result file path
    decision_result_csv = "{result}/decision_result_{sample_type}_{itr}itr.csv".format(result=param["result_directory"],
                                                                                                        sample_type=sample_type,
                                                                                                        itr=itr)
    decision_result_list = []

    # load test file
    normal_test_data,y_true_normal = load_normal_test_data(seed=itr)
    abnormal_test_data,y_true_abnormal,abnormal_data_mean_1,abnormal_data_mean_2,abnormal_data_mean_3,abnormal_size1,abnormal_size2,abnormal_size3 = load_abnormal_test_data()
    y_true_normal=np.array(y_true_normal)
    y_true_abnormal=np.array(y_true_abnormal)
    y_true=np.concatenate([y_true_normal,y_true_abnormal],axis=0)
    test_data=np.concatenate([normal_test_data,abnormal_test_data],axis=0)

    normal_train_data=load_normal_train_data(itr)

    with tf.Graph().as_default():

      # Input tensor define
      normal_input_tensor = tf.placeholder(tf.float32, shape=[None, normal_test_data.shape[1]],name='normal_input_tensor')
      abnormal_input_tensor = tf.placeholder(tf.float32, shape=[None, normal_test_data.shape[1]],name='abnormal_input_tensor')
      Discriminator_normal_label_tensor = tf.placeholder(tf.float32, shape=[None,2], name='Discriminator_normal_label_tensor')
      Discriminator_abnormal_label_tensor = tf.placeholder(tf.float32, shape=[None,2], name='Discriminator_abnormal_label_tensor')
      
      # Build AE
      rebuilt_normal_data,code_normal_data=SSC_AE_model.AE(normal_input_tensor,reuse=tf.AUTO_REUSE)
      rebuilt_abnormal_data,code_abnormal_data=SSC_AE_model.AE(abnormal_input_tensor,reuse=True)

      # Build discriminator
      dis_pred_normal = SSC_AE_model.discriminator(SSC_AE_model.AE(normal_input_tensor,reuse=True)[1],reuse=True)
      dis_pred_abnormal = SSC_AE_model.discriminator(SSC_AE_model.AE(abnormal_input_tensor,reuse=True)[1],reuse=True)

      vars = tf.trainable_variables()

      #test step for AE model 
      def AE_test_step(sess, test_data):

        feed_dict = {normal_input_tensor: test_data}
        rebuilt_normal_data_value = sess.run(rebuilt_normal_data, feed_dict=feed_dict)

        return rebuilt_normal_data_value
      
      #test step for self-supervised classifier model
      def Disc_test_step(sess, test_data):
  
        feed_dict = {normal_input_tensor: test_data}
        disc_pred_value = sess.run(dis_pred_normal, feed_dict=feed_dict)

        return disc_pred_value
      
      # test step for getting AE code
      def Code_test_step(sess, test_data):
  
        feed_dict = {normal_input_tensor:test_data}
        code_pred_value = sess.run(code_normal_data, feed_dict=feed_dict)
        code_pred_value = code_pred_value.reshape((code_pred_value.shape[0], code_pred_value.shape[1]))

        return code_pred_value

      print("============== MODEL LOAD ==============")
      # set model path
      sample_type = 'strawberry'
      model_file = "{model}/model_SSC_AE_{sample_type}_{itr}itr.model".format(model=param["model_directory"],
                                                                      sample_type=sample_type,itr=itr)
      print(model_file)

      print("\n============== BEGIN TEST ==============")
      # load model file
      with tf.Session() as sess:
          
        #load model
        model_saver = tf.train.Saver()
        model_saver.restore(sess,model_file)
        #testing
        rebuilt_test_data = AE_test_step(sess, test_data=test_data)
        disc_pred_test_data = Disc_test_step(sess, test_data=test_data)
        code_pred_test_data = Code_test_step(sess, test_data=test_data)
        train_code_vetor = Code_test_step(sess, test_data=normal_train_data)

        # calculate rebuilt error
        rebuilt_errors = -np.mean(np.square(test_data - rebuilt_test_data), axis=1)
        
        #rebuilt rebuit cosine_similarity error
        rebuilt_cosine_errors = []
        train_rebuilt_vetor = AE_test_step(sess, test_data=normal_train_data)
        for i in range(test_data.shape[0]):
          cos_similarity = cosine_similarity( rebuilt_test_data[i], train_rebuilt_vetor) # shape(len(test), len(train))
          rebuilt_cosine_errors.append(np.mean(cos_similarity))

        errors = np.array(rebuilt_cosine_errors)
        y_pred = -errors

        for i in range(y_true.shape[0]):
          anomaly_score_list.append([y_true[i], y_pred[i]])

        y_pred = np.array(y_pred)

        # save anomaly score
        save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
        com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

        #make normal/abnormal decisions
        decision = np.zeros((y_pred.shape[0]))
        index = np.argsort(y_pred)
        decision[index[0:normal_test_data.shape[0]]]=0
        decision[index[normal_test_data.shape[0]:]]=1

        # save decision results
        save_csv(save_file_path=decision_result_csv, save_data=decision_result_list)
        com.logger.info("decision result ->  {}".format(decision_result_csv))

        print("\n============ END OF TEST ============")

        # caculate statistical results
        auc = metrics.roc_auc_score(y_true, y_pred)
        print('auc:',auc)
        auc_total[itr]=auc

        tn, fp, fn, tp = metrics.confusion_matrix(y_true, decision).ravel()
        prec = tp / np.maximum(tp + fp, sys.float_info.epsilon)
        recall = tp / np.maximum(tp + fn, sys.float_info.epsilon)
        f1 = 2.0 * prec * recall / np.maximum(prec + recall, sys.float_info.epsilon)
        print('prec:',prec)
        print('recall:',recall)
        print('f1:',f1)
        prec_total[itr] = prec
        recall_total[itr] = recall
        f1_total[itr] = f1

        acc_normal = 1 - np.sum(decision[0:y_true_normal.shape[0]]) / y_true_normal.shape[0]
        acc_bruise = np.sum(decision[y_true_normal.shape[0]:y_true_normal.shape[0] + abnormal_size1]) / abnormal_size1
        acc_decay = np.sum(decision[y_true_normal.shape[0] + abnormal_size1:y_true_normal.shape[0] + abnormal_size1 + abnormal_size2]) / abnormal_size2
        acc_contaminated = np.sum(decision[y_true_normal.shape[0] + abnormal_size1 + abnormal_size2:]) / abnormal_size3

        acc_normal_total[itr] = acc_normal
        acc_bruise_total[itr] = acc_bruise
        acc_decay_total[itr] = acc_decay
        acc_contaminated_total[itr] = acc_contaminated

        csv_lines.append(['strawberry_'+str(itr)+'runs', auc, f1, acc_normal, acc_bruise, acc_decay, acc_contaminated])

  # statistical results for 10 runs
  auc_mean = np.mean(auc_total)
  prec_mean = np.mean(prec_total)
  recall_mean = np.mean(recall_total)
  f1_mean = np.mean(f1_total)
  acc_normal_mean = np.mean(acc_normal_total)
  acc_bruise_mean = np.mean(acc_bruise_total)
  acc_decay_mean = np.mean(acc_decay_total)
  acc_contaminated_mean = np.mean(acc_contaminated_total)

  auc_std = np.std(auc_total)
  f1_std = np.std(f1_total)
  acc_normal_std = np.std(acc_normal_total)
  acc_bruise_std = np.std(acc_bruise_total)
  acc_decay_std = np.std(acc_decay_total)
  acc_contaminated_std = np.std(acc_contaminated_total)

  print('auc',auc_total)
  print('f1',f1_total)
  print('acc_normal',acc_normal_total)
  print('acc_bruise',acc_bruise_total)
  print('acc_decay',acc_decay_total)
  print('acc_contaminated',acc_contaminated_total)

  csv_lines.append(['strawberry_10runs_mean', auc_mean, f1_mean, acc_normal_mean, acc_bruise_mean, acc_decay_mean,acc_contaminated_mean])
  csv_lines.append(['strawberry_10runs_std', auc_std, f1_std, acc_normal_std, acc_bruise_std, acc_decay_std, acc_contaminated_std])

  # save results
  result_path = "{result}/{file_name}".format(result=param["result_directory"], file_name='result.csv')
  com.logger.info("statistical results -> {}".format(result_path))
  save_csv(save_file_path=result_path, save_data=csv_lines)

