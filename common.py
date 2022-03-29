"""
 @file   common.py
 @brief  Commonly used script
 @author Yisen Liu
Copyright (C) 2021 Institute of Intelligent Manufacturing, Guangdong Academy of Sciences. All right reserved.
"""

########################################################################
# import python-library
########################################################################
import glob
import logging
import os

import yaml

logging.basicConfig(level=logging.DEBUG, filename="SSC-AE.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

########################################################################
# load parameter.yaml
########################################################################
def yaml_load():
    with open("parameter.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

########################################################################

# select data dirs
def select_dirs(param):

    dir_path = os.path.abspath("{base}/*".format(base=param["data_directory"]))
    dirs = sorted(glob.glob(dir_path))

    return dirs
########################################################################

# normalization_data
def normalize_data(data):
  
  for i in range(len(data)):
    data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())

