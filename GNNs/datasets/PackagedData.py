#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : PackagedData.py
# @Date    : 2020-08-28
# @Author  : mingjian
    描述
"""
import pickle
import time
import os
from GNNs.datasets.NetLearningDatasetDGL import NetLearningDatasetDGL
from utils import DataUtil as DU
ds_idx = 3
save_root = "/home/mingjian/Dataset/SPN/0904/RandData/DS%s/package_data"%str(ds_idx)
data_root = "/home/mingjian/Dataset/SPN/0904/RandData/DS%s/preprocessd_data"%str(ds_idx)
DU.mkdir(save_root)
dataset = NetLearningDatasetDGL(data_root)

start = time.time()
with open(os.path.join(save_root,'dataset.pkl') ,'wb') as f:
        pickle.dump([dataset.train,dataset.test],f)
print('Time (sec):',time.time() - start)
