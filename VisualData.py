#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : VisualData.py
# @Date    : 2020-08-30
# @Author  : mingjian
    描述
"""

from DataVisualization import Visual
from utils import DataUtil as DU
import os

dstype = "RandData"
ds = "DS1"
para_job = -1
# data_loc = "/home/mingjian/Dataset/SGN/paperdataset/0813/%s/ori_data/"%ds
# save_pic_loc = "/home/mingjian/Dataset/SGN/paperdataset/0813/all_pic/%s"%ds
data_loc = "Data/%s/%s/ori_data/" % (dstype,ds)
save_pic_loc = "Pics/%s/%s/"  % (dstype,ds)



train_pic_loc = os.path.join(save_pic_loc,"train_pic")
test_pic_loc = os.path.join(save_pic_loc,"test_pic")
DU.mkdir(save_pic_loc)
all_train_data = DU.load_json(os.path.join(data_loc,"train_data.json"))
all_test_data = DU.load_json(os.path.join(data_loc,"test_data.json"))

DU.mkdir(train_pic_loc)
DU.mkdir(test_pic_loc)
Visual.visual_data(all_train_data,train_pic_loc,para_job)
Visual.visual_data(all_test_data,test_pic_loc,para_job)