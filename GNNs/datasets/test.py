#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : test.py
# @Date    : 2020-08-28
# @Author  : mingjian
    描述
"""
import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl


ds_idx = 1
save_root = "/home/mingjian/Dataset/SGN/paperdataset/0813/DS%s/package_data"%str(ds_idx)
data_root = "/home/mingjian/Dataset/SGN/paperdataset/0813/DS%s/preprocessd_data"%str(ds_idx)
print(save_root[-12:])

