#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : StatisticsDS.py
# @Date    : 2020-09-03
# @Author  : mingjian
    描述
"""
from utils import DataUtil as DU
import os
import numpy as np
from utils.ExcelTools import ExcelTool as ET
dataset_name = "DS3"  # DS1 ~ DS5
data_type = "GridData"  # "GridData", "RandData"
# data_loc = "/home/mingjian/Dataset/SGN/paperdataset/0813/%s/ori_data" % dataset_name
# data_loc = "/home/mingjian/Dataset/SPN/0904/GridData/%s/ori_data" % dataset_name
data_loc = "Data/%s/%s/ori_data" % (data_type,dataset_name)
save_data_loc = "result/%s/excel"%data_type
all_train_data = DU.load_json(os.path.join(data_loc,"train_data.json"))
all_test_data = DU.load_json(os.path.join(data_loc,"test_data.json"))

if data_type == "RandData":
    row_p = [5 + 2 * (i + 1) for i in range(3)]
    col_m = [4 + 6 * (i + 1) for i in range(8)]
else:
    row_p = [5 + 2 * (i + 1) for i in range(5)]
    col_m = [4 + 4 * (i + 1) for i in range(10)]
print(row_p)
print(col_m)

dir_count = np.zeros([len(row_p),len(col_m)],dtype=int)

for data in all_train_data.values():
    petri_net = data["petri_net"]
    v_list = data["arr_vlist"]
    p_num = DU.get_lowest_idx(len(petri_net), row_p)
    m_num = DU.get_lowest_idx(len(v_list), col_m)
    dir_count[p_num - 1][m_num - 1] = dir_count[p_num - 1][m_num - 1] + 1

for data in all_test_data.values():
    petri_net = data["petri_net"]
    v_list = data["arr_vlist"]
    p_num = DU.get_lowest_idx(len(petri_net), row_p)
    m_num = DU.get_lowest_idx(len(v_list), col_m)
    dir_count[p_num - 1][m_num - 1] = dir_count[p_num - 1][m_num - 1] + 1
print(dir_count)
DU.mkdir(save_data_loc)


ET = ET(save_data_loc,'static%s.xls'%dataset_name,dataset_name)
value_title = [["data distribution"], ]

ET.write_xls(value_title)
ET.write_xls_append([row_p])
ET.write_xls_append([col_m])
ET.write_xls_append(dir_count.astype(int).tolist())

ET.read_excel_xls()
print("data toatal num: ",np.sum(dir_count))