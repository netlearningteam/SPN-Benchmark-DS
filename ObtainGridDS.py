#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : ObtainGridDS.py
# @Date    : 2020-09-21
# @Author  : mingjian
    描述
"""
import os
import numpy as np
from utils import DataUtil as DU
from DataGenerate import DataTransformation as dts
import pickle
import time
from GNNs.datasets.NetLearningDatasetDGL import NetLearningDatasetDGL

def get_lowest_idx(va,vec):
    # idx = 0
    for i in range(len(vec)):
        idx = i + 1
        if va < vec[i]:
            return idx
    idx = len(vec)
    return idx

def save_tmp_grid_jsons(tmp_grid_loc,accumulation_data,raw_data_loc):
    for i in range(5):
        for j in range(10):
            DU.mkdir(os.path.join(tmp_grid_loc, "p%s" % str(i + 1), "m%s" % str(j + 1)))

    grid_data_info = {}

    if os.path.exists(os.path.join(tmp_grid_loc, "config.json")) and accumulation_data:
        grid_data_info = DU.load_json(os.path.join(tmp_grid_loc, "config.json"))
        row_p = grid_data_info['row_p']
        col_m = grid_data_info['col_m']
        dir_count = grid_data_info['json_count']

    else:
        row_p = [5 + 2 * (i + 1) for i in range(5)]
        col_m = [4 + 4 * (i + 1) for i in range(10)]
        dir_count = np.zeros([len(row_p), len(col_m)], dtype=int)

    grid_data_info['row_p'] = row_p
    grid_data_info['col_m'] = col_m

    print(dir_count)
    print(row_p)
    print(col_m)

    all_data_json = DU.load_json(raw_data_loc)
    print(len(all_data_json))

    # 分栏保存数据
    for data in all_data_json.values():
        petri_net = data["petri_net"]
        v_list = data["arr_vlist"]
        p_num = get_lowest_idx(len(petri_net), row_p)
        m_num = get_lowest_idx(len(v_list), col_m)
        dir_count[p_num - 1][m_num - 1] = dir_count[p_num - 1][m_num - 1] + 1
        DU.save_data_to_json(os.path.join(tmp_grid_loc, "p%s" % str(p_num), "m%s" % str(m_num),
                                          "data%s.json" % str(int(dir_count[p_num - 1][m_num - 1]))), data)

    print(dir_count)
    # print(get_lowest_idx(7,row_p))
    if isinstance(dir_count, list):
        grid_data_info["json_count"] = dir_count
    else:
        grid_data_info["json_count"] = dir_count.tolist()
    DU.save_data_to_json(os.path.join(tmp_grid_loc, "config.json"), grid_data_info)

def main():
    # 生成网格文件夹
    config = DU.load_json("config/DataConfig/PartitionGrid.json")
    tmp_grid_loc = config['tmp_grid_loc']
    raw_data_loc = config['raw_data_loc']
    print(raw_data_loc)
    accumulation_data = config['accumulation_data']
    save_tmp_grid_jsons(tmp_grid_loc,accumulation_data,raw_data_loc)



    # data_root = "data/griddata2"
    save_grid_loc = config['save_grid_loc']
    grid_data_loc = os.path.join(tmp_grid_loc,"p%s","m%s")
    config_loc = os.path.join(tmp_grid_loc,"config.json")
    grid_config = DU.load_json(config_loc)
    print(grid_config)
    p_upper_limit = config['p_upper_limit']
    m_upper_limit = config['m_upper_limit']
    each_grid_num = config['each_grid_num']
    labda_num = config['labda_num']
    print(save_grid_loc)
    # print(len(DU.sample_dir_json(each_grid_num,grid_data_loc%(str(1),str(1)))))
    all_data = []
    for i in range(p_upper_limit):
        for j in range(m_upper_limit):
            all_sample_list = DU.sample_dir_json(each_grid_num,grid_data_loc%(str(i+1),str(j+1)))
            all_data.extend(all_sample_list)
    print(len(all_data))
    dts_new_datas = []
    for data in all_data:
        new_datas = dts.labda_transformation(data,labda_num)
        dts_new_datas.extend(new_datas)
    print(len(dts_new_datas))

    DU.mkdir(os.path.join(save_grid_loc,"ori_data"))
    all_data_ditc = DU.gen_dict(dts_new_datas)
    DU.save_data_to_json(os.path.join(save_grid_loc,"ori_data","all_data.json"),all_data_ditc)
    DU.partition_datasets(save_grid_loc,16,0.2)
    DU.mkdir(os.path.join(save_grid_loc,"package_data"))
    dataset = NetLearningDatasetDGL(os.path.join(save_grid_loc,"preprocessd_data"))

    start = time.time()
    with open(os.path.join(os.path.join(save_grid_loc,"package_data"),'dataset.pkl') ,'wb') as f:
            pickle.dump([dataset.train,dataset.test],f)
    print('Time (sec):',time.time() - start)

main()