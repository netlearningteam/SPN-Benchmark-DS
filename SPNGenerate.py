#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : SPNGenerate.py
# @Date    : 2020-08-23
# @Author  : mingjian
    描述
"""
from DataGenerate import PetriGenerate as PeGen
from DataVisualization import Visual
from DataGenerate import SPN
from utils import DataUtil as DU
from joblib import Parallel, delayed
import argparse
import os
import shutil
import numpy as np
from DataGenerate import DataTransformation

def generate_spn(config, write_loc, data_idx):
    place_upper_bound = config['place_upper_bound']
    marks_lower_limit = config['marks_lower_limit']
    marks_upper_limit = config['marks_upper_limit']
    prune_flag = config['prune_flag']
    add_token = config['add_token']
    max_place_num = config['max_place_num']
    min_place_num = config['min_place_num']
    finish = False
    while finish == False:
        place_num = np.random.randint(min_place_num, max_place_num+1)
        tran_num = place_num + np.random.randint(-3, 1)
        petri_matrix = PeGen.rand_generate_petri(place_num,tran_num)
        if prune_flag:
            petri_matrix = PeGen.prune_petri(petri_matrix)
        if add_token:
            petri_matrix = PeGen.add_token(petri_matrix)
        results_dict,finish = SPN.filter_spn(petri_matrix, place_upper_bound, marks_lower_limit, marks_upper_limit)
    DU.save_data_to_json(os.path.join(write_loc,"data%s.json"%str(data_idx)),results_dict)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Please give a config.json file ",default="config/DataConfig/SPNGenerate.json")
    args = parser.parse_args()
    config = DU.load_json(args.config)
    print(config)
    write_data_loc = config['write_data_loc']
    parallel_job = config['parallel_job']
    place_upper_bound = config['place_upper_bound']
    marks_lower_limit = config['marks_lower_limit']
    marks_upper_limit = config['marks_upper_limit']
    data_num = config['data_num']
    visual_flag = config['visual_flag']
    pic_loc = config['pic_loc']
    transformation_flag = config['transformation_flag']
    maxtransform_num = config['maxtransform_num']
    DU.mkdir(write_data_loc)
    tmp_write_dir = os.path.join(write_data_loc,"tmp")
    w_pic_loc = os.path.join(write_data_loc,pic_loc)

    print(tmp_write_dir)
    DU.mkdir(tmp_write_dir)
    Parallel(n_jobs=parallel_job)(delayed(generate_spn)(config, tmp_write_dir, i + 1) for i in range(data_num))
    all_data = DU.load_alldata_from_json(tmp_write_dir)
    # print(len(all_data))
    # print(os.path.join(write_data_loc, "all_data.json"))
    new_transfo_datas = {}
    counter = 1
    print("*"*30 + "data transformation begin" + "*"*30)
    if transformation_flag:
        for data in all_data.values():
            all_ex_data = DataTransformation.transformation(data['petri_net'],place_upper_bound,marks_lower_limit,marks_upper_limit)
            print("transformation num : %s" % str(len(all_ex_data)))
            if len(all_ex_data) >= maxtransform_num:
                data_range = np.arange(len(all_ex_data))
                sample_index = np.random.choice(data_range, maxtransform_num, replace=False)
            else:
                sample_index = np.arange(len(all_ex_data))
            for se_idx in sample_index:
                new_transfo_datas["data%s" % str(counter)] = all_ex_data[se_idx]
                counter += 1
        print("*" * 30 + "data transformation finish" + "*" * 30)
            # print("transformation num : %s"%str(len(all_ex_data)))
            # for ex_data in all_ex_data:
            #     new_transfo_datas["data%s"%str(counter+1)] = ex_data
            #     counter += 1

    # all_data = {**all_data,**new_transfo_datas}
    if transformation_flag:
        all_data = new_transfo_datas
    print("total data number : %s"%str(len(all_data)))
    ori_data_loc = "ori_data"
    DU.mkdir(os.path.join(write_data_loc,ori_data_loc))
    DU.save_data_to_json(os.path.join(write_data_loc,ori_data_loc,"all_data.json"), all_data)
    # del tmp dir
    print(all_data.keys())

    if visual_flag:
        DU.mkdir(w_pic_loc)
        counter = 1
        for data in all_data.values():
            Visual.plot_petri(data['petri_net'], os.path.join(w_pic_loc, "data(petri)%s" % str(counter)))
            Visual.plot_spn(data['arr_vlist'], data['arr_edge'], data['arr_tranidx'],
                            data['spn_labda'], data['spn_steadypro'], data['spn_markdens'],
                            data['spn_allmus'], os.path.join(w_pic_loc, "data(arr)%s" % str(counter))
                            )
            counter += 1
    shutil.rmtree(tmp_write_dir)





