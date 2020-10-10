#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : DataTransformation.py
# @Date    : 2020-08-23
# @Author  : mingjian
    描述
"""

import numpy as np
from DataGenerate import SPN


def transformation(petri_matrix,place_upper_bound,marks_lower_limit,marks_upper_limit):
    petri_matrix = np.array(petri_matrix)
    ex_data = []
    col = petri_matrix.shape[1]
    row = len(petri_matrix)
    # *****************  del a edge **********************
    for i in range(row):
        for j in range(col - 1):
            if petri_matrix[i][j] == 1:
                da_tp = petri_matrix.copy()
                da_tp[i][j] = 0
                results_dict,finish = SPN.filter_spn(da_tp, place_upper_bound, marks_lower_limit, marks_upper_limit)
                if finish:
                    ex_data.append(results_dict)
    # *****************  add a  edge **********************
    for i in range(row):
        for j in range(col - 1):
            if petri_matrix[i][j] == 0:
                da_tp = petri_matrix.copy()
                da_tp[i][j] = 1
                results_dict, finish = SPN.filter_spn(da_tp, place_upper_bound, marks_lower_limit,
                                                      marks_upper_limit)
                if finish:
                    ex_data.append(results_dict)
    # *****************  add a token **********************
    for i in range(row):
        da_tp = petri_matrix.copy()
        da_tp[i][col - 1] += 1
        results_dict, finish = SPN.filter_spn(da_tp, place_upper_bound, marks_lower_limit, marks_upper_limit)
        if finish:
            ex_data.append(results_dict)
    # *****************  del a token **********************
    for i in range(row):
        da_tp = petri_matrix.copy()
        if  np.sum(da_tp[:,-1]) > 1 and da_tp[i][col - 1] >= 1:
            da_tp[i][col - 1] -= 1
            results_dict, finish = SPN.filter_spn(da_tp, place_upper_bound, marks_lower_limit, marks_upper_limit)
            if finish:
                ex_data.append(results_dict)
    # *****************  add a place **********************
    for i in range(col - 2):
        row_zero = np.zeros((1,col))
        col_list = [j for j in range(i+1,col-1)]
        for k in col_list:
            da_tp = np.row_stack((petri_matrix, row_zero))
            da_tp[-1][i] = 1
            da_tp[-1][k] = 1
            results_dict, finish = SPN.filter_spn(da_tp, place_upper_bound, marks_lower_limit, marks_upper_limit)
            if finish:
                ex_data.append(results_dict)
    labda_list = []
    ex_num = 0
    while ex_num <=100:
        for ex in ex_data:
            la_num = np.random.randint(1, 10)
            # la_num = 5
            for i in range(la_num):
                da_tp = ex["petri_net"]
                results_dict, finish = SPN.filter_spn(da_tp, place_upper_bound, marks_lower_limit, marks_upper_limit)
                if finish:
                    labda_list.append(results_dict)
        ex_num = len(labda_list)
    la_num = np.random.randint(1, 10)
    # la_num = 5
    for i in range(la_num):
        da_tp = petri_matrix.copy()
        results_dict, finish = SPN.filter_spn(da_tp, place_upper_bound, marks_lower_limit, marks_upper_limit)
        if finish:
            labda_list.append(results_dict)


    ex_data.extend(labda_list)

    return ex_data


def labda_transformation(petri_dict,labda_num):
    all_labda_list = []
    # petri_matrix = np.array(petri_matrix)
    petri_net = petri_dict['petri_net']
    arr_vlist = petri_dict['arr_vlist']
    arr_edge = petri_dict['arr_edge']
    arr_tranidx = petri_dict['arr_tranidx']
    tran_num = (len(petri_net[0]) - 1) // 2
    while len(all_labda_list) < labda_num:
        labda = np.random.randint(1, 11, size=tran_num)
        results_dict, finish = SPN.get_spn(petri_net
                                           ,arr_vlist,
                                           arr_edge,arr_tranidx,
                                           labda)
        if finish:
            all_labda_list.append(results_dict)
    return all_labda_list

