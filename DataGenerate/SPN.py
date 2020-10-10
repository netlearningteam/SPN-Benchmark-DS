#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : SPN.py
# @Date    : 2020-08-22
# @Author  : mingjian
    描述
"""
from scipy.linalg import solve
import numpy as np
from numpy.linalg import lstsq
from numpy.linalg import solve
from DataGenerate import ArrivableGraph as ArrGra



def state_equation(v_list, edage_list, arctrans_list, labda):
    m_num = len(v_list)
    redundant_state_matrix = np.zeros((m_num + 1, m_num), dtype=int)
    y_list = np.zeros(m_num, dtype=int)
    y_list[-1] = 1
    redundant_state_matrix[-1, :] = 1

    for ed_idx in range(len(edage_list)):
        # pb_idx = int(re.findall(r"\d+\.?\d*", str(arctrans_list[ed_idx]))[0]) - 1
        pb_idx = int(arctrans_list[ed_idx])
        redundant_state_matrix[edage_list[ed_idx][0], edage_list[ed_idx][0]] += (- labda[pb_idx])
        redundant_state_matrix[edage_list[ed_idx][1], edage_list[ed_idx][0]] += (labda[pb_idx])

    state_matrix = []
    for i in range(m_num - 1):
        state_matrix.append(redundant_state_matrix[i])
    state_matrix.append(redundant_state_matrix[-1, :])
    return state_matrix,y_list



def avg_mark_nums(v_list, steady_state_prob):
    # 寻找可能的托肯数量
    token_list = []
    for v in v_list:
        token_list.extend(np.unique(v))
    # print(token_list)
    token_list = np.unique(token_list)

    # 初始化密度列表，row为标识位置,col为对应token_list 的下标
    mark_dens_list = np.zeros((len(v_list[0]),len(token_list)))
    arr_v_list = np.array(v_list)
    for pi in range(len(v_list[0])):
        tok_uni = np.unique(arr_v_list[:,pi])
        for tok  in tok_uni:
            tok_loc = np.argwhere(token_list == tok)[0]
            pipeitok_idx = np.argwhere(arr_v_list[:, pi] == tok).flatten()
            # pipeitok_idx = np.array(pipeitok_idx,dtype=int)
            mark_dens_list[pi][tok_loc] = np.sum(steady_state_prob[pipeitok_idx])

    mu_mark_nums = []
    for wd_gl in range(len(mark_dens_list)):
        sum = 0
        for p_idx in range(len(token_list)):
            sum = sum +  token_list[p_idx] * mark_dens_list[wd_gl][p_idx]
        mu_mark_nums.extend([sum])

    return mark_dens_list,mu_mark_nums



def generate_sgn_task(v_list, edage_list, arctrans_list, tran_num):
    # labda = np.array([2,1,1,3,2])
    labda = np.random.randint(1,11,size=tran_num)
    state_matrix,y_list = state_equation(v_list, edage_list, arctrans_list, labda)
    # print(np.array(state_matrix))
    # print(y_list)
    sv = None
    try:
        sv = solve(state_matrix,y_list.T)
        mark_dens_list,mu_mark_nums = avg_mark_nums(v_list, sv)
    except np.linalg.linalg.LinAlgError:
        mark_dens_list,mu_mark_nums = None,None
    return sv,mark_dens_list,mu_mark_nums,labda

def convert_data(npdata):
    return np.array(npdata).astype(int).tolist()


def is_connected_graph(petri_matrix):
    petri_matrix = np.array(petri_matrix)
    trans_num = len(petri_matrix[0]) // 2
    flag = True
    # print(np.sum(petri_matrix == 0))
    for row in range(len(petri_matrix)):
        if np.sum(petri_matrix[row,:-1]) == 0:
            return False
    for col in range(trans_num):
        if np.sum(petri_matrix[:,col]) +  np.sum(petri_matrix[:,col+trans_num]) == 0 :
            return False
    return flag


def filter_spn(petri_matrix, place_upper_bound=10, marks_lower_limit = 4, marks_upper_limit=500):
    v_list,edage_list,arctrans_list,tran_num,bound_flag = ArrGra.get_arr_gra(petri_matrix,place_upper_bound,marks_upper_limit)
    results_dict = {}
    if bound_flag == False or len(v_list) < marks_lower_limit:
        return results_dict,False
    sv,mark_dens_list,mu_mark_nums,labda = generate_sgn_task(v_list, edage_list, arctrans_list, tran_num)
    if sv is None:
        return results_dict,False

    if is_connected_graph(petri_matrix) == False:
        return results_dict, False


    results_dict['petri_net'] = convert_data(petri_matrix)
    results_dict['arr_vlist'] = convert_data(v_list)
    results_dict['arr_edge'] = convert_data(edage_list)
    results_dict['arr_tranidx'] = convert_data(arctrans_list)
    results_dict['spn_labda'] = np.array(labda).tolist()
    results_dict['spn_steadypro'] = np.array(sv).tolist()
    results_dict['spn_markdens'] = np.array(mark_dens_list).tolist()
    results_dict['spn_allmus'] = np.array(mu_mark_nums).tolist()
    results_dict['spn_mu'] = np.sum(mu_mark_nums)

    return results_dict,True


def generate_sgn_task_given_labda(v_list, edage_list, arctrans_list,labda):
    # labda = np.array([2,1,1,3,2])
    # labda = np.random.randint(1,11,size=tran_num)
    state_matrix,y_list = state_equation(v_list, edage_list, arctrans_list, labda)
    # print(np.array(state_matrix))
    # print(y_list)
    sv = None
    try:
        sv = solve(state_matrix,y_list.T)
        mark_dens_list,mu_mark_nums = avg_mark_nums(v_list, sv)
    except np.linalg.linalg.LinAlgError:
        mark_dens_list,mu_mark_nums = None,None
    return sv,mark_dens_list,mu_mark_nums

def get_spn(petri_matrix,v_list, edage_list, arctrans_list,labda):
    results_dict = {}
    sv,mark_dens_list,mu_mark_nums = generate_sgn_task_given_labda(v_list, edage_list, arctrans_list, labda)
    if sv is None:
        return results_dict,False
    if is_connected_graph(petri_matrix) == False:
        return results_dict, False
    results_dict['petri_net'] = convert_data(petri_matrix)
    results_dict['arr_vlist'] = convert_data(v_list)
    results_dict['arr_edge'] = convert_data(edage_list)
    results_dict['arr_tranidx'] = convert_data(arctrans_list)
    results_dict['spn_labda'] = np.array(labda).tolist()
    results_dict['spn_steadypro'] = np.array(sv).tolist()
    results_dict['spn_markdens'] = np.array(mark_dens_list).tolist()
    results_dict['spn_allmus'] = np.array(mu_mark_nums).tolist()
    results_dict['spn_mu'] = np.sum(mu_mark_nums)
    return results_dict,True

def get_spnds3(petri_matrix,labda, place_upper_bound=10, marks_lower_limit = 4, marks_upper_limit=500):
    v_list, edage_list, arctrans_list, tran_num, bound_flag = ArrGra.get_arr_gra(petri_matrix, place_upper_bound,
                                                                                 marks_upper_limit)
    results_dict = {}
    if bound_flag == False or len(v_list) < marks_lower_limit:
        return results_dict, False
    sv,mark_dens_list,mu_mark_nums = generate_sgn_task_given_labda(v_list, edage_list, arctrans_list, labda)
    if sv is None:
        return results_dict, False

    if is_connected_graph(petri_matrix) == False:
        return results_dict, False
    mu_sums  = np.sum(mu_mark_nums)
    if mu_sums < -100 and mu_sums > 100:
        return results_dict, False
    results_dict['petri_net'] = convert_data(petri_matrix)
    results_dict['arr_vlist'] = convert_data(v_list)
    results_dict['arr_edge'] = convert_data(edage_list)
    results_dict['arr_tranidx'] = convert_data(arctrans_list)
    results_dict['spn_labda'] = np.array(labda).tolist()
    results_dict['spn_steadypro'] = np.array(sv).tolist()
    results_dict['spn_markdens'] = np.array(mark_dens_list).tolist()
    results_dict['spn_allmus'] = np.array(mu_mark_nums).tolist()
    results_dict['spn_mu'] = np.sum(mu_mark_nums)
    return results_dict,True