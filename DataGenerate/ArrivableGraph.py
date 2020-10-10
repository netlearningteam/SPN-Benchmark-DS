#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : ArrivableGraph.py
# @Date    : 2020-08-22
# @Author  : mingjian
    描述
"""
import numpy as np


def wherevec(vec, matrix):
    for row in range(len(matrix)):
        if np.sum((matrix[row] - vec) == 0) == len(matrix[0]):
            return row
    return -1

def enable_set(A1,A2,M):
    ena_list = []
    ena_mlist = []

    for i in range(A1.shape[1]):
        # Pre-set
        pro_idx = np.argwhere(A1[:,i]== 1).flatten()
        m_token = M[pro_idx].flatten()
        m_enable = np.argwhere(m_token > 0).flatten()
        if len(m_enable) == len(m_token):
            m_temp = M.copy()
            ena_list.extend([i])
            # Update the mark, subtract 1 from the previous set and add 1 to the post set
            for idx in pro_idx:
                m_temp[idx] = m_temp[idx] - 1
            # post set
            post_idx = np.argwhere(A2[:,i]== 1).flatten()
            for idx in post_idx:
                m_temp[idx] = m_temp[idx] + 1


            ena_mlist.append(m_temp)


    return ena_mlist,ena_list



def get_arr_gra(petri_matrix, place_upper_limit=10, marks_upper_limit = 500):

    """
    Obtain the reachable graph of the petri net.

    :param
        petri_matrix: petri net matrix
        upper_limit: The upper bound of the place in petri net, if > upper_limit : unbound petri , else : bound petri
    :return:
        v_list : The set of all vertices of the reachable graph.
        edage_list : The set of all edges of the reachable graph.
        arctrans_list : The set of arc transitions of the reachable graph.
        tran_num : Number of transitions.
        bound_flag : Whether it is a bounded net. If yes, return True, otherwise False.
    """

    petri_matrix = np.array(petri_matrix)
    bound_flag = True
    tran_num = int(petri_matrix.shape[1] / 2)
    # place_num = petri_matrix.shape[0]
    leftmatrix = petri_matrix[:, 0:tran_num]
    rightmatrix = petri_matrix[:, tran_num:-1]
    M0 = np.array(petri_matrix[:, -1], dtype=int)
    counter = 0
    v_list = [M0]
    new_list = [counter]
    edage_list = []
    arctrans_list = []
    C = (rightmatrix - leftmatrix)
    while len(new_list) > 0 :

        if counter > marks_upper_limit:
            bound_flag = False
            return v_list, edage_list, arctrans_list, tran_num, bound_flag

        new_m = np.random.choice(new_list)
        gra_en_sets,tran_sets = enable_set(leftmatrix,rightmatrix,v_list[new_m])
        for bs in gra_en_sets:
            p_num = np.argwhere(bs > place_upper_limit).flatten()
            if len(p_num) > 0:
                bound_flag = False
                return v_list, edage_list, arctrans_list, tran_num, bound_flag

        if len(gra_en_sets) == 0:
            new_list.remove(new_m)
        else:
            # Traverse all enable marks
            for en_m,ent_idx in zip(gra_en_sets,tran_sets):
                # Calculate the current enable transition, generate a new mark and save it in M_new.

                # Use the t matrix to sample, and obtain the mark increase or decrease under the current transition
                t = np.zeros(tran_num)
                t[ent_idx] = 1
                M_new = np.array(v_list[new_m] + np.dot(C,t),dtype=int)
                M_newidx = wherevec(M_new, v_list)
                if  M_newidx == -1:
                    counter += 1
                    v_list.append(M_new)
                    new_list.append(counter)
                    edage_list.append([new_m, counter])
                else:

                    edage_list.append([new_m,M_newidx])
                arctrans_list.extend([ent_idx])
            new_list.remove(new_m)

    return v_list,edage_list,arctrans_list,tran_num,bound_flag