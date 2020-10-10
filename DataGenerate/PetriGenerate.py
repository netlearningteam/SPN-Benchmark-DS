#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : PetriGenerate.py
# @Date    : 2020-08-22
# @Author  : mingjian
    描述
"""
import numpy as np
from collections import Counter


def is_place(node_idx,place_num):
    if node_idx > place_num :
        return False
    return True

def split_pt(node_list,place_num):
    p_list = []
    t_list = []
    for i in node_list:
        if i > place_num:
            t_list.extend([i])
        else:
            p_list.extend([i])

    return p_list,t_list



def dele_edage(gra_matrix,tran_num):


    for i in range(len(gra_matrix)):
        rowcontent = Counter(gra_matrix[i,0:-1])
        if rowcontent[1] >= 3 :
            itemindex = np.argwhere(gra_matrix[i,0:-1] == 1)
            itemindex = itemindex.reshape(len(itemindex))
            #print(itemindex)
            rmindex = np.random.choice(itemindex,rowcontent[1] - 2)
            gra_matrix[i][rmindex] = 0
    #print(gra_matrix)

    for i in range(2*tran_num):
        colcontent = Counter(gra_matrix[:,i])
        if colcontent[1] >= 3:
            itemindex = np.argwhere(gra_matrix[:,i] == 1)
            itemindex = itemindex.reshape(len(itemindex))
            #print(itemindex)
            rmindex = np.random.choice(itemindex, colcontent[1] - 2)
            #print(rmindex)
            #print(i)
            for rmidx in rmindex:
                gra_matrix[rmidx][i] = 0
    #print(gra_matrix)

    return  gra_matrix

def add_node(petri_matrix, tran_num):
    #print("************add_node***********")
    leftmatrix = petri_matrix[:, 0:tran_num]
    rightmatrix = petri_matrix[:, tran_num:-1]

    #each column must have a 1
    for i in range(2*tran_num):
        colcontent = Counter(petri_matrix[:, i])
        if colcontent[1] < 1:
            rand_idx = np.random.randint(0, len(petri_matrix))
            petri_matrix[rand_idx][i] = 1


    # Each row must have two elements of 1, the left matrix has 1, and the right must also have 1
    for i in range(len(petri_matrix)):
        rowcontent = Counter(leftmatrix[i])
        if rowcontent[1] < 1:
            rand_idx = np.random.randint(0,tran_num)
            petri_matrix[i][rand_idx] = 1

        rowcontent = Counter(rightmatrix[i])
        if rowcontent[1] < 1:
            rand_idx = np.random.randint(0, tran_num)
            petri_matrix[i][rand_idx + tran_num] = 1


    #print(petri_matrix)
    return petri_matrix


def rand_generate_petri(place_num,tran_num):
    sub_graph = []
    remain_node = [i + 1 for i in range(place_num + tran_num)]
    petri_matrix = np.zeros((place_num, 2 * tran_num + 1), dtype=int)
    # The first selected point in the picture, randomly find other points for him to connect
    p_list, t_list = split_pt(remain_node, place_num)
    first_p = np.random.choice(p_list)
    first_t = np.random.choice(t_list)
    # first_node = np.random.randint(1, place_num + tran_num + 1)

    sub_graph.extend([first_p,first_t])
    remain_node.remove(first_p)
    remain_node.remove(first_t)
    rand_num = np.random.rand(0, 1)
    #print(rand_num)
    if rand_num <= 0.5:
        petri_matrix[first_p - 1][first_t - place_num - 1] = 1
    else:
        petri_matrix[first_p - 1][first_t - place_num - 1 + tran_num] = 1
    np.random.shuffle(remain_node)


    for i in range(len(remain_node)):
        # p_list, t_list = split_pt(remain_node, place_num)
        subp_list, subt_list = split_pt(sub_graph, place_num)

        # if len(p_list) > 0 and len(subt_list) > 0:
        #     node1 = np.random.choice(p_list)
        # else:
        #     node1 = np.random.choice(t_list)
        node1 = np.random.choice(remain_node)
        if is_place(node1, place_num):
            node2 = np.random.choice(subt_list)
            rand_num = np.random.rand(0, 1)
            #print(rand_num)
            if rand_num <= 0.5:
                petri_matrix[node1 - 1][node2 - place_num - 1] = 1
            else:
                petri_matrix[node1 - 1][node2 - place_num - 1 + tran_num] = 1
        else:
            node2 = np.random.choice(subp_list)
            rand_num = np.random.rand(0, 1)
            if rand_num <= 0.5:
                petri_matrix[node2 - 1][node1 - place_num - 1] = 1
            else:
                petri_matrix[node2 - 1][node1 - place_num - 1 + tran_num] = 1
        sub_graph.extend([node1])
        remain_node.remove(node1)


    # The front is to prevent isolated subgraphs

    # token
    rand_num = np.random.randint(0, place_num)
    petri_matrix[rand_num][-1] = 1

    for i in range(petri_matrix.shape[0]):
        for j in range(petri_matrix.shape[1]):
            if petri_matrix[i][j] == 0:
                rand_num = np.random.randint(0, 10)
                #print(rand_num)
                if rand_num <= 1:
                    petri_matrix[i][j] = 1

    return petri_matrix

def prune_petri(petri_matrix):
    tran_num = (len(petri_matrix[0]) - 1) // 2
    petri_matrix = dele_edage(petri_matrix, tran_num)
    petri_matrix = add_node(petri_matrix, tran_num)

    return petri_matrix

def add_token(petri_matrix):
    for i in range(len(petri_matrix)):
        rand_num = np.random.randint(0, 10)
        # print(rand_num)
        if rand_num <= 2:
            petri_matrix[i][-1] += 1

    return petri_matrix