import numpy as np
# import os
# import sys
# sys.path.append('../')

import json
from sklearn.model_selection import train_test_split
import os

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("dir is created successful")

def load_data(data_loc):
    data = np.loadtxt(data_loc, delimiter=',')
    return data

def load_json(json_loc):
    with open(json_loc) as f:
        config = json.load(f)
    return config

def load_alldata_from_json(json_loc):
    json_dirs = os.listdir(json_loc)
    json_dirs.sort(key=lambda x: int(x[4:-5]))
    all_data = {}
    for j_file_idx in range(len(json_dirs)):
        j_file = json_dirs[j_file_idx]
        data = load_json(os.path.join(json_loc,j_file))
        all_data['data%s'%str(j_file_idx + 1)] = data
    return all_data


def count_json_num(json_loc):
    json_dirs = os.listdir(json_loc)
    json_dirs.sort(key=lambda x: int(x[4:-5]))
    return len(json_dirs),json_dirs

def load_alldata_from_txt(txt_loc):
    txt_dirs = os.listdir(txt_loc)
    txt_dirs.sort(key=lambda x: int(x[4:-4]))
    all_data = []
    for t_file_idx in range(len(txt_dirs)):
        data = load_data(os.path.join(txt_loc,txt_dirs[t_file_idx]))
        all_data.append(data)
    return all_data

def write_data(data_loc,data):
    np.savetxt(data_loc, data, fmt = "%.0f",delimiter=',')



def red_all_txt(dir):
    filenames = os.listdir(dir)
    # filenames.sort()
    filenames.sort(key=lambda x: int(x[4:]))
    # print(filenames)
    all_txt = []
    for di in filenames:
        dir_txts = os.listdir(dir + di)
        dir_txts.sort(key = lambda x: int(x[4:-4]))
        # print(dir_txts)
        alldir_txt = [dir + di + "/" + i for i in dir_txts]
        # print(alldir_txt)
        all_txt.extend(alldir_txt)
    # print(all_txt)
    # print(len(all_txt))
    return all_txt





def load_arr_gra(arr_loc,arr_i):
    # print(arr_loc+ "pb_data%d.txt"%arr_i)
    pb = load_data(arr_loc+ "pb_data%d.txt"%arr_i).astype(int)

    node = load_data(arr_loc+ "node_data%d.txt"%arr_i).astype(int)
    hu = load_data(arr_loc+ "hu_data%d.txt"%arr_i).astype(int)
    tran_num = load_data(arr_loc+ "tran_data%d.txt"%arr_i).astype(int)

    return node,hu,pb,tran_num

def save_data_to_json(outfile,data):
    with open(outfile, 'w') as f:
        json.dump(data, f)


def load_all_txt(dir):
    dir_txts = os.listdir(dir)
    dir_txts.sort(key=lambda x: int(x[4:-4]))

    return dir_txts

def addpre_to_dict(node_f_num,key,value,pre_dict):
    arr_dict = {}
    n_unit = []
    for row in value['arr_vlist']:
        row_n = np.zeros(node_f_num)
        for i in range(len(row)):
            row_n[i] = row[i]
        row_n[len(row):] = 0
        n_unit.append(row_n.tolist())
    arr_dict["node_f"] = n_unit
    arr_dict["edge_index"] = value['arr_edge']
    arr_dict["edge_f"] = np.array(value['spn_labda'])[[int(pa) for pa in value['arr_tranidx']]].tolist()
    arr_dict["label"] = value['spn_mu']
    pre_dict[key] = arr_dict
    return pre_dict

def sample_dir_json(sample_num,dir_loc):
    all_list = []
    json_nums,json_dirs = count_json_num(dir_loc)
    data_range = np.arange(json_nums)
    sample_jsons = np.random.choice(json_dirs, sample_num, replace=False)
    sample_flag = False
    # sample_jsons = json_dirs[sample_index]
    for sj in sample_jsons:
        while sample_flag == False:
            cur_sj = os.path.join(dir_loc, sj)
            cur_dict = load_json(cur_sj)
            # 去除噪声，防止插入非常不合理的数据
            if cur_dict['spn_mu'] >= -100 and cur_dict['spn_mu'] <= 100:
                sample_flag = True
            else:
                sj = np.random.choice(json_dirs, 1, replace=False)[0]
        all_list.append(cur_dict)
    return all_list

def gen_dict(all_datas):
    counter = 1
    data_dict = {}
    for data in all_datas:
        data_dict["data%s"%str(counter)] = data
        counter += 1
    return data_dict


def load_arr_data(loc,lowlimit,upperlimit):
    """
        parameters：

        loc : string 文件位置，例如：
        "data/100_100data/arr/node_data%s.txt" % (j)

        upperlimit ： j的最大值

        return ：
        所有数据的list

    """
    all_data = []
    for i in range(lowlimit,upperlimit):
        all_data.append(load_data(loc%str(i+1)))

    return all_data

def load_labda_mu(loc,i,lowlimit,upperlimit):
    """
        parameters：

        loc : string 文件位置，例如：
        labda_loc = loc_root + "labda/labda%s/data%s.txt"(i,j)

        upperlimit ： j的最大值

        return ：
        所有数据的list

    """
    all_data = []
    for j in range(lowlimit,upperlimit):
        all_data.append(load_data(loc%(str(i),str(j+1))))

    return all_data
# def packagedata(save_root,data_root):
# # save_root = "/home/mingjian/Dataset/SGN/paperdataset/0813/DS%s/package_data" % str(ds_idx)
# # data_root = "/home/mingjian/Dataset/SGN/paperdataset/0813/DS%s/preprocessd_data" % str(ds_idx)
#     mkdir(save_root)
#     dataset = NetLearningDatasetDGL(data_root)
#
#     start = time.time()
#     with open(os.path.join(save_root,'dataset.pkl') ,'wb') as f:
#         pickle.dump([dataset.train,dataset.test],f)
#     print('Time (sec):',time.time() - start)


def partition_datasets(json_data_loc,node_f_num,ratio = 0.2):
    # json_data_loc = "/home/mingjian/Dataset/SGN/paperdataset/0813/DS4"
    # json_data_loc = "data/SGNData/0823/3"
    ori_data_loc = "ori_data"
    prepro = "preprocessd_data"
    # config_loc = "config/DataConfig/SPNGenerate.json"
    # config = load_json(config_loc)
    # node_f_num = config["max_place_num"] + 1
    edge_f_num = 1
    # DU.mkdir(os.path.join(json_data_loc,ori_data_loc))
    all_data = load_json(os.path.join(json_data_loc, ori_data_loc, "all_data.json"))
    print(len(all_data))
    train_data, test_data = train_test_split(list(all_data.values()), test_size=ratio, random_state=0)
    print(len(train_data))
    # print(train_data[0])
    train_data_dict = gen_dict(train_data)
    test_data_dict = gen_dict(test_data)
    # print(len(train_data_dict))
    # print(train_data_dict["data1"])
    save_data_to_json(os.path.join(json_data_loc, ori_data_loc, "train_data.json"), train_data_dict)
    save_data_to_json(os.path.join(json_data_loc, ori_data_loc, "test_data.json"), test_data_dict)

    pre_train_dict = {}
    for key, value in train_data_dict.items():
        pre_train_dict = addpre_to_dict(node_f_num, key, value, pre_train_dict)

    pre_test_dict = {}
    for key, value in test_data_dict.items():
        pre_test_dict = addpre_to_dict(node_f_num, key, value, pre_test_dict)

    mkdir(os.path.join(json_data_loc, prepro))
    # DU.mkdir(os.path.join(json_data_loc, pre_data_loc))
    save_data_to_json(os.path.join(json_data_loc, prepro, "train_data.json"), pre_train_dict)
    save_data_to_json(os.path.join(json_data_loc, prepro, "test_data.json"), pre_test_dict)
    # packagedata(os.path.join(json_data_loc, "package_data"),os.path.join(json_data_loc, prepro))

def get_lowest_idx(va,vec):
    # idx = 0
    for i in range(len(vec)):
        idx = i + 1
        if va < vec[i]:
            return idx
    idx = len(vec)
    return idx