#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : NetLearningDatasetDGL.py
# @Date    : 2020-08-27
# @Author  : mingjian
    描述
"""
from GNNs.datasets.BasicDatasets import BasicDatasets
import torch
import time
import numpy as np
import dgl
import pickle
import os
class NetLearningDatasetDGL(torch.utils.data.Dataset):
    def __init__(self,data_loc):
        start = time.time()
        print("[I] Loading dataset %s..." % (data_loc))
        if data_loc[-12:] == "package_data" :
            with open(os.path.join(data_loc , "dataset.pkl"), "rb") as f:
                f = pickle.load(f)
                self.train = f[0]
                self.test = f[1]
        else:
            self.train = BasicDatasets(data_loc, 'train')
            self.test = BasicDatasets(data_loc, 'test')
        print('train, test_net:', len(self.train), len(self.test))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        tab_sizes_e = [graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_e]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels, snorm_n, snorm_e