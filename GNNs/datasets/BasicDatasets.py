#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : BasicDatasets.py
# @Date    : 2020-08-27
# @Author  : mingjian
    描述
"""
import torch
import torch.utils.data
from utils import DataUtil as DU
import dgl
import os

class BasicDatasets(torch.utils.data.Dataset):
    def __init__(self, data_dir, split):
        self.split = split
        self.data_dir = data_dir
        if split == "train":
            self.alldata = DU.load_json(os.path.join(data_dir,"train_data.json"))
        elif split == "test" :
            self.alldata = DU.load_json(os.path.join(data_dir, "test_data.json"))
        else:
            print("split shoud be train or test")
            return
        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.alldata)
        self._prepare()



    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (len(self.alldata),self.split))

        for data in self.alldata.values():

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(len(data['node_f']))
            g.ndata['feat'] = torch.tensor(data['node_f']).float()

            for src, dst in data['edge_index']:
                g.add_edges(src, dst)
            g.edata['feat'] = torch.tensor(data['edge_f']).view(-1,1).float()
            self.graph_lists.append(g)
            self.graph_labels.append(torch.tensor(data['label']).view(-1,1).long())

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]

