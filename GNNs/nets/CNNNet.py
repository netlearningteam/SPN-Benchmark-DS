#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : CNNNet.py
# @Date    : 2020-09-16
# @Author  : mingjian
    描述
"""
import torch as th
import torch.nn as nn
import dgl
from GNNs.nets.BasicMLPs import BasicNets

class CNNNet(BasicNets):

    def __init__(self,net_params):
        super(CNNNet,self).__init__(net_params)
        self.out_dim = net_params["out_dim"]
        self.kernal_size = net_params['kernal_size']
        self.layers = nn.ModuleList([nn.Conv2d(1,self.h_dim,self.kernal_size,padding=2),nn.ReLU()])
        for _ in range(self.n_layers - 2):
            self.layers.extend([nn.Conv2d(self.h_dim,self.h_dim,self.kernal_size,padding=2),nn.ReLU()])
        # self.layers.extend([nn.Linear(self.h_dim,1)])
        # self.fc1 = nn.Linear(self.h_dim,1)
        self.fc1 = nn.Linear(1,1)


    def forward(self, g, x, e, snorm_n, snorm_e):
        allgs = dgl.unbatch(g)
        prelist = th.tensor([]).to(self.device)
        for gra in allgs:
            h = gra.ndata['feat'].view(1,1,-1,self.node_in_dim).to(self.device)
            for lin in self.layers:
                h = lin(h)
            h = th.mean(h,dim=0).view(-1,1)
            h = self.fc1(h)
            # pre = th.tensor([th.mean(h)]).to(self.device)
            prelist = th.cat((prelist,th.mean(h).view(-1)),dim=0)
            # prelist.append(th.mean(h))
        del allgs
        # prelist = th.tensor(prelist, requires_grad=True).view(-1,1).to(self.device)
        prelist = prelist.view(-1,1)
        # gt = th.tensor(glist).to(self.device)
        # h = self.in_feat_dropout(gt)
        # for lin in self.layers:
        #     h = lin(h)


        return prelist.to(self.device)