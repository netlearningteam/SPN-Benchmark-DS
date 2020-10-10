#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : MPNNNet.py
# @Date    : 2020-08-27
# @Author  : mingjian
    描述
"""

import torch.nn as nn
import torch.nn.functional as F
import dgl
from GNNs.layers.mlp_readout_layer import MLPReadout
from GNNs.layers.GCNLayers import GCNLayer
from GNNs.nets.BasicGNNNets import BasicNets
class GCNNet(BasicNets):
    def __init__(self, net_params):
        super(GCNNet,self).__init__(net_params)
        in_feat_dim = net_params['node_in_dim']
        self.layers = nn.ModuleList([GCNLayer(in_feat_dim, self.h_dim, F.relu,
                                                self.dropout, self.graph_norm, self.batch_norm, self.residual)])
        self.layers.extend([GCNLayer(self.h_dim, self.h_dim, F.relu,
                                                self.dropout, self.graph_norm, self.batch_norm, self.residual) for _ in
                                     range(self.n_layers - 1)])

        # self.layers.append(MPNN_Layer(self.in_feat_dim, out_dim, F.relu,
        #                             dropout, self.graph_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(self.h_dim, 1)  # 1 out dim since regression problem

    def forward(self, g, x, e, snorm_n, snorm_e):
        # h = self.embedding_h(h)
        h = self.in_feat_dropout(x)

        # h = torch.zeros([g.number_of_edges(),self.h_dim]).float().to(self.device)
        # src, dst = g.all_edges()

        for conv in self.layers:
            h = conv(g, h, snorm_n)


        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return self.MLP_layer(hg)

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss