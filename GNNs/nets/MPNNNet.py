#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : MPNNNet.py
# @Date    : 2020-08-27
# @Author  : mingjian
    描述
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from GNNs.layers.mlp_readout_layer import MLPReadout
from GNNs.layers.MPNNLayers import MPNN_Layer
from GNNs.nets.BasicGNNNets import BasicNets
class MPNNNet(BasicNets):
    def __init__(self, net_params):
        super(MPNNNet,self).__init__(net_params)
        node_in_dim = net_params['node_in_dim']
        edage_in_dim = net_params['edage_in_dim']
        self.edge_f = net_params['edge_f']
        self.dst_f = net_params['dst_f']
        if self.edge_f:
            if self.dst_f:
                in_feat_dim = node_in_dim * 2 + edage_in_dim + self.h_dim
            else:
                in_feat_dim = node_in_dim + edage_in_dim + self.h_dim
        else:
            if self.dst_f:
                in_feat_dim = node_in_dim * 2 + self.h_dim
            else:
                in_feat_dim = node_in_dim + self.h_dim

        self.layers = nn.ModuleList([MPNN_Layer(in_feat_dim, self.h_dim, F.relu,
                                                self.dropout, self.graph_norm, self.batch_norm, self.residual) for _ in
                                     range(self.n_layers)])
        # self.layers.append(MPNN_Layer(self.in_feat_dim, out_dim, F.relu,
        #                             dropout, self.graph_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(self.h_dim, 1)  # 1 out dim since regression problem

    def forward(self, g, x, e, snorm_n, snorm_e):
        # h = self.embedding_h(h)
        # h = self.in_feat_dropout(h)

        h = torch.zeros([g.number_of_edges(),self.h_dim]).float().to(self.device)
        src, dst = g.all_edges()

        for mpnn in self.layers:
            if self.edge_f:
                if self.dst_f:
                    h = mpnn(g, src_feat = x[src], dst_feat = x[dst], e_feat = e, h_feat = h, snorm_e = snorm_e)
                else:
                    h = mpnn(g, src_feat=x[src], e_feat=e, h_feat=h, snorm_e=snorm_e)

            else:
                if self.dst_f:
                    h = mpnn(g, src_feat=x[src], dst_feat=x[dst], h_feat=h, snorm_e=snorm_e)
                else:
                    h = mpnn(g, src_feat=x[src], h_feat=h, snorm_e=snorm_e)


        g.edata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_edges(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_edges(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_edges(g, 'h')
        else:
            hg = dgl.mean_edges(g, 'h')  # default readout is mean nodes

        return self.MLP_layer(hg)

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss