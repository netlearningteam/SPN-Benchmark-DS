#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : BasicGNNNets.py
# @Date    : 2020-08-27
# @Author  : mingjian
    描述
"""

import torch.nn as nn

class BasicNets(nn.Module):
    def __init__(self,net_params):
        super().__init__()
        self.device = net_params['device']
        self.dropout = net_params['dropout']
        self.n_layers = net_params['L']
        self.h_dim = net_params['h_dim']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.in_feat_dropout = nn.Dropout(net_params['in_feat_dropout'])
        self.readout = net_params['readout']

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss