#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : BasicMLPs.py
# @Date    : 2020-09-16
# @Author  : mingjian
    描述
"""

import torch.nn as nn

class BasicNets(nn.Module):
    def __init__(self,net_params):
        super().__init__()
        self.node_in_dim = net_params['node_in_dim']
        self.device = net_params['device']
        self.dropout = net_params['dropout']
        self.n_layers = net_params['L']
        self.h_dim = net_params['h_dim']
        self.in_feat_dropout = nn.Dropout(net_params['in_feat_dropout'])

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss