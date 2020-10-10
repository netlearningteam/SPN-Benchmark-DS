#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : BasicLayers.py
# @Date    : 2020-08-27
# @Author  : mingjian
    描述
"""
import torch
from torch import nn
class BasicLayers(nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout, graph_norm, batch_norm, residual=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(in_dim, out_dim)


