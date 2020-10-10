#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : GCNLayers.py
# @Date    : 2020-08-28
# @Author  : mingjian
    描述
"""
import dgl
import torch
from torch import nn
import dgl.function as fn
from GNNs.layers.BasicLayers import BasicLayers
from dgl.nn import GraphConv

class GCNLayer(BasicLayers):

    def __init__(self, in_dim, out_dim, activation, dropout, graph_norm, batch_norm, residual=False):
        super(GCNLayer, self).__init__(in_dim, out_dim, activation, dropout, graph_norm, batch_norm, residual)
        self.conv = GraphConv(in_dim, out_dim)

    def forward(self, g, feature, snorm_n):
        h_in = feature
        h = self.conv(g, feature)
        if self.graph_norm:
            h = h * snorm_n  # normalize activation w.r.t. graph size

        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual and len(h_in[0]) == len(h[0]):
            h = h_in + h  # residual connection

        h = self.dropout(h)
        return h