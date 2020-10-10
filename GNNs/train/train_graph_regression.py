#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : train_graph_regression.py
# @Date    : 2020-08-27
# @Author  : mingjian
    描述
"""
"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch


from GNNs.train.metrics import MAE,ErrorRate


def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    epoch_train_errra = 0
    nb_data = 0
    gpu_mem = 0

    for iter, (batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_targets = batch_targets.to(device)
        batch_snorm_n = batch_snorm_n.to(device)  # num x 1
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets).detach().item()
        epoch_train_errra += ErrorRate(batch_scores, batch_targets).detach().item()
        nb_data += batch_targets.size(0)
        del batch_x,batch_e
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    epoch_train_errra /= (iter + 1)

    return epoch_loss, epoch_train_mae, optimizer


def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    epoch_test_errra = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_targets = batch_targets.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets).detach().item()
            epoch_test_errra += ErrorRate(batch_scores, batch_targets).detach().item()
            nb_data += batch_targets.size(0)
            del batch_x, batch_e
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        epoch_test_errra /= (iter + 1)

    return epoch_test_loss, epoch_test_mae,epoch_test_errra