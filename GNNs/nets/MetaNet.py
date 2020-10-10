import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""

# from layers.gcn_layer import GCNLayer
from GNNs.layers.mlp_readout_layer import MLPReadout
from GNNs.layers.Meta_layers import Meta_Edge_Layer
from GNNs.layers.Meta_layers import Meta_Node_Layer
from GNNs.nets.BasicGNNNets import BasicNets

class MetaNet(BasicNets):
    def __init__(self, net_params):
        super(MetaNet,self).__init__(net_params)

        self.node_in_dim = net_params['node_in_dim']
        self.edage_in_dim = net_params['edage_in_dim']
        self.edge_f = net_params['edge_f']
        self.dst_f = net_params['dst_f']

        self.node_update = net_params['node_update']

        if self.edge_f :
            if self.dst_f:
                in_feat_dim = self.node_in_dim * 2 + self.edage_in_dim + self.h_dim
            else :
                in_feat_dim = self.node_in_dim + self.edage_in_dim + self.h_dim
        else:
            if self.dst_f :
                in_feat_dim = self.node_in_dim * 2  + self.h_dim
            else :
                in_feat_dim = self.node_in_dim  + self.h_dim


        # self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)


        self.edge_layers = nn.ModuleList([Meta_Edge_Layer(in_feat_dim, self.h_dim, F.relu,
                                              self.dropout, self.graph_norm, self.batch_norm, self.residual) for _ in
                                     range(self.n_layers)])

        self.node_layers = nn.ModuleList([Meta_Node_Layer(in_feat_dim, self.node_in_dim, F.relu,
                                              self.dropout, self.graph_norm, self.batch_norm, self.residual) for _ in
                                     range(self.n_layers)])
        # self.layers.append(MPNN_Layer(self.in_feat_dim, out_dim, F.relu,
        #                             dropout, self.graph_norm, self.batch_norm, self.residual))

        self.MLP_layer = MLPReadout(self.h_dim, 1)  # 1 out dim since regression problem
        # self.Global_MLP_layer = MLPReadout(self.h_dim + self.node_in_dim, 1)
        if self.node_update:
            self.Global_MLP_layer = MLPReadout(self.h_dim + self.node_in_dim, 1)
        else:
            self.Global_MLP_layer = MLPReadout(self.h_dim * 2, 1)
        self.Global_MLP_layer_Node_Update = MLPReadout(self.h_dim +self.node_in_dim, 1)
        self.edge_MLPReadout = MLPReadout(self.h_dim, 1)

    def forward(self, g, x, e, snorm_n, snorm_e):
        # snorm_n batch中用到的
        # h = self.embedding_h(h)
        # h = self.in_feat_dropout(h)

        h_node = torch.zeros([g.number_of_nodes(),self.node_in_dim]).float().to(self.device)
        h_edge = torch.zeros([g.number_of_edges(),self.h_dim]).float().to(self.device)
        src, dst = g.all_edges()

        for edge_layer, node_layer in zip(self.edge_layers, self.node_layers):
            if self.edge_f:
                if self.dst_f:
                    h_edge = edge_layer(g, src_feat = x[src], dst_feat = x[dst], e_feat = e, h_feat = h_edge, snorm_e = snorm_e)
                    h_node = node_layer(g, src_feat=x[src], dst_feat=x[dst], e_feat=e, h_feat=h_node, snorm_e=snorm_e, n_feat = x)
                else:
                    h_edge = edge_layer(g, src_feat=x[src], e_feat=e, h_feat=h_edge, snorm_e=snorm_e)
                    h_node = node_layer(g, src_feat=x[src], e_feat=e, h_feat=h_node, snorm_e=snorm_e, n_feat = x)

            else:
                if self.dst_f:
                    h_edge = edge_layer(g, src_feat=x[src], dst_feat=x[dst], h_feat=h_edge, snorm_e=snorm_e)
                    h_node = node_layer(g, src_feat=x[src], dst_feat=x[dst], h_feat=h_node, snorm_e=snorm_e, n_feat = x)
                else:
                    h_edge = edge_layer(g, src_feat=x[src], h_feat=h_edge, snorm_e=snorm_e)
                    h_node = node_layer(g, src_feat=x[src], h_feat=h_node, snorm_e=snorm_e, n_feat = x)


        g.edata['h'] = h_edge
        if self.node_update:
            g.ndata['h'] = h_node

        # print("g.data:", g.ndata['h'][0].shape)

        if self.readout == "sum":
            he = dgl.sum_edges(g, 'h')
            hn = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            he = dgl.max_edges(g, 'h')
            hn = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            he = dgl.mean_edges(g, 'h')
            hn = dgl.mean_nodes(g, 'h')
        else:
            he = dgl.mean_edges(g, 'h')  # default readout is mean nodes
            hn = dgl.mean_nodes(g, 'h')

        # print(torch.cat([he, hn], dim=1).shape)
        # used to global task

        out = self.Global_MLP_layer(torch.cat([he, hn], dim=1))

        # used to transition task
        edge_out = self.edge_MLPReadout(h_edge)

        # return self.MLP_layer(he)
        return out

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss

