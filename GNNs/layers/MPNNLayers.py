import dgl
import torch
from torch import nn
import dgl.function as fn
from GNNs.layers.BasicLayers import BasicLayers
# msg = fn.copy_src(src='h', out='m')
def message(edges):
    return {'m': edges.data['h']}

def reduce(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

class MPNN_Layer(BasicLayers):

    def __init__(self,in_dim, out_dim, activation, dropout, graph_norm, batch_norm, residual=False):
        super(MPNN_Layer,self).__init__(in_dim, out_dim, activation, dropout, graph_norm, batch_norm, residual)



    def forward(self, g, src_feat = None, dst_feat = None, e_feat = None, h_feat = None, snorm_e = None):
        h_in = h_feat  # to be used for residual connection
        g.edata['h'] = h_in
        g.update_all(message, reduce)
        h = g.edata['h']

        if e_feat != None:
            if dst_feat != None:
                # print(src_feat.size())
                # print(dst_feat.size())
                # print(e_feat.size())
                # print(h.size())
                h = torch.cat([src_feat,dst_feat,e_feat,h],dim=1)
                # print("tongguo ")
            else :
                # print(src_feat.size(), e_feat.size(), h.size())
                h = torch.cat([src_feat, e_feat, h], dim=1)
        else:
            if dst_feat != None:
                h = torch.cat([src_feat,dst_feat,h],dim=1)
            else :
                h = torch.cat([src_feat, h], dim=1)


        h = self.lin(h)

        if self.graph_norm:
            h = h * snorm_e  # normalize activation w.r.t. graph size

        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # residual connection

        h = self.dropout(h)
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.residual)




