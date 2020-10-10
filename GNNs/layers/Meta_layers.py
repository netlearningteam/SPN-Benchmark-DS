import dgl
import torch
from torch import nn
from torch_scatter import scatter_mean
from torch_scatter import scatter
import dgl.function as fn
from GNNs.layers.BasicLayers import BasicLayers
# msg = fn.copy_src(src='h', out='m')
def message(edges):
    return {'m': edges.data['h']}

def reduce(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}


class Meta_Edge_Layer(BasicLayers):

    def __init__(self,in_dim, out_dim, activation, dropout, graph_norm, batch_norm, residual=False, dgl_builtin=False):
        super(Meta_Edge_Layer,self).__init__(in_dim, out_dim, activation, dropout, graph_norm, batch_norm, residual)




    def forward(self, g, src_feat = None, dst_feat = None, e_feat = None, h_feat = None, snorm_e = None, n_feat = None):
        h_in = h_feat  # to be used for residual connection
        g.edata['h'] = h_in
        g.update_all(message, reduce)
        h = g.edata['h']
        h_node = g.ndata['h']

        if e_feat != None:
            if dst_feat != None:
                h = torch.cat([src_feat,dst_feat,e_feat,h],dim=1)
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

class Meta_Node_Layer(BasicLayers):

    def __init__(self, in_dim, out_dim, activation, dropout, graph_norm, batch_norm, residual=False,
                 ):
        super(Meta_Node_Layer,self).__init__(in_dim, out_dim, activation, dropout, graph_norm, batch_norm, residual)


    def forward(self, g, src_feat=None, dst_feat=None, e_feat=None, h_feat=None, snorm_e=None, n_feat=None):
        h_node_in = n_feat
        g.ndata['h'] = h_node_in
        g.update_all(message, reduce)
        h = g.edata['h']
        h_node = g.ndata['h']

        src, dst = g.all_edges()


        src = torch.tensor(src).to(src_feat.device)
        dst = torch.tensor(dst).to(src_feat.device)

        h_src = scatter_mean(src_feat, src, dim=0, dim_size=len(h_node))
        h_dst = scatter_mean(dst_feat, dst, dim=0, dim_size=len(h_node))
        h_e = scatter_mean(e_feat, dst, dim=0, dim_size=len(h_node))

        h_node = torch.cat([h_src, h_dst, h_e, h_node], dim=1)

        h_node = self.lin(h_node)


        # if self.graph_norm:
        #     h_node = h_node * snorm_e  # normalize activation w.r.t. graph size

        if self.batch_norm:
            h_node = self.batchnorm_h(h_node)  # batch normalization

        if self.activation:
            h_node = self.activation(h_node)

        if self.residual:
            h_node = h_node_in + h_node # residual connection

        h_node = self.dropout(h_node)
        return h_node

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.residual)




