"""
    Utility file to select GraphNN model as
    selected by the user
"""

from GNNs.nets.MPNNNet import MPNNNet
from GNNs.nets.MetaNet import MetaNet
from GNNs.nets.GCNNet import GCNNet
from GNNs.nets.MLPNet import MLPNet
from GNNs.nets.CNNNet import CNNNet
def MPNN(net_params):
    return MPNNNet(net_params)

def GN(net_params):
    return MetaNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def CNN(net_params):
    return CNNNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'MPNN': MPNN,
        'GN' :  GN,
        "GCN" : GCN,
        "MLP" : MLP,
        "CNN" : CNN
    }
        
    return models[MODEL_NAME](net_params)