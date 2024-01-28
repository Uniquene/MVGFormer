"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.graph_transformer_net_classification import GraphTransformerNet

def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'MVGFormer': GraphTransformer
    }
        
    return models[MODEL_NAME](net_params)