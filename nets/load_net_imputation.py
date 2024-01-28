"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.graph_transformer_net_imputation import GraphTransformerNet

def GraphTransformer(net_params,args):
    return GraphTransformerNet(net_params,args)

def gnn_model(MODEL_NAME, net_params,args):
    models = {
        'MVGFormer': GraphTransformer
    }
        
    return models[MODEL_NAME](net_params,args)