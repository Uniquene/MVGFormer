import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer
    
"""
from layers.graph_transformer_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout
import math


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        print("[!] Adding transformer positional encoding.")

        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x -- [batch_size, series_length, fea_dim]
        :param x:
        :return:
        '''
        # 将位置编码矩阵添加到输入张量中
        x_pe = x + self.pe[:x.size(0), :]
        return x_pe


def positional_encoding(X, num_features, dropout_p=0.1, max_len=512):
    r'''
        给输入加入位置编码
    参数：
        - num_features: 输入进来的维度
        - dropout_p: dropout的概率，当其为非零时执行dropout
        - max_len: 句子的最大长度，默认512

    形状：
        - 输入： [batch_size, seq_length, num_features]
        - 输出： [batch_size, seq_length, num_features]
    '''

    dropout = nn.Dropout(dropout_p)
    P = torch.zeros((1, max_len, num_features))
    X_ = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
        10000,
        torch.arange(0, num_features, 2, dtype=torch.float32) / num_features)
    P[:, :, 0::2] = torch.sin(X_)
    P[:, :, 1::2] = torch.cos(X_)
    X = X + P[:, :X.shape[1], :].to(X.device)
    print("===X.shape:",X.shape)
    return dropout(X)


class GraphTransformerNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        self.hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        self.tf_pos_enc = net_params["tf_pos_enc"]   # the original Transformer position encoding
        max_wl_role_index = 100


        if self.tf_pos_enc:
            self.embedding_tf_pos_enc = PositionalEncoder(hidden_dim)
        
        print("===in_dim_node:",in_dim_node)
        print("===hidden_dim:",hidden_dim)
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)   # node feat is an integer
        self.embedding_h_linear = nn.Linear(in_dim_node, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        print("===out_dim:",out_dim)

        # 维度不发生变化
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads,
                                              dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual))

        # Graph Pool方法1： 原论文 MLPReadout
        self.MLP_layer = MLPReadout(out_dim, n_classes)

        # Graph Pool方法2：graph classification
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_dim, n_classes)


    def forward(self, batch_size, g, h, h_lap_pos_enc=None, h_wl_pos_enc=None):
        # 先embeddding，然后在GNN层不改动维度

        V = int(len(h)/batch_size)   # the length of time series
        # input embedding,
        if self.tf_pos_enc:
            # 如果每个节点携带上多通道原数值作为原始特征的话，则用nn.linear()
            h = self.embedding_h_linear(h)
        else:
            h = self.embedding_h(h)


        if self.tf_pos_enc:
            h = h.view(batch_size, V, -1)
            h = self.embedding_tf_pos_enc(h)
            h = h.view(batch_size * V, -1)

        h = self.in_feat_dropout(h)
        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(g, h)

        h = h.view(batch_size, V, -1).transpose(2, 1)
        h = self.avgpool(h)  # graph-GAP
        h = h.reshape(h.shape[0], -1)

        # Graph Pool 方法1： 原论文 MLPReadout（对比下来，发现MLPReadout不如方法2：GAP+FC）
        # h_out = self.MLP_layer(h)

        # Graph Pool 方法2：Graph GAP + FC
        h_out = self.fc(h)

        return h_out, h
    
    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        # criterion = nn.CrossEntropyLoss(weight=weight)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        return loss
