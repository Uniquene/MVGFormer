#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time: 2023/11/15 9:50
# file: VG_save.py
# author: chenTing
import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import time
import dgl
import networkx as nx
from ts2vg import NaturalVG, HorizontalVG

def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']


        This function is called inside a function in MVGsDataset class.
    """

    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())

    return new_g



def generate_VG(data_matrix, vg_type='NVG'):
    '''
    data_matrix: [channel, seq_len]
    generate visibility graph
    '''

    multi_ts = data_matrix[:, :]
    cur_MTS =data_matrix[:, :].transpose()
    mg = nx.Graph()
    t1 = time.time()
    # adj_mix = np.empty((data_matrix.shape[1], data_matrix.shape[1]))
    for j in range(data_matrix.shape[0]):

        ts = data_matrix[j, :]  # 当前通道的time series ==multi_ts[0]

        if vg_type=='NVG':
            g = NaturalVG(directed=None).build(ts)  # each dimension transform into a visibility graph
            # g = NaturalVG(directed='left_to_right').build(ts)  # each dimension transform into a visibility graph
            # g = NaturalVG(penetrable_limit=2).build(ts)  # each dimension transform into a visibility graph
        if vg_type=='HVG':
            g = HorizontalVG(directed=None).build(ts)
        nxg = g.as_networkx()
        # adj_mix =adj_mix +  np.array(nx.adjacency_matrix(nxg).todense())
        mg = nx.compose(mg, nxg)  # compose multiplex graph
        # g_list.append((dgl.from_networkx(mg),torch.tensor(label[i], dtype=torch.int32)))

    cur_dgl = dgl.from_networkx(mg)
    cur_dgl.ndata['feat'] = torch.FloatTensor(cur_MTS)  # 用original multi-channel value作为节点特征(其实最好用degree作为节点特征)

    # full_g = make_full_graph(cur_dgl)
    cur_dgl = self_loop(cur_dgl)
    # 获取边的数量
    # print("the number of edge for visibility graph:",cur_dgl.number_of_edges())
    # print("the number of edge for full graph:", full_g.number_of_edges())

    return cur_dgl


class Dataset_ETT_hour_ts2vg_save():
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, save_path=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.save_path = save_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.graph_list = []


    def __gen_vg__(self):

        data_len = len(self.data_x)
        for index in range(data_len):
            s_begin = index
            s_end = s_begin + self.seq_len
            if s_end < data_len:
                seq_x = self.data_x[s_begin:s_end]
                seq_x_g = generate_VG(seq_x.transpose())  # 生成visibility graph
                # 构建保存文件名，这里假设使用索引作为文件名
                filename = os.path.join(self.save_path, f'graph_{index}.bin')
                # print("=='node_feats': seq_x_g.ndata['feat']:,'node_feats':",seq_x_g.ndata['feat'])
                # 将图数据保存为二进制文件
                dgl.save_graphs(filename, [seq_x_g])
            else:
                break

        return

    def save(self):

        self.__gen_vg__()
        # 遍历图数据列表并保存
        for i, g in enumerate(self.graph_list):
            # 构建保存文件名，这里假设使用索引作为文件名
            filename = os.path.join(self.save_path, f'graph_{i}.bin')
            # 将图数据保存为二进制文件
            dgl.save_graphs(filename, {'g': g})


def graph_save(args, flag):

    timeenc = 0 if args.embed != 'timeF' else 1
    save_path = os.path.join(args.root_path,'{}-graph'.format(args.data_path.split('.')[0]), flag)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        os.remove(save_path)
    gen_vg = Dataset_ETT_hour_ts2vg_save(root_path=args.root_path, flag=flag, size=[args.seq_len, args.label_len, args.pred_len],
                     features=args.features, data_path='ETTh1.csv',
                     target=args.target, scale=True, timeenc=timeenc, freq=args.freq, seasonal_patterns=None, save_path=save_path)
    gen_vg.save()

    # # 从文件加载图
    # loaded_g, _ = dgl.load_graphs('E:\phd\coding\DATA\Public Dataset\Forecasting\\all_datasets\ETT-small\ETTh1-graph\\train\\graph_10.bin')
    # print(loaded_g)
    # print("====a:",a)