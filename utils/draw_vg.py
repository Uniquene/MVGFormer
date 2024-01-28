#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time: 2023/12/2 18:51
# file: draw_vg.py
# author: chenTing
from ts2vg import NaturalVG,HorizontalVG
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_vg(ts, vg_type='NVG'):
    '''
    draw raw time series & visibility graph & networkx graph
    :param ts:
    :return:
    '''
    # 1. Build visibility graph
    if vg_type == 'NVG':
        g = NaturalVG(directed=None).build(ts)  # each dimension transform into a visibility graph
        title = 'Visibility Graph'
    if vg_type == 'HVG':
        g = HorizontalVG(directed=None).build(ts)
        title = 'Horizontal Visibilty Graph'
    nxg = g.as_networkx()  # transform visibility graph to networkx

    degree_distribution = nx.degree_histogram(nxg)  # 度的分布
    degree_list = [z for z in degree_distribution]
    degree_distribution = np.array(degree_distribution) / len(nxg.nodes)

    node_degree = nxg.degree()
    cur_graph_degree = []
    for node_id in range(len(node_degree)):
        cur_graph_degree.append(node_degree[node_id])
    hubs_degree = sorted(cur_graph_degree,reverse = True)[:15]
    # # hubs_degree = [160,199,209,184,142,101]
    # hubs_degree = [52, 74, 135]
    hubs = {}
    for i in hubs_degree:
        hubs[cur_graph_degree.index(i)]=i

    labels = {}
    for node in nxg.nodes():
        if node in hubs:
            labels[node] = hubs[node]

    # 2. Make plots
    fig, [ax0, ax1, ax2, ax3, ax4] = plt.subplots(ncols=5, figsize=(25, 5))
    # fig, [ax0, ax1, ax3] = plt.subplots(nrows=3, figsize=(7, 4))


    ax0.plot(ts)
    ax0.set_title('Original Time Series')

    mark_node = hubs.keys()
    color_map = []
    node_size = []
    for node in nxg:
        if node in mark_node:
            color_map.append('red')
            node_size.append(15)
        else:
            color_map.append((0, 0, 0, 1))
            node_size.append(2)


    graph_plot_options = {
        'with_labels': False,
        # 'node_size': node_size,
        'node_size': 2,
        'node_color': [(0, 0, 0, 1)],
        # 'node_color': color_map,
        'edge_color': [(0, 0, 0, 0.05)],
    }

    nx.draw_networkx(nxg, ax=ax1, pos=g.node_positions(), labels=labels,font_size=16, font_color='r', **graph_plot_options)
    ax1.tick_params(bottom=True, labelbottom=True)
    ax1.plot(ts)
    ax1.set_title(title)


    nx.draw_networkx(nxg, ax=ax2, pos=nx.kamada_kawai_layout(nxg), labels=labels ,font_size=16, font_color='r', **graph_plot_options)
    ax2.set_title(title)


    # 每个节点度的曲线图
    ax3.plot(cur_graph_degree)
    # ax3.scatter(hubs.keys(), hubs.values(), color='red', s=10)  # 在最大值点上绘制一个红色的圆点
    ax3.set_title('Node Degree')
    #
    # ax4.plot(degree_distribution)
    ax4.loglog(degree_list, '.', c='blue')
    ax4.set_title('Degree Distribution')

    plt.savefig('E:\phd\presentation\\time-series\paper_writing\MVGFormer\Figures\\VG_regular_graph_AtrialFibrillation.pdf', format='pdf',bbox_inches='tight')

    plt.show()



def plot_ts_in_one_fig(X):
    # Visualize the timeseries in the train and test set

    # colors = ['r', 'b', 'c', 'g', 'y','o']
    # colors = ['blue', 'orange', 'green', 'red', 'purple','brown','pink','gray','olive','cyan']

    view_num = X.shape[1]

    NUM_COLORS = view_num
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    # colors = ['pink','aquamarine','orange']
    # colors = ['palevioletred','turquoise','orange']

    plt.figure(figsize=(6, 5))
    # plt.suptitle('The timeseries in the {} set'.format(dataset_name))
    # cm = plt.get_cmap('gist_rainbow')
    for i in range(view_num):
        plt.subplot(view_num, 1, 1 + i)
        plt.plot(range(len(X[:,i])), X[:,i], c=colors[i], linewidth=1.5)
        # plt.plot(range(len(ts)), ts, c=cm(label*20))
    # plt.savefig('E:\phd\presentation\\time-series\paper_writing\MVGFormer\Figures\\original_time_series.pdf', format='pdf',bbox_inches='tight')
    plt.show()
