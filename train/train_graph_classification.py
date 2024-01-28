#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time: 2023/9/16 19:59
# file: train_graph_classification.py
# author: chenTing

"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

from utils.metrics import MAE
import pandas as pd
import numpy as np
import os


def plot_tsne(data, y, epoch, dataset_name, acc_test):
    '''
    plot tsne to visualize the graph embedding
    :param data: numpy
    :param y: list
    :return:
    '''

    # visible final_embed by PCA/TSNE
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    n_components = 2
    tsne = TSNE(n_components)

    result = tsne.fit_transform(data)
    # Plot the result of our TSNE with the label color coded
    tsne_result_df = pd.DataFrame({'tsne_1': result[:, 0], 'tsne_2': result[:, 1],'label':y})

    num_classes = len(set(y))
    NUM_COLORS = len(set(y))
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    color_y = [colors[i] for i in y]

    # y = [i*10 for i in y]

    fig, ax = plt.subplots(1)
    # sns.scatterplot(x='tsne_1', y='tsne_2', hue='label',data=tsne_result_df, ax=ax, s=120)
    scatter = ax.scatter(result[:, 0], result[:, 1], c=y, s=25, cmap="gist_rainbow")
    # plt.scatter(result[:, 0], result[:, 1], c=y, s=25, label="PCA")
    lim = (result.min() - 5, result.max() + 5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')

    if num_classes > 20:
        ncols = 2
    else:
        ncols = 1
    legend1 = ax.legend(*scatter.legend_elements(num=num_classes),
                        bbox_to_anchor=(1.05, 1), loc=2, title="Classes",ncol = ncols)
    ax.add_artist(legend1)

    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    save_path = "E:\phd\coding\Multi-Visibility-Graph\MVG_GraphFormer\output\classification\\tsne\\{}".format(dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'epoch_{}_ACC_{}.pdf'.format(epoch, round(acc_test,4))), format='pdf', bbox_inches='tight')
    # plt.show()


def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    result = []
    targets = []


    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):

        batch_size = len(batch_targets)
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        # batch_e = batch_graphs.edata['feat'].to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        try:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0;
            sign_flip[sign_flip < 0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None

        try:
            batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
        except:
            batch_wl_pos_enc = None

        batch_scores, batch_graph_embedding = model.forward(batch_size, batch_graphs, batch_x, batch_lap_pos_enc, batch_wl_pos_enc)
        result.append(batch_scores.cpu().detach())
        targets += batch_targets.cpu().detach().tolist()

        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

        # epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)


    # compute accuracy
    result = torch.cat(result, dim=0)
    y_true = torch.LongTensor(targets).to(device)
    pred = result.max(1, keepdim=True)[1]
    correct = pred.eq(y_true.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(pred))

    return epoch_loss, acc_train, optimizer


def evaluate_network(model, device, data_loader, epoch, dataset_name):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    result = []
    targets = []
    test_graph_emb = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):

            batch_size = len(batch_targets)

            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            # batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            except:
                batch_lap_pos_enc = None

            try:
                batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
            except:
                batch_wl_pos_enc = None

            batch_scores, batch_graph_embedding = model.forward(batch_size, batch_graphs, batch_x, batch_lap_pos_enc, batch_wl_pos_enc)
            result.append(batch_scores.cpu().detach())
            targets += batch_targets.cpu().detach().tolist()
            test_graph_emb += (batch_graph_embedding.cpu().detach().tolist())

            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            # epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)

    # compute accuracy
    result = torch.cat(result, dim=0)
    y_true = torch.LongTensor(targets).to(device)
    pred = result.max(1, keepdim=True)[1]
    correct = pred.eq(y_true.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(pred))

    plot_tsne(np.array(test_graph_emb), targets, epoch, dataset_name,acc_test)

    return epoch_test_loss, acc_test