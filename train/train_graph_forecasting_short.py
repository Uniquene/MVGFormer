#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time: 2023/11/16 11:35
# file: train_graph_forecasting_short.py
# author: chenTing


"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import numpy as np
import os
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from data_provider.m4 import M4Meta
from data_provider.data_loader import generate_VG
from utils.metrics import MAE
from utils.m4_summary import M4Summary
import time
import pandas


import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
file_handler = logging.FileHandler("run.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(file_handler)



def select_criterion(loss_name='MSE'):
    if loss_name == 'MSE':
        return nn.MSELoss()
    elif loss_name == 'MAPE':
        return mape_loss()
    elif loss_name == 'MASE':
        return mase_loss()
    elif loss_name == 'SMAPE':
        return smape_loss()


def train_epoch(args, model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0

    t1 = time.time()
    record_edge = []

    criterion = select_criterion(args.loss)


    for iter, (batch_x, batch_graphs, batch_y, batch_x_mark, batch_y_mark, batch_num_edges) in enumerate(data_loader):


        record_edge.append(torch.mean(batch_num_edges.float()).item())
        batch_size = len(batch_y)
        batch_graphs = batch_graphs.to(device)
        batch_node_feat = batch_graphs.ndata['feat'].float().to(device)  # num x feat
        batch_y = batch_y.float().to(device)
        batch_x = batch_x.float().to(device)
        optimizer.zero_grad()
        try:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None

        try:
            batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
        except:
            batch_wl_pos_enc = None

        outputs = model.forward(batch_size, batch_graphs, batch_node_feat, batch_lap_pos_enc, batch_wl_pos_enc)

        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:].float().to(device)
        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

        batch_y_mark = batch_y_mark[:, -args.pred_len:, f_dim:].float().to(device)
        loss_value = criterion(batch_x, args.frequency_map, outputs, batch_y, batch_y_mark)
        # loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
        loss = loss_value  # + loss_sharpness * 1e-5

        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)


    return epoch_loss, optimizer


def evaluate_network(args, model, device, test_data, test_loader, train_loader, epoch):

    x, _ = train_loader.dataset.last_insample_window()
    y = test_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(device)
    x = x.unsqueeze(-1)

    if args.data == 'm4':
        args.pred_len = M4Meta.horizons_map[args.seasonal_patterns]  # Up to M4 config
        args.seq_len = 2 * args.pred_len  # input_len = 2*pred_len
        args.label_len = args.pred_len
        args.frequency_map = M4Meta.frequency_map[args.seasonal_patterns]


    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    preds = []
    trues = []
    criterion = select_criterion(args.loss)

    folder_path = './output/forecasting/' + args.data_path.split('.')[0] + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(device)
        dec_inp = torch.cat([x[:, - args.label_len:, :], dec_inp], dim=1).float()
        # encoder - decoder
        outputs = torch.zeros((B, args.pred_len + args.seq_len, C)).float().to(device)
        id_list = np.arange(0, B, 1)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):

            batch_graphs, _ = generate_VG(x[id_list[i]:id_list[i + 1]].squeeze(0).numpy().transpose())
            batch_node_feat = batch_graphs.ndata['feat'].float().to(device)
            batch_size = 1

            outputs[id_list[i]:id_list[i + 1], :, :] = model.forward(batch_size, batch_graphs, batch_node_feat, None, None)


            if id_list[i] % 1000 == 0:
                print(id_list[i])

        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        outputs = outputs.detach().cpu().numpy()

        preds = outputs
        trues = y
        x = x.detach().cpu().numpy()

        for i in range(0, preds.shape[0], preds.shape[0] // 10):
            gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
            pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
            visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    print('test shape:', preds.shape)

    # result save
    folder_path = './output/forecasting/' + args.data + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(args.pred_len)])
    forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
    forecasts_df.index.name = 'id'
    forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
    forecasts_df.to_csv(folder_path + args.seasonal_patterns + '_forecast.csv')

    file_path = './output/forecasting/' + args.data + '/'
    if 'Weekly_forecast.csv' in os.listdir(file_path) \
            and 'Monthly_forecast.csv' in os.listdir(file_path) \
            and 'Yearly_forecast.csv' in os.listdir(file_path) \
            and 'Daily_forecast.csv' in os.listdir(file_path) \
            and 'Hourly_forecast.csv' in os.listdir(file_path) \
            and 'Quarterly_forecast.csv' in os.listdir(file_path):
        m4_summary = M4Summary(file_path, args.root_path)
        # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
        smape_results, owa_results, mape, mase = m4_summary.evaluate()
        logger.info('smape:{}'.format(smape_results))
        logger.info('mape:{}'.format(mape))
        logger.info('mase:{}'.format(mase))
        logger.info('owa:{}'.format(owa_results))
    else:
        print('After all 6 tasks are finished, you can calculate the averaged index')
    return

