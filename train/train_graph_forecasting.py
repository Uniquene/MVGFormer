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
import numpy as np
import os
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss

from utils.metrics import MAE

import time

# criterion = nn.MSELoss()


from tensorboardX import SummaryWriter
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


def train_epoch(args, model, optimizer, device, data_loader, epoch, setting):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0

    t1 = time.time()
    record_edge = []

    criterion = select_criterion(args.loss)

    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)


    for iter, (batch_x, batch_graphs, batch_y, batch_x_mark, batch_y_mark, batch_num_edges) in enumerate(data_loader):


        record_edge.append(torch.mean(batch_num_edges.float()).item())
        batch_size = len(batch_y)
        batch_graphs = batch_graphs.to(device)
        batch_node_feat = batch_graphs.ndata['feat'].float().to(device)  # num x feat
        # batch_e = batch_graphs.edata['feat'].to(device)
        batch_y = batch_y.float().to(device)
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
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()


    # print("record_edge:",record_edge)
    ave_edge = torch.mean(torch.tensor(record_edge).float()).item()

    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)

    best_model_path = path + '/' + 'checkpoint.pth'
    torch.save(model.state_dict(), best_model_path)


    return epoch_loss, optimizer


def evaluate_network(args, model, device, test_data, data_loader, epoch, setting):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    preds = []
    trues = []
    criterion = select_criterion(args.loss)

    folder_path =os.path.join( './output/forecasting/', setting)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with torch.no_grad():
        for iter, (batch_x, batch_graphs, batch_y, batch_x_mark, batch_y_mark, batch_num_edges) in enumerate(data_loader):

            batch_size = len(batch_y)

            batch_graphs = batch_graphs.to(device)
            batch_node_feat = batch_graphs.ndata['feat'].float().to(device)

            batch_y = batch_y.float().to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            except:
                batch_lap_pos_enc = None

            try:
                batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
            except:
                batch_wl_pos_enc = None

            outputs = model.forward(batch_size, batch_graphs, batch_node_feat, batch_lap_pos_enc, batch_wl_pos_enc)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            loss = criterion(outputs, batch_y)
            epoch_test_loss += loss.detach().item()

            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            if test_data.scale and args.inverse:
                shape = outputs.shape
                outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

            pred = outputs
            true = batch_y

            preds.append(pred)
            trues.append(true)

            if iter % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                if test_data.scale and args.inverse:
                    shape = input.shape
                    input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(iter)+'.pdf'))

    epoch_test_loss /= (iter + 1)
    epoch_test_mae /= (iter + 1)

    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    logger.info('test shape:{},{}'.format(preds.shape, trues.shape))

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    logger.info('[!!!!!!!]mse:{}, mae:{}'.format(mse, mae))
    #
    # result save
    folder_path = './output/forecasting/result/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    f = open("result_long_term_forecast.txt", 'a')
    f.write(setting + "  \n")
    f.write('mse:{}, mae:{}'.format(mse, mae))
    f.write('\n')
    f.write('\n')
    f.close()

    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(folder_path + 'pred.npy', preds)
    np.save(folder_path + 'true.npy', trues)

    return epoch_test_loss

