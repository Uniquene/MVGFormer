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
    time_now = time.time()
    train_steps = len(data_loader)
    iter_count = 0

    for iter, (batch_x, batch_mask, batch_graphs, batch_y, batch_x_mark, batch_y_mark, batch_num_edges) in enumerate(data_loader):


        record_edge.append(torch.mean(batch_num_edges.float()).item())
        batch_size = len(batch_y)
        batch_graphs = batch_graphs.to(device)
        batch_node_feat = batch_graphs.ndata['feat'].float().to(device)  # num x feat
        batch_y = batch_y.float().to(device)
        optimizer.zero_grad()
        iter_count += 1

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

        outputs = model.forward(batch_size, batch_mask, batch_graphs, batch_node_feat, batch_lap_pos_enc, batch_wl_pos_enc)

        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, :, f_dim:]
        loss = criterion(outputs[batch_mask == 0], batch_x[batch_mask == 0].float())


        if (iter + 1) % 100 == 0:
            logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(iter + 1, epoch + 1, loss.item()))
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((args.epochs - epoch) * train_steps - iter)
            logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()

        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)

    best_model_path = path + '/' + 'checkpoint.pth'
    torch.save(model.state_dict(), best_model_path)

    return epoch_loss, optimizer


def evaluate_network(args, model, device, test_data, data_loader, epoch, setting):
    model.eval()
    epoch_test_loss = 0
    preds = []
    trues = []
    masks = []
    criterion = select_criterion(args.loss)

    folder_path = os.path.join( './output/imputation/', setting)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with torch.no_grad():
        for iter, (batch_x, batch_mask, batch_graphs, batch_y, batch_x_mark, batch_y_mark, batch_num_edges) in enumerate(data_loader):

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

            outputs = model.forward(batch_size, batch_mask, batch_graphs, batch_node_feat, batch_lap_pos_enc, batch_wl_pos_enc)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]

            loss = criterion(outputs[batch_mask == 0], batch_x[batch_mask == 0].float())
            epoch_test_loss += loss.detach().item()

            outputs = outputs.detach().cpu().numpy()
            pred = outputs
            true = batch_x.detach().cpu().numpy()
            preds.append(pred)
            trues.append(true)
            masks.append(batch_mask.detach().cpu())

            if iter % 20 == 0:
                filled = true[0, :, -1].copy()
                filled = filled * batch_mask[0, :, -1].detach().cpu().numpy() + \
                         pred[0, :, -1] * (1 - batch_mask[0, :, -1].detach().cpu().numpy())
                visual(true[0, :, -1], filled, os.path.join(folder_path, str(iter) + '.pdf'))



    epoch_test_loss /= (iter + 1)
    preds = np.concatenate(preds, 0)
    trues = np.concatenate(trues, 0)
    masks = np.concatenate(masks, 0)
    print('test shape:', preds.shape, trues.shape)

    # result save
    folder_path = './output/imputation/result/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
    logger.info('======mse:{}, mae:{}'.format(mse, mae))
    f = open("result_imputation.txt", 'a')
    f.write(setting + "  \n")
    f.write('mse:{}, mae:{}'.format(mse, mae))
    f.write('\n')
    f.write('\n')
    f.close()

    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(folder_path + 'pred.npy', preds)
    np.save(folder_path + 'true.npy', trues)


    return epoch_test_loss, mse

