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
from utils.tools import EarlyStopping, adjust_learning_rate, visual,adjustment
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


import torch.nn.functional as F
from tensorboardX import SummaryWriter
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
file_handler = logging.FileHandler("run_anomaly.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(file_handler)



criterion = nn.MSELoss()
def train_epoch(args, model, optimizer, device, data_loader, epoch, setting):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0

    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    for iter, (batch_x, batch_graphs, batch_y) in enumerate(data_loader):

        batch_size = len(batch_x)
        batch_x = batch_x.float().to(device)
        batch_graphs = batch_graphs.to(device)
        batch_node_feat = batch_graphs.ndata['feat'].float().to(device)  # num x feat
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
        outputs = outputs[:, :, f_dim:]
        loss = criterion(outputs, batch_x)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)

    best_model_path = path + '/' + 'checkpoint.pth'
    torch.save(model.state_dict(), best_model_path)
    # model.load_state_dict(torch.load(best_model_path))

    return epoch_loss, optimizer


def evaluate_network(args, model, device, train_loader, test_data, test_loader, epoch, setting):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    preds = []
    trues = []

    attens_energy = []
    folder_path =os.path.join( './output/anomaly_detection/', setting)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    anomaly_criterion = nn.MSELoss(reduce=False)

    # (1) stastic on the train set
    with torch.no_grad():
        for i, (batch_x, batch_graphs, batch_y) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)

            batch_size = len(batch_x)
            batch_graphs = batch_graphs.to(device)
            batch_node_feat = batch_graphs.ndata['feat'].float().to(device)

            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            except:
                batch_lap_pos_enc = None

            try:
                batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
            except:
                batch_wl_pos_enc = None

            # reconstruction
            outputs = model.forward(batch_size, batch_graphs, batch_node_feat, batch_lap_pos_enc, batch_wl_pos_enc)

            # criterion
            score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)

    # (2) find the threshold
    attens_energy = []
    test_labels = []
    for iter, (batch_x, batch_graphs, batch_y) in enumerate(test_loader):

        batch_size = len(batch_y)
        batch_x = batch_x.float().to(device)

        batch_graphs = batch_graphs.to(device)
        batch_node_feat = batch_graphs.ndata['feat'].float().to(device)

        # batch_e = batch_graphs.edata['feat'].to(device)
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

        # criterion
        score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
        score = score.detach().cpu().numpy()
        attens_energy.append(score)
        test_labels.append(batch_y)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    threshold = np.percentile(combined_energy, 100 - args.anomaly_ratio)
    logger.info("Threshold : {}".format(threshold))

    # (3) evaluation on the test set
    pred = (test_energy > threshold).astype(int)  # 1-normal; 0-abnormal
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_labels = np.array(test_labels)
    gt = test_labels.astype(int)

    logger.info("pred: {}  ".format(pred.shape))
    logger.info("gt: {}    ".format(gt.shape))

    # (4) detection adjustment
    gt, pred = adjustment(gt, pred)

    pred = np.array(pred)
    gt = np.array(gt)
    logger.info("pred:{}".format(pred.shape))
    logger.info("gt:{} ".format(gt.shape))

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    logger.info("=====================================================")
    logger.info("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision,
        recall, f_score))

    f = open("result_anomaly_detection.txt", 'a')
    f.write(setting + "  \n")
    f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision,
        recall, f_score))
    f.write('\n')
    f.write('\n')
    f.close()

    return

