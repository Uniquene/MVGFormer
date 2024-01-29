#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time: 2023/9/15 21:46
# file: run_classification.py
# author: chenTing


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from ts2vg import NaturalVG,HorizontalVG
import networkx as nx
import argparse
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import torch.optim as optim
from tqdm import tqdm
import random
import time

from data_provider.data_factory import data_provider

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



"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


if __name__ == "__main__":

    data_info={
        'ETTh1': {"root_path": "ETT-small", "data_path": "ETTh1.csv", "data": "ETTh1", "enc_in": 7, "dec_in": 7,
                  "c_out": 7, "d_model": 16, "seq_len": 96, "label_len": 48, "train_epochs": 30},
        'ETTh2': {"root_path": "ETT-small", "data_path": "ETTh2.csv", "data": "ETTh2", "enc_in": 7, "dec_in": 7,
                  "c_out": 7, "d_model": 32, "seq_len": 96, "label_len": 48, "train_epochs": 5},
        'illness': {"root_path": "illness", "data_path": "national_illness.csv", "data": "custom", "enc_in": 7,
                    "dec_in": 7, "c_out": 7, "d_model": 768, "seq_len": 36, "label_len": 18, "train_epochs": 30},
        'exchange_rate': {"root_path": "exchange_rate", "data_path": "exchange_rate.csv", "data": "custom", "enc_in": 8,
                          "dec_in": 8, "c_out": 8, "d_model": 64, "seq_len": 96, "label_len": 48, "train_epochs": 6},
        'weather': {"root_path": "weather", "data_path": "weather.csv", "data": "custom", "enc_in": 21, "dec_in": 21,
                    "c_out": 21, "d_model": 32, "seq_len": 96, "label_len": 48, "train_epochs": 25}
        }


    for data, info in data_info.items():
        logger.info("=======starting============data:{}".format(data))
        logger.info("====info:{}".format(str(info)))
        if data =="illness":
            pred_len_list = [24, 36, 48, 60]
        else:
            pred_len_list = [96, 192, 336, 720]

        for pred_len in pred_len_list:

            logger.info("=======pred len:{}".format(pred_len))

            parser = argparse.ArgumentParser(description='MVGFormer')

            # basic config
            parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                                help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
            parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
            parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
            parser.add_argument('--model', type=str, required=False, default='MVGFormer',
                                help='model name, options: [Autoformer, Transformer, TimesNet]')

            # data loader
            parser.add_argument('--data', type=str, required=False, default=info['data'], help='dataset type')
            parser.add_argument('--root_path', type=str, default='/dataset/all_datasets/ETT-small',
                                help='root path of the data file')
            parser.add_argument('--data_path', type=str, default=info['data_path'], help='data file')
            parser.add_argument('--features', type=str, default='M',
                                help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
            parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
            parser.add_argument('--freq', type=str, default='h',
                                help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
            parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

            # forecasting task
            parser.add_argument('--seq_len', type=int, default=info['seq_len'], help='input sequence length')
            parser.add_argument('--label_len', type=int, default=info['label_len'], help='start token length')
            parser.add_argument('--pred_len', type=int, default=pred_len, help='prediction sequence length')
            parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
            parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

            # inputation task
            parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

            # anomaly detection task
            parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

            # model define
            parser.add_argument('--enc_in', type=int, default=info['enc_in'], help='encoder input size')
            parser.add_argument('--dec_in', type=int, default=info['dec_in'], help='decoder input size')
            parser.add_argument('--c_out', type=int, default=info['c_out'], help='output size')
            parser.add_argument('--d_model', type=int, default=info['d_model'], help='dimension of model(default:512)')
            parser.add_argument('--distil', action='store_false',
                                help='whether to use distilling in encoder, using this argument means not using distilling',
                                default=True)
            parser.add_argument('--embed', type=str, default='timeF',
                                help='time features encoding, options:[timeF, fixed, learned]')
            parser.add_argument('--activation', type=str, default='gelu', help='activation')
            parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

            # optimization
            parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers(0:main worker)')
            parser.add_argument('--itr', type=int, default=1, help='experiments times')
            parser.add_argument('--batch_size', type=int, default=32,
                                help='input batch size for training (default: 32)')
            parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
            parser.add_argument('--des', type=str, default='test', help='exp description')
            parser.add_argument('--loss', type=str, default='MSE', help='loss function')
            parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
            parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

            # GPU
            parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
            parser.add_argument('--gpu', type=int, default=0, help='gpu')
            parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
            parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

            # de-stationary projector params
            parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                                help='hidden layer dimensions of projector (List)')
            parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')


            # graph transformer
            parser.add_argument('--is_shuffle',type=bool,default=True, help='is shuffle data?(default:True)')
            parser.add_argument('--epochs', type=int, default=info['train_epochs'],
                                help='number of epochs to train (default: 350)')
            parser.add_argument('--lr', type=float, default=0.001,
                                help='learning rate (default: 0.01)')
            parser.add_argument('--seed', type=int, default=0,
                                help='random seed for splitting the dataset into 10 (default: 0)')
            parser.add_argument('--filename', type=str, default="",
                                help='output file')
            parser.add_argument('--kernel_size_1D', type=int, default=41,
                                help='kernel_size_1D')
            parser.add_argument('--VG_type', type=str, default='NVG',
                                help='Setting the type of visibility graph, NaturalVG-NVG or HorizontalVG-HVG (default:"NVG")')

            parser.add_argument('--config', type=str, default='configs/config.json', help="Please give a config.json file with training/model/data/param details")
            parser.add_argument('--gpu_id', help="Please give a value for gpu id")
            parser.add_argument('--init_lr', help="Please give a value for init_lr")
            parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
            parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
            parser.add_argument('--min_lr', help="Please give a value for min_lr")
            parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
            parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
            parser.add_argument('--L', help="Please give a value for L")
            parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
            parser.add_argument('--out_dim', help="Please give a value for out_dim")
            parser.add_argument('--residual', help="Please give a value for residual")
            parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
            parser.add_argument('--readout', help="Please give a value for readout")
            parser.add_argument('--n_heads', help="Please give a value for n_heads")
            parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
            parser.add_argument('--dropout', help="Please give a value for dropout")
            parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
            parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
            parser.add_argument('--self_loop', help="Please give a value for self_loop")
            parser.add_argument('--max_time', help="Please give a value for max_time")

            args = parser.parse_args()

            log_writer = SummaryWriter()

            train_data, train_loader = data_provider(args, flag='train')
            vali_data, vali_loader = data_provider(args, flag='val')
            test_data, test_loader = data_provider(args, flag='test')


            import json
            with open(args.config) as f:
                config = json.load(f)

            params = config['params']
            net_params = config['net_params']
            net_params['n_classes'] = args.c_out
            if net_params['tf_pos_enc']:
                net_params['in_dim'] = args.enc_in   # the dim of node feature

            device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
            net_params['device'] = device

            net_params['hidden_dim'] = int(args.d_model)
            net_params['out_dim'] = int(args.d_model)


            if args.seed is not None:
                params['seed'] = int(args.seed)
            if args.epochs is not None:
                params['epochs'] = int(args.epochs)
            if args.batch_size is not None:
                params['batch_size'] = int(args.batch_size)
            if args.init_lr is not None:
                params['init_lr'] = float(args.init_lr)
            if args.lr_reduce_factor is not None:
                params['lr_reduce_factor'] = float(args.lr_reduce_factor)
            if args.lr_schedule_patience is not None:
                params['lr_schedule_patience'] = int(args.lr_schedule_patience)
            if args.min_lr is not None:
                params['min_lr'] = float(args.min_lr)
            if args.weight_decay is not None:
                params['weight_decay'] = float(args.weight_decay)
            if args.print_epoch_interval is not None:
                params['print_epoch_interval'] = int(args.print_epoch_interval)
            if args.max_time is not None:
                params['max_time'] = float(args.max_time)
            # network parameters
            net_params = config['net_params']
            net_params['device'] = device
            net_params['gpu_id'] = config['gpu']['id']
            net_params['batch_size'] = params['batch_size']
            if args.L is not None:
                net_params['L'] = int(args.L)
            if args.hidden_dim is not None:
                net_params['hidden_dim'] = int(args.hidden_dim)
            if args.out_dim is not None:
                net_params['out_dim'] = int(args.out_dim)
            if args.residual is not None:
                net_params['residual'] = True if args.residual == 'True' else False
            if args.edge_feat is not None:
                net_params['edge_feat'] = True if args.edge_feat == 'True' else False
            if args.readout is not None:
                net_params['readout'] = args.readout
            if args.n_heads is not None:
                net_params['n_heads'] = int(args.n_heads)
            if args.in_feat_dropout is not None:
                net_params['in_feat_dropout'] = float(args.in_feat_dropout)
            if args.dropout is not None:
                net_params['dropout'] = float(args.dropout)
            if args.layer_norm is not None:
                net_params['layer_norm'] = True if args.layer_norm == 'True' else False
            if args.batch_norm is not None:
                net_params['batch_norm'] = True if args.batch_norm == 'True' else False
            if args.self_loop is not None:
                net_params['self_loop'] = True if args.self_loop == 'True' else False


            # setting seeds
            random.seed(params['seed'])
            np.random.seed(params['seed'])
            torch.manual_seed(params['seed'])
            if device.type == 'cuda':
                torch.cuda.manual_seed(params['seed'])

            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}'.format(
                args.task_name,
                args.model,
                args.data_path.split('.')[0],
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model
            )
            logger.info("===setting:{}".format(setting))


            from nets.load_net_forecast import gnn_model

            MODEL_NAME = config['model']
            model = gnn_model(MODEL_NAME, net_params, args)
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=params['lr_reduce_factor'],
                                                             patience=params['lr_schedule_patience'],
                                                             verbose=True)


            epoch_train_losses, epoch_val_losses = [], []
            epoch_train_accs, epoch_val_accs = [], []
            best_test_acc = 0
            best_epoch = 0
            logger.info("start training!-----------------------------------")

            t0 = time.time()

            from train.train_graph_forecasting import train_epoch, evaluate_network
            try:
                with tqdm(range(params['epochs'])) as t:
                    for epoch in t:
                        t.set_description('Epoch %d' % epoch)

                        start_time = time.time()
                        logger.info("-----{}-----epoch {}/{}:".format(args.data, epoch, params['epochs']))

                        epoch_train_loss, optimizer = train_epoch(args, model, optimizer, device, train_loader,
                                                                                   epoch, setting)

                        logger.info("====train done!")
                        logger.info("train cost {} seconds:".format(time.time() - start_time))

                        test_strat_time = time.time()
                        epoch_test_loss = evaluate_network(args, model, device, test_data, test_loader, epoch, setting)
                        logger.info("test cost {} seconds:".format(time.time() - test_strat_time))

                        epoch_train_losses.append(epoch_train_loss)

                        log_writer.add_scalar('Loss/train', float(epoch_train_loss), epoch)
                        log_writer.add_scalar('Loss/test', float(epoch_test_loss), epoch)

                        logger.info("cost {} seconds:".format(time.time() - start_time))
                        logger.info("training loss:{}, testing loss:{}".format(epoch_train_loss, epoch_test_loss))

            except KeyboardInterrupt:
                print('-' * 89)
                print('Exiting from training early because of KeyboardInterrupt')


