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
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import warnings
warnings.filterwarnings('ignore')
import torch.optim as optim
from tqdm import tqdm
import random
import pandas as pd
import time

import torch.nn.functional as F
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

from torch.utils.data import DataLoader, Dataset


def file_parse(dataset_name,data_path):
    '''
    parse dataset information,like,dimensions,number of class,labels and so on.
    :param dataset_name: the name of dataset
    :return: the dataset information
    '''

    file_dict = {'AtrialFibrillation':['AtrialFibrillation_TRAIN.tsv','AtrialFibrillation_TEST.tsv'],
                 'StandWalkJump':['StandWalkJump_TRAIN.tsv','StandWalkJump_TEST.tsv'],
                 'SelfRegulationSCP1':['SelfRegulationSCP1_TRAIN.tsv','SelfRegulationSCP1_TEST.tsv'],
                 'SelfRegulationSCP2':['SelfRegulationSCP2_TRAIN.tsv','SelfRegulationSCP2_TEST.tsv'],
                 'HandMovementDirection':['HandMovementDirection_TRAIN.tsv','HandMovementDirection_TEST.tsv'],
                 'FingerMovements':['FingerMovements_TRAIN.tsv','FingerMovements_TEST.tsv'],
                 'ArticularyWordRecognition':['ArticularyWordRecognition_TRAIN.tsv','ArticularyWordRecognition_TEST.tsv'],
                 'BasicMotions':['BasicMotions_TRAIN.tsv','BasicMotions_TEST.tsv'],
                 'Cricket':['Cricket_TRAIN.tsv','Cricket_TEST.tsv'],
                 'DuckDuckGeese': ['DuckDuckGeese_TRAIN.tsv', 'DuckDuckGeese_TEST.tsv'],
                 'Epilepsy':['Epilepsy_TRAIN.tsv','Epilepsy_TEST.tsv'],
                 'EthanolConcentration':['EthanolConcentration_TRAIN.tsv','EthanolConcentration_TEST.tsv'],
                 'Handwriting':['Handwriting_TRAIN.tsv','Handwriting_TEST.tsv'],
                 'Heartbeat':['Heartbeat_TRAIN.tsv','Heartbeat_TEST.tsv'],
                 'Libras':['Libras_TRAIN.tsv','Libras_TEST.tsv'],
                 'MotorImagery':['MotorImagery_TRAIN.tsv','MotorImagery_TEST.tsv'],
                 'NATOPS':['NATOPS_TRAIN.tsv','NATOPS_TEST.tsv'],
                 'RacketSports':['RacketSports_TRAIN.tsv','RacketSports_TEST.tsv'],
                 'UWaveGestureLibrary':['UWaveGestureLibrary_TRAIN.tsv','UWaveGestureLibrary_TEST.tsv'],
                 'FaceDetection':['FaceDetection_TRAIN.tsv','FaceDetection_TEST.tsv'],
                 'LSST':['LSST_TRAIN.tsv','LSST_TEST.tsv'],
                 'PenDigits':['PenDigits_TRAIN.tsv','PenDigits_TEST.tsv'],
                 'PEMS-SF':['PEMS-SF_TRAIN.tsv','PEMS-SF_TEST.tsv'],
                 'PhonemeSpectra':['PhonemeSpectra_TRAIN.tsv','PhonemeSpectra_TEST.tsv'],
                 'ERing':['ERing_TRAIN.tsv','ERing_TEST.tsv']
                 }

    train_file = os.path.join(data_path, dataset_name, file_dict[dataset_name][0])
    test_file = os.path.join(data_path, dataset_name, file_dict[dataset_name][1])
    data_info = {}

    if 'StandWalkJump' in train_file:
        data_info['ski_num'] = 20
        data_info['dimensions'] = 4
        data_info['class_num'] = 3
        data_info['label_map'] = {'standing':0, 'walking':1,'jumping':2}
    if 'AtrialFibrillation' in train_file:
        data_info['ski_num'] = 22
        data_info['dimensions'] = 2
        data_info['class_num'] = 3
        data_info['label_map'] = {'n':0, 's':1, 't':2}
    if 'SelfRegulationSCP1' in train_file:
        data_info['ski_num'] = 53
        data_info['dimensions'] = 6
        data_info['class_num'] = 2
        data_info['label_map'] = {'positivity':0, 'negativity':1}
    if 'SelfRegulationSCP2' in train_file:
        data_info['ski_num'] = 59
        data_info['dimensions'] = 7
        data_info['class_num'] = 2
        data_info['label_map'] = {'positivity':0, 'negativity':1}
    if 'HandMovementDirection' in train_file:
        data_info['ski_num'] = 58
        data_info['dimensions'] = 10
        # data_info['dimensions'] = 1
        data_info['class_num'] = 4
        data_info['label_map'] = {'forward': 0, 'left': 1, 'backward': 2, 'right': 3}
    if 'FingerMovements' in train_file:
        data_info['ski_num'] = 34
        data_info['dimensions'] = 28
        # data_info['dimensions'] = 2
        data_info['class_num'] = 2
        data_info['label_map'] = {'left':0, 'right':1}
    if 'ArticularyWordRecognition' in train_file:
        data_info['ski_num'] = 31
        data_info['dimensions'] = 9
        data_info['class_num'] = 25
        data_info['label_map'] = {1.0:0, 2.0:1, 3.0:2, 4.0:3, 5.0:4, 6.0:5, 7.0:6, 8.0:7, 9.0:8, 10.0:9, 11.0:10, 12.0:11, 13.0:12, 14.0:13, 15.0:14, 16.0:15, 17.0:16, 18.0:17, 19.0:18, 20.0:19, 21.0:20, 22.0:21, 23.0:22, 24.0:23, 25.0:24}
    if 'BasicMotions' in train_file:
        data_info['ski_num'] = 13
        data_info['dimensions'] = 6
        data_info['class_num'] = 4
        data_info['label_map'] = {'Standing':0, 'Running':1, 'Walking':2, 'Badminton':3}
    if 'Cricket' in train_file:
        data_info['ski_num'] = 27
        data_info['dimensions'] = 6
        data_info['class_num'] = 12
        data_info['label_map'] = {1.0:0, 2.0:1, 3.0:2, 4.0:3, 5.0:4, 6.0:5, 7.0:6, 8.0:7, 9.0:8, 10.0:9, 11.0:10, 12.0:11}
    if 'DuckDuckGeese' in train_file:
        data_info['ski_num'] = 31
        data_info['dimensions'] = 1345
        data_info['class_num'] = 5
        data_info['label_map'] = {'Black-bellied_Whistling_Duck':0, 'Canadian_Goose':1, 'Greylag_Goose':2, 'Pink-footed_Goose':3,'White-faced_Whistling_Duck':4}
    if 'Epilepsy' in train_file:
        data_info['ski_num'] = 42
        data_info['dimensions'] = 3
        data_info['class_num'] = 4
        data_info['label_map'] = {'EPILEPSY':0, 'WALKING':1, 'RUNNING':2, 'SAWING':3}
    if 'EthanolConcentration' in train_file:
        data_info['ski_num'] = 29
        data_info['dimensions'] = 3
        data_info['class_num'] = 4
        data_info['label_map'] = {'E35':0, 'E38':1, 'E40':2, 'E45':3}
    if 'Handwriting' in train_file:
        data_info['ski_num'] = 16
        data_info['dimensions'] = 3
        data_info['class_num'] = 26
        data_info['label_map'] = {1.0:0, 2.0:1, 3.0:2, 4.0:3, 5.0:4, 6.0:5, 7.0:6, 8.0:7, 9.0:8, 10.0:9, 11.0:10, 12.0:11, 13.0:12, 14.0:13, 15.0:14, 16.0:15, 17.0:16, 18.0:17, 19.0:18, 20.0:19, 21.0:20, 22.0:21, 23.0:22, 24.0:23, 25.0:24, 26.0:25}
    if 'Heartbeat' in train_file:
        data_info['ski_num'] = 26
        data_info['dimensions'] = 61
        data_info['class_num'] = 2
        data_info['label_map'] = {'normal':0, 'abnormal':1}
    if 'Libras' in train_file:
        data_info['ski_num'] = 121
        data_info['dimensions'] = 2
        data_info['class_num'] = 15
        data_info['label_map'] = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 12:11, 13:12, 14:13, 15:14}
    if 'MotorImagery' in train_file:
        data_info['ski_num'] = 39
        data_info['dimensions'] = 64
        data_info['class_num'] = 2
        data_info['label_map'] = {'finger':0, 'tongue':1}
    if 'NATOPS' in train_file:
        data_info['ski_num'] = 53
        data_info['dimensions'] = 24
        data_info['class_num'] = 6
        data_info['label_map'] = {1.0:0, 2.0:1, 3.0:2, 4.0:3, 5.0:4, 6.0:5}
    if 'RacketSports' in train_file:
        data_info['ski_num'] = 15
        data_info['dimensions'] = 6
        data_info['class_num'] = 4
        data_info['label_map'] = {'Badminton_Smash':0, 'Badminton_Clear':1, 'Squash_ForehandBoast':2, 'Squash_BackhandBoast':3}
    if 'UWaveGestureLibrary' in train_file:
        data_info['ski_num'] = 14
        data_info['dimensions'] = 3
        data_info['class_num'] = 8
        data_info['label_map'] = {1.0:0, 2.0:1, 3.0:2, 4.0:3, 5.0:4, 6.0:5, 7.0:6, 8.0:7}
    if 'FaceDetection' in train_file:
        data_info['ski_num'] = 18
        data_info['dimensions'] = 144
        data_info['class_num'] = 2
        data_info['label_map'] = {0:0, 1:1}
    if 'LSST' in train_file:
        data_info['ski_num'] = 55
        data_info['dimensions'] = 6
        data_info['class_num'] = 14
        data_info['label_map'] = {6:0, 15:1, 16:2, 42:3, 52:4, 53:5, 62:6, 64:7, 65:8, 67:9, 88:10, 90:11, 92:12, 95:13}
    if 'PenDigits' in train_file:
        data_info['ski_num'] = 15
        data_info['dimensions'] = 2
        data_info['class_num'] = 10
        data_info['label_map'] = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}
    if 'PEMS-SF' in train_file:
        data_info['ski_num'] = 24
        data_info['dimensions'] = 963
        data_info['class_num'] = 7
        data_info['label_map'] = {1.0:0, 2.0:1, 3.0:2, 4.0:3, 5.0:4, 6.0:5, 7.0:6}
    if 'PhonemeSpectra' in train_file:
        data_info['ski_num'] = 22
        data_info['dimensions'] = 11
        data_info['class_num'] = 39
        data_info['label_map'] = {'AA':0,'AE':1,'AH':2,'AO':3, 'AW':4, 'AY':5, 'B':6, 'CH':7, 'D':8, 'DH':9, 'EH':10, 'ER':11, 'EY':12,
                                  'F':13, 'G':14, 'HH':15, 'IH':16, 'IY':17, 'JH':18, 'K':19, 'L':20, 'M':21, 'N':22, 'NG':23, 'OW':24, 'OY':25,
                                  'P':26, 'R':27, 'S':28, 'SH':29, 'T':30, 'TH':31, 'UH':32, 'UW':33, 'V':34, 'W':35, 'Y':36, 'Z':37, 'ZH':38}
    if 'ERing' in train_file:
        data_info['ski_num'] = 15
        data_info['dimensions'] = 4
        data_info['class_num'] = 6
        data_info['label_map'] = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5}

    return train_file,test_file,data_info


def data_preprocessing_ISRUC(data, label, sample_num):
    '''
    preprocessing ISRUC dataset
    :param data:
    :param label:
    :return:
    '''
    logger.info("##### Data Preprocessing ISRUC ######")

    idx = random.sample(list(range(data.shape[0])),sample_num)
    # idx = list(range(data.shape[0]))
    data = data[idx,:,:]
    label = label[idx]
    logger.info("resample dataset shape:{}".format(data.shape))

    # split_idx = int((data.shape[0]/10)*7)
    # x_train = data[:split_idx,:,:]
    # y_train = label[:split_idx]
    # x_test = data[split_idx:,:,:]
    # y_test = label[split_idx:]
    # logger.info("x_train dataset shape:{}".format(x_train.shape))
    # logger.info("x_test dataset shape:{}".format(x_test.shape))
    #
    # return x_train, y_train, x_test, y_test

    return data, label

from sklearn.model_selection import StratifiedShuffleSplit
def kFoldGenerator(data, label, test_size):

    # logger.info("##### Generate k-Fold dataset ######")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)

    x_train = {}
    x_test = {}

    for train_index, test_index in sss.split(data['node_feature_matrix'], label):
        x_train['adj_matrix'] = data['adj_matrix'][train_index]
        x_train['node_feature_matrix'] = data['node_feature_matrix'][train_index]
        y_train = label[train_index]

        x_test['adj_matrix'] = data['adj_matrix'][test_index]
        x_test['node_feature_matrix'] = data['node_feature_matrix'][test_index]
        y_test = label[test_index]

    #
    # split_idx = int((len(label)/10)*9)
    #
    #
    # # shuffle dataset
    # # dim = 0
    # # idx = torch.randperm(data['adj_matrix'].shape[dim])
    # # data['adj_matrix'] = data['adj_matrix'][idx,:,:]
    # # data['node_feature_matrix'] = data['node_feature_matrix'][idx,:,:,:]
    # # label = label[idx]
    #
    # # np.random.seed(12)
    # # np.random.shuffle(data['adj_matrix'])
    # # np.random.seed(12)
    # # np.random.shuffle(data['node_feature_matrix'])
    # # np.random.seed(12)
    # # np.random.shuffle(label)
    #
    #
    # x_train = {}
    # x_train['adj_matrix'] = data['adj_matrix'][:split_idx]
    # # x_train['adj_matrix'] = data['adj_matrix'][split_idx:]
    # x_train['node_feature_matrix'] = data['node_feature_matrix'][:split_idx]
    # # x_train['node_feature_matrix'] = data['node_feature_matrix'][split_idx:]
    # y_train = label[:split_idx]
    # # y_train = label[split_idx:]
    #
    # x_test = {}
    # x_test['adj_matrix'] = data['adj_matrix'][split_idx:]
    # x_test['node_feature_matrix'] = data['node_feature_matrix'][split_idx:]
    # y_test = label[split_idx:]
    #
    # # x_train = data[:split_idx]
    # # y_train = label[:split_idx]
    # # x_test = data[split_idx:]
    # # y_test = label[split_idx:]


    return x_train, y_train, x_test, y_test

    # return data[:split_idx], data[split_idx:]


def kFoldGenerator_split_raw_data(data, label, test_size):
    # logger.info("##### Generate k-Fold dataset ######")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)


    for train_index, test_index in sss.split(data, label):
        x_train = data[train_index]
        y_train = label[train_index]

        x_test = data[test_index]
        y_test = label[test_index]

    return x_train, y_train, x_test, y_test


def data_preprocessing(train_file,test_file):
    '''
    data preprocessing
    :param data_info: the information about dataset, including ski_num,dimensions,class_num,label_map
    :param is_shuffle: True default.
    :return: x_train,y_train,x_test,y_test
    '''
    logger.info("##### Data Preprocessing ######")
    train_data = pd.read_csv(train_file, sep=',|:', skiprows=list(range(0, args.ski_num)), header=None)
    test_data = pd.read_csv(test_file, sep=',|:', skiprows=list(range(0, args.ski_num)), header=None)
    # replace label name to a specific number
    train_data.iloc[:,-1].replace(args.label_map,inplace=True)
    test_data.iloc[:,-1].replace(args.label_map,inplace=True)
    train_data.iloc[:,-1] = train_data.iloc[:,-1].astype(int)
    test_data.iloc[:,-1] = test_data.iloc[:,-1].astype(int)

    if args.is_shuffle:
        train_data = shuffle(train_data)
        test_data = shuffle(test_data)


    x_train = train_data.iloc[:,:-1].values
    y_train = train_data.iloc[:,-1].values
    x_test = test_data.iloc[:,:-1].values
    y_test = test_data.iloc[:,-1].values

    # reshape为m维度数据,so, we get a matrix [sample_num*dimension*point_value]
    x_train = x_train.reshape(x_train.shape[0], args.dimensions, int(x_train.shape[1]/args.dimensions))
    x_test = x_test.reshape(x_test.shape[0], args.dimensions, int(x_test.shape[1]/args.dimensions))
    logger.info("x_train dataset shape:{}".format(x_train.shape))
    logger.info("x_test dataset shape:{}".format(x_test.shape))

    # return x_train[:,:3,:], y_train, x_test[:,:3,:], y_test
    return x_train, y_train, x_test, y_test



import dgl
def generate_VG(data_matrix, label, vg_type='NVG'):
    '''
    generate visibility graph
    '''

    g_list = []
    for i in range(data_matrix.shape[0]):  #
        multi_ts = data_matrix[i, :, :]
        cur_MTS =data_matrix[i, :, :].transpose()
        mg = nx.Graph()
        for j in range(data_matrix.shape[1]):
            if j==0:
                multi_ts = data_matrix[i, :, :]
            else:
                multi_ts = np.roll(multi_ts, -1, axis=0)
            ts = data_matrix[i, j, :]

            if vg_type=='NVG':
                g = NaturalVG(directed=None).build(ts)  # each dimension transform into a visibility graph
            if vg_type=='HVG':
                g = HorizontalVG(directed=None).build(ts)
            nxg = g.as_networkx()
            # adj = nx.adjacency_matrix(nxg).todense()
            mg = nx.compose(mg, nxg)

            # g_list.append((dgl.from_networkx(mg),torch.tensor(label[i], dtype=torch.int32)))
        cur_dgl = dgl.from_networkx(mg)
        cur_dgl.ndata['feat'] = torch.FloatTensor(cur_MTS)
        g_list.append([cur_dgl, label[i]])

    return g_list

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

    data_path = "E:\phd\coding\DATA\Public Dataset\Multivariate_ts"

    dataset_name = ['AtrialFibrillation','StandWalkJump', 'SelfRegulationSCP1', 'SelfRegulationSCP2', 'HandMovementDirection',
                    'FingerMovements','ArticularyWordRecognition', 'BasicMotions', 'Cricket', 'DuckDuckGeese', 'Epilepsy', 'EthanolConcentration',
                    'Handwriting', 'Heartbeat', 'Libras', 'MotorImagery', 'NATOPS', 'RacketSports', 'UWaveGestureLibrary','FaceDetection',
                    'LSST','PenDigits','PEMS-SF','PhonemeSpectra','ERing']

    max_node_degree = 0


    now_date = time.strftime("%Y-%m-%d", time.localtime())
    ouput_path = './output/classification/model_{}'.format(now_date)
    if not os.path.exists(ouput_path):
        os.makedirs(ouput_path)
    result_path = os.path.join(ouput_path,'result')
    model_save_path = os.path.join(ouput_path,'save_model')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    global ISRUC_data, ISRUC_label
    repeat_num = 10

    L = [0, 1, 4, 5, 6, 7, 10, 12, 14, 17, 18, 16,20, 21,  24, 8]
    dataset = 'UEA'

    for i in L:
        if dataset == 'UEA':
            logger.info(" ")
            logger.info(" ")
            logger.info("=======start process dataset:{}".format(dataset_name[i]))
            train_file, test_file, data_info = file_parse(dataset_name[i],data_path)
            view_num = data_info['dimensions']
            num_classes = data_info['class_num']
            cur_time = time.strftime("%H_%M_%S", time.localtime())
            save_model_path = os.path.join(model_save_path, dataset_name[i]+"-{}".format(cur_time))
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            result_save_path = os.path.join(result_path, dataset_name[i])
            if not os.path.exists(result_save_path):
                os.makedirs(result_save_path)
        if dataset == 'ISRUC-S3':
            logger.info(" ")
            logger.info(" ")
            logger.info("=======start process dataset: {}".format(dataset))
            ISRUC_data = np.load('E:\phd\coding\DATA\Public Dataset\ISRUC\ISRUC-S3-resample300\ISRUC_S3_data.npy')
            ISRUC_label = np.load('E:\phd\coding\DATA\Public Dataset\ISRUC\ISRUC-S3-resample300\ISRUC_S3_label.npy')
            new_slpit_save_path = "E:\phd\coding\DATA\Public Dataset\Multivariate_ts_new_split"

            # 0-Wake:0、3373, 1-N1:340、5956, 2-N2:161, 3-N3:113, 4-REM:1695

            # plot_physiological_time_series(ISRUC_data[label_id,:,:],channels=ISRUC_data.shape[1])
            # communityDetection(ISRUC_data[label_id,1,:])
            cur_time = time.strftime("%H_%M_%S", time.localtime())
            save_model_path = os.path.join(model_save_path, 'ISRUC-S3' + "-{}".format(cur_time))
            result_save_path = os.path.join(result_path, 'ISRUC-S3')
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            if not os.path.exists(result_save_path):
                os.makedirs(result_save_path)

            logger.info("ISRUC_data.shape:{}".format(ISRUC_data.shape))
            logger.info("ISRUC_label.shape:{}".format(ISRUC_label.shape))
            data_info = {}
            data_info['ski_num'] = 0
            data_info['dimensions'] = 10
            data_info['class_num'] = 5
            data_info['series_len'] = ISRUC_data.shape[-1]
            data_info['label_map'] = {0:0, 1:1, 2:2, 3:3, 4:4}
            view_num = data_info['dimensions']
            num_classes = data_info['class_num']


        parser = argparse.ArgumentParser()
        parser.add_argument('--class_num',type=int,default=data_info['class_num'],help='the number of class')
        parser.add_argument('--dimensions',type=int,default=data_info['dimensions'],help='the number of dimension')
        parser.add_argument('--label_map',type=dict,default=data_info['label_map'],help='map label onto a specific number')
        parser.add_argument('--ski_num', type=int, default=data_info['ski_num'], help='ski_num to get data')
        parser.add_argument('--is_shuffle',type=bool,default=True, help='is shuffle data?(default:True)')
        parser.add_argument('--device', type=int, default=0,
                            help='which gpu to use if any (default: 0)')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='input batch size for training (default: 32)')
        parser.add_argument('--iters_per_epoch', type=int, default=10,
                            help='number of iterations per each epoch (default: 50)')
        parser.add_argument('--epochs', type=int, default=300,
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
        parser.add_argument('--model', help="Please give a value for model name")
        parser.add_argument('--dataset', help="Please give a value for dataset name")
        parser.add_argument('--out_dir', help="Please give a value for out_dir")
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
        # device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cpu")

        if view_num == 1:
            args.use_final_layer_attention = 0
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        # 1.preprocess train/test data
        if dataset == 'UEA':
            if dataset_name[i] not in ['Libras','RacketSports','LSST','PenDigits']:
                args.kernel_size_1D = 41
            else:
                args.kernel_size_1D = 21
            logger.info("args.kernel_size_1D:{}".format(args.kernel_size_1D))

            logger.info("==using original dataset")
            x_train, y_train, x_test, y_test = data_preprocessing(train_file, test_file)
            from utils.draw_vg import plot_vg
            # plot_vg(x_train[10, 0, :], vg_type='NVG')

            logger.info("train label distribution:{}".format(np.bincount(y_train)))
            logger.info("test label distribution:{}".format(np.bincount(y_test)))
            # adapative batch size
            if y_test.shape[0] < args.batch_size or 0.3*y_train.shape[0]< args.batch_size:
                args.batch_size = 32
            if y_test.shape[0] < args.batch_size or 0.3*y_train.shape[0]< args.batch_size:
                args.batch_size = 16
            if y_test.shape[0] < args.batch_size or 0.3*y_train.shape[0]< args.batch_size:
                args.batch_size = 4
            logger.info("args.batch_size:{}".format(args.batch_size))
        if dataset == 'ISRUC-S3':
            # x_train, y_train, x_test, y_test = data_preprocessing_ISRUC(ISRUC_data, ISRUC_label)
            # ISRUC_data, ISRUC_label = data_preprocessing_ISRUC(ISRUC_data, ISRUC_label, sample_num=500)
            # logger.info("label distribution:{}".format(np.bincount(ISRUC_label)))
            args.kernel_size_1D = 41

            resave_data_path = os.path.join(new_slpit_save_path, "ISRUC_S3_resample2000")
            if not os.path.exists(resave_data_path):
                print("=====split train & test.....")
                os.makedirs(resave_data_path)

                # shuffle data
                np.random.seed(12)
                np.random.shuffle(ISRUC_data)
                np.random.seed(12)
                np.random.shuffle(ISRUC_label)

                # split train and test
                x_train, y_train, x_test, y_test = kFoldGenerator_split_raw_data(ISRUC_data, ISRUC_label, test_size=0.3)
                # save train_data, test_data
                np.save(os.path.join(resave_data_path,'train_dataset.npy'),x_train)
                np.save(os.path.join(resave_data_path,'train_label.npy'),y_train)
                np.save(os.path.join(resave_data_path,'test_dataset.npy'),x_test)
                np.save(os.path.join(resave_data_path,'test_label.npy'),y_test)

                # logger.info("label distribution:{}".format(np.bincount(ISRUC_label)))
                # print("===ISRUC_data.shape:",ISRUC_data.shape)
            else:
                print("=====load train & test.....")
                x_train = np.load(os.path.join(resave_data_path, 'train_dataset.npy'))
                y_train = np.load(os.path.join(resave_data_path, 'train_label.npy'))
                x_test = np.load(os.path.join(resave_data_path, 'test_dataset.npy'))
                y_test = np.load(os.path.join(resave_data_path, 'test_label.npy'))

            logger.info("train label distribution:{}".format(np.bincount(y_train)))
            logger.info("test label distribution:{}".format(np.bincount(y_test)))


        logger.info("-----------generate visibility graph----------------")
        t1 = time.time()
        train_VG_data = generate_VG(x_train, y_train, vg_type=args.VG_type)
        test_VG_data = generate_VG(x_test, y_test, vg_type=args.VG_type)
        logger.info("generate graph cost {} seconds.".format(time.time()-t1))

        from data_provider.data import MVGsDataset
        trainset = MVGsDataset(train_VG_data)
        testset = MVGsDataset(test_VG_data)

        import json
        with open(args.config) as f:
            config = json.load(f)

        params = config['params']
        net_params = config['net_params']
        net_params['n_classes'] = args.class_num
        if net_params['tf_pos_enc']:
            net_params['in_dim'] = x_train.shape[1]   # the dim of node feature
        else:
            net_params['in_dim'] = x_train.shape[-1]   # the length of time series
        device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
        net_params['device'] = device

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

        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=trainset.collate, drop_last=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, collate_fn=testset.collate, drop_last=True)

        # setting seeds
        random.seed(params['seed'])
        np.random.seed(params['seed'])
        torch.manual_seed(params['seed'])
        if device.type == 'cuda':
            torch.cuda.manual_seed(params['seed'])
        print("Training Graphs: ", len(trainset))
        print("Test Graphs: ", len(testset))
        print("Number of Classes: ", net_params['n_classes'])

        from nets.load_net_classification import gnn_model

        MODEL_NAME = config['model']
        model = gnn_model(MODEL_NAME, net_params)
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

        from train.train_graph_classification import train_epoch, evaluate_network
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            with tqdm(range(params['epochs'])) as t:
                for epoch in t:
                    t.set_description('Epoch %d' % epoch)

                    start_time = time.time()

                    epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader,
                                                                               epoch)

                    # epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch)
                    epoch_test_loss, epoch_test_acc = evaluate_network(model, device, test_loader, epoch, dataset_name[i])

                    epoch_train_losses.append(epoch_train_loss)
                    epoch_train_accs.append(epoch_train_acc)

                    log_writer.add_scalar('Loss/train', float(epoch_train_loss), epoch)
                    log_writer.add_scalar('Loss/test', float(epoch_test_loss), epoch)
                    log_writer.add_scalar('Acc/train', float(epoch_train_acc), epoch)
                    log_writer.add_scalar('Acc/test', float(epoch_test_acc), epoch)


                    # print("===epoch {}: train_acc:{}, epoch_test_acc:{}".format(epoch, epoch_train_acc,epoch_test_acc))
                    logger.info("-----{}-----epoch {}/{}:".format(dataset_name[i], epoch, params['epochs']))
                    logger.info("cost {} seconds:".format(time.time() - start_time))
                    logger.info("training loss:{}, testing loss:{}".format(epoch_train_loss, epoch_test_loss))
                    logger.info("accuracy train:{},accuracy test:{}".format(epoch_train_acc, epoch_test_acc))

                    if epoch_test_acc == best_test_acc or epoch_test_acc > best_test_acc:
                        best_test_acc = epoch_test_acc
                        best_epoch = epoch
                        torch.save(model.state_dict(), save_model_path+'/best_ACC_{}_Epoch_{}.pt'.format(round(best_test_acc,6),best_epoch))

                    logger.info("current best test acc:{} in epoch {}.".format(best_test_acc, best_epoch))


        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early because of KeyboardInterrupt')
        

        total_execution_time = (time.time() - t0)/params['epochs']
        df_metric = pd.DataFrame(data=[[best_test_acc, best_epoch, total_execution_time]],columns=('best_test_acc','best_epoch','avg_execution time'))
        df_metric.to_csv(os.path.join(result_save_path,'df_metric.csv'),index=False,mode='a')
