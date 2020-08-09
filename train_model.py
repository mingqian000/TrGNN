# CUDA_VISIBLE_DEVICES=0 nohup python train_model.py -m TrGNN [-D sg_expressway_8weeks -p TrGNN_1581343606_100epoch.cpt -c 1] &
import pandas as pd
import time
from datetime import date, timedelta
from datetime import datetime as dt
import os
import folium
from utils import *
import random
import numpy as np
from math import radians, degrees, sin, cos, asin, acos, sqrt
import pickle as pkl
import networkx as nx
from metrics import *
from trajectory_transition import extract_trajectory_transition
from road_graph import extract_road_adj
from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import argparse


# Arguments
parser = argparse.ArgumentParser(description='train_model')
parser.add_argument('-m', '--model_name', help='TrGNN', required=True)
parser.add_argument('-D', '--dataset', help='sg_expressway_8weeks', default='sg_expressway_8weeks')
parser.add_argument('-p', '--pre_trained', help='pre-trained model path. E.g. TrGNN_1581343606_100epoch.cpt', default='')
parser.add_argument('-c', '--calibrate', help='flow calibration on a daily basis', default=1)
args = parser.parse_args()
model_name, dataset, model_path, calibrate = args.model_name, args.dataset, args.pre_trained, bool(args.calibrate)


start_time = time.time()


# Model and log
models = {'TrGNN':Model_TrGNN, 'TrGNN-':Model_GNN}
model = models[model_name]()
if model_path == '': # if no pre-trained model path
    prefix = '%s_%s'%(model_name, int(start_time))
    checkpoint_epoch = -1
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))
    prefix = '_'.join(model_path.split('_')[:2])
    checkpoint_epoch = int(model_path.split('_')[-1][:-9])
model_path = 'model/%s_%sepoch.cpt'%(prefix, '%d')
log_path = 'log/%s.log'%prefix


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print_log(device, log_path)


# Dataset
# 'sg_expressway_4weeks', 'sg_expressway_8weeks'
road_adj = extract_road_adj() # directed adj

if dataset == 'demo':
    start_date, end_date = '20160314', '20160314'
    calibrate = False
elif dataset == 'sg_expressway_8weeks':
    start_date, end_date = '20160314', '20160424' # train period + validation period
else:
    start_date, end_date = '20160401', '20160421' # train period + validation period
trajectory_transition = extract_trajectory_transition(start_date, end_date)
# smoothing with binary road_adj, in case no historical flow is recorded.
road_adj_mask = np.zeros(road_adj.shape)
road_adj_mask[road_adj > 0] = 1
np.fill_diagonal(road_adj_mask, 0)
for i in range(len(trajectory_transition)):
    trajectory_transition[i] = trajectory_transition[i] + road_adj_mask

if dataset == 'demo':
    start_date, end_date = '20160314', '20160314'
    calibrate = False
elif dataset == 'sg_expressway_8weeks':
    start_date, end_date = '20160314', '20160508' # train (5 weeks) + validation (1 week) + test (2 weeks)
else:
    start_date, end_date = '20160401', '20160428' # train + validation + test
dates = date_range(start_date, end_date)
flow_df = pd.concat([pd.read_csv('data/flow_%s_%s.csv'%(date, date), index_col=0) for date in dates])
flow_df.columns = pd.Index(int(road_id) for road_id in flow_df.columns)
# flow calibration on a daily basis
if calibrate:
    print_log('Calibrating flow...', log_path)
    trajectory_metadata = pd.read_csv('data/trajectory_metadata.csv') # read trajectory metadata
    multipliers = np.repeat(np.array(trajectory_metadata['vehicles'][0] / trajectory_metadata['vehicles']), 96)
    multipliers[multipliers==np.inf]=0
    flow_df = flow_df.mul(multipliers, axis=0)
print_log(flow_df.shape, log_path)
print_log('Total flow: %d'%(flow_df.sum().sum()), log_path)


if dataset == 'demo': # 20160314
    indices = {'train': list(range(56)), # first 14 hours
               'val': list(range(56, 68)), # next 3 hours
               'test': list(range(68, 92))} # last 6 hours
    weekdays = np.array([0]) # day 0 (i.e. 20160314) is a weekday
elif dataset == 'sg_expressway_8weeks': # version 20160314-20160508
    indices = {'train': list(range(3220)), # first 5 weeks 20160314-20160417 (24-1)*(60/15)*56
               'val': list(range(3220, 3864)), # 6th week 20160418-20160424 (24-1)*(60/15)*7
               'test': list(range(3864, 5152))} # 7th-8th weeks 20160425-20160508 (24-1)*(60/15)*14
    # indices of weekdays (exclude weekends and PHs)
    weekdays = np.array([0, 1, 2, 3, 4, 
                         7, 8, 9, 10, # PH: 25th May, Friday
                         14, 15, 16, 17, 18,
                         21, 22, 23, 24, 25,
                         28, 29, 30, 31, 32, 
                         35, 36, 37, 38, 39,
                         42, 43, 44, 45, 46, 
                         50, 51, 52, 53]) # PH: 2nd May, Monday
else: # version 20160401-20160428
    indices = {'train': list(range(1288)), # first two weeks (24-1)*(60/15)*14
               'val': list(range(1288, 1932)), # third week (24-1)*(60/15)*7
               'test': list(range(1932, 2576))} # fourth week (24-1)*(60/15)*7


scaler = StandardScaler().fit(flow_df.iloc[indices['train'] + indices['val']].values) # normalize flow


# Train model
loss_fn = nn.MSELoss()
learning_rate = 0.004
num_epochs = 100
min_mae = 10 # initialize
early_stop_threshold = 3.0 # for val_mae
# result_function = result_analysis2 if dataset == 'sg_expressway_8weeks' else result_analysis


def validate(mode='val'):
    # mode: ['val', 'test']. Validate on validation set or test set.
    
    running_loss = 0
    n_samples = 0
    
    h_init = torch.zeros(5, 2404, 1) # (gru_num_layers, n_road, hidden_size)
    h_init = h_init.to(device)
    
    Y_true = np.zeros((len(indices[mode]), 2404)) # (n_sample, n_road)
    Y_pred = np.zeros((len(indices[mode]), 2404))
    for i in indices[mode]:

        d = i // 92
        t = i % 92

        X = normalized_flows[d*96+t : d*96+t+4]
        T = tuple(transitions_ToD[t:t+4])
        # W passed to device already
        y_true = normalized_flows[d*96+t+4]
        
        ToD = torch.from_numpy(np.eye(24)[np.full((2404), ((t+4) * 15 // 60) % 24)]).float().to(device) # one-hot encoding: hour of day. (n_road, 24)
        DoW = torch.from_numpy(np.full((2404, 1), int(d in weekdays))).float().to(device) # indicator: 1 for weekdays, 0 for weekends/PHs. (n_road, 1)
        y_pred = model(X, T, W, h_init, W_norm, ToD, DoW)
        
        Y_true[n_samples] = flow_df.iloc[d*96+t+4].values
        Y_pred[n_samples] = scaler.inverse_transform(y_pred.detach().cpu().numpy())
        
        loss = loss_fn(y_pred, y_true)
        loss.detach_()

        running_loss += loss.item()
        n_samples += 1
    
    Y_pred[Y_pred < 0] = 0 # correction for negative values
    mae = MAE(Y_pred, Y_true, main_roads=False)
    mape = MAPE(Y_pred, Y_true, main_roads=False)
    rmse = RMSE(Y_pred, Y_true, main_roads=False)
    print_log('>> %s_loss: %.3f, MAE: %.3f, MAPE: %.3f, RMSE: %.3f'%(mode, running_loss/n_samples, mae, mape, rmse), log_path)
    
    return running_loss/n_samples, Y_pred, Y_true, mae

    
# preprocessing
print_log('Preprocessing...', log_path)
normalized_flows = torch.from_numpy(scaler.transform(flow_df.values)).float().to(device) # for X. normalized
transitions_ToD = [to_sparse_tensor(normalize_adj(trajectory_transition[i])).to(device) for i in range(len(trajectory_transition))] # for T. time of day
W = torch.from_numpy(road_adj).to(device) # for W
W_norm = torch.from_numpy(normalize_adj(road_adj, mode='aggregation')).to(device) # for normalized W
print_log('Preprocessing completed. Clock: %.0f seconds'%(time.time() - start_time), log_path)

print_log('Training model...', log_path)
for epoch in range(checkpoint_epoch+1, num_epochs):
    
    print_log('Epoch %d'%epoch, log_path)
    
    if epoch%30 == 0:
        learning_rate /= 2
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    h_init = torch.zeros(5, 2404, 1) # (gru_num_layers, n_road, hidden_size)
    h_init = h_init.to(device)
    
    running_loss = 0
    n_samples = 0
    
    np.random.shuffle(indices['train'])
    for index in range(len(indices['train'])):
        
        i = indices['train'][index]
        d = i // 92
        t = i % 92
        
        X = normalized_flows[d*96+t : d*96+t+4] # tensor: (n_timestamp, n_road)
        T = tuple(transitions_ToD[t:t+4]) # tuple of n_timestamp sparse_tensors: (n_road, n_road)
        # W passed to device already # sparse_tensor: (n_road, n_road)
        y_true = normalized_flows[d*96+t+4] # (n_road)
        
        optimizer.zero_grad()
        ToD = torch.from_numpy(np.eye(24)[np.full((2404), ((t+4) * 15 // 60) % 24)]).float().to(device) # one-hot encoding: hour of day. (n_road, 24)
        DoW = torch.from_numpy(np.full((2404, 1), int(d in weekdays))).float().to(device) # indicator: 1 for weekdays, 0 for weekends/PHs. (n_road, 1)
        y_pred = model(X, T, W, h_init, W_norm, ToD, DoW)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        n_samples += 1
        if n_samples % 500 == 0:
            print_log('Epoch %d, %d samples, clock: %.0f seconds'%(epoch, n_samples, time.time() - start_time), log_path)
    
    train_loss = running_loss/n_samples
    print_log('Validating...', log_path)
    val_loss, Y_pred, Y_true, val_mae = validate(mode='val')
    print_log('Testing...', log_path)
    test_loss, Y_pred, Y_true, test_mae = validate(mode='test')
    line = 'Epoch %d, time spent: %.0f seconds, train_loss: %.3f, val_loss: %.3f, test_loss: %.3f'%(epoch, time.time()-start_time, train_loss, val_loss, test_loss)
    print_log(line, log_path)
    
    if val_mae < min_mae:
        min_mae = val_mae
        print_log('Saving model...', log_path)
        torch.save(model.state_dict(), model_path%epoch)
        print_log('Saving results...', log_path)
        with open('result/%s_Y_true.pkl'%(prefix), 'wb') as f:
            pkl.dump(Y_true, f)
        with open('result/%s_%sepoch_Y_pred.pkl'%(prefix, epoch), 'wb') as f:
            pkl.dump(Y_pred, f)
#         result_function(Y_pred, Y_true, model_type='ours', log_path=log_path) # result analysis on test results
        
    if min_mae < early_stop_threshold:
        print_log('Early stop.', log_path)
        break
