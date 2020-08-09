# nohup python baseline.py -m MA [-p 0 -H 5 -n 10 -D sg_expressway_8weeks -c 1] &
import pandas as pd
import time
from datetime import date, timedelta
from datetime import datetime as dt
import os
from utils import *
import random
import numpy as np
from math import radians, degrees, sin, cos, asin, acos, sqrt
import pickle as pkl
from metrics import *
from trajectory_transition import extract_trajectory_transition
from road_graph import extract_road_adj
from model import *
from sklearn.preprocessing import StandardScaler
import argparse
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.ensemble import RandomForestRegressor


################ baseline models ################

def baseline_MA(df, history_window=4, prediction_window=1, test_ratio=0.25):
    # moving average: average of previous timestamps for predicted timestamp

    n_timestamp, n_road = df.shape
    n_timestamp_train = int(round(n_timestamp * (1 - test_ratio)))
    n_timestamp_test = n_timestamp - n_timestamp_train
    
    X_test = np.concatenate([np.expand_dims(
        flow_df[(n_timestamp_train - history_window - prediction_window + 1 + i) 
                : (n_timestamp - history_window - prediction_window + 1 + i)], axis=2) 
                             for i in range(history_window)], axis=2)  # (n_sample, n_road, history_window)
    y_test = np.array(flow_df.iloc[n_timestamp_train:]) # (n_sample, n_road)
    y_pred = np.average(X_test, axis=2) # (n_sample, n_road)
    
    return y_pred, y_test


def baseline_static(df, history_window=None, prediction_window=1, test_ratio=0.25):
    # static: previous timestamp for predicted timestamp
    # history_window: always set to 1.
    
    y_pred, y_test = baseline_MA(df, history_window=1, prediction_window=prediction_window, test_ratio=test_ratio)
    
    return y_pred, y_test


def baseline_HA(df, previous_weeks=None, interval=15, test_ratio=0.25, history_window=None, prediction_window=None):
    # historical average: average of same timestamp in previous weeks for next timestamp
    # previous_weeks: if None, use as many previous weeks as possible
    # interval: 15 minutes
    # history_window: not in use
    # prediction_window: not in use

    n_timestamp, n_road = df.shape
    n_timestamp_train = int(round(n_timestamp * (1 - test_ratio)))
    n_timestamp_test = n_timestamp - n_timestamp_train
    if previous_weeks is None:
        previous_weeks = n_timestamp_train // int( 7 * 24 * 60 / interval )
    
    X_test = np.concatenate([np.expand_dims(
        flow_df[int(n_timestamp_train - 7 * 24 * 60 / interval * (i + 1)) 
                : int(n_timestamp_train - 7 * 24 * 60 / interval * (i + 1) + n_timestamp_test)], axis=2) 
                             for i in range(previous_weeks)], axis=2) # (n_sample, n_road, history_window)
    y_test = np.array(flow_df.iloc[n_timestamp_train:]) # (n_sample, n_road)
    y_pred = np.average(X_test, axis=2) # (n_sample, n_road)
    
    return y_pred, y_test


# new version, considering only small neighborhood. updated 20200408.
def baseline_VAR(flow_df, road_adj, hops=5, history_window=4, prediction_window=1, test_ratio=0.25):
    
    n_timestamp, n_road = flow_df.shape
    n_timestamp_train = int(round(n_timestamp * (1 - test_ratio)))
    n_timestamp_test = n_timestamp - n_timestamp_train
    
    # find neighbors for each node
    symm_adj = road_adj+road_adj.transpose()
    neighbor_adj = symm_adj
    for hop in range(hops-1):
        neighbor_adj = np.matmul(neighbor_adj, symm_adj) + symm_adj
    np.fill_diagonal(neighbor_adj, 0) # exclude self
    
    train_data = np.array(flow_df.iloc[:n_timestamp_train]) # (n_timestamp_train, n_road)
    test_data = np.array(flow_df.iloc[n_timestamp_train:]) # (n_timestamp_test, n_road)
    
    Y_true = test_data[history_window + (prediction_window-1) : n_timestamp_test] # (n_sample, n_road)
    Y_pred = np.zeros(Y_true.shape) # (n_sample, n_road)
    
    for road_index in range(n_road): 
        
        filtered_roads = [road_index]+list(np.where(neighbor_adj[road_index]>0)[0])
        filtered_train_data = np.array(train_data[:, filtered_roads])
        filtered_test_data = np.array(test_data[:, filtered_roads])
        
        model = VAR(filtered_train_data)
        model_fitted = model.fit(history_window)
        

        X_test = np.concatenate([np.expand_dims(filtered_test_data[i : (n_timestamp_test - history_window - prediction_window + 1 + i)], axis=2) for i in range(history_window)], axis=2) # (n_sample, n_road, history_window)
        for i in range(Y_pred.shape[0]): # n_sample
            Y_pred[i, road_index] = model_fitted.forecast(X_test[i].transpose(), steps=prediction_window)[-1, :][0]
    
#     max_value = Y_true.max()
#     print((Y_pred > max_value).sum()) # no super large values
#     print((Y_pred < 0).sum()) # negative values account for 0.2%
    Y_pred[Y_pred < 0] = 0 # correct negative values
    
    return Y_pred, Y_true


# new version, considering only small neighborhood. updated 20200417.
def baseline_RF(df, road_adj, hops=5, n_estimators=10, history_window=4, prediction_window=1, test_ratio=0.25):
    # random forest
    
    n_timestamp, n_road = df.shape
    n_timestamp_train = int(round(n_timestamp * (1 - test_ratio)))
    n_timestamp_test = n_timestamp - n_timestamp_train
    
    # find neighbors for each node
    symm_adj = road_adj+road_adj.transpose()
    neighbor_adj = symm_adj
    for hop in range(hops-1):
        neighbor_adj = np.matmul(neighbor_adj, symm_adj) + symm_adj
    np.fill_diagonal(neighbor_adj, 0) # exclude self
    
    train_data = np.array(df.iloc[:n_timestamp_train]) # (n_timestamp_train, n_road)
    test_data = np.array(df.iloc[n_timestamp_train:]) # (n_timestamp_test, n_road)
    
    Y_true = test_data[history_window + (prediction_window-1) : n_timestamp_test] # (n_sample, n_road)
    Y_pred = np.zeros(Y_true.shape) # (n_sample, n_road)
    
    print('Fitting RF model...')
    start_time = time.time()
    
    for road_index in range(n_road): 
        
        filtered_roads = [road_index]+list(np.where(neighbor_adj[road_index]>0)[0])
        
        n_sample_train = n_timestamp_train - history_window - prediction_window + 1 
        X_train = np.concatenate([np.expand_dims(train_data[i : (n_sample_train + i), filtered_roads], axis=2) for i in range(history_window)], axis=2) # (n_sample, n_filtered_road, history_window)
        X_train = np.reshape(X_train, (n_sample_train, -1)) # (n_sample, n_filtered_road * history_window)
        y_train = train_data[history_window + prediction_window - 1 : , [road_index]] # (n_sample, 1)
        
        n_sample_test = n_timestamp_test - history_window - prediction_window + 1 
        X_test = np.concatenate([np.expand_dims(test_data[i : (n_sample_test + i), filtered_roads], axis=2) for i in range(history_window)], axis=2) # (n_sample, n_filtered_road, history_window)
        X_test = np.reshape(X_test, (n_sample_test, -1)) # (n_sample, n_filtered_road * history_window)
        y_test = test_data[history_window + prediction_window - 1 : , [road_index]] # (n_sample, 1)
        
        model = RandomForestRegressor(n_estimators, random_state=0)
        model.fit(X_train, y_train)
        
        for i in range(Y_pred.shape[0]): # n_sample
            Y_pred[i, road_index] = model.predict(X_test[[i]])[0]
    
    print('Time Spent: %.2f s'%(time.time()-start_time))
#     max_value = Y_true.max()
#     print((Y_pred > max_value).sum()) # no super large values
#     print((Y_pred < 0).sum())
    Y_pred[Y_pred < 0] = 0 # correct negative values (should be zero or super-small portion)
    
    return Y_pred, Y_true


################ run baseline model ##################

if __name__ == '__main__':
    
    
    # Arguments
    parser = argparse.ArgumentParser(description='baseline')
    parser.add_argument('-m', '--model_name', help='MA', required=True)
    parser.add_argument('-p', '--previous_weeks', help='for HA. 0 means None.', default=0)
    parser.add_argument('-H', '--hops', help='for VAR and RF', default=5)
    parser.add_argument('-n', '--n_estimators', help='for RF', default=10)
    parser.add_argument('-D', '--dataset', help='sg_expressway_8weeks', default='sg_expressway_8weeks')
    parser.add_argument('-c', '--calibrate', help='flow calibration on a daily basis', default=1)
    args = parser.parse_args()
    model_name, dataset, hops, n_estimators, calibrate, previous_weeks = args.model_name, args.dataset, int(args.hops), int(args.n_estimators), bool(args.calibrate), int(args.previous_weeks)
    
    # model and log
    models = {'MA': baseline_MA, 'static': baseline_static, 'HA':baseline_HA, 'VAR':baseline_VAR, 'RF':baseline_RF}
    model = models[model_name]
    model_path = 'model/%s.cpt'%model_name # not in use for MA, HA, static
    log_path = 'log/%s.log'%model_name

    # Dataset: flow
    # 'sg_expressway_4weeks', 'sg_expressway_8weeks'
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
    if calibrate:
        print_log('Calibrating flow...', log_path)
        trajectory_metadata = pd.read_csv('data/trajectory_metadata.csv') # read trajectory metadata
        multipliers = np.repeat(np.array(trajectory_metadata['vehicles'][0] / trajectory_metadata['vehicles']), 96)
        multipliers[multipliers==np.inf]=0
        flow_df = flow_df.mul(multipliers, axis=0)
    print_log('flow_df: ' + str(flow_df.shape), log_path)
    print_log('Total flow: %d'%(flow_df.sum().sum()), log_path)
    # Dataset: road_adj
    road_adj = extract_road_adj()
    road_adj # upper triangular
    print_log('road_adj: ' + str(road_adj.shape), log_path)

    # parameters
    history_window = 4
    prediction_window = 1
    test_ratio = 0.25
    previous_weeks = None if previous_weeks==0 else previous_weeks # for HA
    hops = hops # for VAR and RF. 5 by default.
    n_estimators = n_estimators # for RF. 10 by default.
    
    # run model
    start_time = time.time()
    if model_name == 'HA':
        prefix = model_name if previous_weeks is None else model_name + str(previous_weeks)
        Y_pred, Y_true = model(flow_df, previous_weeks=previous_weeks, history_window=history_window, prediction_window=prediction_window, test_ratio=test_ratio)
    if model_name in ['static', 'MA']:
        prefix = model_name
        Y_pred, Y_true = model(flow_df, history_window=history_window, prediction_window=prediction_window, test_ratio=test_ratio)
    if model_name == 'VAR':
        prefix = 'VAR_%dhop'%hops
        Y_pred, Y_true = model(flow_df, road_adj, hops=hops, history_window=history_window, prediction_window=prediction_window, test_ratio=test_ratio)
    if model_name == 'RF':
        prefix = 'RF_%dhop_%destimator'%(hops, n_estimators)
        Y_pred, Y_true = model(flow_df, road_adj, hops=hops, n_estimators=n_estimators, history_window=history_window, prediction_window=prediction_window, test_ratio=test_ratio)
    time_spent = time.time() - start_time
    mae = MAE(Y_pred, Y_true, main_roads=False)
    mape = MAPE(Y_pred, Y_true, main_roads=False)
    rmse = RMSE(Y_pred, Y_true, main_roads=False)
    print_log('Model: %s, Prediction window: %s'%(prefix, prediction_window), log_path)
    print_log('MAE: %.3f, MAPE: %.3f, RMSE: %.3f, Time spent: %.2f s'%(mae, mape, rmse, time_spent), log_path)
    print_log('Saving results...', log_path)
    with open('result/%s_Y_true.pkl'%prefix, 'wb') as f:
        pkl.dump(Y_true, f)
    with open('result/%s_Y_pred.pkl'%prefix, 'wb') as f:
        pkl.dump(Y_pred, f)
    
    # result analysis
#     model_type = 'VAR' if model_name in ['VAR', 'RF'] else 'baseline' # for result analysis
#     result_function = result_analysis2 if dataset == 'sg_expressway_8weeks' else result_analysis
#     result_function(Y_pred, Y_true, model_type=model_type, log_path=log_path)

    