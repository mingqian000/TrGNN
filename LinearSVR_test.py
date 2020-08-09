# nohup python LinearSVR_test.py &
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
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from metrics import MAE, MAPE, RMSE


start_date, end_date = '20160401', '20160428'
dates = date_range(start_date, end_date)
start_time = time.time()
flow_df = pd.concat([pd.read_csv('data/flow_%s_%s.csv'%(date, date), index_col=0) for date in dates])
flow_df.columns = pd.Index(int(road_id) for road_id in flow_df.columns)


def result_analysis(Y_pred, Y_true):
    # for LinearSVR prediction_window=1, y_pred.shape=(668, 2404)
    
    # abnormal day: 22nd Apr
    _Y_pred = Y_pred[:92]
    _Y_true = Y_true[:92]
    mae = MAE(_Y_pred, _Y_true, main_roads=False)
    mape = MAPE(_Y_pred, _Y_true, main_roads=False)
    rmse = RMSE(_Y_pred, _Y_true, main_roads=False)
    line = '>> result analysis - abnormal day 22nd Apr. MAE: %.3f, MAPE: %.3f, RMSE: %.3f'%(mae, mape, rmse)
    print(line)
    with open('log/LinearSVR.log', 'a') as f:
        f.write(line)

    
    # abnormal hours: 8am-9am
    _Y_pred = np.zeros((4*7, 2404)) # (n_sample, n_road)
    _Y_true = np.zeros((4*7, 2404))
    for i in range(7):
        _Y_pred[i*4:(i+1)*4] = Y_pred[((8-1)*4+96*i):((9-1)*4+96*i)]
        _Y_true[i*4:(i+1)*4] = Y_true[((8-1)*4+96*i):((9-1)*4+96*i)]
    mae = MAE(_Y_pred, _Y_true, main_roads=False)
    mape = MAPE(_Y_pred, _Y_true, main_roads=False)
    rmse = RMSE(_Y_pred, _Y_true, main_roads=False)
    line = '>> result analysis - abnormal hours 8am-9am. MAE: %.3f, MAPE: %.3f, RMSE: %.3f'%(mae, mape, rmse)
    print(line)
    with open('log/LinearSVR.log', 'a') as f:
        f.write(line)
    
    # abnormal hours: 11pm-12am
    _Y_pred = np.zeros((4*7, 2404)) # (n_sample, n_road)
    _Y_true = np.zeros((4*7, 2404))
    for i in range(7):
        _Y_pred[i*4:(i+1)*4] = Y_pred[((23-1)*4+96*i):((24-1)*4+96*i)]
        _Y_true[i*4:(i+1)*4] = Y_true[((23-1)*4+96*i):((24-1)*4+96*i)]
    mae = MAE(_Y_pred, _Y_true, main_roads=False)
    mape = MAPE(_Y_pred, _Y_true, main_roads=False)
    rmse = RMSE(_Y_pred, _Y_true, main_roads=False)
    line = '>> result analysis - abnormal hours 11pm-12am. MAE: %.3f, MAPE: %.3f, RMSE: %.3f'%(mae, mape, rmse)
    print(line)
    with open('log/LinearSVR.log', 'a') as f:
        f.write(line)
    
    return


for prediction_window in [1]: # 1, 2, 3, 4
    line = 'Prediction window: %d'%prediction_window
    print(line)
    with open('log/LinearSVR.log', 'a') as f:
        f.write(line)
        
    history_window=4
    test_ratio=0.25

    n_timestamp, n_road = flow_df.shape
    n_timestamp_train = int(round(n_timestamp * (1 - test_ratio)))
    n_timestamp_test = n_timestamp - n_timestamp_train

    train_data = np.array(flow_df.iloc[:n_timestamp_train]) # (n_timestamp_train, n_road)
    test_data = np.array(flow_df.iloc[n_timestamp_train:]) # (n_timestamp_test, n_road)

    n_sample_train = n_timestamp_train - history_window - prediction_window + 1 
    X_train = np.concatenate([np.expand_dims(train_data[i : (n_sample_train + i)], axis=2) for i in range(history_window)], axis=2) # (n_sample, n_road, history_window)
    X_train = np.reshape(X_train, (n_sample_train, -1)) # (n_sample, n_road * history_window)
    y_train = train_data[history_window + prediction_window - 1 : ] # (n_sample, n_road)

    n_sample_test = n_timestamp_test - history_window - prediction_window + 1 
    X_test = np.concatenate([np.expand_dims(test_data[i : (n_sample_test + i)], axis=2) for i in range(history_window)], axis=2) # (n_sample, n_road, history_window)
    X_test = np.reshape(X_test, (n_sample_test, -1)) # (n_sample, n_road * history_window)
    y_test = test_data[history_window + prediction_window - 1 : ] # (n_sample, n_road)

    y_pred_transposed = np.zeros(np.transpose(y_test).shape) # (n_sample, n_road)
    print('Fitting LinearSVR model...')
    start_time = time.time()
    for i in range(n_road):
        model = LinearSVR(random_state=0)
        model.fit(X_train, y_train[:, i])
        y_pred_transposed[i] = model.predict(X_test)
        if i%200 == 0:
            y_pred = np.transpose(y_pred_transposed)
            with open('LinearSVR_y_pred_window%s.pkl'%prediction_window, 'wb') as f:
                pkl.dump(y_pred, f)
            mae = MAE(y_pred[:, :i+1], y_test[:, :i+1], main_roads=False)
            mape = MAPE(y_pred[:, :i+1], y_test[:, :i+1], main_roads=False)
            rmse = RMSE(y_pred[:, :i+1], y_test[:, :i+1], main_roads=False)
            time_spent = time.time() - start_time
            line = 'Road: %d, MAE: %.3f, MAPE: %.3f, RMSE: %.3f, Time spent: %.2f\n'%(i, mae, mape, rmse, time_spent)
            print(line)
            with open('log/LinearSVR.log', 'a') as f:
                f.write(line)
    
    if prediction_window==1:
        result_analysis(y_pred, y_test)
        
    line = 'Road: %d, MAE: %.3f, MAPE: %.3f, RMSE: %.3f, Time spent: %.2f\n'%(i, mae, mape, rmse, time_spent)
    print(line)
    with open('log/LinearSVR.log', 'a') as f:
        f.write(line)

    
y_pred = np.transpose(y_pred_transposed)
print(y_pred.shape, y_test.shape)
print(y_pred[:10, 1])
print(y_test[:10, 1])