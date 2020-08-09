# Evaluation Metrics
# E.g. mae = MAE(y_pred, y_test, main_roads=False)

import pandas as pd
import numpy as np
from utils import print_log


def read_main_road_index(file_path='data/road_list_main.csv'):
    main_roads = pd.read_csv(file_path)
    main_roads = main_roads.set_index(main_roads.columns[0])
    main_road_index = np.array(main_roads.index)
    return main_road_index


def MAE(y_pred, y_test, main_roads=False):
    if main_roads:
        main_road_index = read_main_road_index()
        y_pred_, y_test_ = y_pred[:, main_road_index], y_test[:, main_road_index] 
    else:
        y_pred_, y_test_ = y_pred, y_test
    return np.average(np.abs(y_pred_ - y_test_))


def MAPE(y_pred, y_test, main_roads=False):
    if main_roads:
        main_road_index = read_main_road_index()
        y_pred_, y_test_ = y_pred[:, main_road_index], y_test[:, main_road_index] 
    else:
        y_pred_, y_test_ = y_pred, y_test
    mask = (y_test_ != 0).astype(np.int)
    return np.sum(np.nan_to_num(np.abs((y_pred_ - y_test_).astype(np.float32) / y_test_)) * mask) / np.sum(mask)


def RMSE(y_pred, y_test, main_roads=False):
    if main_roads:
        main_road_index = read_main_road_index()
        y_pred_, y_test_ = y_pred[:, main_road_index], y_test[:, main_road_index] 
    else:
        y_pred_, y_test_ = y_pred, y_test
    return np.sqrt(np.average(np.square(y_pred_ - y_test_)))


# Our models. dataset version 20160401-20160428
def result_analysis(Y_pred, Y_true, log_path='nohup.out'):
    # abnormal day: 22nd Apr
    _Y_pred = Y_pred[:92]
    _Y_true = Y_true[:92]
    mae = MAE(_Y_pred, _Y_true, main_roads=False)
    mape = MAPE(_Y_pred, _Y_true, main_roads=False)
    rmse = RMSE(_Y_pred, _Y_true, main_roads=False)
    print_log('>> result analysis - abnormal day 22nd Apr. MAE: %.3f, MAPE: %.3f, RMSE: %.3f'%(mae, mape, rmse), log_path)
    
    # abnormal hours: 8am-9am
    _Y_pred = np.zeros((4*7, 2404)) # (n_sample, n_road)
    _Y_true = np.zeros((4*7, 2404))
    for i in range(7):
        _Y_pred[i*4:(i+1)*4] = Y_pred[((8-1)*4+92*i):((9-1)*4+92*i)]
        _Y_true[i*4:(i+1)*4] = Y_true[((8-1)*4+92*i):((9-1)*4+92*i)]
    mae = MAE(_Y_pred, _Y_true, main_roads=False)
    mape = MAPE(_Y_pred, _Y_true, main_roads=False)
    rmse = RMSE(_Y_pred, _Y_true, main_roads=False)
    print_log('>> result analysis - abnormal hours 8am-9am. MAE: %.3f, MAPE: %.3f, RMSE: %.3f'%(mae, mape, rmse), log_path)
    
    # abnormal hours: 11pm-12am
    _Y_pred = np.zeros((4*7, 2404)) # (n_sample, n_road)
    _Y_true = np.zeros((4*7, 2404))
    for i in range(7):
        _Y_pred[i*4:(i+1)*4] = Y_pred[((23-1)*4+92*i):((24-1)*4+92*i)]
        _Y_true[i*4:(i+1)*4] = Y_true[((23-1)*4+92*i):((24-1)*4+92*i)]
    mae = MAE(_Y_pred, _Y_true, main_roads=False)
    mape = MAPE(_Y_pred, _Y_true, main_roads=False)
    rmse = RMSE(_Y_pred, _Y_true, main_roads=False)
    print_log('>> result analysis - abnormal hours 11pm-12am. MAE: %.3f, MAPE: %.3f, RMSE: %.3f'%(mae, mape, rmse), log_path)
    
    return


# dataset version 20160314-20160508
# updated 20200503. corrected weekday indices
# added peak hour, non-peak hour
def result_analysis2(Y_pred, Y_true, model_type='ours', log_path='nohup.out'): 
    # for HA, static, MA, model_type = 'baseline'. (96*14, 2404) full.
    # for our models, model_type = 'ours'. (92*14, 2404) missing the first hour for each day.
    # for VAR and RF, model_type = 'VAR'. (96*14-4, 2404) missing the first hour for the first day.
    # 14 days in test period
    ToD = 96 if model_type in ['baseline', 'VAR']  else 92 # number of timestamps in day for test
    interval_offset = 4 if model_type in ['VAR', 'ours'] else 0 # number of shifted timestamp indices of day
    
    # overall
    mae = MAE(Y_pred, Y_true, main_roads=False)
    mape = MAPE(Y_pred, Y_true, main_roads=False)
    rmse = RMSE(Y_pred, Y_true, main_roads=False)
    print_log('>> Overall. MAE: %.3f, MAPE: %.3f, RMSE: %.3f'%(mae, mape, rmse), log_path)

    # hourly.
    # weekdays average. exclude weekends (day 5,6,12,13) and PH (day 7).
    weekdays = np.array([0,1,2,3,4,8,9,10,11])
    weekday_indices = (np.repeat(weekdays.reshape(-1, 1), 4, axis=1)*4 + np.arange(4).reshape(1, -1)).reshape(-1)
    hours = np.arange(1, 24)
    for hour in hours:
        
        _Y_pred = np.zeros((4*14, 2404)) # (n_sample, n_road)
        _Y_true = np.zeros((4*14, 2404))
        for i in range(14):
            _Y_pred[i*4:(i+1)*4] = Y_pred[(hour*4-interval_offset+ToD*i):((hour+1)*4-interval_offset+ToD*i)]
            _Y_true[i*4:(i+1)*4] = Y_true[(hour*4-interval_offset+ToD*i):((hour+1)*4-interval_offset+ToD*i)]
        _Y_pred = _Y_pred[weekday_indices]
        _Y_true = _Y_true[weekday_indices]
        mae = MAE(_Y_pred, _Y_true, main_roads=False)
        mape = MAPE(_Y_pred, _Y_true, main_roads=False)
        rmse = RMSE(_Y_pred, _Y_true, main_roads=False)
        print_log('>> Peak hours %d-%d. MAE: %.3f, MAPE: %.3f, RMSE: %.3f'%(hour, hour+1, mae, mape, rmse), log_path)
    
    # peak hours: 7-9am & 2-4pm.
    # weekdays average. exclude weekends (day 5,6,12,13) and PH (day 7).
    weekdays = np.array([0,1,2,3,4,8,9,10,11])
    weekday_indices = (np.repeat(weekdays.reshape(-1, 1), 8, axis=1)*8 + np.arange(8).reshape(1, -1)).reshape(-1)

    _Y_pred = np.zeros((8*14, 2404)) # (n_sample, n_road)
    _Y_true = np.zeros((8*14, 2404))
    for i in range(14):
        _Y_pred[i*8:(i+1)*8] = Y_pred[(7*4-interval_offset+ToD*i):(9*4-interval_offset+ToD*i)]
        _Y_true[i*8:(i+1)*8] = Y_true[(7*4-interval_offset+ToD*i):(9*4-interval_offset+ToD*i)]
    _Y_pred = _Y_pred[weekday_indices]
    _Y_true = _Y_true[weekday_indices]
    mae = MAE(_Y_pred, _Y_true, main_roads=False)
    mape = MAPE(_Y_pred, _Y_true, main_roads=False)
    rmse = RMSE(_Y_pred, _Y_true, main_roads=False)
    print_log('>> Peak hours %d-%d. MAE: %.3f, MAPE: %.3f, RMSE: %.3f'%(7, 9, mae, mape, rmse), log_path)
    
    _Y_pred = np.zeros((8*14, 2404)) # (n_sample, n_road)
    _Y_true = np.zeros((8*14, 2404))
    for i in range(14):
        _Y_pred[i*8:(i+1)*8] = Y_pred[(14*4-interval_offset+ToD*i):(16*4-interval_offset+ToD*i)]
        _Y_true[i*8:(i+1)*8] = Y_true[(14*4-interval_offset+ToD*i):(16*4-interval_offset+ToD*i)]
    _Y_pred = _Y_pred[weekday_indices]
    _Y_true = _Y_true[weekday_indices]
    mae = MAE(_Y_pred, _Y_true, main_roads=False)
    mape = MAPE(_Y_pred, _Y_true, main_roads=False)
    rmse = RMSE(_Y_pred, _Y_true, main_roads=False)
    print_log('>> Peak hours %d-%d. MAE: %.3f, MAPE: %.3f, RMSE: %.3f'%(14, 16, mae, mape, rmse), log_path)
    
    # MRT breakdown
    # duration: 25th Apr, 20:15-21:30
    # affected roads: west Singapore (longitude < 103.85)
    _Y_pred = Y_pred[(ToD-interval_offset-15):(ToD-interval_offset-10)]
    _Y_true = Y_true[(ToD-interval_offset-15):(ToD-interval_offset-10)]
    mae = MAE(_Y_pred, _Y_true, main_roads=True) # use affected roads only. 'data/road_list_main.csv'
    mape = MAPE(_Y_pred, _Y_true, main_roads=True) # use affected roads only. 'data/road_list_main.csv'
    rmse = RMSE(_Y_pred, _Y_true, main_roads=True) # use affected roads only. 'data/road_list_main.csv'
    print_log('>> 25th Apr MRT breakdown. MAE: %.3f, MAPE: %.3f, RMSE: %.3f'%(mae, mape, rmse), log_path)
    
    return