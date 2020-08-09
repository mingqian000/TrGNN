# nohup python flow.py -d 20160325 -i 5 >> log/flow0325.log &
import os
import time
import pandas as pd
import numpy as np
import argparse
from datetime import datetime as dt
from datetime import date, timedelta
import networkx as nx
from utils import time_difference, round_time, df_to_csv
from road_graph import get_road_list, road_graph


def generate_time_intervals(start_date='20160325', end_date='20160325', interval=5):
    start_time = int(time.mktime(dt.strptime(start_date, '%Y%m%d').timetuple()))
    end_time = int(time.mktime((dt.strptime(end_date, '%Y%m%d') + timedelta(1)).timetuple()))
    interval = interval * 60 # convert minutes to seconds
    time_intervals = []
    for timestamp in range(start_time, end_time, interval):
        time_intervals.append(dt.fromtimestamp(timestamp / interval * interval).strftime('%d/%m/%Y %H:%M:%S'))
    return time_intervals


if __name__ == '__main__':
    
    # Arguments
    parser = argparse.ArgumentParser(description='flow')
    parser.add_argument('-d', '--date', help='%Y%m%d', required=True)
    parser.add_argument('-t', '--test_mode', default=0)
    parser.add_argument('-i', '--interval', help='in minutes', default=5)
    args = parser.parse_args()
    date, test_mode, interval = args.date, int(args.test_mode), int(args.interval)

    # Parameter Settings
    # start_date = '20160325'
    # end_date = '20160325'
    # interval = 5
    start_date = date
    end_date = date
    interval = interval # in minutes
    flow_path = 'data/flow_%s_%s.csv'%(start_date, end_date)
    checkpoint_path = 'data/flow_%s_%s.checkpoint'%(start_date, end_date)
    trajectory_path = 'data/recovered_trajectory_df_%s_%s.csv'%(start_date, end_date)
    road_list_path = 'data/road_list.csv'
    test_mode = test_mode
    
    
    # read trajectories
    print('Reading trajectories')
    if test_mode:
        print('------- test mode -------')
        recovered_trajectory_df = pd.read_csv(trajectory_path, nrows=200)
    else:
        recovered_trajectory_df = pd.read_csv(trajectory_path)

    # initialize flow_df
    print('Initializing flow_df')
    if os.path.exists(flow_path):
        print('Flow file exists')
        flow_df = pd.read_csv(flow_path, index_col=0)
        flow_df.columns = pd.Index(int(road_id) for road_id in flow_df.columns)
        print('Existing total flow:', flow_df.sum().sum())
        with open(checkpoint_path, 'r') as f:
            checkpoint = int(f.read()) + 1
    else:
        print('Creating new flow file')
        road_list = get_road_list(road_df=None, out_path=road_list_path, update=False)
        flow_df = pd.DataFrame(columns=list(road_list['road_id']))
        time_intervals = generate_time_intervals(start_date=start_date, end_date=end_date, interval=interval)
        for time_interval in time_intervals:
            flow_df.loc[time_interval] = 0
        checkpoint = 0

    # aggregate and save flow
    # flow is saved to file in overwrite mode
    print('Total number of points:', len(recovered_trajectory_df))
    start_time = time.time()
    i = -1 # dummy
    for i, row in recovered_trajectory_df.iterrows():
        if i < checkpoint:
            pass
        elif i == 0:
            flow_df.loc[round_time(row['time'], interval=interval), row['road_id']] += 1
        else:
            if i % 10000 == 0:
                print('Saving result at index %s. Time spent: %s s'%(i, int(time.time() - start_time)))
                df_to_csv(flow_df, flow_path, index=True)
                with open(checkpoint_path, 'w') as f:
                    f.write(str(i))
            if row['vehicle_id'] != previous_vehicle_id or row['trajectory_id'] != previous_trajectory_id: # new trajectory
                flow_df.loc[round_time(row['time'], interval=interval), row['road_id']] += 1
            elif row['road_id'] != previous_road_id: # appear in this road
                flow_df.loc[round_time(row['time'], interval=interval), row['road_id']] += 1
        previous_vehicle_id, previous_trajectory_id, previous_road_id = row['vehicle_id'], row['trajectory_id'], row['road_id']
    print('Saving result at index', i)
    df_to_csv(flow_df, flow_path, index=True)
    with open(checkpoint_path, 'w') as f:
        f.write(str(i))
    print('New total flow:', flow_df.sum().sum())
    print('Finished flow aggregation. Total time spent: %.2f hour.'%((time.time() - start_time)/3600))
