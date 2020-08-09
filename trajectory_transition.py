# nohup python trajectory_transition.py -d1 20160401 -d2 20160401 >> log/transition0401.log &
import pandas as pd
import time
from datetime import date, timedelta
from datetime import datetime as dt
import os
from utils import *
import numpy as np
from math import radians, degrees, sin, cos, asin, acos, sqrt
import pickle as pkl
import argparse
from road_graph import get_road_list


def extract_trajectory_transition(start_date, end_date, interval=15):
    # start_date, end_date = '20160401', '20160421'
    # interval is in minutes, and should divide 60.
    
    total_file_path = 'data/trajectory_transition_%s_%s.pkl'%(start_date, end_date)
    if os.path.exists(total_file_path):
        with open(total_file_path, 'rb') as f:
            print('Total file exists')
            total_trajectory_transition = pkl.load(f)
    else:
        print('Reading road list')
        road_list = get_road_list()
        road_list = road_list.reset_index().rename(columns={'index':'road_index'})
        
        print('Creating empty total_trajectory_transition')
        total_trajectory_transition = np.zeros((60//interval*24, len(road_list), len(road_list)), dtype=np.int8)
        
        date_list = date_range(start_date, end_date)
        
        for current_date in date_list:
            
            print('Date %s'%(current_date))
            file_path = 'data/trajectory_transition_%s_%s.pkl'%(current_date, current_date)
            if os.path.exists(file_path):
                print('File exists')
                with open(file_path, 'rb') as f:
                    trajectory_transition = pkl.load(f)
                
            else:
                start_time = time.time()
                
                print('Reading recovered_trajectory_df')
                trajectory_path = 'data/recovered_trajectory_df_%s_%s.csv'%(current_date, current_date)
                recovered_trajectory_df = pd.read_csv(trajectory_path)
                print('Time spent till now: %.2f seconds'%(time.time() - start_time))

                print('Extracting road index')
                recovered_trajectory_df = recovered_trajectory_df.merge(road_list, on='road_id', how='left')
                print('Time spent till now: %.2f seconds'%(time.time() - start_time))

                print('Extracting time_index')
                def extract_time_index(time):
                    hour = int(time[-8:-6])
                    minute = int(time[-5:-3])
                    time_index = hour * 4 + minute // 15
                    return time_index
                recovered_trajectory_df['time_index'] = recovered_trajectory_df['time'].apply(extract_time_index)
                print('Time spent till now: %.2f seconds'%(time.time() - start_time))

                print('Creating empty trajectory_transition')
                trajectory_transition = np.zeros((60//interval*24, len(road_list), len(road_list)), dtype=np.int16)
                print('Time spent till now: %.2f seconds'%(time.time() - start_time))

                print('Calculating trajectory_transition')
                for i, row in recovered_trajectory_df.iterrows():
                    if i != 0:
                        if previous_row['vehicle_id'] == row['vehicle_id'] and \
                        previous_row['trajectory_id'] == row['trajectory_id'] and \
                        previous_row['road_id'] != row['road_id']:
                            trajectory_transition[previous_row['time_index'], previous_row['road_index'], row['road_index']] += 1
                    if i % 100000 == 0:
                        print(i, 'at %.2f seconds'%(time.time() - start_time))
                    previous_row = row
                print('Time spent till now: %.2f seconds'%(time.time() - start_time))

                print('Saving trajectory_transition')
                with open(file_path, 'wb') as f:
                    pkl.dump(trajectory_transition, f)
                print('Time spent till now: %.2f seconds'%(time.time() - start_time))
            
            print('Merging trajectory transition')
            total_trajectory_transition = total_trajectory_transition + trajectory_transition
            print('Total count: %d'%(total_trajectory_transition.sum()))
        
        print('Saving total_trajectory_transition')
        with open(total_file_path, 'wb') as f:
            pkl.dump(total_trajectory_transition, f)
        
    return total_trajectory_transition


if __name__ == '__main__':
    
    # Arguments
    parser = argparse.ArgumentParser(description='trajectory_transition')
    parser.add_argument('-d1', '--start_date', help='%Y%m%d', required=True)
    parser.add_argument('-d2', '--end_date', help='%Y%m%d', required=True)
    args = parser.parse_args()
    start_date, end_date = args.start_date, args.end_date
    
    # start_date, end_date = '20160401', '20160421'
    trajectory_transition = extract_trajectory_transition(start_date, end_date)