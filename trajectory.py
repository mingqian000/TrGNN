# nohup python trajectory.py -d 20160325 >> log/trajectory0325.log &
import os
import time
import pandas as pd
import numpy as np
import argparse
from datetime import datetime as dt
from datetime import date, timedelta
import networkx as nx
from utils import time_difference
from road_graph import get_road_list, road_graph


def read_GPS_dataset(date_range=['20160325', '20160325'], in_path='data/ParsedTaxiData_%s.csv', test_mode=False):
    # date_range = [start_date, end_date]
    print('Reading GPS dataset')
    start_date, end_date = date_range
    start_date, end_date = dt.strptime(start_date, '%Y%m%d'), dt.strptime(end_date, '%Y%m%d')
    date_list = [(start_date + timedelta(i)).strftime('%Y%m%d') for i in range((end_date - start_date).days+1)]
#     column_names = ['vehicle_id', 'lon', 'lat', 'speed', 'direction', 'status', 'time', 
#                     'closest_road_id', 'matched_lon', 'matched_lat', 'matched_road_id', 'matched_road_name']
    if test_mode:
        df_list = [pd.read_csv(in_path%(date), nrows=100000).drop_duplicates() for date in date_list]
    else:
        df_list = [pd.read_csv(in_path%(date)).drop_duplicates() for date in date_list]
    df = pd.concat(df_list)
    return df


def shortest_distance(G, road_id1, road_id2):
    if road_id1 == road_id2:
        return 0
    shortest_path_length = nx.dijkstra_path_length(G, road_id1, road_id2)
    shortest_distance = shortest_path_length - G.node[road_id1]['length']/2 - G.node[road_id2]['length']/2
    return shortest_distance


def extract_vehicle_readings(df, vehicle_id):
    # Extract GPS readings for this vehicle
    vehicle_df = df[df['vehicle_id']==vehicle_id][['vehicle_id', 'time', 'matched_road_id']].rename(columns={'matched_road_id':'road_id'})
    vehicle_df = vehicle_df.merge(road_list, on='road_id', how='inner')
    vehicle_df = vehicle_df.sort_values(by='time') # compulsory step: sort by time
    vehicle_df = vehicle_df.reset_index().drop('index', axis=1)
    return vehicle_df


def extract_trajectory(vehicle_df, G, time_gap=10, stay_duration=10, speed_limit=120, verb=True):
    
    # Trajectory extraction
    trajectory_df = pd.DataFrame(columns=['vehicle_id', 'trajectory_id', 'time', 'road_id', 'scenario'])
    for i, row in vehicle_df.iterrows():
        if i == 0: # Scenario 0.1: for the first reading, just record
            row['scenario'] = 0.1
            stay_start_time = row['time'] # to record the start of stay at the same road segment
            trajectory_id = 0
        else: 
            if row['time'] == previous_row['time']: # Scenario 0.2: same timing for multiple records, skip the following records
                if verb: print('Point %s S0.2. same timing for multiple records, skip the following records'%(i))
                row['scenario'] = 0.2
                continue
            if previous_row['road_id'] != row['road_id']:
                stay_start_time = row['time'] # update stay start time
            # Scenarios to start a new trajectory
            if time_difference(row['time'], previous_row['time']) > 60 * time_gap: # Scenario 1.1. time gap is too long
                if verb: print('Point %s S1.1. time gap is too long'%(i))
                row['scenario'] = 1.1
                trajectory_id += 1
            elif time_difference(row['time'], stay_start_time) > 60 * stay_duration: # Scenario 1.2. stay at the same road segment for too long
                if verb: print('Point %s S1.2. stay at the same road segment for too long'%(i))
                row['scenario'] = 1.2
                drop_indices = trajectory_df[trajectory_df['time'].apply(
                    lambda t: (time_difference(t, stay_start_time) > 0) & (time_difference(row['time'], t) > 0))].index
                trajectory_df = trajectory_df.drop(drop_indices) # drop intermediate points
                trajectory_id += 1 # keep the two end points, but in different trajectories
            elif not nx.has_path(G, previous_row['road_id'], row['road_id']): # Scenario 1.3. cannot find path between two points
                if verb: print('Point %s S1.3. cannot find path between two points'%(i))
                row['scenario'] = 1.3
                trajectory_id += 1
            elif shortest_distance(G, previous_row['road_id'], row['road_id']) / time_difference(row['time'], previous_row['time']) * 3600 > 120:  # Scenario 1.4. driver drives exceptionally fast
                if verb: print('Point %s S1.4. driver drives exceptionally fast'%(i))
                row['scenario'] = 1.4
                if verb: print(shortest_distance(G, previous_row['road_id'], row['road_id']))
                if verb: print(time_difference(row['time'], previous_row['time']))
                trajectory_id += 1
        row['trajectory_id'] = trajectory_id
        trajectory_df = trajectory_df.append(row, ignore_index=True)
        previous_row = row
    
    return trajectory_df


def clean_trajectory(trajectory_df):
    
    # Trajectory cleaning
    # Scenario 2.1. Remove trajectories with single point
    drop_indices = []   
    for i in range(len(trajectory_df)):
        if i == 0:
            if len(trajectory_df) ==1 or trajectory_df.loc[i, 'trajectory_id'] != trajectory_df.loc[i + 1, 'trajectory_id']:
                drop_indices.append(i)
        elif i == len(trajectory_df)-1:
            if trajectory_df.loc[i, 'trajectory_id'] != trajectory_df.loc[i - 1, 'trajectory_id']:
                drop_indices.append(i)
        elif trajectory_df.loc[i, 'trajectory_id'] != trajectory_df.loc[i - 1, 'trajectory_id'] and trajectory_df.loc[i, 'trajectory_id'] != trajectory_df.loc[i + 1, 'trajectory_id']:
                drop_indices.append(i)
    cleaned_trajectory_df = trajectory_df.drop(drop_indices).reset_index().drop('index', axis=1)
    
    return cleaned_trajectory_df


def recover_trajectory(cleaned_trajectory_df, G):
    
    # Trajectory recovery
    # Scenario 3.1. For points that are not adjacent,
    # apply Dijkstra's shortest path algorithm to recover intermediate points.
    # Timing follows D time in O-D.
    recovered_trajectory_df = pd.DataFrame(columns=['vehicle_id', 'trajectory_id', 'time', 'road_id', 'scenario'])
    for i, row in cleaned_trajectory_df.iterrows():
        if i == 0:
            point_index = 0 # index of a point in a trajectory
        else:
            point_index = point_index + 1 if previous_row['trajectory_id'] == row['trajectory_id'] else 0
        if point_index ==0:
            recovered_trajectory_df = recovered_trajectory_df.append(row, ignore_index=True)
        if point_index > 0:
            O = previous_row['road_id']
            D = row['road_id']
            if O == D:
                recovered_trajectory_df = recovered_trajectory_df.append(row, ignore_index=True)
            else: # O != D
                road_ids = nx.dijkstra_path(G, O, D)
                for road_id in road_ids[1:]: # add intermediate points and end points
                    row['road_id'] = road_id
                    row['scenario'] = 3.1
                    recovered_trajectory_df = recovered_trajectory_df.append(row, ignore_index=True)
        previous_row = row
    
    return recovered_trajectory_df


def get_trajectory(df, vehicle_id, G, time_gap=10, stay_duration=2, speed_limit=120, verb=False):
    vehicle_df = extract_vehicle_readings(df, vehicle_id)
    trajectory_df = extract_trajectory(vehicle_df, G, verb=False, time_gap=time_gap, stay_duration=stay_duration, speed_limit=speed_limit)
    cleaned_trajectory_df = clean_trajectory(trajectory_df)
    recovered_trajectory_df = recover_trajectory(cleaned_trajectory_df, G)
    if verb:
        print('Trajectories: %s'%(recovered_trajectory_df['trajectory_id'].nunique()))
        print('Road segments: %s'%(len(recovered_trajectory_df)))
    return recovered_trajectory_df


if __name__ == '__main__':
    
    
    # Arguments
    parser = argparse.ArgumentParser(description='trajectory')
    parser.add_argument('-d', '--date', help='%Y%m%d', required=True)
    parser.add_argument('-t', '--test_mode', default=0)
    args = parser.parse_args()
    date, test_mode = args.date, int(args.test_mode)


    # Parameter Settings
    # start_date = '20160325'
    # end_date = '20160325'
    start_date = date
    end_date = date
    time_gap = 10 # threshold for trajectory extraction
    stay_duration = 2 # threshold for trajectory extraction
    speed_limit = 120 # threshold for trajectory extraction
    GPS_path = 'data/ParsedTaxiData_%s.csv'
    road_list_path = 'data/road_list.csv'
    graph_path = 'data/road_graph.gml'
    trajectory_path = 'data/recovered_trajectory_df_%s_%s.csv'%(start_date, end_date)
    test_mode = test_mode
    
    
    # Read road network within selected region
    # Set road_df to None: use existing road_list and graph
    road_list = get_road_list(road_df=None, out_path=road_list_path, update=False)
    G = road_graph(road_df=None, out_path=graph_path, update=False)
    
    if test_mode:
        print('------- test mode -------')
    # Read GPS dataset
    df = read_GPS_dataset(date_range=[start_date, end_date], in_path=GPS_path, test_mode=test_mode)
    # GPS within selected region
    df = df.merge(road_list.rename(columns={'road_id':'matched_road_id'}), on='matched_road_id', how='inner')
    print('Num GPS points:', len(df))
    print('Num vehicles:', df['vehicle_id'].nunique())
    
    # Extract and save trajectories for all vehicles
    # Trajecotories are saved to file in append mode
    print('Extracting trajectories for all vehicles to %s'%(trajectory_path))
    vehicle_ids = df['vehicle_id'].unique()
    temp_trajectory_path = '%s_temp'%(trajectory_path)
    if os.path.exists(trajectory_path):
        os.system('cp %s %s'%(trajectory_path, temp_trajectory_path))
        recovered_trajectory_df = pd.read_csv(temp_trajectory_path)
        num_vehicles = recovered_trajectory_df['vehicle_id'].nunique()
        print('Num vehicles processed:', num_vehicles)
        if num_vehicles == 0:
            first_index = 0
        else:
            last_vehicle_id = recovered_trajectory_df.iloc[-1]['vehicle_id']
            last_index = np.where(vehicle_ids==last_vehicle_id)[0][0]
            print('Last vehicle index:', last_index)
            first_index = last_index + 1
    else:
        first_index = 0

    start_time = time.time()
    recovered_trajectory_df = pd.DataFrame(columns=['vehicle_id', 'trajectory_id', 'time', 'road_id', 'scenario'])
    for i in range(first_index, len(vehicle_ids)):
        vehicle_id = vehicle_ids[i]
        print('Vehicle #%s: %s. Time spent: %s s'%(i, vehicle_id, int(time.time() - start_time)))
        if i == 0:
            print('Saving header to file')
            recovered_trajectory_df.to_csv(temp_trajectory_path, index=False) # save header to file
        if i % 50 == 0:
            print('Appending result to file')
            recovered_trajectory_df.to_csv(temp_trajectory_path, mode='a', index=False, header=False)
            recovered_trajectory_df = pd.DataFrame(columns=['vehicle_id', 'trajectory_id', 'time', 'road_id', 'scenario'])
        recovered_trajectory_df = recovered_trajectory_df.append(get_trajectory(df, vehicle_id, G, time_gap=time_gap, stay_duration=stay_duration, speed_limit=speed_limit, verb=True), ignore_index=True)
    print('Appending result to file')
    recovered_trajectory_df.to_csv(temp_trajectory_path, mode='a', index=False, header=False)
    if os.path.exists(trajectory_path):
        os.system('rm %s'%(trajectory_path))
    os.system('mv %s %s'%(temp_trajectory_path, trajectory_path))
    print('Finished trajectory extraction. Total time spent: %.2f hour.'%((time.time() - start_time)/3600))
       
