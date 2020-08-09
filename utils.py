import os
import folium
import numpy as np
import geopy.distance
import time
from datetime import date, timedelta
from datetime import datetime as dt
import scipy.sparse as sp
import torch


def to_sparse_tensor(dense_matrix):
    coo = sp.coo_matrix(dense_matrix)

    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    values = torch.FloatTensor(coo.data)
    shape = coo.shape

    sparse_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
    
    return sparse_tensor


def date_range(date1, date2):
    # date1, date2 = '20160401', '20160428'
    datetime1 = dt.strptime(date1, '%Y%m%d')
    datetime2 = dt.strptime(date2, '%Y%m%d')
    days = (datetime2 - datetime1).days + 1
    date_list = [(datetime1 + timedelta(day)).strftime('%Y%m%d') for day in range(days)]
    return date_list


def time_difference(time1, time2):
    # format: '25/03/2016 00:00:04'
    # time_difference = time1 - time2
    return (dt.strptime(time1, '%d/%m/%Y %H:%M:%S') - dt.strptime(time2, '%d/%m/%Y %H:%M:%S')).total_seconds()


def df_to_csv(df, file_path, index=False):
    print('Saving to file at %s'%(file_path))
    if os.path.exists(file_path):
        temp_file_path = '%s_temp'%(file_path)
        df.to_csv(temp_file_path, index=index)
        os.system('rm %s'%(file_path))
        os.system('mv %s %s'%(temp_file_path, file_path))
    else:
        df.to_csv(file_path, index=index)
    print('Saved.')

    
def print_log(line, log_path):
    with open(log_path, 'a') as f:
        f.write(str(line)+'\n')
    
    
def round_time(t, interval=5):
    # t = '25/03/2016 12:26:45'
    # output: '25/03/2016 12:25:00'
    # interval: in minutes
    interval = interval * 60 # convert minutes to seconds
    datetime = dt.strptime(t, '%d/%m/%Y %H:%M:%S')
    new_datetime = dt.fromtimestamp(int(time.mktime(datetime.timetuple())) // interval * interval)
    return new_datetime.strftime('%d/%m/%Y %H:%M:%S')


class Point():
    def __init__(self, latitude=None, longitude=None, time=None):
        self.lat = latitude
        self.lon = longitude
        self.time = time
        
    def __str__(self):
        return '%s, %s'%(self.lat, self.lon, self.time)
    

def get_bearing(p1, p2):    
    # Returns compass bearing from p1 to p2
    
    long_diff = np.radians(p2.lon - p1.lon)
    
    lat1 = np.radians(p1.lat)
    lat2 = np.radians(p2.lat)
    
    x = np.sin(long_diff) * np.cos(lat2)
    y = (np.cos(lat1) * np.sin(lat2) 
        - (np.sin(lat1) * np.cos(lat2) 
        * np.cos(long_diff)))
    bearing = np.degrees(np.arctan2(x, y))
    
    # adjusting for compass bearing
    if bearing < 0:
        return bearing + 360
    return bearing


def get_arrow(locations, color='#3388ff', size=6, n_arrows=3, road_id=''):
    
    # get arrow for a road segment to indicate the direction
    # locations e.g. [(1.3096, 103.9081), (1.3103, 103.9079)]
    
    # creating point from our Point named tuple
    p1 = Point(locations[0][0], locations[0][1])
    p2 = Point(locations[1][0], locations[1][1])
    
    # getting the rotation needed for our marker.  
    # Subtracting 90 to account for the marker's orientation
    # of due East(get_bearing returns North)
    rotation = get_bearing(p1, p2) - 90
    
    # get an evenly space list of lats and lons for our arrows
    # note that I'm discarding the first and last for aesthetics
    # as I'm using markers to denote the start and end
#     arrow_lats = np.linspace(p1.lat, p2.lat, n_arrows + 2)[1:n_arrows+1]
#     arrow_lons = np.linspace(p1.lon, p2.lon, n_arrows + 2)[1:n_arrows+1]
    arrow_lat = p2.lat
    arrow_lon = p2.lon
    
    arrows = []
    
    #creating each "arrow" and appending them to our arrows list
#     for points in zip(arrow_lats, arrow_lons):
    arrow = folium.RegularPolygonMarker(location=(arrow_lat, arrow_lon), 
                  weight=1, color=color, fill_color=color, number_of_sides=3, 
                  radius=size, rotation=rotation, popup='%s, %s, %s'%(arrow_lat, arrow_lon, road_id))
    return arrow


def display_road(start_lon, start_lat, end_lon, end_lat, color='#3388ff', weight=3, m=None, tiles='OpenStreetMap', road_id='', arrow=True):
    center_lat = (start_lat + end_lat) / 2
    center_lon = (start_lon + end_lon) / 2
    
    # plot map
    if m is None:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=20, tiles=tiles)
    # add road line
    folium.PolyLine([(start_lat, start_lon), (end_lat, end_lon)], color=color, weight=weight).add_to(m)
    # add direction arrow to road
    if arrow:
        get_arrow([(start_lat, start_lon), (end_lat, end_lon)], color=color, road_id=road_id).add_to(m)
    
    return m


def display_roads(road_ids, road_df, color='#3388ff', m=None, tiles='OpenStreetMap', arrow=True):
    # E.g. display_roads([103067603, 103106763], road_df)
    
    # extract roads
    df = road_df.set_index('road_id')
    roads = df.loc[road_ids]
    
    # locate the center of the map
    min_lat = min(min(roads['start_lat']), min(roads['end_lat']))
    max_lat = max(max(roads['start_lat']), max(roads['end_lat']))
    min_lon = min(min(roads['start_lon']), min(roads['end_lon']))
    max_lon = max(max(roads['start_lon']), max(roads['end_lon']))
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # plot map
    if m is None:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=tiles)
    
    # add road
    for _, road in roads.iterrows():
        display_road(road['start_lon'], road['start_lat'], road['end_lon'], road['end_lat'], color=color, m=m, road_id=road.name, arrow=arrow)
    
    return m


def display_roads_heatmap(road_ids, road_df, colors=None, weights=None, m=None, tiles='OpenStreetMap', arrow=True):
    # E.g. display_roads([103067603, 103106763], road_df, ['#ff0000', '#ffa500'], [3, 3])
    
    # extract roads
    df = road_df.set_index('road_id')
    roads = df.loc[road_ids]
    
    # locate the center of the map
    min_lat = min(min(roads['start_lat']), min(roads['end_lat']))
    max_lat = max(max(roads['start_lat']), max(roads['end_lat']))
    min_lon = min(min(roads['start_lon']), min(roads['end_lon']))
    max_lon = max(max(roads['start_lon']), max(roads['end_lon']))
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # plot map
    if m is None:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=tiles)
    
    # add road
    if colors is None: colors = ['#3388ff'] * len(road_ids)
    if weights is None: weights = [3] * len(road_ids)
    for (_, road), color, weight in zip(roads.iterrows(), colors, weights):
        display_road(road['start_lon'], road['start_lat'], road['end_lon'], road['end_lat'], color=color, weight=weight, m=m, road_id=road.name, arrow=arrow)
    
    return m


def display_road_network(min_lat, max_lat, min_lon, max_lon, road_df, color='#3388ff', m=None, tiles='OpenStreetMap', arrow=True):
    # E.g. display_road_network(1.310, 1.315, 103.90, 103.91, road_df, color='red', tiles='cartodbpositron')
    
    # extract roads
    df = road_df[(road_df['start_lat']>=min_lat) & (road_df['start_lat']<=max_lat) & 
                 (road_df['end_lat']>=min_lat) & (road_df['end_lat']<=max_lat) & 
                 (road_df['start_lon']>=min_lon) & (road_df['start_lon']<=max_lon) & 
                 (road_df['end_lon']>=min_lon) & (road_df['end_lon']<=max_lon)]
    road_ids = list(df['road_id'])
    df = df.set_index('road_id')
    roads = df.loc[road_ids]
    
    # locate the center of the map
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # plot map
    if m is None:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=tiles)
    
    # add road
    for _, road in roads.iterrows():
        display_road(road['start_lon'], road['start_lat'], road['end_lon'], road['end_lat'], color=color, m=m, arrow=arrow)
    
    return m


def display_trajectory(points, m=None, tiles='OpenStreetMap'):
    
    min_lat, max_lat, min_lon, max_lon = (91, -90, 181, -181)
    markers = []
    for point in points:
        min_lat = min(min_lat, point.lat)
        max_lat = max(max_lat, point.lat)
        min_lon = min(min_lon, point.lon)
        max_lon = max(max_lon, point.lon)
        markers.append(folium.Marker([point.lat, point.lon], popup='%s'%(point.time)))
    
    # locate the center of the map
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # plot map
    if m is None:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=tiles)
    
    # add point
    for marker in markers:
        marker.add_to(m)
    
    return m


def display_vehicle_raw_trajectory(vehicle_id, start_time, end_time, df, m=None, tiles='OpenStreetMap'):
    # extract points of the vehicle
    vehicle_df = df[(df['vehicle_id']==vehicle_id) &
                    (df['time'].apply(lambda t: time_difference(t, start_time) >= 0)) &
                    (df['time'].apply(lambda t: time_difference(end_time, t) >= 0))
                   ].drop_duplicates()
    points = []
    for _, row in vehicle_df.iterrows():
        points.append(Point(row['lat'], row['lon'], row['time']))
    
    # display trajectory
    m = display_trajectory(points, m=m, tiles=tiles)
    
    return m


def display_vehicle_matched_trajectory(vehicle_id, start_time, end_time, df, m=None, tiles='OpenStreetMap'):
    # extract points of the vehicle
    vehicle_df = df[(df['vehicle_id']==vehicle_id) &
                    (df['time'].apply(lambda t: time_difference(t, start_time) >= 0)) &
                    (df['time'].apply(lambda t: time_difference(end_time, t) >= 0))
                   ].drop_duplicates()
    points = []
    for _, row in vehicle_df.iterrows():
        points.append(Point(row['matched_lat'], row['matched_lon'], row['time']))
    
    # display trajectory
    m = display_trajectory(points, m=m, tiles=tiles)
    
    return m


def geodistance(coords_1, coords_2):
    return geopy.distance.great_circle(coords_1, coords_2).m