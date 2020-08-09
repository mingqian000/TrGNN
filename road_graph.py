import os
import pandas as pd
import networkx as nx
import pickle as pkl
from math import radians, degrees, sin, cos, asin, acos, sqrt
import numpy as np


# Parameter Settings
min_lat, max_lat, min_lon, max_lon = 1.31, 1.37, 103.7, 103.8
road_path = 'LTA_cleaned.txt'
road_list_path = 'data/road_list.csv'
graph_path = 'data/road_graph.gml'


def great_circle(lon1, lat1, lon2, lat2):
    # great circle distance in km
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    return 6371 * ( acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)) )


def read_road_dataset(boundary=None, road_path='data/LTA_cleaned.txt'):
    # boundary: None, or [min_lat, max_lat, min_lon, max_lon]. If None, return full road dataset.
    column_names = ['road_id', 'road_type','lane_id',
                    'zone_id','road_name', 'vertex_num', 
                    'start_lon', 'start_lat', 'end_lon', 'end_lat']
    road_df = pd.read_csv(road_path, header=None, names=column_names)
    if boundary is not None:
        min_lat, max_lat, min_lon, max_lon = boundary
        road_df = road_df[(road_df['start_lat']>=min_lat) & (road_df['start_lat']<=max_lat) & 
                 (road_df['end_lat']>=min_lat) & (road_df['end_lat']<=max_lat) & 
                 (road_df['start_lon']>=min_lon) & (road_df['start_lon']<=max_lon) & 
                 (road_df['end_lon']>=min_lon) & (road_df['end_lon']<=max_lon)]
    road_df['start'] = road_df.apply(lambda row: (row['start_lat'], row['start_lon']), axis=1)
    road_df['end'] = road_df.apply(lambda row: (row['end_lat'], row['end_lon']), axis=1)
    road_df['length'] = road_df.apply(lambda row: great_circle(row['start_lon'], row['start_lat'], row['end_lon'], row['end_lat']), axis=1)
    return road_df


def get_road_list(road_df=None, out_path='data/road_list.csv', update=False):
    # road_list: the mapping of road_id. for adjacency matrix
    if not update and os.path.exists(out_path):
        print('Road list exists')
        road_list = pd.read_csv(out_path)
    else:
        print('Generating new road list from road df')
        road_list = road_df[['road_id']].reset_index().drop('index', axis=1)
        road_list.to_csv(out_path, index=False)
    return road_list


def road_graph(road_df=None, out_path='data/road_graph.gml', update=False):
    if not update and os.path.exists(out_path):
        print('Graph exists')
        G = nx.read_gml(out_path)
        G = nx.relabel_nodes(G, int)
    else:
        print('Generating new graph from road df')
        G = nx.DiGraph()

        node_list = list(road_df['road_id'])
        G.add_nodes_from(node_list)
        lengths = dict(zip(road_df['road_id'], road_df['length']))
        nx.set_node_attributes(G, lengths, 'length')

        road_df = road_df.copy()
        road_df = road_df[['road_id', 'start', 'end', 'length']]
        road_df['key'] = 0
        adj_df = road_df.merge(road_df, on='key', how='inner')
        adj_df = adj_df[((adj_df['end_x'] == adj_df['start_y']) & (adj_df['start_x'] != adj_df['end_y'])) # x -> y
                        | (adj_df['road_id_x'] == adj_df['road_id_y'])]  # x self
        adj_df['distance'] = adj_df.apply(lambda row: 
                                          (row['length_x'] + row['length_y']) / 2 if row['road_id_x'] != row['road_id_y'] 
                                          else 0, axis=1)
        adj_df['edge'] = adj_df.apply(lambda row: (row['road_id_x'], row['road_id_y'], {'weight': row['distance']}), axis=1)
        edge_list = list(adj_df['edge'])
        G.add_edges_from(edge_list)
        
        nx.write_gml(G, out_path)
        
    return G


def extract_road_adj(G=None, road_list=None):
    
    file_path = 'data/road_adj.pkl'
    if os.path.exists(file_path):
        print('Road adj exists')
        with open(file_path, 'rb') as f:
            road_adj = pkl.load(f)
    else:
        print('Extracting road adj from graph')
        
        from road_graph import road_graph
        G = road_graph(road_df=None, out_path='data/road_graph.gml', update=False)
        road_adj = np.zeros((len(G.nodes), len(G.nodes)), dtype=np.float32)
        
        from road_graph import get_road_list
        road_list = get_road_list()
        def road_index(road_id):
            return road_list[road_list['road_id']==road_id].index[0]

        # masked exponential kernel. Set lambda = 1.
        lambda_ = 1
        # lambda: for future consideration
        # total_weight = 0
        # total_count = 0

        # for O in list(G.nodes):
        #     for D in list(G.successors(O)):
        #         total_weight += G.edges[O, D]['weight']
        #         total_count += 1

        # lambda_ = total_weight / total_count
        for O in list(G.nodes):
            for D in list(G.successors(O)):
                road_adj[road_index(O), road_index(D)] = lambda_ * np.exp(- lambda_ * G.edges[O, D]['weight'])

        with open(file_path, 'wb') as f:
            pkl.dump(road_adj, f)
    
    return road_adj


if __name__ == '__main__':
    # road_df = read_road_dataset()
    road_df = read_road_dataset(boundary=[min_lat, max_lat, min_lon, max_lon], road_path=road_path)
    road_list = get_road_list(road_df, out_path=road_list_path, update=False)
    G = road_graph(road_df, out_path=graph_path, update=False)