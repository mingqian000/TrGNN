import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import to_sparse_tensor


def normalize_adj(adj, mode='random walk'):
    # mode: 'random walk', 'aggregation'
    if mode == 'random walk': # for T. avg weight for sending node
        deg = np.sum(adj, axis=1).astype(np.float32)
        inv_deg = np.reciprocal(deg, out=np.zeros_like(deg), where=deg!=0)
        D_inv = np.diag(inv_deg)
        normalized_adj = np.matmul(D_inv, adj)
    if mode == 'aggregation': # for W. avg weight for receiving node
        deg = np.sum(adj, axis=0).astype(np.float32)
        inv_deg = np.reciprocal(deg, out=np.zeros_like(deg), where=deg!=0)
        D_inv = np.diag(inv_deg)
        normalized_adj = np.matmul(adj, D_inv)
    return normalized_adj


def graph_propagation_sparse(x, A, hop=10, dual=False):
    # sparse version
    # x: graph signal vector. tensor. (n_road)
    # A: adjacency matrix. tranposed. sparse_tensor. (n_road, n_road)
    # hop: # propagation steps
    # output: propagation result. tensor. (n_road, hop+1)
    
    y = x.unsqueeze(1)
    X = y
    if dual: # dual random walk
        for i in range(hop):
            y_down = A.mm(X) # downstream
            y_up = A.transpose(0, 1).mm(X) # upstream
            X = torch.cat([y, y_down, y_up], dim=1)
    else: # downstream random walk only
        for i in range(hop):
            y = A.mm(y)
            X = torch.cat([X, y], dim=1)
    return X


from torch.nn import Parameter
from torch.nn import init
import math

class ChannelFullyConnected(nn.Module):
        
    __constants__ = ['bias', 'in_features', 'channels']

    def __init__(self, in_features, channels, bias=True):
        super(ChannelFullyConnected, self).__init__()
        self.in_features = in_features
        self.channels = channels
        self.weight = Parameter(torch.Tensor(channels, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.mul(input, self.weight).sum(dim=1) + self.bias

    def extra_repr(self):
        return 'in_features={}, channels={}, bias={}'.format(
            self.in_features, self.channels, self.bias is not None
        )
    

class ChannelAttention(nn.Module):
        
    __constants__ = ['bias', 'in_features', 'channels']

    def __init__(self, in_features, out_features, channels, bias=True):
        super(ChannelAttention, self).__init__()
        self.in_features = in_features
        self.channels = channels
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(channels, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(channels, out_features)) # out_features=1+demand_hop
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input): # input: (history_window, channels=n_road, in_features=1+status_hop)
        return F.mul(input, self.weight).sum(dim=2) + self.bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, channels={}, bias={}'.format(
            self.in_features, self.out_features, self.channels, self.bias is not None
        )
    
    
class Model_TrGNN(nn.Module):
    # TrGNN.
    
    def __init__(self, input_size=1, output_size=1, demand_hop=75, status_hop=3):
        super(Model_TrGNN, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.demand_hop = demand_hop
        self.status_hop = status_hop
        
        # attention
        self.attention_layer = ChannelAttention(2**(status_hop+1)-1, demand_hop+1, channels=2404, bias=True) # channels=n_road
                
        # linear output
        self.output_layer = ChannelFullyConnected(in_features=4+24+1, channels=2404) # channels=n_road
        

    def forward(self, X, T, W, h_init, W_norm, ToD, DoW):
        # X: graph signal. normalized. tensor: (history_window, n_road)
        # T: trajectory transition. normalized. tuple of history_window sparse_tensors: (n_road, n_road)
        # W: weighted road adjacency matrix. # sparse_tensor: (n_road, n_road)
        # h_init: for GRU. (gru_num_layers, n_road, hidden_size)
        # ToD: road-wise one-hot encoding of hour of day. (n_road, 24)
        # DoW: road-wise indicator. 1 for weekdays, 0 for weekends/PHs. (n_road, 1)
        
        # graph propagation
        H = torch.cat([graph_propagation_sparse(x, A.transpose(0, 1), hop=self.demand_hop).unsqueeze(0) for x, A in zip(torch.unbind(X, dim=0), T)], dim=0)

        # attention
        S = torch.cat([graph_propagation_sparse(x, W_norm, hop=self.status_hop, dual=True).unsqueeze(0) for x in torch.unbind(X, dim=0)], dim=0)
        att = self.attention_layer(S.unsqueeze(3)) # specify weights and bias for each road segment
        att = F.softmax(att, dim=2) # attention weights across hops sum up to 1. (history_window, n_road, demand_hop+1)
        H = torch.mul(H, att) # (history_window, n_road, demand_hop+1)
        H = torch.sum(H, dim=2) # (history_window, n_road)
        
        # add ToD, DoW features
        H = torch.cat([H.transpose(0, 1), ToD, DoW], dim=1) # (n_road, history_window+24+1)
        
        # linear output. specify weights and bias for each road segment
        Y = self.output_layer(H) # (1, 1, n_road)

        return Y.squeeze(0).squeeze(0) 
    
    
class Model_GNN(nn.Module):
    # TrGNN-. Remove trajectory information. Replace T in TrGNN with W_norm.
    
    def __init__(self, input_size=1, output_size=1, demand_hop=75, status_hop=3):
        super(Model_GNN, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.demand_hop = demand_hop
        self.status_hop = status_hop
        
        # attention
        self.attention_layer = ChannelAttention(2**(status_hop+1)-1, demand_hop+1, channels=2404, bias=True) # channels=n_road
                
        # linear output
        self.output_layer = ChannelFullyConnected(in_features=4+24+1, channels=2404) # channels=n_road
        

    def forward(self, X, T, W, h_init, W_norm, ToD, DoW):
        # X: graph signal. normalized. tensor: (history_window, n_road)
        # T: trajectory transition. normalized. tuple of history_window sparse_tensors: (n_road, n_road)
        # W: weighted road adjacency matrix. # sparse_tensor: (n_road, n_road)
        # h_init: for GRU. (gru_num_layers, n_road, hidden_size)
        # ToD: road-wise one-hot encoding of hour of day. (n_road, 24)
        # DoW: road-wise indicator. 1 for weekdays, 0 for weekends/PHs. (n_road, 1)
        
        # graph propagation
        H = torch.cat([graph_propagation_sparse(x, W_norm, hop=self.demand_hop).unsqueeze(0) for x in torch.unbind(X, dim=0)], dim=0)

        # attention
        S = torch.cat([graph_propagation_sparse(x, W_norm, hop=self.status_hop, dual=True).unsqueeze(0) for x in torch.unbind(X, dim=0)], dim=0)
        att = self.attention_layer(S.unsqueeze(3)) # specify weights and bias for each road segment
        att = F.softmax(att, dim=2) # attention weights across hops sum up to 1. (history_window, n_road, demand_hop+1)
        H = torch.mul(H, att) # (history_window, n_road, demand_hop+1)
        H = torch.sum(H, dim=2) # (history_window, n_road)
        
        # add ToD, DoW features
        H = torch.cat([H.transpose(0, 1), ToD, DoW], dim=1) # (n_road, history_window+24+1)
        
        # linear output. specify weights and bias for each road segment
        Y = self.output_layer(H) # (1, 1, n_road)

        return Y.squeeze(0).squeeze(0)