import torch
import time
from torch_geometric import utils
from torch_sparse import SparseTensor
import torch_geometric as pyg
import numpy as np
import h5py
import sys


def get_batch(file, wait=10, trials=60):
    while trials > 0:
        try:
            return False, torch.load(file)
            
        except FileNotFoundError as e:
            trials -= 1
            print(e, file=sys.stderr)
            print(f"waiting for {wait} sec...", file=sys.stderr)
            time.sleep(wait)
        return True, "The directory is not reachable..."

def get_data(file_name, group, index, wait=10, trials=60):
    while trials > 0:
        try:
            with h5py.File(file_name, 'r') as file:
                return False, file[group][index][0]
            
        except FileNotFoundError as e:
            trials -= 1
            print(e, file=sys.stderr)
            print(f"waiting for {wait} sec...", file=sys.stderr)
            time.sleep(wait)
    return True, "The directory is not reachable..."


def to_heterodata(nodes:dict, edges:dict, weights:dict, features, labels):

    geometric_ei = torch.from_numpy(geometric_ei)
    geometric_ea = torch.pow(torch.sigmoid(torch.from_numpy(geometric_ea)), -1)
    
    geometric_ei, geometric_ea = utils.add_self_loops(geometric_ei, geometric_ea, 2)
    
    momentum_ei, momentum_ea = momentum_connection(x[:,1])
    n = len(x)

    for node in nodes: # nodes: dict[str, numpy array]
        ...
        ##### TOOOOOOOOO DOOOOOOOOOOOOOO!!!!!!!!!!!!!
    
    for node, edge in zip(nodes, edges):
        edge_index  = edges[edge][0]
        edge_weight = edges[edge][1]

        adj = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=edge_weight, 
            sparse_sizes=(n, n)
        )
        ...
    
    mom_adj_t = SparseTensor(
        row=momentum_ei[0],
        col=momentum_ei[1],
        value=geometric_ea, 
        sparse_sizes=(n, n)
    )
    
    pos_adj_t = SparseTensor(
        row=geometric_ei[0],
        col=geometric_ei[1],
        value=geometric_ea, 
        sparse_sizes=(n, n)
    )
    
    x = torch.cat([
        x, (adj.to_torch_sparse_coo_tensor() @ x[:,1]).unsqueeze(1)
    ], dim=1)
    
    hdata = pyg.data.HeteroData({
        'features': features, 'y': torch.from_numpy(gl),
        'particle': {
            'x': x },
        ('particle', 'r_con', 'particle'):
        { 'adj_t': pos_adj_t},
        ('particle', 'p_con', 'particle'):
        { 'adj_t': mom_adj_t},
    })
    return hdata


def to_data(x, edges, weights, features, labels, momentum_con=False):
    x, features = extend_features(torch.from_numpy(x), features[0][0])

    if momentum_con:
        edges, weights = momentum_connection(x[:,1])
    else:
        weights = torch.pow(torch.sigmoid(torch.from_numpy(weights)), -1)
        edges, weights = utils.to_undirected(torch.from_numpy(edges), weights)
        edges, weights = utils.add_self_loops(edges, weights, 2)
    n = len(x)
    adj = SparseTensor(row=edges[0], col=edges[1], value=weights, sparse_sizes=(n, n))
    x = torch.cat([x, (adj.to_torch_sparse_coo_tensor() @ x[:,1]).unsqueeze(1)], dim=1)
    return pyg.data.Data(x=x, features=features, adj_t=adj, y=torch.from_numpy(labels))

def extend_features(x, nv):
    graph_features = [nv]

    for p_type in [11, 13, 22, 211, 130]:
        mask = torch.abs(x[:, 5]) == p_type
        x = torch.cat([
            x, (mask).unsqueeze(1)
        ], dim=1)
        graph_features.append(mask.sum())

    mask = torch.abs(x[:, 5]) < 2
    graph_features.append(mask.sum())
    x[:, 5] = (mask)

    graph_features = torch.tensor(graph_features).unsqueeze(1)
    return x, graph_features

def momentum_connection(pt, tr=0.3):
    pt_t = torch.abs(pt.unsqueeze(1) - pt.unsqueeze(0))
    mask = pt_t < tr
    edge_inds  = mask.nonzero().t()
    edge_attr = torch.pow(torch.sigmoid(pt_t[mask]), -1)
    return edge_inds, edge_attr



def check_tensors(*tensors):
    nnan = 0
    ninf = 0
    for tensor in tensors:
        nnan += tensor.isnan().sum()
        ninf += tensor.isinf().sum()   
    return nnan > 0, ninf > 0 

def check_data(data, b_ind=-1):
    if hasattr(data, 'adj_t'):
        inf_status, nan_status = check_tensors(data.x, data.features, data.y)
    else:
        inf_status, nan_status = check_tensors(data.edge_attr, data.edge_index, data.x, data.features, data.y)
        
    clean = not (inf_status or nan_status)
    err   = None
    
    if not clean:
        if inf_status and nan_status:
            err = f"`nan` and `inf` detected in batch no. {b_ind}. Skipping the batch..."

        elif inf_status:
            err = f"`inf` detected in batch no. {b_ind}. Skipping the batch..."

        elif nan_status:
            err = f"`nan` detected in batch no. {b_ind}. Skipping the batch..."
    
    return clean, err


def remove_nan_node(node_features, edge_index, edge_attributes):    
    node_ID, feature_ID = np.where(np.isnan(node_features))
    node_features       = np.delete(node_features, node_ID, axis=0)
    
    for node in node_ID:
        _, edge    = np.where(edge_index == node)
        edge_index = np.delete(edge_index, edge, axis=1)
        mask = edge_index > node
        if mask.sum() > 0:
            edge_index[mask] = edge_index[mask] - 1
        edge_attributes  = np.delete(edge_attributes, edge, axis=0)
    
    return node_features, edge_index, edge_attributes

def remove_inf_node(node_features, edge_index, edge_attributes):    
    node_ID, feature_ID = np.where(np.isinf(node_features))
    node_features       = np.delete(node_features, node_ID, axis=0)
    
    for node in node_ID:
        _, edge    = np.where(edge_index == node)
        edge_index = np.delete(edge_index, edge, axis=1)
        mask = edge_index > node
        if mask.sum() > 0:
            edge_index[mask] = edge_index[mask] - 1
        edge_attributes = np.delete(edge_attributes, edge, axis=0)
    
    return node_features, edge_index, edge_attributes

def extend_features(x, nv):
    graph_features = [nv]

    for p_type in [11, 13, 22, 211, 130]:
        mask = torch.abs(x[:, 5]) == p_type
        x = torch.cat([
            x, (mask).unsqueeze(1)
        ], dim=1)
        graph_features.append(mask.sum())

    mask = torch.abs(x[:, 5]) < 2
    graph_features.append(mask.sum())
    x[:, 5] = (mask)

    graph_features = torch.tensor(graph_features).unsqueeze(1)
    return x, graph_features