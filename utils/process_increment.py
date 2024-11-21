import os
import pickle
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import random
from copy import deepcopy

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def unscaled_metrics(y_pred, y, scaler):
    y = scaler.inverse_transform(y.detach().cpu())
    y_pred = scaler.inverse_transform(y_pred.detach().cpu())
    # mse
    mse = ((y_pred - y) ** 2).mean()
    # RMSE
    rmse = torch.sqrt(mse)
    # MAE
    mae = torch.abs(y_pred - y).mean()

    return {
        'mse': mse.detach(),
        'rmse': rmse.detach(),
        'mae': mae.detach(),
    }

def load_dataset(name, adj_mx_name, num_clients, pred_len):
    rootpath = 'data/'
    random.seed(0)
    total_num = 325
    if name == 'METR-LA':
        total_num = 207
    selected_nodes = sorted(random.sample(range(0, total_num), num_clients))

    adj_mx_path = os.path.join(rootpath, 'sensor_graph', adj_mx_name)
    _, _, adj_mx = load_pickle(adj_mx_path)
    adj_mx_ts = torch.from_numpy(adj_mx).float()
    train_adj_mx_ts = adj_mx_ts[selected_nodes, :][:, selected_nodes]

    train_edge_index, train_edge_attr = dense_to_sparse(train_adj_mx_ts)
    datapath = os.path.join(rootpath, name)
    raw_data = np.load(os.path.join(datapath, str(pred_len) + '_series.npz'))

    FEATURE_START, FEATURE_END = 0, 1
    ATTR_START, ATTR_END = 1, 2

    train_features = raw_data['x'][:, :, :, FEATURE_START:FEATURE_END]
    train_features = train_features[:, :, selected_nodes, :]
    train_features = train_features.reshape(-1, train_features.shape[-1])
    feature_scaler = StandardScaler(
        mean=train_features.mean(axis=0), std=train_features.std(axis=0)
    )
    attr_scaler = StandardScaler(
        mean=0, std=1
    )
    x = feature_scaler.transform(raw_data['x'][:, :, selected_nodes, FEATURE_START:FEATURE_END])
    y = feature_scaler.transform(raw_data['y'][:, :, selected_nodes, FEATURE_START:FEATURE_END])
    x_attr = attr_scaler.transform(raw_data['x'][:, :, selected_nodes, ATTR_START:ATTR_END])
    y_attr = attr_scaler.transform(raw_data['y'][:, :, selected_nodes, ATTR_START:ATTR_END])
    data = {}
    data.update(
        x=torch.from_numpy(x).float(), y=torch.from_numpy(y).float(),
        x_attr=torch.from_numpy(x_attr).float(),
        y_attr=torch.from_numpy(y_attr).float(),
        edge_index=train_edge_index, edge_attr=train_edge_attr,
        adj_mx=train_adj_mx_ts,
        feature_scaler=feature_scaler,
        attr_scaler=attr_scaler
    )
    return data, selected_nodes