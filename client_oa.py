import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from collections import defaultdict
from utils.process_increment import unscaled_metrics
from models.fl_model import GRU
from copy import deepcopy
import numpy as np
import scipy.stats

class Client(object):

    def __init__(self, client_id, client_dataset, feature_scaler,
                 input_size, output_size, args):
        self.client_id = client_id
        self.client_dataset = client_dataset
        self.feature_scaler = feature_scaler
        self.input_size = input_size
        self.output_size = output_size
        self.args = args
        self.lr = self.args.lr
        self.batch_size = self.args.batch_size
        self.dataloader = DataLoader(self.client_dataset, batch_size=self.batch_size)
        self.model = GRU(input_size, self.args.hidden_size, output_size, self.args.dropout, self.args.num_layers)

        self.state_dict = None
        self.h_client_dataset = None
        self.selected = False

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')


    def local_execute(self, state_dict_to_load):
        print('Training on client #{}'.format(self.client_id))
        self.dataloader = DataLoader(self.client_dataset, batch_size=self.batch_size)
        if self.selected:
            if state_dict_to_load is not None:
                self.model.load_state_dict(state_dict_to_load)
            self.model.to(self.device)
            self.model.train()
            with torch.enable_grad():
                for epoch_i in range(self.args.epoch):
                    num_samples = 0
                    epoch_log = defaultdict(lambda: 0.0)
                    for batch in self.dataloader:
                        x, y, x_attr, y_attr = batch
                        x = x.to(self.device) if (x is not None) else None
                        y = y.to(self.device) if (y is not None) else None
                        x_attr = x_attr.to(self.device) if (x_attr is not None) else None
                        y_attr = y_attr.to(self.device) if (y_attr is not None) else None
                        data = dict(
                            x=x, x_attr=x_attr, y=y, y_attr=y_attr
                        )
                        y_pred = self.model(data)
                        loss = nn.MSELoss()(y_pred, y)
                        loss.backward()
                        for param in self.model.parameters():
                            param.data = param.data - self.lr * param.grad.data
                            param.grad.data.zero_()
                        num_samples += x.shape[0]
                        metrics = unscaled_metrics(y_pred, y, self.feature_scaler)
                        epoch_log['loss'] += loss.detach() * x.shape[0]
                        for k in metrics:
                            epoch_log[k] += metrics[k] * x.shape[0]
                    for k in epoch_log:
                        epoch_log[k] /= num_samples  # 做平均化操作
                        epoch_log[k] = epoch_log[k].cpu()
            self.h_client_dataset = deepcopy(self.client_dataset)
        else:
            self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                num_samples = 0
                epoch_log = defaultdict(lambda: 0.0)
                for batch in self.dataloader:
                    x, y, x_attr, y_attr = batch
                    x = x.to(self.device) if (x is not None) else None
                    y = y.to(self.device) if (y is not None) else None
                    x_attr = x_attr.to(self.device) if (x_attr is not None) else None
                    y_attr = y_attr.to(self.device) if (y_attr is not None) else None
                    data = dict(
                        x=x, x_attr=x_attr, y=y, y_attr=y_attr
                    )
                    y_pred = self.model(data)
                    loss = nn.MSELoss()(y_pred, y)

                    num_samples += x.shape[0]
                    metrics = unscaled_metrics(y_pred, y, self.feature_scaler)
                    epoch_log['loss'] += loss.detach() * x.shape[0]
                    for k in metrics:
                        epoch_log[k] += metrics[k] * x.shape[0]
                for k in epoch_log:
                    epoch_log[k] /= num_samples
                    epoch_log[k] = epoch_log[k].cpu()


        self.selected = False
        self.state_dict = deepcopy(self.model.to(self.device).state_dict())

        epoch_log['num_samples'] = num_samples
        epoch_log = dict(**epoch_log)
        self.model.to(self.device)
        self.local_result = {
            'state_dict': self.state_dict, 'log': epoch_log
        }

    def eval_dataset(self):
        if self.h_client_dataset == None:
            self.selected = True
        else:
            self.dataloader = DataLoader(self.client_dataset, batch_size=self.batch_size)
            for batch in self.dataloader:
                x, y, x_attr, y_attr = batch
                data_now = np.array(x.flatten().tolist())
            self.h_dataloader = DataLoader(self.h_client_dataset, batch_size=self.batch_size)
            for batch in self.h_dataloader:
                x, y, x_attr, y_attr = batch
                data_h = np.array(x.flatten().tolist())
            data_h = self.feature_scaler.inverse_transform(data_h)
            data_now = self.feature_scaler.inverse_transform(data_now)
            KL = scipy.stats.entropy(data_now, data_h)
            if KL > self.args.kl_threshold:
                self.selected = True
            else:
                self.selected = False