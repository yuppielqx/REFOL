from client_oa import Client
import numpy as np
import torch
import xlsxwriter
from utils.process_increment import load_dataset
from torch.utils.data import TensorDataset
from copy import deepcopy
from models.AggregationGCN import AttGCN

class REFOL(object):
    def __init__(self, config):
        self.config = config

    def boot(self):
        print('Booting {} fl-server...'.format(self.config.agg_model))
        self.num_clients = self.config.num_clients
        print('Total clients: {}'.format(self.num_clients))
        data, selected_node = load_dataset(name=self.config.dataset
                                           , adj_mx_name=self.config.adj_mx
                                           , num_clients=self.num_clients
                                           , pred_len=self.config.pred_steps
                                           )

        self.data = data
        input_size = self.data['x'].shape[-1] + self.data['x_attr'].shape[-1]
        output_size = self.data['y'].shape[-1]
        self.max_epoch = data['x'].shape[0]
        self.train_per_num_samples = 1

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        clients = []
        for client_i in range(self.num_clients):
            client_dataset = TensorDataset(
                data['x'][:self.train_per_num_samples, :, client_i:client_i + 1, :],
                data['y'][:self.train_per_num_samples, :, client_i:client_i + 1, :],
                data['x_attr'][:self.train_per_num_samples, :, client_i:client_i + 1, :],
                data['y_attr'][:self.train_per_num_samples, :, client_i:client_i + 1, :]
            )

            client_tmp = Client(client_id=client_i,
                                client_dataset=client_dataset,
                                feature_scaler=self.data['feature_scaler'],
                                input_size=input_size,
                                output_size=output_size,
                                args=self.config)
            clients.append(client_tmp)

        self.clients = clients
        self.gcn = AttGCN()

        self.server_datasets = TensorDataset(
            self.data['x'], self.data['y'],
            self.data['x_attr'], self.data['y_attr'])

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.global_model = None

    def run(self):
        rounds = self.max_epoch - 1 - self.train_per_num_samples
        rounds = 10
        workbook = xlsxwriter.Workbook('{}_{}.xlsx'.format(self.config.dataset, self.config.pred_steps))
        sheet_round = workbook.add_worksheet('round')
        sheet_round.write(0, 0, 'round_num')
        sheet_round.write(0, 1, '{}_rmse'.format(self.config.agg_model))
        sheet_round.write(0, 2, '{}_mae'.format(self.config.agg_model))


        for rround in range(1, rounds + 1):
            print('**** Round {}/{} ****'.format(rround, rounds))
            train_log = self.train_round(rround)
            train_loss = train_log['log']['rmse'].item()
            train_mae = train_log['log']['mae'].item()

            sheet_round.write(rround, 0, rround)
            sheet_round.write(rround, 1, train_loss)
            sheet_round.write(rround, 2, train_mae)
            print('prediction rmse is: {} '.format(train_loss))

        workbook.close()

    def train_round(self, rround):
        self.update_train_data(rround, self.clients)#所有客户端更新本次的训练数据
        agg_id_list = []
        for client in self.clients:
            if client.selected:
                agg_id_list.append(client.client_id)
        print('selected clients：', agg_id_list)

        local_logs = []
        agg_state_dict = []
        for idx, client in enumerate(self.clients):
            if client.selected:
                client.local_execute(state_dict_to_load=deepcopy(self.global_model))
                agg_state_dict.append(deepcopy(client.local_result['state_dict']))
                local_logs.append(client.local_result['log'])
            else:
                client.local_execute(state_dict_to_load=None)
                local_logs.append(client.local_result['log'])


        agg_local_train_results = self.aggregate_local_train_results(local_logs, agg_state_dict, agg_id_list, rround)
        agg_log = agg_local_train_results['log']
        log = agg_log
        return {
            'loss': torch.tensor(0).float(),
            'progress_bar': log,
            'log': log
        }

    def update_train_data(self, rround, sample_clients):
        for client in sample_clients:
            client_i = client.client_id
            client.client_dataset = TensorDataset(
                self.data['x'][rround - 1:self.train_per_num_samples + rround - 1, :, client_i:client_i + 1, :],
                self.data['y'][rround - 1:self.train_per_num_samples + rround - 1, :, client_i:client_i + 1, :],
                self.data['x_attr'][rround - 1:self.train_per_num_samples + rround - 1, :, client_i:client_i + 1, :],
                self.data['y_attr'][rround - 1:self.train_per_num_samples + rround - 1, :, client_i:client_i + 1, :]
            )
            client.eval_dataset()

    def aggregate_local_train_results(self, local_logs, local_states, agg_id_list, round):
        self.aggregate_local_train_state_dicts(local_states, agg_id_list, round)
        return {
            'log': self.aggregate_local_logs(local_logs)
        }

    # 聚合客户端本地模型
    def aggregate_local_train_state_dicts(self, local_states, agg_id_list, round):
        edge_index = np.array(deepcopy(self.data['edge_index']))
        sample_id = np.array(agg_id_list)
        mask = np.isin(edge_index, sample_id)
        mask1 = np.isin(np.sum(mask, axis=0), 2)
        edge_index = edge_index[:, mask1]
        table = np.zeros(sample_id.max() + 1, np.int64)
        table[sample_id] = np.arange(sample_id.size)
        edge_index = torch.from_numpy(table[edge_index])
        tmp = np.full((sample_id.shape), len(sample_id))
        tmp = np.stack((table[sample_id], tmp))
        tmp = np.hstack((tmp, [[len(sample_id)],[len(sample_id)]]))
        edge_index = torch.from_numpy(np.concatenate((edge_index, tmp), axis=1))

        tmp_model = self.global_model
        if tmp_model is None:
            tmp_model = deepcopy(local_states[0])
        local_states.append(tmp_model)

        local_results = []
        for i, local_train_result in enumerate(local_states):
            for name in local_train_result:
                local_results += local_train_result[name].flatten().tolist()
        local_results = torch.Tensor(local_results).view((len(sample_id) + 1, -1))

        self.gcn.to(self.device)
        local_results = self.gcn(
            x=local_results.to(self.device)
            , edge_index=edge_index.to(self.device)
        )
        global_model = local_results[-1]
        agg_state_dict = {}
        len_start = 0
        for name in local_train_result:
            length = len(local_train_result[name].flatten().tolist())
            agg_state_dict[name] = global_model[len_start:len_start + length].reshape_as(local_train_result[name])
            len_start += length
        self.global_model = agg_state_dict


    def aggregate_local_logs(self, local_logs):
        agg_log = deepcopy(local_logs[0])
        for k in agg_log:
            agg_log[k] = 0
            for local_log_idx, local_log in enumerate(local_logs):
                if k == 'num_samples':
                    agg_log[k] += local_log[k]
                else:
                    agg_log[k] += local_log[k] * local_log['num_samples']
        for k in agg_log:
            if k != 'num_samples':
                agg_log[k] /= agg_log['num_samples']
        return agg_log