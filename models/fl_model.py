import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, gru_num_layers):
        super().__init__()
        self.decoder = nn.GRU(
            input_size, hidden_size, num_layers=gru_num_layers, dropout=dropout
        )
        self.out_net = nn.Linear(hidden_size, output_size)


    def forward(self, data):
        # B x T x N x F
        x, x_attr, y, y_attr = data['x'], data['x_attr'], data['y'], data['y_attr']
        batch_num, node_num = x.shape[0], x.shape[2]
        x_input = torch.cat((x, x_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2) # T x (B x N) x F

        y_input = torch.cat((y, y_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2)
        y_input = torch.cat((x_input[-1:], y_input[:-1]), dim=0)
        out_hidden, _ = self.decoder(y_input)
        out = self.out_net(out_hidden)
        out = out.view(out.shape[0], batch_num, node_num, out.shape[-1]).permute(1, 0, 2, 3)
        return out