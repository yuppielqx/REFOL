import torch
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import degree


class GCNConv(MessagePassing):
    def __init__(self):
        super(GCNConv, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        row, col = edge_index
        num_objects = x.size(0)
        deg = degree(col, num_objects, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        weights = scatter(norm, col, dim=0, reduce="sum")
        for i, node in enumerate(col):
            norm[i] /= weights[node]

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class AttGCN(torch.nn.Module):
    def __init__(self):
        super(AttGCN, self).__init__()
        self.conv1 = GCNConv()
        self.conv2 = GCNConv()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x