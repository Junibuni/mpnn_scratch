import torch
import torch.nn as nn
import torch.nn.functional as F

def scatter_add(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max()) + 1
    out = torch.zeros(dim_size, src.size(-1), device=src.device)
    return out.index_add(dim, index, src)

class MessagePassingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.edge_mlp = nn.Linear(edge_dim, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.edge_mlp.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.lin(x)

        row, col = edge_index  # i <- j

        x_j = x[col]  # source node features
        msg = self.message(x_j, edge_attr)
        aggr = self.aggregate(msg, row, dim_size=x.size(0))
        out = self.update(aggr)
        return out

    def message(self, x_j, edge_attr):
        return x_j + self.edge_mlp(edge_attr)

    def aggregate(self, messages, index, dim_size):
        return scatter_add(messages, index, dim=0, dim_size=dim_size)

    def update(self, aggr_out):
        return F.relu(aggr_out)


class ScratchMPNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, edge_dim):
        super().__init__()
        self.conv1 = MessagePassingLayer(in_channels, hidden_channels, edge_dim)
        self.conv2 = MessagePassingLayer(hidden_channels, hidden_channels, edge_dim)
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.classifier.reset_parameters()

    def global_mean_pool(self, x, batch):
        out = scatter_add(x, batch, dim=0)
        count = torch.bincount(batch)
        return out / count.unsqueeze(-1).to(out.dtype)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.global_mean_pool(x, batch)
        return self.classifier(x)
