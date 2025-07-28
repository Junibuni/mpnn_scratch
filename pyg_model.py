import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

class SimpleMPNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SimpleMPNN, self).__init__(aggr='add')  # 'add', 'mean', 'max'
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.edge_mlp = torch.nn.Linear(1, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            edge_embedding = self.edge_mlp(edge_attr)
            return x_j + edge_embedding
        return x_j

    def update(self, aggr_out):
        return F.relu(aggr_out)

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GNN, self).__init__()
        self.conv1 = SimpleMPNN(in_channels, hidden_channels)
        self.conv2 = SimpleMPNN(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return self.lin(x)
