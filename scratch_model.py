# scratch_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def scatter_add(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max()) + 1
    out = torch.zeros(dim_size, src.size(-1), device=src.device)
    return out.index_add(dim, index, src)

class ScratchMPNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ScratchMPNNLayer, self).__init__()
        self.message_mlp = nn.Linear(in_channels, out_channels)
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index):
        row, col = edge_index  # row: i (target), col: j (source)
        messages = self.message_mlp(x[col])  # x_j
        aggr = scatter_add(messages, row, dim=0, dim_size=x.size(0))
        updated = self.update_mlp(torch.cat([x, aggr], dim=1))
        return updated

class ScratchMPNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(ScratchMPNN, self).__init__()
        self.conv1 = ScratchMPNNLayer(in_channels, hidden_channels)
        self.conv2 = ScratchMPNNLayer(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        # Global mean pooling
        batch_size = int(batch.max()) + 1
        out = scatter_add(x, batch, dim=0, dim_size=batch_size)
        count = torch.bincount(batch)
        out = out / count.unsqueeze(-1).to(out.dtype).to(out.device)

        return self.classifier(out)
