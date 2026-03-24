from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv


class GraphTransformerNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=False))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout, concat=False))
        self.convs.append(TransformerConv(hidden_channels, out_channels, heads=1, dropout=dropout, concat=False))

    def forward(self, x, edge_index, return_attention: bool = False):
        attn_info: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i, conv in enumerate(self.convs):
            if return_attention:
                x, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
                attn_info.append((ei, alpha.detach()))
            else:
                x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention:
            return x, attn_info
        return x


class GCNNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MLPNet(nn.Module):
    def __init__(self, in_channels: int, hidden_dims: Optional[List[int]] = None, out_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]
        layers = []
        prev = in_channels
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
