"""
GCN: Graph Convolutional Network
=================================

Simplified GCN implementation supporting both:
- Node classification (single graph)
- Graph classification (multiple graphs with pooling)

Based on: Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool


class GCN(nn.Module):
    """
    Graph Convolutional Network for node and graph classification.

    Args:
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden units
        out_channels (int): Number of output classes
        num_layers (int): Number of GCN layers (default: 3)
        dropout (float): Dropout rate (default: 0.5)
        batch_norm (bool): Use batch normalization (default: False)
        layer_norm (bool): Use layer normalization (default: False)
        residual (bool): Use residual connections (default: False)
        pooling (str): Pooling method for graph classification: 'mean', 'add', 'max', None (default: None)
                      If None, performs node classification
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.5, batch_norm=False, layer_norm=False,
                 residual=False, pooling=None):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.residual = residual
        self.pooling = pooling

        # GCN Layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, normalize=True))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True))

        # Residual connection linear projections
        if self.residual:
            self.residual_lins = nn.ModuleList()
            self.residual_lins.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.residual_lins.append(nn.Linear(hidden_channels, hidden_channels))

        # Normalization layers
        if self.batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(num_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        if self.layer_norm:
            self.lns = nn.ModuleList()
            for _ in range(num_layers):
                self.lns.append(nn.LayerNorm(hidden_channels))

        # Output layer
        self.output_layer = nn.Linear(hidden_channels, out_channels)

        # Pooling for graph classification
        if self.pooling == 'mean':
            self.pool = global_mean_pool
        elif self.pooling == 'add':
            self.pool = global_add_pool
        elif self.pooling == 'max':
            self.pool = global_max_pool
        elif self.pooling is None:
            self.pool = None
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        if self.residual:
            for lin in self.residual_lins:
                lin.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
        if self.layer_norm:
            for ln in self.lns:
                ln.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.

        Args:
            x (Tensor): Node feature matrix [num_nodes, in_channels]
            edge_index (Tensor): Graph connectivity [2, num_edges]
            batch (Tensor, optional): Batch vector [num_nodes] for graph classification
                                     Assigns each node to a specific graph

        Returns:
            Tensor: Output predictions
                   - Node classification: [num_nodes, out_channels]
                   - Graph classification: [num_graphs, out_channels]
        """
        for i, conv in enumerate(self.convs):
            x_input = x

            # GCN convolution
            x = conv(x, edge_index)

            # Residual connection
            if self.residual:
                x = x + self.residual_lins[i](x_input)

            # Normalization
            if self.batch_norm:
                x = self.bns[i](x)
            elif self.layer_norm:
                x = self.lns[i](x)

            # Activation
            x = F.relu(x)

            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling for graph classification
        if self.pool is not None:
            if batch is None:
                raise ValueError("batch must be provided for graph classification (pooling enabled)")
            x = self.pool(x, batch)

        # Output layer
        x = self.output_layer(x)

        return x

    def get_embeddings(self, x, edge_index, batch=None):
        """
        Get node/graph embeddings before the final classification layer.

        Returns:
            Tensor: Embeddings [num_nodes, hidden_channels] or [num_graphs, hidden_channels]
        """
        for i, conv in enumerate(self.convs):
            x_input = x
            x = conv(x, edge_index)

            if self.residual:
                x = x + self.residual_lins[i](x_input)

            if self.batch_norm:
                x = self.bns[i](x)
            elif self.layer_norm:
                x = self.lns[i](x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply pooling if graph classification
        if self.pool is not None and batch is not None:
            x = self.pool(x, batch)

        return x
