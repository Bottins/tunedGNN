"""
Dataset Loaders for Node and Graph Classification
==================================================

Supports:
- Node Classification: Single graph with node-level labels
- Graph Classification: Multiple graphs with graph-level labels
"""

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import (
    Amazon, Coauthor, HeterophilousGraphDataset, WikiCS,
    Planetoid, TUDataset
)
from torch_geometric.loader import DataLoader
from ogb.nodeproppred import NodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
import numpy as np


# ============================================================================
# Dataset Availability
# ============================================================================

NODE_CLASSIFICATION_DATASETS = {
    'small': ['cora', 'citeseer', 'pubmed'],
    'medium': [
        'amazon-photo', 'amazon-computer',
        'coauthor-cs', 'coauthor-physics',
        'wikics',
        'chameleon', 'squirrel'
    ],
    'heterophilous': ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'],
    'large': ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins']
}

GRAPH_CLASSIFICATION_DATASETS = {
    'bioinformatics': ['MUTAG', 'PROTEINS', 'DD', 'NCI1', 'ENZYMES', 'PTC_MR'],
    'social': ['REDDIT-BINARY', 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI'],
    'molecules': ['ogbg-molhiv', 'ogbg-molpcba']
}


def get_available_datasets():
    """Returns all available datasets."""
    return {
        'node_classification': NODE_CLASSIFICATION_DATASETS,
        'graph_classification': GRAPH_CLASSIFICATION_DATASETS
    }


# ============================================================================
# Node Classification Dataset Wrapper
# ============================================================================

class NCDataset(object):
    """
    Node Classification Dataset wrapper.

    Attributes:
        graph (dict): Dictionary containing edge_index, node_feat, edge_feat, num_nodes
        label (Tensor): Node labels
        train_idx, valid_idx, test_idx (Tensor): Train/validation/test indices
    """

    def __init__(self, name):
        self.name = name
        self.graph = {}
        self.label = None
        self.train_idx = None
        self.valid_idx = None
        self.test_idx = None

    def get_idx_split(self):
        """Returns train/valid/test split indices."""
        if self.train_idx is None:
            raise ValueError("Dataset has no predefined split. Use random split instead.")
        return {
            'train': self.train_idx,
            'valid': self.valid_idx,
            'test': self.test_idx
        }

    def __getitem__(self, idx):
        assert idx == 0, 'Node classification dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'


# ============================================================================
# Graph Classification Dataset Wrapper
# ============================================================================

class GCDataset(object):
    """
    Graph Classification Dataset wrapper.

    Attributes:
        dataset: PyG dataset
        num_features (int): Number of node features
        num_classes (int): Number of graph classes
        train_loader, valid_loader, test_loader: DataLoaders
    """

    def __init__(self, name, root, dataset):
        self.name = name
        self.root = root
        self.dataset = dataset
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def create_loaders(self, batch_size=32, train_split=0.8, valid_split=0.1):
        """Create train/valid/test data loaders."""
        num_graphs = len(self.dataset)
        num_train = int(num_graphs * train_split)
        num_valid = int(num_graphs * valid_split)

        # Random split
        indices = torch.randperm(num_graphs)
        train_indices = indices[:num_train]
        valid_indices = indices[num_train:num_train + num_valid]
        test_indices = indices[num_train + num_valid:]

        train_dataset = self.dataset[train_indices]
        valid_dataset = self.dataset[valid_indices]
        test_dataset = self.dataset[test_indices]

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return self.train_loader, self.valid_loader, self.test_loader

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name}, graphs={len(self)})'


# ============================================================================
# Dataset Loaders
# ============================================================================

def load_dataset(name, data_dir='./data', task_type='auto'):
    """
    Load dataset for node or graph classification.

    Args:
        name (str): Dataset name
        data_dir (str): Root directory for datasets
        task_type (str): 'node', 'graph', or 'auto' (auto-detect)

    Returns:
        NCDataset or GCDataset
    """

    # Auto-detect task type
    if task_type == 'auto':
        all_node = sum(NODE_CLASSIFICATION_DATASETS.values(), [])
        all_graph = sum(GRAPH_CLASSIFICATION_DATASETS.values(), [])

        if name in all_node:
            task_type = 'node'
        elif name in all_graph:
            task_type = 'graph'
        else:
            raise ValueError(f"Unknown dataset: {name}")

    # Load dataset based on task type
    if task_type == 'node':
        return _load_node_classification_dataset(name, data_dir)
    elif task_type == 'graph':
        return _load_graph_classification_dataset(name, data_dir)
    else:
        raise ValueError(f"Invalid task_type: {task_type}. Use 'node', 'graph', or 'auto'")


# ============================================================================
# Node Classification Loaders
# ============================================================================

def _load_node_classification_dataset(name, data_dir):
    """Load node classification dataset."""

    if name in ('cora', 'citeseer', 'pubmed'):
        return _load_planetoid(name, data_dir)
    elif name in ('amazon-photo', 'amazon-computer'):
        return _load_amazon(name, data_dir)
    elif name in ('coauthor-cs', 'coauthor-physics'):
        return _load_coauthor(name, data_dir)
    elif name == 'wikics':
        return _load_wikics(data_dir)
    elif name in ('roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'):
        return _load_heterophilous(name, data_dir)
    elif name in ('ogbn-arxiv', 'ogbn-products', 'ogbn-proteins'):
        return _load_ogbn(name, data_dir)
    else:
        raise ValueError(f"Unknown node classification dataset: {name}")


def _load_planetoid(name, data_dir):
    """Load Planetoid datasets (Cora, Citeseer, Pubmed)."""
    torch_dataset = Planetoid(root=f'{data_dir}/Planetoid', name=name)
    data = torch_dataset[0]

    dataset = NCDataset(name)
    dataset.graph = {
        'edge_index': data.edge_index,
        'node_feat': data.x,
        'edge_feat': None,
        'num_nodes': data.num_nodes
    }
    dataset.label = data.y
    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    return dataset


def _load_amazon(name, data_dir):
    """Load Amazon datasets (Photo, Computer)."""
    dataset_name = name.split('-')[1].capitalize()
    torch_dataset = Amazon(root=f'{data_dir}/Amazon', name=dataset_name)
    data = torch_dataset[0]

    dataset = NCDataset(name)
    dataset.graph = {
        'edge_index': data.edge_index,
        'node_feat': data.x,
        'edge_feat': None,
        'num_nodes': data.num_nodes
    }
    dataset.label = data.y

    # Random split (you can modify this)
    dataset.train_idx, dataset.valid_idx, dataset.test_idx = _random_split(
        data.num_nodes, train_ratio=0.6, valid_ratio=0.2
    )

    return dataset


def _load_coauthor(name, data_dir):
    """Load Coauthor datasets (CS, Physics)."""
    dataset_name = name.split('-')[1].upper()
    torch_dataset = Coauthor(root=f'{data_dir}/Coauthor', name=dataset_name)
    data = torch_dataset[0]

    dataset = NCDataset(name)
    dataset.graph = {
        'edge_index': data.edge_index,
        'node_feat': data.x,
        'edge_feat': None,
        'num_nodes': data.num_nodes
    }
    dataset.label = data.y

    dataset.train_idx, dataset.valid_idx, dataset.test_idx = _random_split(
        data.num_nodes, train_ratio=0.6, valid_ratio=0.2
    )

    return dataset


def _load_wikics(data_dir):
    """Load WikiCS dataset."""
    torch_dataset = WikiCS(root=f'{data_dir}/WikiCS')
    data = torch_dataset[0]

    dataset = NCDataset('wikics')
    dataset.graph = {
        'edge_index': data.edge_index,
        'node_feat': data.x,
        'edge_feat': None,
        'num_nodes': data.num_nodes
    }
    dataset.label = data.y
    dataset.train_idx = torch.where(data.train_mask[:, 0])[0]
    dataset.valid_idx = torch.where(data.val_mask[:, 0])[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    return dataset


def _load_heterophilous(name, data_dir):
    """Load heterophilous datasets."""
    torch_dataset = HeterophilousGraphDataset(root=data_dir, name=name.capitalize())
    data = torch_dataset[0]

    dataset = NCDataset(name)
    dataset.graph = {
        'edge_index': data.edge_index,
        'node_feat': data.x,
        'edge_feat': None,
        'num_nodes': data.num_nodes
    }
    dataset.label = data.y
    dataset.train_idx = torch.where(data.train_mask[:, 0])[0]
    dataset.valid_idx = torch.where(data.val_mask[:, 0])[0]
    dataset.test_idx = torch.where(data.test_mask[:, 0])[0]

    return dataset


def _load_ogbn(name, data_dir):
    """Load OGB node classification datasets."""
    torch_dataset = NodePropPredDataset(name=name, root=data_dir)
    split_idx = torch_dataset.get_idx_split()
    data = torch_dataset[0]

    dataset = NCDataset(name)
    dataset.graph = {
        'edge_index': torch.as_tensor(data[0]['edge_index']),
        'node_feat': torch.as_tensor(data[0]['node_feat']),
        'edge_feat': None,
        'num_nodes': data[0]['num_nodes']
    }
    dataset.label = torch.as_tensor(data[1]).squeeze()
    dataset.train_idx = split_idx['train']
    dataset.valid_idx = split_idx['valid']
    dataset.test_idx = split_idx['test']

    return dataset


# ============================================================================
# Graph Classification Loaders
# ============================================================================

def _load_graph_classification_dataset(name, data_dir):
    """Load graph classification dataset."""

    if name.startswith('ogbg-'):
        return _load_ogbg(name, data_dir)
    else:
        # TUDataset
        return _load_tu_dataset(name, data_dir)


def _load_tu_dataset(name, data_dir):
    """Load TU datasets (MUTAG, PROTEINS, etc.)."""
    torch_dataset = TUDataset(root=f'{data_dir}/TUDataset', name=name)

    dataset = GCDataset(name, data_dir, torch_dataset)
    return dataset


def _load_ogbg(name, data_dir):
    """Load OGB graph classification datasets."""
    torch_dataset = PygGraphPropPredDataset(name=name, root=data_dir)

    dataset = GCDataset(name, data_dir, torch_dataset)
    split_idx = torch_dataset.get_idx_split()

    # Create loaders with OGB splits
    dataset.train_loader = DataLoader(torch_dataset[split_idx['train']], batch_size=32, shuffle=True)
    dataset.valid_loader = DataLoader(torch_dataset[split_idx['valid']], batch_size=32, shuffle=False)
    dataset.test_loader = DataLoader(torch_dataset[split_idx['test']], batch_size=32, shuffle=False)

    return dataset


# ============================================================================
# Utility Functions
# ============================================================================

def _random_split(num_nodes, train_ratio=0.6, valid_ratio=0.2):
    """Generate random train/valid/test split."""
    indices = torch.randperm(num_nodes)
    num_train = int(num_nodes * train_ratio)
    num_valid = int(num_nodes * valid_ratio)

    train_idx = indices[:num_train]
    valid_idx = indices[num_train:num_train + num_valid]
    test_idx = indices[num_train + num_valid:]

    return train_idx, valid_idx, test_idx
