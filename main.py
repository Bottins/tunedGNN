"""
GCN Optimizer Comparison
========================

Compare standard Adam optimizer with T_Adam (customizable Adam) on GCN training.

Usage:
    # Node classification (single graph)
    python main.py --dataset cora --task node

    # Graph classification (multiple graphs)
    python main.py --dataset MUTAG --task graph

    # Use GPU
    python main.py --dataset cora --task node --device cuda
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from models import GCN
from datasets import load_dataset, get_available_datasets
from optimizers import T_Adam
from utils import evaluate_model, OptimizerComparison, compute_gradient_norm


def parse_args():
    parser = argparse.ArgumentParser(description='GCN Optimizer Comparison')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Dataset name (e.g., cora, MUTAG)')
    parser.add_argument('--task', type=str, default='node', choices=['node', 'graph'],
                        help='Task type: node or graph classification')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')

    # Model settings
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GCN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--batch_norm', action='store_true',
                        help='Use batch normalization')
    parser.add_argument('--layer_norm', action='store_true',
                        help='Use layer normalization')
    parser.add_argument('--residual', action='store_true',
                        help='Use residual connections')

    # Training settings
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for graph classification')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs for averaging results')

    # Optimizer settings
    parser.add_argument('--optimizers', type=str, nargs='+', default=['Adam', 'T_Adam'],
                        help='Optimizers to compare (Adam, T_Adam)')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'add', 'max'],
                        help='Pooling method for graph classification')

    # Device settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output settings
    parser.add_argument('--save_plot', type=str, default='optimizer_comparison.png',
                        help='Path to save comparison plot')
    parser.add_argument('--save_results', type=str, default='optimizer_results.json',
                        help='Path to save results JSON')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def train_node_classification(model, optimizer, graph, label, split_idx, criterion, device):
    """
    Single training step for node classification.

    Returns:
        tuple: (train_metrics, gradient_norm)
    """
    model.train()
    optimizer.zero_grad()

    edge_index = graph['edge_index'].to(device)
    node_feat = graph['node_feat'].to(device)
    label = label.to(device)
    train_idx = split_idx['train'].to(device)

    # Forward pass
    out = model(node_feat, edge_index)
    loss = criterion(out[train_idx], label[train_idx])

    # Backward pass
    loss.backward()

    # Compute gradient norm before optimizer step
    grad_norm = compute_gradient_norm(model)

    # Optimizer step
    optimizer.step()

    # Compute training accuracy
    pred = out[train_idx].argmax(dim=-1)
    train_acc = (pred == label[train_idx]).float().mean().item()

    train_metrics = {'loss': loss.item(), 'accuracy': train_acc}

    return train_metrics, grad_norm


def train_graph_classification(model, optimizer, train_loader, criterion, device):
    """
    Single training epoch for graph classification.

    Returns:
        tuple: (train_metrics, gradient_norm)
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_graphs = 0
    total_grad_norm = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)

        # Backward pass
        loss.backward()

        # Compute gradient norm
        grad_norm = compute_gradient_norm(model)
        total_grad_norm += grad_norm

        # Optimizer step
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=-1)
        total_correct += (pred == batch.y).sum().item()
        total_graphs += batch.num_graphs

    avg_loss = total_loss / total_graphs
    avg_acc = total_correct / total_graphs
    avg_grad_norm = total_grad_norm / len(train_loader)

    train_metrics = {'loss': avg_loss, 'accuracy': avg_acc}

    return train_metrics, avg_grad_norm


def run_experiment(args, optimizer_name, dataset, device):
    """
    Run full training experiment with a specific optimizer.

    Returns:
        dict: Results containing best test metrics and training history
    """
    print(f"\n{'='*60}")
    print(f"Training with {optimizer_name}")
    print(f"{'='*60}")

    # Determine task-specific parameters
    if args.task == 'node':
        graph, label = dataset[0]
        in_channels = graph['node_feat'].shape[1]
        num_classes = int(label.max().item()) + 1
        split_idx = dataset.get_idx_split()
        pooling = None
    else:  # graph classification
        if not hasattr(dataset, 'train_loader') or dataset.train_loader is None:
            dataset.create_loaders(batch_size=args.batch_size)
        in_channels = dataset.num_features
        num_classes = dataset.num_classes
        pooling = args.pooling

    # Create model
    model = GCN(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        layer_norm=args.layer_norm,
        residual=args.residual,
        pooling=pooling
    ).to(device)

    # Create optimizer
    if optimizer_name == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_name == 'T_Adam':
        optimizer = T_Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0
    best_test_metrics = None
    best_epoch = 0
    start_time = time.time()

    # Create comparison tracker
    comparison = OptimizerComparison()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Training
        if args.task == 'node':
            train_metrics, grad_norm = train_node_classification(
                model, optimizer, graph, label, split_idx, criterion, device
            )
        else:
            train_metrics, grad_norm = train_graph_classification(
                model, optimizer, dataset.train_loader, criterion, device
            )

        # Validation
        if args.task == 'node':
            val_metrics = evaluate_model(
                model, (graph, label), criterion, device, task='node', split_idx=split_idx['valid']
            )
            test_metrics = evaluate_model(
                model, (graph, label), criterion, device, task='node', split_idx=split_idx['test']
            )
        else:
            val_metrics = evaluate_model(
                model, dataset.valid_loader, criterion, device, task='graph'
            )
            test_metrics = evaluate_model(
                model, dataset.test_loader, criterion, device, task='graph'
            )

        epoch_time = time.time() - epoch_start

        # Log metrics
        comparison.log_epoch(
            optimizer_name, epoch, train_metrics, val_metrics, test_metrics,
            epoch_time=epoch_time, gradient_norm=grad_norm
        )

        # Track best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_test_metrics = test_metrics
            best_epoch = epoch

        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Test Acc: {test_metrics['accuracy']:.4f} | "
                  f"Time: {epoch_time:.2f}s")

    total_time = time.time() - start_time

    # Log final results
    comparison.log_final_results(optimizer_name, best_test_metrics, total_time, best_epoch)

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Final Test Accuracy: {best_test_metrics['accuracy']:.4f}")
    print(f"Total Training Time: {total_time:.2f}s")

    return {
        'comparison': comparison,
        'best_test_metrics': best_test_metrics,
        'total_time': total_time,
        'best_epoch': best_epoch
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    print("\n" + "="*80)
    print("GCN OPTIMIZER COMPARISON")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Device: {args.device}")
    print(f"Optimizers: {', '.join(args.optimizers)}")
    print("="*80)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, data_dir=args.data_dir, task_type=args.task)
    print(f"Dataset loaded successfully!")

    if args.task == 'node':
        graph, label = dataset[0]
        print(f"  Nodes: {graph['num_nodes']}")
        print(f"  Edges: {graph['edge_index'].shape[1]}")
        print(f"  Features: {graph['node_feat'].shape[1]}")
        print(f"  Classes: {int(label.max().item()) + 1}")
    else:
        print(f"  Graphs: {len(dataset)}")
        print(f"  Features: {dataset.num_features}")
        print(f"  Classes: {dataset.num_classes}")

    # Device setup
    device = torch.device(args.device)

    # Run experiments for each optimizer
    all_comparisons = OptimizerComparison()

    for optimizer_name in args.optimizers:
        # Reset seed before each optimizer to ensure fair comparison
        set_seed(args.seed)
        result = run_experiment(args, optimizer_name, dataset, device)

        # Merge results into global comparison
        for key, value in result['comparison'].history[optimizer_name].items():
            all_comparisons.history[optimizer_name][key] = value
        all_comparisons.final_results[optimizer_name] = result['comparison'].final_results[optimizer_name]

    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    all_comparisons.print_summary()

    # Save results
    all_comparisons.save_results(args.save_results)

    # Plot comparison
    all_comparisons.plot_comparison(save_path=args.save_plot)

    print("\nExperiment completed!")
    print(f"Results saved to: {args.save_results}")
    print(f"Plot saved to: {args.save_plot}")


if __name__ == '__main__':
    main()
