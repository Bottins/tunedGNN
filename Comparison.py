"""
Comprehensive T_Adam Dual Scaling Comparison
============================================

Rigorous comparison of all T_Adam configurations across multiple datasets.

Tests:
- Standard Adam (baseline)
- T_Adam (no scaling)
- T_Adam + TRF only
- T_Adam + Anti-Hub only
- T_Adam + Homophily only
- T_Adam + Ricci only
- T_Adam + TRF + Anti-Hub (full dual)
- T_Adam + TRF + Homophily (full dual)
- T_Adam + TRF + Ricci (full dual)

Datasets:
- Node classification: Cora, CiteSeer, PubMed
- Graph classification: MUTAG, PROTEINS, ENZYMES

Multiple runs with controlled seeds for statistical significance.
"""

import argparse
import time
import json
import os
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models import GCN
from datasets import load_dataset
from optimizers import T_Adam
from utils import evaluate_model, compute_gradient_norm


# Configuration constants
NODE_DATASETS = ['cora', 'citeseer']#, 'pubmed']
GRAPH_DATASETS = ['MUTAG', 'PROTEINS', 'ENZYMES']

# Optimizer configurations to test
OPTIMIZER_CONFIGS = [
    {'name': 'Adam', 'type': 'Adam'},
    {'name': 'T_Adam', 'type': 'T_Adam', 'use_trf': False, 'gradient_scaling': None},
    {'name': 'T_Adam+TRF', 'type': 'T_Adam', 'use_trf': True, 'gradient_scaling': None},
    {'name': 'T_Adam+AntiHub', 'type': 'T_Adam', 'use_trf': False, 'gradient_scaling': 'anti_hub'},
    {'name': 'T_Adam+Homophily', 'type': 'T_Adam', 'use_trf': False, 'gradient_scaling': 'homophily'},
    {'name': 'T_Adam+Ricci', 'type': 'T_Adam', 'use_trf': False, 'gradient_scaling': 'ricci'},
    {'name': 'T_Adam+TRF+AntiHub', 'type': 'T_Adam', 'use_trf': True, 'gradient_scaling': 'anti_hub'},
    {'name': 'T_Adam+TRF+Homophily', 'type': 'T_Adam', 'use_trf': True, 'gradient_scaling': 'homophily'},
    {'name': 'T_Adam+TRF+Ricci', 'type': 'T_Adam', 'use_trf': True, 'gradient_scaling': 'ricci'},
]


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_optimizer(model, config, graph_data=None, lr=0.01, weight_decay=5e-4):
    """
    Create optimizer based on configuration.

    Args:
        model: PyTorch model
        config: Optimizer configuration dict
        graph_data: Graph data for T_Adam scaling
        lr: Learning rate
        weight_decay: Weight decay

    Returns:
        optimizer instance
    """
    if config['type'] == 'Adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif config['type'] == 'T_Adam':
        # Prepare TRF weights if using TRF
        trf_weights = None
        if config.get('use_trf', False):
            trf_weights = {
                'wA': 1.0, 'wNc': 1.0, 'wEc': 1.0,
                'wE': 1.0, 'wL': 1.0, 'wmu': 1.0
            }

        # Prepare graph data if using any scaling
        optimizer_graph_data = None
        if config.get('use_trf', False) or config.get('gradient_scaling') is not None:
            optimizer_graph_data = graph_data

        # Node grad indices for local scaling
        node_grad_indices = [0] if config.get('gradient_scaling') is not None else None

        return T_Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            graph_data=optimizer_graph_data,
            trf_weights=trf_weights,
            beta_trf=0.1,
            gradient_scaling_mode=config.get('gradient_scaling'),
            node_grad_indices=node_grad_indices
        )

    else:
        raise ValueError(f"Unknown optimizer type: {config['type']}")


def train_node_classification(model, optimizer, graph, label, split_idx, criterion, device):
    """Single training step for node classification."""
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
    grad_norm = compute_gradient_norm(model)
    optimizer.step()

    # Training accuracy
    pred = out[train_idx].argmax(dim=-1)
    train_acc = (pred == label[train_idx]).float().mean().item()

    return loss.item(), train_acc, grad_norm


def train_graph_classification(model, optimizer, train_loader, criterion, device):
    """Single training epoch for graph classification."""
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
        grad_norm = compute_gradient_norm(model)
        total_grad_norm += grad_norm
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=-1)
        total_correct += (pred == batch.y).sum().item()
        total_graphs += batch.num_graphs

    avg_loss = total_loss / total_graphs
    avg_acc = total_correct / total_graphs
    avg_grad_norm = total_grad_norm / len(train_loader)

    return avg_loss, avg_acc, avg_grad_norm


def run_single_experiment(dataset_name, task, config, run_idx, device, epochs=200, lr=0.01,
                         hidden_channels=64, num_layers=3, dropout=0.5, batch_size=32):
    """
    Run a single training experiment.

    Returns:
        dict with results
    """
    # Set seed for this specific run
    seed = 42 + run_idx
    set_seed(seed)

    print(f"\n{'='*80}")
    print(f"Run {run_idx+1} | Dataset: {dataset_name} | Config: {config['name']}")
    print(f"Seed: {seed}")
    print(f"{'='*80}")

    # Load dataset
    dataset = load_dataset(dataset_name, data_dir='./data', task_type=task)

    # Prepare data based on task
    if task == 'node':
        graph, label = dataset[0]
        in_channels = graph['node_feat'].shape[1]
        num_classes = int(label.max().item()) + 1
        split_idx = dataset.get_idx_split()
        pooling = None

        # Prepare graph data for T_Adam
        graph_data = {
            'edge_index': graph['edge_index'],
            'num_nodes': graph['num_nodes'],
            'node_features': graph['node_feat']
        }
    else:  # graph classification
        if not hasattr(dataset, 'train_loader') or dataset.train_loader is None:
            dataset.create_loaders(batch_size=batch_size)
        in_channels = dataset.num_features
        num_classes = dataset.num_classes
        pooling = 'mean'

        # Use first graph as representative for T_Adam
        first_batch = next(iter(dataset.train_loader))
        graph_data = {
            'edge_index': first_batch.edge_index[:, :first_batch.ptr[1]],
            'num_nodes': int(first_batch.ptr[1]),
            'node_features': first_batch.x[:first_batch.ptr[1]]
        }

    # Create model
    model = GCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        batch_norm=False,
        layer_norm=False,
        residual=False,
        pooling=pooling
    ).to(device)

    # Create optimizer
    optimizer = create_optimizer(model, config, graph_data, lr=lr, weight_decay=5e-4)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0
    training_time = 0

    train_losses = []
    val_accs = []
    test_accs = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training
        if task == 'node':
            train_loss, train_acc, grad_norm = train_node_classification(
                model, optimizer, graph, label, split_idx, criterion, device
            )
        else:
            train_loss, train_acc, grad_norm = train_graph_classification(
                model, optimizer, dataset.train_loader, criterion, device
            )

        # Validation & Testing
        if task == 'node':
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
        training_time += epoch_time

        val_acc = val_metrics['accuracy']
        test_acc = test_metrics['accuracy']

        train_losses.append(train_loss)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch

        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
                  f"Test: {test_acc:.4f} | Best Test: {best_test_acc:.4f}")

    total_time = time.time() - start_time

    print(f"\nBest Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Test Acc at Best Val: {best_test_acc:.4f}")
    print(f"Total Time: {total_time:.2f}s")

    return {
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'total_time': total_time,
        'train_losses': train_losses,
        'val_accs': val_accs,
        'test_accs': test_accs,
        'seed': seed
    }


def run_comparison(task='node', datasets=None, runs=5, epochs=200, device='cuda',
                  output_dir='./comparison_results'):
    """
    Run comprehensive comparison across all configurations and datasets.

    Args:
        task: 'node' or 'graph'
        datasets: List of dataset names (None = use defaults)
        runs: Number of runs per configuration
        epochs: Number of training epochs
        device: Device to use
        output_dir: Directory to save results
    """
    if datasets is None:
        datasets = NODE_DATASETS if task == 'node' else GRAPH_DATASETS

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Results storage
    all_results = defaultdict(lambda: defaultdict(list))

    print("\n" + "="*80)
    print(f"COMPREHENSIVE T_ADAM COMPARISON - {task.upper()} CLASSIFICATION")
    print("="*80)
    print(f"Task: {task}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Configurations: {len(OPTIMIZER_CONFIGS)}")
    print(f"Runs per config: {runs}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("="*80)

    # Run experiments
    total_experiments = len(datasets) * len(OPTIMIZER_CONFIGS) * runs
    current_experiment = 0

    for dataset_name in datasets:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#'*80}")

        for config in OPTIMIZER_CONFIGS:
            print(f"\n{'-'*80}")
            print(f"Configuration: {config['name']}")
            print(f"{'-'*80}")

            for run_idx in range(runs):
                current_experiment += 1
                print(f"\nProgress: {current_experiment}/{total_experiments} "
                      f"({100*current_experiment/total_experiments:.1f}%)")

                try:
                    result = run_single_experiment(
                        dataset_name=dataset_name,
                        task=task,
                        config=config,
                        run_idx=run_idx,
                        device=torch.device(device),
                        epochs=epochs
                    )

                    all_results[dataset_name][config['name']].append(result)

                except Exception as e:
                    print(f"ERROR in experiment: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    # Compute statistics
    statistics = compute_statistics(all_results)

    # Save results
    results_file = os.path.join(output_dir, f'results_{task}_{timestamp}.json')
    save_results(all_results, statistics, results_file)

    # Generate plots
    plot_file = os.path.join(output_dir, f'comparison_{task}_{timestamp}.png')
    plot_results(statistics, task, plot_file)

    # Print summary
    print_summary(statistics, task)

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETED!")
    print(f"Results saved to: {results_file}")
    print(f"Plot saved to: {plot_file}")
    print("="*80)

    return all_results, statistics


def compute_statistics(all_results):
    """Compute mean and std for all configurations across runs."""
    statistics = defaultdict(lambda: defaultdict(dict))

    for dataset_name, configs in all_results.items():
        for config_name, runs in configs.items():
            test_accs = [r['best_test_acc'] for r in runs]
            val_accs = [r['best_val_acc'] for r in runs]
            times = [r['total_time'] for r in runs]

            statistics[dataset_name][config_name] = {
                'test_acc_mean': np.mean(test_accs),
                'test_acc_std': np.std(test_accs),
                'test_acc_min': np.min(test_accs),
                'test_acc_max': np.max(test_accs),
                'val_acc_mean': np.mean(val_accs),
                'val_acc_std': np.std(val_accs),
                'time_mean': np.mean(times),
                'time_std': np.std(times),
                'num_runs': len(runs)
            }

    return statistics


def save_results(all_results, statistics, filepath):
    """Save results to JSON file."""
    # Convert to serializable format
    serializable_results = {}
    for dataset_name, configs in all_results.items():
        serializable_results[dataset_name] = {}
        for config_name, runs in configs.items():
            serializable_results[dataset_name][config_name] = [
                {k: v for k, v in r.items() if k not in ['train_losses', 'val_accs', 'test_accs']}
                for r in runs
            ]

    output = {
        'statistics': dict(statistics),
        'all_results': serializable_results,
        'timestamp': datetime.now().isoformat()
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def plot_results(statistics, task, filepath):
    """Generate comparison plots."""
    datasets = list(statistics.keys())
    configs = list(statistics[datasets[0]].keys())

    # Create figure with subplots
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset_name in enumerate(datasets):
        ax = axes[idx]

        # Extract data
        config_names = []
        means = []
        stds = []

        for config_name in configs:
            stats = statistics[dataset_name][config_name]
            config_names.append(config_name)
            means.append(stats['test_acc_mean'])
            stds.append(stats['test_acc_std'])

        # Create bar plot
        x = np.arange(len(config_names))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)

        # Color bars: Adam in gray, T_Adam variants in colors
        colors = ['gray' if 'Adam' == name else 'steelblue' if 'T_Adam' == name
                  else 'orange' if 'TRF' in name and '+' not in name.replace('+TRF', '')
                  else 'green' if 'AntiHub' in name and 'TRF' not in name
                  else 'purple' if 'Homophily' in name and 'TRF' not in name
                  else 'red' if 'Ricci' in name and 'TRF' not in name
                  else 'darkgreen' if 'AntiHub' in name and 'TRF' in name
                  else 'darkviolet' if 'Homophily' in name and 'TRF' in name
                  else 'darkred'
                  for name in config_names]

        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel('Configuration', fontsize=10)
        ax.set_ylabel('Test Accuracy', fontsize=10)
        ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.02, f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=7)

    plt.suptitle(f'T_Adam Dual Scaling Comparison - {task.upper()} Classification',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")
    plt.close()


def print_summary(statistics, task):
    """Print summary table of results."""
    print(f"\n{'='*100}")
    print(f"SUMMARY - {task.upper()} CLASSIFICATION")
    print(f"{'='*100}\n")

    for dataset_name, configs in statistics.items():
        print(f"\n{dataset_name}:")
        print("-" * 100)
        print(f"{'Configuration':<25} {'Test Acc (mean±std)':<25} {'Best':<10} {'Worst':<10} {'Time (s)':<15}")
        print("-" * 100)

        # Sort by test accuracy
        sorted_configs = sorted(configs.items(),
                               key=lambda x: x[1]['test_acc_mean'],
                               reverse=True)

        for config_name, stats in sorted_configs:
            print(f"{config_name:<25} "
                  f"{stats['test_acc_mean']:.4f} ± {stats['test_acc_std']:.4f}        "
                  f"{stats['test_acc_max']:.4f}    "
                  f"{stats['test_acc_min']:.4f}    "
                  f"{stats['time_mean']:.1f} ± {stats['time_std']:.1f}")

        print()


def main():
    parser = argparse.ArgumentParser(description='Comprehensive T_Adam Comparison')
    parser.add_argument('--task', type=str, default='both', choices=['node', 'graph', 'both'],
                       help='Task type (node, graph, or both)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='Specific datasets to test (default: all for task)')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of runs per configuration')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Run comparisons
    if args.task == 'both':
        print("\n" + "#"*100)
        print("# RUNNING NODE CLASSIFICATION COMPARISON")
        print("#"*100)
        run_comparison(
            task='node',
            datasets=args.datasets if args.datasets else NODE_DATASETS,
            runs=args.runs,
            epochs=args.epochs,
            device=args.device,
            output_dir=args.output_dir
        )

        print("\n" + "#"*100)
        print("# RUNNING GRAPH CLASSIFICATION COMPARISON")
        print("#"*100)
        run_comparison(
            task='graph',
            datasets=args.datasets if args.datasets else GRAPH_DATASETS,
            runs=args.runs,
            epochs=args.epochs,
            device=args.device,
            output_dir=args.output_dir
        )
    else:
        run_comparison(
            task=args.task,
            datasets=args.datasets,
            runs=args.runs,
            epochs=args.epochs,
            device=args.device,
            output_dir=args.output_dir
        )


if __name__ == '__main__':
    main()
