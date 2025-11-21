"""
Metrics for Optimizer Comparison
=================================

Provides evaluation metrics and comparison utilities for analyzing
optimizer performance on GNN training tasks.
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json


def evaluate_model(model, data_loader, criterion, device, task='node', split_idx=None):
    """
    Evaluate model performance.

    Args:
        model: GNN model
        data_loader: DataLoader for graph classification or tuple (graph, label) for node classification
        criterion: Loss function
        device: torch device
        task: 'node' or 'graph'
        split_idx: For node classification, indices to evaluate on

    Returns:
        dict: Dictionary with loss, accuracy, and optionally other metrics
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        if task == 'node':
            # Node classification
            graph, label = data_loader
            edge_index = graph['edge_index'].to(device)
            node_feat = graph['node_feat'].to(device)
            label = label.to(device)

            out = model(node_feat, edge_index)

            if split_idx is not None:
                out = out[split_idx]
                label = label[split_idx]

            loss = criterion(out, label)
            pred = out.argmax(dim=-1)

            all_preds = pred.cpu().numpy()
            all_labels = label.cpu().numpy()
            total_loss = loss.item()

        elif task == 'graph':
            # Graph classification
            for batch in data_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)

                total_loss += loss.item() * batch.num_graphs
                pred = out.argmax(dim=-1)

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

            total_loss /= len(data_loader.dataset)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)

    results = {
        'loss': total_loss,
        'accuracy': accuracy
    }

    # Add F1 score for multi-class
    if len(np.unique(all_labels)) > 2:
        results['f1_macro'] = f1_score(all_labels, all_preds, average='macro')
        results['f1_micro'] = f1_score(all_labels, all_preds, average='micro')

    return results


class OptimizerComparison:
    """
    Tracks and compares optimizer performance during training.

    This class collects training metrics for multiple optimizers and provides
    analysis and visualization tools for comparison.
    """

    def __init__(self):
        self.history = defaultdict(lambda: {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'epoch_times': [],
            'gradient_norms': [],
            'param_updates': []
        })
        self.final_results = {}

    def log_epoch(self, optimizer_name, epoch, train_metrics, val_metrics, test_metrics=None,
                  epoch_time=None, gradient_norm=None, param_update_norm=None):
        """
        Log metrics for a single epoch.

        Args:
            optimizer_name (str): Name of the optimizer
            epoch (int): Epoch number
            train_metrics (dict): Training metrics (loss, accuracy, etc.)
            val_metrics (dict): Validation metrics
            test_metrics (dict, optional): Test metrics
            epoch_time (float, optional): Time taken for the epoch
            gradient_norm (float, optional): Norm of gradients
            param_update_norm (float, optional): Norm of parameter updates
        """
        history = self.history[optimizer_name]

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])

        if test_metrics is not None:
            history['test_loss'].append(test_metrics['loss'])
            history['test_acc'].append(test_metrics['accuracy'])

        if epoch_time is not None:
            history['epoch_times'].append(epoch_time)

        if gradient_norm is not None:
            history['gradient_norms'].append(gradient_norm)

        if param_update_norm is not None:
            history['param_updates'].append(param_update_norm)

    def log_final_results(self, optimizer_name, test_metrics, total_time, best_val_epoch):
        """
        Log final results after training.

        Args:
            optimizer_name (str): Name of the optimizer
            test_metrics (dict): Final test metrics
            total_time (float): Total training time
            best_val_epoch (int): Epoch with best validation performance
        """
        self.final_results[optimizer_name] = {
            'test_accuracy': test_metrics['accuracy'],
            'test_loss': test_metrics['loss'],
            'total_time': total_time,
            'best_val_epoch': best_val_epoch,
            'avg_epoch_time': total_time / len(self.history[optimizer_name]['train_loss'])
        }

        if 'f1_macro' in test_metrics:
            self.final_results[optimizer_name]['test_f1_macro'] = test_metrics['f1_macro']

    def plot_comparison(self, save_path=None):
        """
        Generate comparison plots.

        Args:
            save_path (str, optional): Path to save the figure
        """
        optimizers = list(self.history.keys())
        if len(optimizers) == 0:
            print("No data to plot")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Optimizer Comparison', fontsize=16, fontweight='bold')

        # Plot 1: Training Loss
        ax = axes[0, 0]
        for opt_name in optimizers:
            ax.plot(self.history[opt_name]['train_loss'], label=opt_name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Validation Loss
        ax = axes[0, 1]
        for opt_name in optimizers:
            ax.plot(self.history[opt_name]['val_loss'], label=opt_name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Training Accuracy
        ax = axes[0, 2]
        for opt_name in optimizers:
            ax.plot(self.history[opt_name]['train_acc'], label=opt_name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Accuracy')
        ax.set_title('Training Accuracy over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Validation Accuracy
        ax = axes[1, 0]
        for opt_name in optimizers:
            ax.plot(self.history[opt_name]['val_acc'], label=opt_name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('Validation Accuracy over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 5: Gradient Norms
        ax = axes[1, 1]
        for opt_name in optimizers:
            if len(self.history[opt_name]['gradient_norms']) > 0:
                ax.plot(self.history[opt_name]['gradient_norms'], label=opt_name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norms over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 6: Convergence Speed (Epoch Times)
        ax = axes[1, 2]
        for opt_name in optimizers:
            if len(self.history[opt_name]['epoch_times']) > 0:
                cumulative_time = np.cumsum(self.history[opt_name]['epoch_times'])
                ax.plot(cumulative_time, self.history[opt_name]['val_acc'], label=opt_name, linewidth=2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('Convergence Speed (Accuracy vs Time)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def print_summary(self):
        """Print summary comparison table."""
        if len(self.final_results) == 0:
            print("No final results to display")
            return

        print("\n" + "=" * 80)
        print("OPTIMIZER COMPARISON SUMMARY")
        print("=" * 80)

        print(f"\n{'Optimizer':<20} {'Test Acc':<12} {'Test Loss':<12} {'Time (s)':<12} {'Best Epoch':<12}")
        print("-" * 80)

        for opt_name, results in self.final_results.items():
            print(f"{opt_name:<20} "
                  f"{results['test_accuracy']:.4f}      "
                  f"{results['test_loss']:.4f}      "
                  f"{results['total_time']:.2f}      "
                  f"{results['best_val_epoch']}")

        # Find best optimizer
        best_opt = max(self.final_results.items(), key=lambda x: x[1]['test_accuracy'])
        print("\n" + "-" * 80)
        print(f"BEST OPTIMIZER: {best_opt[0]} (Test Accuracy: {best_opt[1]['test_accuracy']:.4f})")
        print("=" * 80 + "\n")

    def save_results(self, path='optimizer_comparison_results.json'):
        """Save all results to JSON file."""
        results = {
            'history': dict(self.history),
            'final_results': self.final_results
        }

        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        print(f"Results saved to {path}")


def compute_gradient_norm(model):
    """
    Compute the L2 norm of gradients.

    Args:
        model: PyTorch model

    Returns:
        float: L2 norm of all gradients
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_param_update_norm(optimizer):
    """
    Compute the L2 norm of parameter updates.

    This requires storing parameter values before the optimizer step.

    Args:
        optimizer: PyTorch optimizer with stored 'param_before' in state

    Returns:
        float: L2 norm of parameter updates
    """
    total_norm = 0.0
    for group in optimizer.param_groups:
        for p in group['params']:
            if 'param_before' in optimizer.state[p]:
                param_before = optimizer.state[p]['param_before']
                update = p.data - param_before
                total_norm += update.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
