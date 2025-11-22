"""
Example: Using T_Adam with Dual Topological Scaling
====================================================

This example demonstrates how to use the T_Adam optimizer with:
1. TRF-based global learning rate scaling
2. Three different local gradient scaling modes

The example can be run with different gradient scaling modes:
- anti_hub: Penalizes hub nodes
- homophily: Emphasizes homogeneous regions
- ricci: Emphasizes bridge nodes between communities
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from optimizers.t_adam import T_Adam


class GCN(torch.nn.Module):
    """Simple 2-layer GCN for node classification."""
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train_with_t_adam(dataset_name='Cora', gradient_scaling_mode=None, epochs=200):
    """
    Train GCN with T_Adam optimizer.

    Args:
        dataset_name: Name of the dataset (Cora, CiteSeer, PubMed)
        gradient_scaling_mode: 'anti_hub', 'homophily', 'ricci', or None
        epochs: Number of training epochs
    """
    print(f"\n{'='*80}")
    print(f"Training on {dataset_name} with T_Adam")
    print(f"Gradient Scaling Mode: {gradient_scaling_mode if gradient_scaling_mode else 'None (standard Adam)'}")
    print(f"{'='*80}\n")

    # Load dataset
    dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
    data = dataset[0]

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(
        num_features=dataset.num_features,
        hidden_channels=16,
        num_classes=dataset.num_classes
    ).to(device)
    data = data.to(device)

    # Prepare graph data for T_Adam
    graph_data = {
        'edge_index': data.edge_index,
        'num_nodes': data.num_nodes,
        'node_features': data.x  # Optional, needed for homophily mode
    }

    # Configure TRF weights (all set to 1.0 by default)
    trf_weights = {
        'wA': 1.0,    # Algebraic connectivity
        'wNc': 1.0,   # Node connectivity
        'wEc': 1.0,   # Edge connectivity
        'wE': 1.0,    # Global efficiency
        'wL': 1.0,    # Average path length
        'wmu': 1.0    # mu_G multiplier
    }

    # Initialize T_Adam optimizer with dual scaling
    optimizer = T_Adam(
        model.parameters(),
        lr=0.01,
        graph_data=graph_data,
        trf_weights=trf_weights,
        beta_trf=0.1,  # TRF scaling parameter
        gradient_scaling_mode=gradient_scaling_mode,
        node_grad_indices=[0] if gradient_scaling_mode else None  # Apply to first layer's weights
    )

    # Register gradient scaling hooks if using local gradient scaling
    if gradient_scaling_mode is not None:
        optimizer.apply_gradient_scaling_to_model(model, layer_index=0)

    print("\nStarting training...\n")

    # Training loop
    model.train()
    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)

                train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc

                print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | '
                      f'Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}')

            model.train()

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy at Best Val: {best_test_acc:.4f}")
    print(f"{'='*80}\n")

    return best_test_acc


def compare_all_modes():
    """Compare all gradient scaling modes on Cora dataset."""
    modes = [None, 'anti_hub', 'homophily', 'ricci']
    results = {}

    for mode in modes:
        mode_name = mode if mode else 'standard'
        results[mode_name] = train_with_t_adam(
            dataset_name='Cora',
            gradient_scaling_mode=mode,
            epochs=200
        )

    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    for mode_name, acc in results.items():
        print(f"{mode_name:20s}: {acc:.4f}")
    print("="*80 + "\n")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode not in ['anti_hub', 'homophily', 'ricci', 'standard', 'compare']:
            print("Usage: python example_t_adam_dual_scaling.py [anti_hub|homophily|ricci|standard|compare]")
            sys.exit(1)

        if mode == 'compare':
            compare_all_modes()
        elif mode == 'standard':
            train_with_t_adam(gradient_scaling_mode=None)
        else:
            train_with_t_adam(gradient_scaling_mode=mode)
    else:
        # Default: compare all modes
        print("Running comparison of all gradient scaling modes...")
        compare_all_modes()
