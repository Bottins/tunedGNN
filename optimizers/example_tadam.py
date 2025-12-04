"""
Simple usage example of TAdam optimizer with PyTorch Geometric.

This example shows how to use TAdam with a real GNN model on a citation network.
"""

import torch
import torch.nn.functional as F
from tadam import TAdam

# Try to use PyTorch Geometric if available
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    
    class GCN(torch.nn.Module):
        """Simple 2-layer GCN for node classification."""
        
        def __init__(self, num_features, num_classes, hidden_dim=16):
            super().__init__()
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, num_classes)
            self.dropout = 0.5
            
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
    
    def train_with_tadam_on_cora():
        """Train a GCN on Cora dataset using TAdam."""
        
        print("Loading Cora dataset...")
        dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
        data = dataset[0]
        
        print(f"Dataset info:")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.edge_index.shape[1]}")
        print(f"  Features: {data.num_features}")
        print(f"  Classes: {dataset.num_classes}")
        
        # Initialize model
        model = GCN(
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            hidden_dim=16
        )
        
        # Prepare graph data for TAdam
        graph_data = {
            'edge_index': data.edge_index,
            'num_nodes': data.num_nodes,
            'node_features': data.x  # Pass features for homophily computation
        }
        
        # Test different TAdam configurations
        configs = [
            ('Adam (baseline)', None, None),
            ('TAdam with degree scaling', 'degree', graph_data),
            ('TAdam with homophily scaling', 'homophily', graph_data),
            ('TAdam with combined scaling', 'combined', graph_data),
        ]
        
        results = {}
        
        for config_name, local_scaling, graph_info in configs:
            print(f"\n{'='*50}")
            print(f"Training with {config_name}")
            print('='*50)
            
            # Reset model
            model = GCN(
                num_features=dataset.num_features,
                num_classes=dataset.num_classes,
                hidden_dim=16
            )
            
            # Create optimizer
            if graph_info is None:
                # Standard Adam
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            else:
                # TAdam with topology awareness
                optimizer = TAdam(
                    model.parameters(),
                    lr=0.01,
                    weight_decay=5e-4,
                    graph_data=graph_info,
                    local_scaling=local_scaling,
                    topology_update_freq=20,  # Update topology metrics every 20 steps
                    trf_weights={
                        'wA': 1.0,    # Algebraic connectivity weight
                        'wNc': 1.0,   # Node connectivity weight
                        'wEc': 1.0,   # Edge connectivity weight
                        'wE': 1.0,    # Global efficiency weight
                        'wL': 1.0,    # Average path length weight
                        'wmu': 1.0    # Graph density multiplier weight
                    }
                )
            
            # Training loop
            model.train()
            train_losses = []
            val_accuracies = []
            
            for epoch in range(200):
                # Training step
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                
                # Validation
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        pred = model(data.x, data.edge_index).argmax(dim=1)
                        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                        val_accuracies.append(val_acc.item())
                    model.train()
                    
                    print(f"Epoch {epoch:3d}: Loss = {loss:.4f}, Val Acc = {val_acc:.4f}")
            
            # Final evaluation
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
            
            results[config_name] = {
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'test_accuracy': test_acc.item()
            }
            
            print(f"\nFinal Test Accuracy: {test_acc:.4f}")
        
        # Print comparison
        print("\n" + "="*60)
        print("RESULTS COMPARISON")
        print("="*60)
        
        for config_name, metrics in results.items():
            print(f"\n{config_name}:")
            print(f"  Final training loss: {metrics['train_losses'][-1]:.4f}")
            print(f"  Test accuracy: {metrics['test_accuracy']:.4f}")
            print(f"  Convergence (loss reduction): {metrics['train_losses'][0] - metrics['train_losses'][-1]:.4f}")
        
        # Find best configuration
        best_config = max(results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n🏆 Best configuration: {best_config[0]}")
        print(f"   Test accuracy: {best_config[1]['test_accuracy']:.4f}")
        
        return results
    
    if __name__ == "__main__":
        print("TAdam Example with PyTorch Geometric")
        print("="*60)
        results = train_with_tadam_on_cora()

except ImportError:
    print("PyTorch Geometric not installed!")
    print("Install with: pip install torch-geometric")
    print("\nShowing simple example instead...")
    
    # Fallback example without PyG
    import torch.nn as nn
    
    def simple_example():
        """Simple example without PyTorch Geometric."""
        
        # Create a simple graph (triangle)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                                  [1, 0, 2, 1, 0, 2]], dtype=torch.long)
        num_nodes = 3
        node_features = torch.randn(3, 10)
        
        # Simple linear model
        model = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        
        # Create TAdam optimizer with degree-based scaling
        optimizer = TAdam(
            model.parameters(),
            lr=0.01,
            graph_data=(edge_index, num_nodes),
            local_scaling='degree',
            topology_update_freq=10
        )
        
        print("Training simple model with TAdam...")
        
        # Training loop
        for epoch in range(100):
            optimizer.zero_grad()
            
            # Simple forward pass
            output = model(node_features)
            target = torch.tensor([0, 1, 0])
            loss = F.cross_entropy(output, target)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
        
        print("\nTraining completed!")
        
        # Show optimizer state
        print(f"\nOptimizer state:")
        print(f"  TRF scale factor: {optimizer._trf_scale:.4f}")
        print(f"  Update frequency: {optimizer.metrics.update_frequency}")
        
        # Test state dict save/load
        state = optimizer.state_dict()
        print(f"\nState dict keys: {state.keys()}")
        
        # Create new optimizer and load state
        new_optimizer = TAdam(
            model.parameters(),
            graph_data=(edge_index, num_nodes)
        )
        new_optimizer.load_state_dict(state)
        print("✅ State successfully loaded into new optimizer")
    
    if __name__ == "__main__":
        print("Simple TAdam Example")
        print("="*60)
        simple_example()
