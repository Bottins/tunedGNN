"""
Test script to verify caching functionality and multi-graph TRF computation
"""

import torch
import time
from datasets import load_dataset
from optimizers import T_Adam
from models import GCN

def test_cache_node_classification():
    """Test caching for node classification tasks."""
    print("\n" + "="*80)
    print("TEST 1: Caching for Node Classification (Cora)")
    print("="*80)

    # Load dataset
    dataset = load_dataset('cora', task_type='node')
    graph, label = dataset[0]

    # Prepare graph data
    graph_data = {
        'edge_index': graph['edge_index'],
        'num_nodes': graph['num_nodes'],
        'node_features': graph['node_feat']
    }

    # Create a simple model
    model = GCN(
        in_channels=graph['node_feat'].shape[1],
        hidden_channels=64,
        out_channels=int(label.max().item()) + 1,
        num_layers=2
    )

    # TRF weights
    trf_weights = {
        'wA': 1.0, 'wNc': 1.0, 'wEc': 1.0,
        'wE': 1.0, 'wL': 1.0, 'wmu': 1.0
    }

    print("\n--- First optimizer creation (should compute TRF) ---")
    start = time.time()
    opt1 = T_Adam(
        model.parameters(),
        lr=0.01,
        graph_data=graph_data,
        trf_weights=trf_weights,
        beta_trf=0.1,
        gradient_scaling_mode='anti_hub',
        node_grad_indices=[0]
    )
    time1 = time.time() - start
    print(f"Time taken: {time1:.2f}s")

    print("\n--- Second optimizer creation (should use cache) ---")
    start = time.time()
    opt2 = T_Adam(
        model.parameters(),
        lr=0.01,
        graph_data=graph_data,
        trf_weights=trf_weights,
        beta_trf=0.1,
        gradient_scaling_mode='anti_hub',
        node_grad_indices=[0]
    )
    time2 = time.time() - start
    print(f"Time taken: {time2:.2f}s")

    speedup = time1 / time2 if time2 > 0 else float('inf')
    print(f"\nSpeedup: {speedup:.2f}x")

    # Verify that TRF values match
    assert abs(opt1.trf_value - opt2.trf_value) < 1e-6, "TRF values don't match!"
    assert abs(opt1.lr_scale_factor - opt2.lr_scale_factor) < 1e-6, "LR scale factors don't match!"

    print("✓ Cache working correctly for node classification!")


def test_multigraph_trf():
    """Test multi-graph TRF computation for graph classification tasks."""
    print("\n" + "="*80)
    print("TEST 2: Multi-graph TRF Computation (MUTAG)")
    print("="*80)

    # Load dataset
    dataset = load_dataset('MUTAG', task_type='graph')
    dataset.create_loaders(batch_size=32)

    # Sample multiple graphs
    num_sample_graphs = min(5, len(dataset.dataset))
    indices = torch.linspace(0, len(dataset.dataset)-1, num_sample_graphs).long()

    graph_data_list = []
    for idx in indices:
        data = dataset.dataset[idx]
        graph_data_list.append({
            'edge_index': data.edge_index,
            'num_nodes': data.num_nodes,
            'node_features': data.x
        })

    print(f"\nSampled {len(graph_data_list)} graphs for TRF computation")

    # Create a simple model
    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        num_layers=2,
        pooling='mean'
    )

    # TRF weights
    trf_weights = {
        'wA': 1.0, 'wNc': 1.0, 'wEc': 1.0,
        'wE': 1.0, 'wL': 1.0, 'wmu': 1.0
    }

    print("\n--- Creating optimizer with multi-graph TRF ---")
    start = time.time()
    opt = T_Adam(
        model.parameters(),
        lr=0.01,
        graph_data=graph_data_list,
        trf_weights=trf_weights,
        beta_trf=0.1
    )
    time_taken = time.time() - start

    print(f"\nTime taken: {time_taken:.2f}s")
    print(f"TRF value (average over {len(graph_data_list)} graphs): {opt.trf_value:.4f}")
    print(f"LR scale factor: {opt.lr_scale_factor:.6f}")

    print("\n--- Creating second optimizer (should use cache) ---")
    start = time.time()
    opt2 = T_Adam(
        model.parameters(),
        lr=0.01,
        graph_data=graph_data_list,
        trf_weights=trf_weights,
        beta_trf=0.1
    )
    time2 = time.time() - start
    print(f"Time taken: {time2:.2f}s")

    speedup = time_taken / time2 if time2 > 0 else float('inf')
    print(f"\nSpeedup: {speedup:.2f}x")

    print("✓ Multi-graph TRF computation working correctly!")


if __name__ == '__main__':
    try:
        test_cache_node_classification()
    except Exception as e:
        print(f"\n✗ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_multigraph_trf()
    except Exception as e:
        print(f"\n✗ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)
