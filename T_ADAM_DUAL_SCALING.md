# T_Adam: Dual Topological Scaling Optimizer

## Overview

T_Adam is a custom Adam optimizer that implements **dual topological scaling** for Graph Neural Networks:

1. **Global Scaling**: Learning rate adjusted by the Topological Relevance Function (TRF)
2. **Local Scaling**: Per-node gradient weighting based on topological properties

## Architecture

### 1. Global Scaling: TRF-based Learning Rate Adaptation

The Topological Relevance Function (TRF) quantifies graph complexity:

```
TRF(G) = 1 + μG × [wA×(n-A)/A + wNc×(n-1-Nc)/Nc + wEc×(n-1-Ec)/Ec + wE×(1-E)/E + wL×(L-1)]
```

Where:
- **μG = wμ × m/n**: Edge-to-node ratio multiplier
- **A**: Algebraic connectivity (2nd smallest eigenvalue of normalized Laplacian)
- **Nc**: Node connectivity (min nodes to disconnect graph)
- **Ec**: Edge connectivity (min edges to disconnect graph)
- **E**: Global efficiency (average inverse shortest path length)
- **L**: Average path length

The learning rate is then adjusted as:
```
lr_adjusted = lr_base × exp(-β × TRF(G))
```

Where **β** (default 0.1) controls the scaling intensity.

**Intuition**: Higher TRF indicates more complex topology → reduce learning rate for stability.

---

### 2. Local Scaling: Per-Node Gradient Weighting

The gradient update becomes:
```
g_t = Σ_v∈V α(v) × ∇L(v)
```

Where **α(v)** is a topological weight for each node, computed using one of three modes:

#### Mode 1: Anti-Hub Dominance (`anti_hub`)

```
α(v) = 1 / log(degree(v) + ε)
```

- **Purpose**: Penalize gradients from hub nodes
- **Effect**: Emphasize peripheral nodes, reduce over-smoothing
- **Best for**: Scale-free networks with dominant hubs

**Rationale**: In scale-free networks, hubs can dominate gradient flow. This mode rebalances by giving more weight to low-degree nodes.

#### Mode 2: Homophily-Based Weighting (`homophily`)

```
α(v) = 1 + tanh(H_v)
```

Where **H_v** is the mean cosine similarity between node v's features and its neighbors' features.

- **Purpose**: Accelerate learning in homogeneous regions
- **Effect**: Faster convergence where nodes are similar, slower in heterophilic regions
- **Best for**: Graphs with mixed homophily patterns

**Rationale**: Homophilic regions are easier to learn → increase their gradient weight. Heterophilic regions need more careful updates → reduce their weight.

#### Mode 3: Ricci Curvature (`ricci`)

```
α(v) = exp(-κ_v)
```

Where **κ_v** is the Ollivier-Ricci curvature (approximated via neighborhood overlap).

- **Purpose**: Emphasize bridge nodes between communities
- **Effect**: Higher weight on nodes connecting different clusters
- **Best for**: Community-structured graphs

**Rationale**: Bridge nodes are critical for inter-community learning. Positive curvature (high overlap) → lower weight. Negative curvature (bridge position) → higher weight.

---

## Usage

### Basic Usage (Global Scaling Only)

```python
from optimizers.t_adam import T_Adam

# Prepare graph data
graph_data = {
    'edge_index': data.edge_index,  # [2, num_edges]
    'num_nodes': data.num_nodes
}

# Initialize optimizer with TRF-based global scaling
optimizer = T_Adam(
    model.parameters(),
    lr=0.01,
    graph_data=graph_data,
    beta_trf=0.1  # TRF scaling parameter
)
```

### Advanced Usage (Dual Scaling)

```python
from optimizers.t_adam import T_Adam

# Prepare graph data with node features (needed for homophily mode)
graph_data = {
    'edge_index': data.edge_index,
    'num_nodes': data.num_nodes,
    'node_features': data.x  # [num_nodes, num_features]
}

# Custom TRF weights
trf_weights = {
    'wA': 1.0,    # Algebraic connectivity
    'wNc': 1.0,   # Node connectivity
    'wEc': 1.0,   # Edge connectivity
    'wE': 1.0,    # Global efficiency
    'wL': 1.0,    # Average path length
    'wmu': 1.0    # mu_G multiplier
}

# Initialize with dual scaling
optimizer = T_Adam(
    model.parameters(),
    lr=0.01,
    graph_data=graph_data,
    trf_weights=trf_weights,
    beta_trf=0.1,
    gradient_scaling_mode='anti_hub',  # or 'homophily' or 'ricci'
    node_grad_indices=[0]  # Indices of params to apply local scaling
)
```

### Parameter Guidelines

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lr` | 0.001 | [1e-5, 1e-1] | Base learning rate |
| `beta_trf` | 0.1 | [0.01, 1.0] | TRF scaling intensity (higher = stronger damping) |
| `wA, wNc, wEc, wE, wL, wmu` | 1.0 | [0.0, 5.0] | TRF component weights |
| `gradient_scaling_mode` | None | {None, 'anti_hub', 'homophily', 'ricci'} | Local scaling strategy |

---

## When to Use Each Mode

| Graph Property | Recommended Mode | Reason |
|----------------|------------------|--------|
| Scale-free (power-law degree) | `anti_hub` | Prevent hub dominance |
| High homophily | `homophily` | Accelerate in homogeneous regions |
| Community structure | `ricci` | Emphasize inter-community links |
| Small-world | `homophily` or `ricci` | Balance local/global structure |
| Dense graph | Global only | Local scaling less critical |
| Sparse graph | `anti_hub` | Prevent over-smoothing |

---

## Example: Running Experiments

```bash
# Compare all modes on Cora dataset
python example_t_adam_dual_scaling.py compare

# Train with specific mode
python example_t_adam_dual_scaling.py anti_hub
python example_t_adam_dual_scaling.py homophily
python example_t_adam_dual_scaling.py ricci

# Train with standard Adam (no topological scaling)
python example_t_adam_dual_scaling.py standard
```

---

## Implementation Details

### TRF Computation

- **One-time calculation**: TRF is computed once during optimizer initialization
- **Fallbacks**: For large graphs (>500 nodes), uses approximations for connectivity metrics
- **Efficiency optimizations**:
  - Algebraic connectivity: Lanczos for <1000 nodes, sparse eigsh for larger
  - Connectivity: Degree-based approximation for large graphs
  - Average path length: Computed on largest connected component

### Local Gradient Scaling

- **Applied per-parameter**: Only affects parameters specified in `node_grad_indices`
- **Automatic reshaping**: Handles both 1D and 2D gradient tensors
- **Device-aware**: Automatically moves weights to gradient device

### Computational Complexity

| Component | Complexity | Notes |
|-----------|------------|-------|
| TRF calculation | O(n² + n log n) | One-time at initialization |
| Anti-hub weights | O(n) | Linear in nodes |
| Homophily weights | O(n × d) | d = avg degree |
| Ricci weights | O(n × d²) | Most expensive |
| Per-step overhead | O(1) | Negligible after initialization |

---

## Theory and Intuition

### Why TRF-based Global Scaling?

Graphs with high TRF are topologically complex (low connectivity, high path length, low efficiency).
Such graphs require more careful optimization to avoid:
- **Over-smoothing**: Features become indistinguishable
- **Gradient explosion**: Long paths amplify gradients
- **Unstable convergence**: Complex topology makes loss landscape rugged

By reducing the learning rate proportionally to TRF, we ensure stable learning on complex graphs.

### Why Per-Node Gradient Weighting?

Standard Adam treats all gradient components equally. But in graphs:
- **Hub nodes** contribute disproportionately due to high degree
- **Homophilic regions** learn faster (nodes are similar)
- **Bridge nodes** are critical for cross-community learning

By reweighting gradients based on topology, we:
- Balance hub vs. peripheral contributions
- Adapt update speed to local graph structure
- Prioritize structurally important nodes

---

## Hyperparameter Tuning Guide

### Start with Defaults

```python
optimizer = T_Adam(
    model.parameters(),
    lr=0.01,
    graph_data=graph_data,
    beta_trf=0.1,
    gradient_scaling_mode='anti_hub'
)
```

### If Training is Unstable

1. **Increase β_trf** (0.1 → 0.5): Stronger TRF damping
2. **Reduce lr** (0.01 → 0.001): Lower base learning rate
3. **Try different scaling mode**: e.g., switch to 'homophily'

### If Training is Too Slow

1. **Decrease β_trf** (0.1 → 0.01): Weaker TRF damping
2. **Increase lr** (0.01 → 0.05): Higher base learning rate
3. **Disable local scaling**: Set `gradient_scaling_mode=None`

### If Overfitting

1. **Increase weight_decay**: Add L2 regularization
2. **Try 'ricci' mode**: Emphasizes inter-community generalization
3. **Increase TRF component weights**: Make TRF more sensitive

### If Underfitting

1. **Increase lr**: Allow faster learning
2. **Decrease β_trf**: Reduce TRF damping
3. **Try 'homophily' mode**: Accelerate in easy regions

---

## Citation

If you use T_Adam in your research, please cite:

```bibtex
@software{t_adam_dual_scaling,
  title={T\_Adam: Dual Topological Scaling for Graph Neural Networks},
  author={tunedGNN Project},
  year={2025},
  url={https://github.com/Bottins/tunedGNN}
}
```

---

## References

- Kingma & Ba (2015): Adam optimizer
- Chung (1997): Spectral Graph Theory (algebraic connectivity)
- Latora & Marchiori (2001): Efficient Behavior of Small-World Networks (global efficiency)
- Ollivier (2009): Ricci Curvature of Metric Spaces (Ricci curvature on graphs)
- Zhu et al. (2020): Beyond Homophily in Graph Neural Networks (homophily in GNNs)
