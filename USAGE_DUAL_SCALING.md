# T_Adam Dual Scaling - Usage Guide

## Quick Start

### 1. Standard Comparison (Baseline)

Compare standard Adam vs T_Adam without any topological scaling:

```bash
python main.py --dataset cora --task node
```

---

## Local Gradient Scaling Only

Choose one of three local gradient scaling modes without global TRF scaling:

### 2. Anti-Hub Dominance Mode

**Formula**: `α(v) = 1/log(degree(v) + ε)`

**Best for**: Scale-free networks with dominant hubs (e.g., social networks, citation networks)

**Effect**: Penalizes hub node gradients, emphasizes peripheral nodes

```bash
python main.py --dataset cora --task node --gradient_scaling anti_hub
```

### 3. Homophily-Based Mode

**Formula**: `α(v) = 1 + tanh(H_v)` where H_v is cosine similarity with neighbors

**Best for**: Networks with mixed homophily patterns

**Effect**: Accelerates learning in homogeneous regions, slows down in heterophilic areas

```bash
python main.py --dataset cora --task node --gradient_scaling homophily
```

### 4. Ricci Curvature Mode

**Formula**: `α(v) = exp(-κ_v)` where κ_v is approximate Ollivier-Ricci curvature

**Best for**: Community-structured graphs

**Effect**: Emphasizes bridge nodes connecting different communities

```bash
python main.py --dataset cora --task node --gradient_scaling ricci
```

---

## Global TRF Scaling Only

Enable TRF-based learning rate adaptation without local gradient scaling:

```bash
python main.py --dataset cora --task node --use_trf
```

**Effect**: Learning rate adjusted as `lr_adjusted = lr × exp(-β × TRF(G))`

---

## Full Dual Scaling (Recommended)

Combine both global TRF scaling and local gradient scaling:

### Anti-Hub + TRF
```bash
python main.py --dataset cora --task node --use_trf --gradient_scaling anti_hub
```

### Homophily + TRF
```bash
python main.py --dataset cora --task node --use_trf --gradient_scaling homophily
```

### Ricci + TRF
```bash
python main.py --dataset cora --task node --use_trf --gradient_scaling ricci
```

---

## Advanced Options

### Custom TRF Beta Parameter

Control the intensity of TRF-based learning rate damping (default: 0.1):

```bash
# Stronger damping
python main.py --dataset cora --task node --use_trf --beta_trf 0.5

# Weaker damping
python main.py --dataset cora --task node --use_trf --beta_trf 0.01
```

### Custom TRF Component Weights

Adjust weights for TRF calculation (order: wA wNc wEc wE wL wμ):

```bash
# Emphasize algebraic connectivity and efficiency
python main.py --dataset cora --task node --use_trf --trf_weights 2.0 1.0 1.0 2.0 1.0 1.0

# Equal weighting (default)
python main.py --dataset cora --task node --use_trf --trf_weights 1.0 1.0 1.0 1.0 1.0 1.0
```

### Apply Gradient Scaling to Different Layers

Choose which layer receives local gradient scaling (default: 0 = first layer):

```bash
# Apply to first layer (default)
python main.py --dataset cora --task node --gradient_scaling anti_hub --node_grad_layer 0

# Apply to second layer
python main.py --dataset cora --task node --gradient_scaling anti_hub --node_grad_layer 1
```

### Other Useful Options

```bash
# Change learning rate
python main.py --dataset cora --task node --gradient_scaling anti_hub --lr 0.001

# More epochs
python main.py --dataset cora --task node --gradient_scaling anti_hub --epochs 500

# Use GPU
python main.py --dataset cora --task node --gradient_scaling anti_hub --device cuda

# Different dataset
python main.py --dataset citeseer --task node --gradient_scaling homophily
```

---

## Quick Test Script

Run all modes sequentially to compare results:

```bash
# Test on Cora with 50 epochs each
./test_dual_scaling.sh cora 50

# Test on CiteSeer with 100 epochs each
./test_dual_scaling.sh citeseer 100
```

---

## Comparing Multiple Configurations

To compare different configurations, run them sequentially and check the saved results:

```bash
# Run 1: Baseline
python main.py --dataset cora --task node --save_results results_baseline.json --save_plot plot_baseline.png

# Run 2: Anti-Hub
python main.py --dataset cora --task node --gradient_scaling anti_hub --save_results results_antihub.json --save_plot plot_antihub.png

# Run 3: Full Dual Scaling
python main.py --dataset cora --task node --use_trf --gradient_scaling anti_hub --save_results results_dual.json --save_plot plot_dual.png
```

Results will be saved to JSON files and plots will be generated.

---

## Troubleshooting

### "Could not compute [metric]" warnings

These are normal for very large graphs. The optimizer will use safe fallback values.

### Out of memory

Try reducing:
- `--hidden_channels` (default: 64)
- `--num_layers` (default: 3)
- `--batch_size` (for graph classification, default: 32)

### Training is unstable

Try:
- Increasing `--beta_trf` to strengthen TRF damping
- Reducing `--lr`
- Different gradient scaling mode

### Training is too slow

Try:
- Decreasing `--beta_trf`
- Disabling TRF: remove `--use_trf`
- Simpler scaling mode or disable: remove `--gradient_scaling`

---

## Example Workflows

### Workflow 1: Find Best Local Scaling Mode

```bash
# Test all three modes
python main.py --dataset cora --task node --gradient_scaling anti_hub --save_results anti_hub.json
python main.py --dataset cora --task node --gradient_scaling homophily --save_results homophily.json
python main.py --dataset cora --task node --gradient_scaling ricci --save_results ricci.json

# Compare results in the JSON files
```

### Workflow 2: Hyperparameter Tuning

```bash
# Test different beta_trf values
for beta in 0.01 0.05 0.1 0.2 0.5; do
  python main.py --dataset cora --task node --use_trf --beta_trf $beta --save_results results_beta_${beta}.json
done
```

### Workflow 3: Dataset Comparison

```bash
# Test on multiple datasets
for dataset in cora citeseer pubmed; do
  python main.py --dataset $dataset --task node --use_trf --gradient_scaling anti_hub --save_results results_${dataset}.json
done
```

---

## Summary Table

| Command Flag | Options | Default | Description |
|--------------|---------|---------|-------------|
| `--use_trf` | flag | False | Enable TRF-based global LR scaling |
| `--gradient_scaling` | anti_hub, homophily, ricci | None | Local gradient scaling mode |
| `--beta_trf` | float | 0.1 | TRF scaling intensity |
| `--trf_weights` | 6 floats | [1.0]*6 | TRF component weights (wA, wNc, wEc, wE, wL, wμ) |
| `--node_grad_layer` | int | 0 | Layer index for local scaling |

---

## Citation

If you use T_Adam dual scaling in your research, please cite:

```bibtex
@software{t_adam_dual_scaling,
  title={T\_Adam: Dual Topological Scaling for Graph Neural Networks},
  author={tunedGNN Project},
  year={2025},
  url={https://github.com/Bottins/tunedGNN}
}
```
