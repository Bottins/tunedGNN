# Comprehensive T_Adam Comparison Guide

## Overview

`Comparison.py` performs a rigorous, systematic comparison of all T_Adam configurations across multiple datasets with multiple runs and controlled seeds.

## What It Tests

### Optimizer Configurations (9 total)

1. **Adam** - Standard Adam optimizer (baseline)
2. **T_Adam** - T_Adam without any topological scaling
3. **T_Adam+TRF** - T_Adam with TRF-based global LR scaling only
4. **T_Adam+AntiHub** - T_Adam with Anti-Hub local gradient scaling only
5. **T_Adam+Homophily** - T_Adam with Homophily-based local scaling only
6. **T_Adam+Ricci** - T_Adam with Ricci curvature local scaling only
7. **T_Adam+TRF+AntiHub** - Full dual scaling (TRF + Anti-Hub)
8. **T_Adam+TRF+Homophily** - Full dual scaling (TRF + Homophily)
9. **T_Adam+TRF+Ricci** - Full dual scaling (TRF + Ricci)

### Datasets

**Node Classification:**
- Cora (2708 nodes, 7 classes)
- CiteSeer (3327 nodes, 6 classes)
- PubMed (19717 nodes, 3 classes)

**Graph Classification:**
- MUTAG (188 graphs, 2 classes)
- PROTEINS (1113 graphs, 2 classes)
- ENZYMES (600 graphs, 6 classes)

### Rigor & Reproducibility

- **Multiple runs**: Default 5 runs per configuration
- **Controlled seeds**: `seed = 42 + run_idx` for each run
- **Deterministic**: Sets `torch.backends.cudnn.deterministic = True`
- **Same initialization**: Each optimizer starts from the same model initialization
- **Statistical analysis**: Computes mean, std, min, max across runs

---

## Usage

### Basic Usage

Run comparison for both node and graph classification:

```bash
python Comparison.py
```

This will:
- Test all 9 configurations
- On 3 node classification datasets (Cora, CiteSeer, PubMed)
- On 3 graph classification datasets (MUTAG, PROTEINS, ENZYMES)
- With 5 runs each (270 total experiments!)
- Save results to `./comparison_results/`

---

### Quick Tests

#### Node Classification Only

```bash
python Comparison.py --task node
```

#### Graph Classification Only

```bash
python Comparison.py --task graph
```

#### Single Dataset

```bash
# Test only on Cora
python Comparison.py --task node --datasets cora

# Test only on MUTAG
python Comparison.py --task graph --datasets MUTAG
```

#### Multiple Specific Datasets

```bash
# Test on Cora and CiteSeer only
python Comparison.py --task node --datasets cora citeseer
```

---

### Advanced Options

#### More Runs for Better Statistics

```bash
# 10 runs per configuration
python Comparison.py --runs 10
```

#### Fewer Epochs for Quick Testing

```bash
# Only 50 epochs (for debugging)
python Comparison.py --epochs 50
```

#### CPU Only

```bash
python Comparison.py --device cpu
```

#### Custom Output Directory

```bash
python Comparison.py --output_dir ./my_results
```

---

## Example Workflows

### Workflow 1: Quick Sanity Check

Test on one dataset with fewer epochs:

```bash
python Comparison.py --task node --datasets cora --runs 3 --epochs 50
```

### Workflow 2: Thorough Node Classification Study

Test all node datasets with many runs:

```bash
python Comparison.py --task node --runs 10 --epochs 300
```

### Workflow 3: Graph Classification Focus

Test graph datasets with standard settings:

```bash
python Comparison.py --task graph --runs 5 --epochs 200
```

### Workflow 4: Full Comprehensive Study

Test everything with high rigor:

```bash
python Comparison.py --task both --runs 10 --epochs 300
```

**Warning**: This will run **540 experiments** (9 configs × 6 datasets × 10 runs) and take many hours!

---

## Output Files

### Results JSON

Saved to: `./comparison_results/results_{task}_{timestamp}.json`

Contains:
- **statistics**: Mean, std, min, max for each configuration
- **all_results**: Individual results for each run
- **timestamp**: When the comparison was run

Example structure:
```json
{
  "statistics": {
    "cora": {
      "Adam": {
        "test_acc_mean": 0.8150,
        "test_acc_std": 0.0080,
        "test_acc_min": 0.8050,
        "test_acc_max": 0.8250,
        "time_mean": 15.2,
        "time_std": 0.5,
        "num_runs": 5
      },
      "T_Adam+TRF+AntiHub": {
        "test_acc_mean": 0.8320,
        "test_acc_std": 0.0065,
        ...
      }
    }
  }
}
```

### Comparison Plot

Saved to: `./comparison_results/comparison_{task}_{timestamp}.png`

Features:
- One subplot per dataset
- Bar plot with error bars (std)
- Color-coded by configuration type
- Value labels on each bar

**Color Scheme:**
- **Gray**: Standard Adam
- **Steel Blue**: T_Adam (no scaling)
- **Orange**: TRF only
- **Green**: Anti-Hub only
- **Purple**: Homophily only
- **Red**: Ricci only
- **Dark Green**: TRF + Anti-Hub
- **Dark Violet**: TRF + Homophily
- **Dark Red**: TRF + Ricci

---

## Understanding Results

### Console Output

During execution, you'll see:

1. **Progress tracking**: Current experiment number and percentage
2. **Per-epoch updates**: Every 50 epochs, shows loss and accuracy
3. **Per-run summary**: Best validation and test accuracy
4. **Final statistics table**: Sorted by test accuracy

Example output:
```
cora:
----------------------------------------------------------------------------------------------------
Configuration              Test Acc (mean±std)      Best      Worst     Time (s)
----------------------------------------------------------------------------------------------------
T_Adam+TRF+AntiHub        0.8320 ± 0.0065        0.8410    0.8250    16.2 ± 0.8
T_Adam+TRF+Homophily      0.8280 ± 0.0072        0.8380    0.8190    16.5 ± 0.9
Adam                      0.8150 ± 0.0080        0.8250    0.8050    15.2 ± 0.5
...
```

### Statistical Significance

With 5+ runs, you can assess:
- **Mean difference**: Is one config consistently better?
- **Standard deviation**: How stable is the config?
- **Min/Max range**: What's the performance variability?

**Rule of thumb**: A difference is likely significant if:
```
|mean_A - mean_B| > 2 * sqrt(std_A^2 + std_B^2)
```

---

## Interpreting Configurations

### Expected Patterns

Based on theory:

1. **T_Adam+TRF+AntiHub** should excel on:
   - Scale-free networks (Cora, CiteSeer)
   - Networks with hub dominance

2. **T_Adam+TRF+Homophily** should excel on:
   - Homophilic networks (most citation networks)
   - Balanced connectivity

3. **T_Adam+TRF+Ricci** should excel on:
   - Community-structured graphs
   - Networks with clear clustering

4. **TRF-only** configurations should provide:
   - Stable learning on complex topologies
   - Better than standard T_Adam on irregular graphs

5. **Local-scaling-only** configurations should show:
   - Benefits from gradient reweighting
   - Less impact than full dual scaling

---

## Troubleshooting

### Out of Memory

Reduce model size or batch size (edit defaults in script):
```python
hidden_channels=32  # Instead of 64
num_layers=2       # Instead of 3
batch_size=16      # Instead of 32
```

### Very Slow Execution

Options:
1. Reduce runs: `--runs 3`
2. Reduce epochs: `--epochs 100`
3. Test fewer datasets: `--datasets cora`
4. Use GPU: `--device cuda`

### TRF Computation Warnings

Warnings like "Could not compute node connectivity" are normal for large graphs. The optimizer uses safe fallback values.

### Import Errors

Ensure all dependencies are installed:
```bash
pip install torch torch-geometric networkx scipy numpy matplotlib seaborn
```

---

## Customization

### Add Your Own Configuration

Edit `OPTIMIZER_CONFIGS` in `Comparison.py`:

```python
OPTIMIZER_CONFIGS.append({
    'name': 'T_Adam+TRF+Custom',
    'type': 'T_Adam',
    'use_trf': True,
    'gradient_scaling': 'anti_hub',
    # Add custom parameters here
})
```

### Test Your Own Dataset

Add dataset name to the lists:

```python
NODE_DATASETS = ['cora', 'citeseer', 'pubmed', 'your_dataset']
```

Make sure your dataset is compatible with the `load_dataset()` function.

### Change Hyperparameters

Modify default values in `run_single_experiment()`:
```python
lr=0.005          # Lower learning rate
hidden_channels=128  # Larger model
dropout=0.3       # Less dropout
```

---

## Performance Expectations

### Time Estimates (per configuration)

**Node Classification (200 epochs, 5 runs):**
- Cora: ~1-2 min (CPU), ~30 sec (GPU)
- CiteSeer: ~2-3 min (CPU), ~45 sec (GPU)
- PubMed: ~15-20 min (CPU), ~3-5 min (GPU)

**Graph Classification (200 epochs, 5 runs):**
- MUTAG: ~3-5 min (CPU), ~1-2 min (GPU)
- PROTEINS: ~15-20 min (CPU), ~5-8 min (GPU)
- ENZYMES: ~10-15 min (CPU), ~3-5 min (GPU)

**Total for full comparison** (9 configs × 6 datasets × 5 runs):
- **CPU**: ~20-30 hours
- **GPU**: ~3-5 hours

### Memory Requirements

- **Node Classification**: 2-4 GB RAM, 1-2 GB VRAM
- **Graph Classification**: 4-8 GB RAM, 2-4 GB VRAM

---

## Best Practices

1. **Start small**: Test on one dataset first
   ```bash
   python Comparison.py --task node --datasets cora --runs 3 --epochs 50
   ```

2. **Use GPU**: Dramatically faster
   ```bash
   python Comparison.py --device cuda
   ```

3. **Save intermediate results**: The script saves after completion, so don't interrupt!

4. **Monitor progress**: Watch the console output to track progress

5. **Multiple runs matter**: Use at least 5 runs for reliable statistics

6. **Compare fairly**: Same hyperparameters for all configs

---

## Citation

If you use this comparison framework in your research:

```bibtex
@software{t_adam_comparison,
  title={Comprehensive T\_Adam Dual Scaling Comparison Framework},
  author={tunedGNN Project},
  year={2025},
  url={https://github.com/Bottins/tunedGNN}
}
```

---

## Quick Reference

```bash
# Full comparison (both tasks, all datasets)
python Comparison.py

# Node classification only
python Comparison.py --task node

# Graph classification only
python Comparison.py --task graph

# Single dataset, quick test
python Comparison.py --task node --datasets cora --runs 3 --epochs 50

# Thorough study with 10 runs
python Comparison.py --runs 10 --epochs 300

# CPU only
python Comparison.py --device cpu

# Custom output directory
python Comparison.py --output_dir ./my_results
```
