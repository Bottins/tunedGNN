# T_Adam Dual Scaling - Comprehensive Comparison

## 🎯 Quick Start

### Run Full Comparison

```bash
# Interactive mode (recommended for first time)
./run_comparison.sh full

# Quick sanity check (3 min on GPU)
./run_comparison.sh quick

# Node classification only
./run_comparison.sh node

# Graph classification only
./run_comparison.sh graph
```

### Direct Python Usage

```bash
# Full comparison
python Comparison.py

# Node classification only, quick test
python Comparison.py --task node --datasets cora --runs 3 --epochs 50

# Both tasks, thorough study
python Comparison.py --task both --runs 10 --epochs 300
```

---

## 📊 What Gets Tested

### 9 Optimizer Configurations

| # | Configuration | Description |
|---|---------------|-------------|
| 1 | **Adam** | Standard Adam (baseline) |
| 2 | **T_Adam** | T_Adam without scaling |
| 3 | **T_Adam+TRF** | TRF global scaling only |
| 4 | **T_Adam+AntiHub** | Anti-Hub local scaling only |
| 5 | **T_Adam+Homophily** | Homophily local scaling only |
| 6 | **T_Adam+Ricci** | Ricci local scaling only |
| 7 | **T_Adam+TRF+AntiHub** | 🔥 Full dual scaling |
| 8 | **T_Adam+TRF+Homophily** | 🔥 Full dual scaling |
| 9 | **T_Adam+TRF+Ricci** | 🔥 Full dual scaling |

### 6 Datasets (Default)

**Node Classification:**
- Cora (2708 nodes, 7 classes, citation network)
- CiteSeer (3327 nodes, 6 classes, citation network)
- PubMed (19717 nodes, 3 classes, biomedical)

**Graph Classification:**
- MUTAG (188 graphs, molecular compounds)
- PROTEINS (1113 graphs, protein structures)
- ENZYMES (600 graphs, enzyme classes)

### 📈 Output

For each dataset and configuration:
- **Mean test accuracy** (across multiple runs)
- **Standard deviation**
- **Min/Max accuracy**
- **Training time**
- **Statistical significance**

Plus:
- ✅ JSON file with all results
- ✅ Comparison plots (bar charts with error bars)
- ✅ Summary table printed to console

---

## 🚀 Usage Examples

### Example 1: Quick Sanity Check

Test everything works on one dataset:

```bash
./run_comparison.sh quick
```

**Time**: ~3 minutes on GPU, ~15 minutes on CPU

**Output**: Tests all 9 configs on Cora with 3 runs of 50 epochs each

---

### Example 2: Full Node Classification Study

Rigorous comparison on all node datasets:

```bash
./run_comparison.sh node
```

**Time**: ~1-2 hours on GPU, ~8-12 hours on CPU

**Output**: All node datasets (Cora, CiteSeer, PubMed) with 5 runs of 200 epochs

---

### Example 3: Graph Classification Study

```bash
./run_comparison.sh graph
```

**Time**: ~2-3 hours on GPU, ~10-15 hours on CPU

**Output**: All graph datasets (MUTAG, PROTEINS, ENZYMES)

---

### Example 4: Complete Comparison

Everything, everywhere, all at once:

```bash
./run_comparison.sh full
```

**Time**: ~3-5 hours on GPU, ~20-30 hours on CPU

**Output**: All 6 datasets, 9 configs, 5 runs, 200 epochs = 270 experiments!

---

### Example 5: Thorough Research-Grade Study

Maximum rigor with 10 runs and 300 epochs:

```bash
./run_comparison.sh thorough
```

**Time**: ~8-12 hours on GPU, ~40+ hours on CPU

**Output**: 540 experiments with high statistical confidence

---

### Example 6: Custom Configuration

```bash
./run_comparison.sh custom --task node --datasets cora citeseer --runs 5 --epochs 300
```

Or directly:

```bash
python Comparison.py --task node --datasets cora citeseer --runs 5 --epochs 300
```

---

## 📁 File Structure

```
tunedGNN/
├── Comparison.py              # Main comparison script
├── COMPARISON_GUIDE.md        # Detailed usage guide
├── README_COMPARISON.md       # This file
├── run_comparison.sh          # Quick launcher script
├── comparison_results/        # Output directory (auto-created)
│   ├── results_node_*.json
│   ├── results_graph_*.json
│   ├── comparison_node_*.png
│   └── comparison_graph_*.png
└── ...
```

---

## 🔬 Rigor & Reproducibility

### Controlled Variables

✅ **Fixed seeds**: `seed = 42 + run_idx` for each run
✅ **Deterministic operations**: Sets `cudnn.deterministic = True`
✅ **Same initialization**: All optimizers start from identical model
✅ **Multiple runs**: Default 5 runs per config (customizable)
✅ **Statistical analysis**: Mean, std, min, max computed

### Fair Comparison

- ✅ Same hyperparameters for all configs
- ✅ Same model architecture
- ✅ Same number of epochs
- ✅ Same train/val/test splits
- ✅ Same data preprocessing

### What Changes

- ❌ **Only the optimizer** and its topological scaling settings
- ❌ Nothing else!

---

## 📊 Understanding Results

### Console Output Example

```
cora:
----------------------------------------------------------------------------------------------------
Configuration              Test Acc (mean±std)      Best      Worst     Time (s)
----------------------------------------------------------------------------------------------------
T_Adam+TRF+AntiHub        0.8320 ± 0.0065        0.8410    0.8250    16.2 ± 0.8
T_Adam+TRF+Homophily      0.8280 ± 0.0072        0.8380    0.8190    16.5 ± 0.9
T_Adam+TRF+Ricci          0.8240 ± 0.0088        0.8350    0.8140    16.8 ± 1.0
T_Adam+TRF                0.8200 ± 0.0075        0.8300    0.8110    15.8 ± 0.7
T_Adam+AntiHub            0.8180 ± 0.0082        0.8290    0.8070    15.5 ± 0.6
Adam                      0.8150 ± 0.0080        0.8250    0.8050    15.2 ± 0.5
T_Adam                    0.8140 ± 0.0085        0.8240    0.8040    15.3 ± 0.5
...
```

### Interpreting Statistics

**Mean ± Std**: Average performance and variability
- Lower std = more stable/reliable
- Higher mean = better performance

**Best/Worst**: Range of performance across runs
- Narrow range = consistent behavior
- Wide range = high variance

**Time**: Training duration
- Shows computational overhead of topological scaling
- TRF computation adds one-time cost at initialization

### Statistical Significance

A configuration is significantly better if:

```
|mean_A - mean_B| > 2 × sqrt(std_A² + std_B²)
```

Example:
- Config A: 0.8320 ± 0.0065
- Config B: 0.8150 ± 0.0080
- Difference: 0.0170
- Threshold: 2 × sqrt(0.0065² + 0.0080²) = 0.0206

**Conclusion**: Not quite significant (needs more runs or larger difference)

---

## 🎨 Visualization

The script generates bar plots with:

- **One subplot per dataset**
- **Color-coded bars** by configuration type:
  - Gray: Standard Adam
  - Blue: T_Adam variants
  - Orange/Green/Purple/Red: Single scaling modes
  - Dark colors: Full dual scaling
- **Error bars**: Show standard deviation
- **Value labels**: Exact accuracy on each bar
- **Sorted by performance**: Easy to see winners

---

## ⚙️ Configuration Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `both` | Task type: `node`, `graph`, or `both` |
| `--datasets` | All | Specific datasets to test |
| `--runs` | `5` | Number of runs per config |
| `--epochs` | `200` | Training epochs |
| `--device` | `cuda` | Device: `cuda` or `cpu` |
| `--output_dir` | `./comparison_results` | Output directory |

### Quick Reference

```bash
# Task selection
--task node              # Node classification only
--task graph             # Graph classification only
--task both              # Both tasks (default)

# Dataset selection
--datasets cora          # Single dataset
--datasets cora citeseer # Multiple datasets

# Rigor level
--runs 3 --epochs 50     # Quick test
--runs 5 --epochs 200    # Standard (default)
--runs 10 --epochs 300   # Thorough

# Device
--device cpu             # CPU only
--device cuda            # GPU (default if available)

# Output
--output_dir ./my_results  # Custom output directory
```

---

## 🔧 Troubleshooting

### Problem: Out of Memory

**Solution**: Reduce model size (edit `Comparison.py`):
```python
hidden_channels=32  # Instead of 64
num_layers=2        # Instead of 3
```

### Problem: Very Slow Execution

**Solutions**:
1. Use GPU: `--device cuda`
2. Fewer runs: `--runs 3`
3. Fewer epochs: `--epochs 100`
4. Single dataset: `--datasets cora`

### Problem: TRF Warnings

Messages like "Could not compute node connectivity" are **normal** for large graphs. The optimizer uses safe fallback values.

### Problem: Import Errors

**Solution**: Install dependencies:
```bash
pip install torch torch-geometric networkx scipy numpy matplotlib seaborn
```

---

## 📝 Example Complete Workflow

### Step 1: Quick Test

Verify everything works:

```bash
./run_comparison.sh quick
```

Expected time: ~3 min on GPU

### Step 2: Single Dataset Study

Test on your target dataset:

```bash
python Comparison.py --task node --datasets cora --runs 5 --epochs 200
```

Expected time: ~15-20 min on GPU

### Step 3: Full Comparison

Run complete comparison:

```bash
./run_comparison.sh full
```

Expected time: ~3-5 hours on GPU

### Step 4: Analyze Results

Check outputs:

```bash
# View JSON results
cat comparison_results/results_node_*.json

# View plots
open comparison_results/comparison_node_*.png

# Or on Linux
xdg-open comparison_results/comparison_node_*.png
```

---

## 🎓 Expected Outcomes

Based on theoretical foundations:

### Node Classification (Citation Networks)

**Best performers** likely to be:
1. T_Adam+TRF+AntiHub (scale-free structure)
2. T_Adam+TRF+Homophily (high homophily)
3. T_Adam+TRF (stable learning)

**Why**: Citation networks are scale-free with high homophily

### Graph Classification

**Best performers** likely to be:
1. T_Adam+TRF+Ricci (community structure)
2. T_Adam+TRF (complex topologies)
3. T_Adam+Homophily (intra-graph similarity)

**Why**: Molecular/protein graphs have distinct substructures

---

## 📊 Performance Metrics

### Time Estimates (5 runs, 200 epochs)

**Node Classification:**
| Dataset | CPU | GPU |
|---------|-----|-----|
| Cora | ~10 min | ~2 min |
| CiteSeer | ~15 min | ~3 min |
| PubMed | ~100 min | ~15 min |

**Graph Classification:**
| Dataset | CPU | GPU |
|---------|-----|-----|
| MUTAG | ~20 min | ~5 min |
| PROTEINS | ~90 min | ~20 min |
| ENZYMES | ~70 min | ~15 min |

**Full comparison** (9 configs × 6 datasets × 5 runs):
- **GPU**: ~3-5 hours
- **CPU**: ~20-30 hours

---

## 🏆 Best Practices

1. ✅ **Start with quick test**: Verify setup before long runs
2. ✅ **Use GPU if available**: 5-10× speedup
3. ✅ **Monitor first few experiments**: Catch errors early
4. ✅ **Don't interrupt**: Results saved only at end
5. ✅ **Multiple runs matter**: Use ≥5 for reliability
6. ✅ **Document your runs**: Save console output

---

## 🔬 For Researchers

### Publication-Quality Results

For paper submission:

```bash
# Run thorough comparison
python Comparison.py --task both --runs 10 --epochs 300 --output_dir ./paper_results

# This provides:
# - High statistical confidence (10 runs)
# - Converged models (300 epochs)
# - Complete coverage (all datasets)
```

### Reporting Results

Include in paper:
- Mean ± std for all configurations
- Statistical significance tests
- Training time comparison
- Hyperparameters used
- Hardware specifications

### Citation

```bibtex
@software{t_adam_dual_scaling,
  title={T\_Adam: Dual Topological Scaling for Graph Neural Networks},
  author={tunedGNN Project},
  year={2025},
  url={https://github.com/Bottins/tunedGNN}
}
```

---

## 📞 Support

For issues or questions:

1. Check `COMPARISON_GUIDE.md` for detailed documentation
2. Review console output for specific errors
3. Try quick test first: `./run_comparison.sh quick`
4. Verify dependencies are installed

---

## 🎉 Happy Comparing!

This comparison framework provides rigorous, reproducible evaluation of T_Adam's dual topological scaling across diverse graph learning tasks. Good luck with your experiments!
