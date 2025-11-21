# GCN Optimizer Comparison Framework

Un framework per confrontare ottimizzatori su Graph Convolutional Networks (GCN), con focus sul confronto tra Adam standard e varianti personalizzate.

## 🎯 Obiettivo

Questo progetto permette di confrontare l'ottimizzatore **Adam standard** con **T_Adam** (un ottimizzatore Adam personalizzabile) su task di:
- **Node Classification** (classificazione nodi su singolo grafo)
- **Graph Classification** (classificazione di grafi multipli)

## 📁 Struttura del Progetto

```
tunedGNN/
│
├── models/                  # Architetture GNN
│   ├── __init__.py
│   └── gcn.py              # Graph Convolutional Network
│
├── optimizers/              # Ottimizzatori
│   ├── __init__.py
│   └── t_adam.py           # T_Adam: Ottimizzatore Adam personalizzabile
│
├── datasets/                # Dataset loaders
│   ├── __init__.py
│   └── loaders.py          # Caricamento dataset (node + graph classification)
│
├── utils/                   # Utilities
│   ├── __init__.py
│   ├── metrics.py          # Metriche e confronto ottimizzatori
│   └── logger.py           # Logging risultati
│
├── main.py                  # Script principale per esperimenti
├── requirements.txt         # Dipendenze Python
├── .gitignore
└── README.md
```

## 🚀 Setup

### Installazione Dipendenze

```bash
pip install torch torch-geometric
pip install -r requirements.txt
```

### Verifica CUDA (per RTX su WSL)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
```

## 📊 Dataset Supportati

### Node Classification (Single Graph)

**Small:**
- `cora`, `citeseer`, `pubmed`

**Medium:**
- `amazon-photo`, `amazon-computer`
- `coauthor-cs`, `coauthor-physics`
- `wikics`
- `chameleon`, `squirrel`

**Heterophilous:**
- `roman-empire`, `amazon-ratings`, `minesweeper`, `tolokers`, `questions`

**Large:**
- `ogbn-arxiv`, `ogbn-products`, `ogbn-proteins`

### Graph Classification (Multiple Graphs)

**Bioinformatics:**
- `MUTAG`, `PROTEINS`, `DD`, `NCI1`, `ENZYMES`, `PTC_MR`

**Social Networks:**
- `REDDIT-BINARY`, `COLLAB`, `IMDB-BINARY`, `IMDB-MULTI`

**Molecules (OGB):**
- `ogbg-molhiv`, `ogbg-molpcba`

## 🏃 Utilizzo

### Esempio Base: Node Classification

```bash
# Confronta Adam vs T_Adam su Cora (CPU)
python main.py --dataset cora --task node --device cpu

# Usa GPU (RTX su WSL)
python main.py --dataset cora --task node --device cuda

# Più epoche e runs per risultati robusti
python main.py --dataset cora --task node --epochs 500 --runs 10 --device cuda
```

### Esempio: Graph Classification

```bash
# Confronta ottimizzatori su MUTAG
python main.py --dataset MUTAG --task graph --device cuda

# Usa pooling diverso
python main.py --dataset PROTEINS --task graph --pooling add --device cuda
```

### Parametri Principali

```bash
python main.py \
  --dataset cora \                    # Nome dataset
  --task node \                        # node o graph
  --device cuda \                      # cuda o cpu
  --hidden_channels 64 \              # Dimensione hidden layer
  --num_layers 3 \                    # Numero layer GCN
  --dropout 0.5 \                     # Dropout rate
  --epochs 200 \                      # Numero epoche
  --lr 0.01 \                         # Learning rate
  --weight_decay 5e-4 \               # Weight decay (L2 regularization)
  --runs 5 \                          # Numero di run per media
  --optimizers Adam T_Adam \          # Ottimizzatori da confrontare
  --batch_norm \                      # Usa batch normalization
  --residual \                        # Usa connessioni residuali
  --save_plot comparison.png \        # Salva plot confronto
  --save_results results.json         # Salva risultati JSON
```

## 🔧 Modifica T_Adam

Il file `optimizers/t_adam.py` contiene l'ottimizzatore **T_Adam** che puoi modificare liberamente.

### Come Modificare

1. Apri `optimizers/t_adam.py`
2. Modifica il metodo `_t_adam_update()` (linee 143-192)
3. Esempi di modifiche:
   - Cambia i coefficienti beta1, beta2
   - Modifica la bias correction
   - Aggiungi warmup del learning rate
   - Implementa gradient clipping
   - Sperimenta con adaptive learning rates

### Esempio di Modifica: Gradient Clipping

```python
# In _t_adam_update(), prima dell'update dei parametri:

# Clip gradients
max_grad_norm = 1.0
grad_norm = grad.norm(2)
if grad_norm > max_grad_norm:
    grad = grad * (max_grad_norm / grad_norm)
```

### Esempio: Warmup Learning Rate

```python
# In step(), prima del loop:
warmup_steps = 1000
if state_steps[0] < warmup_steps:
    warmup_factor = state_steps[0] / warmup_steps
    lr = group['lr'] * warmup_factor
else:
    lr = group['lr']
```

## 📈 Output e Metriche

### Metriche Tracciate

- **Loss** (train, validation, test)
- **Accuracy** (train, validation, test)
- **F1 Score** (per multi-class)
- **Gradient Norms** (per ogni epoca)
- **Tempo di Training** (totale e per epoca)

### Visualizzazioni

Il programma genera automaticamente:

1. **Plot di confronto** (`optimizer_comparison.png`):
   - Training/Validation Loss
   - Training/Validation Accuracy
   - Gradient Norms
   - Convergence Speed (Accuracy vs Time)

2. **Risultati JSON** (`optimizer_results.json`):
   - History completa di training
   - Risultati finali per ogni ottimizzatore

### Esempio Output

```
================================================================================
OPTIMIZER COMPARISON SUMMARY
================================================================================

Optimizer            Test Acc     Test Loss    Time (s)     Best Epoch
--------------------------------------------------------------------------------
Adam                 0.8150       0.5234       45.67        142
T_Adam               0.8320       0.4891       46.23        138

--------------------------------------------------------------------------------
BEST OPTIMIZER: T_Adam (Test Accuracy: 0.8320)
================================================================================
```

## 🧪 Esperimenti Suggeriti

### 1. Confronto Base
```bash
python main.py --dataset cora --task node --device cuda
```

### 2. Test su Dataset Grande
```bash
python main.py --dataset ogbn-arxiv --task node --hidden_channels 128 --num_layers 3 --device cuda
```

### 3. Graph Classification
```bash
python main.py --dataset MUTAG --task graph --batch_size 64 --device cuda
```

### 4. Ablation Study (Normalization)
```bash
# Senza normalizzazione
python main.py --dataset cora --task node --device cuda

# Con batch normalization
python main.py --dataset cora --task node --batch_norm --device cuda

# Con layer normalization
python main.py --dataset cora --task node --layer_norm --device cuda
```

### 5. Confronto Multi-Dataset

Crea uno script per testare su più dataset:

```bash
#!/bin/bash
for dataset in cora citeseer pubmed; do
    python main.py --dataset $dataset --task node --device cuda \
      --save_plot ${dataset}_comparison.png \
      --save_results ${dataset}_results.json
done
```

## 🎓 Background: Perché Confrontare Ottimizzatori?

### Adam Optimizer

Adam (Adaptive Moment Estimation) è uno degli ottimizzatori più usati per deep learning:
- Combina momentum (RMSprop) con adaptive learning rates
- Mantiene medie mobili di gradienti (primo momento) e gradienti al quadrato (secondo momento)
- Funziona bene out-of-the-box su molti problemi

### Motivazione per T_Adam

GNN hanno caratteristiche uniche:
- **Graph structure**: La topologia del grafo influenza i gradienti
- **Over-smoothing**: Layer profondi tendono a convergere verso stesse rappresentazioni
- **Heterophily**: Nodi simili possono non essere connessi

Modificare Adam può:
- Migliorare convergenza su grafi specifici
- Ridurre over-smoothing
- Adattarsi meglio a grafi eterogenei

## 🛠️ Tips per WSL + RTX

### Verifica Driver NVIDIA

```bash
nvidia-smi
```

### Se CUDA Non Funziona

1. Installa CUDA toolkit per WSL:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
   sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /"
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

2. Reinstalla PyTorch con CUDA:
   ```bash
   pip uninstall torch torch-geometric
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   pip install torch-geometric
   ```

## 📚 Riferimenti

- **GCN**: Kipf & Welling (2017) - *Semi-Supervised Classification with Graph Convolutional Networks*
- **Adam**: Kingma & Ba (2015) - *Adam: A Method for Stochastic Optimization*
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

## 📝 License

MIT License - Vedi file LICENSE

## 🤝 Contributi

Questo è un framework di ricerca personale. Sentiti libero di:
- Modificare T_Adam con le tue idee
- Aggiungere nuovi dataset
- Sperimentare con architetture diverse
- Creare pull request con miglioramenti

---

**Buona sperimentazione! 🚀**
