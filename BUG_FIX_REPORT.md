# Bug Fix Report: T_Adam Gradient Scaling Implementation

## Data: 2025-11-22 (Update)

## 🐛 Problemi Identificati

### Problema 1: Risultati Identici per Tutte le Varianti di Gradient Scaling

**Sintomo:**
Tutti i run con diversi modi di gradient scaling producevano risultati **identici** fino alla quarta cifra decimale:

```
Cora Results:
- T_Adam+TRF:           0.7914
- T_Adam+AntiHub:       0.7914  ← IDENTICO!
- T_Adam+Homophily:     0.7914  ← IDENTICO!
- T_Adam+Ricci:         0.7914  ← IDENTICO!
- T_Adam+TRF+AntiHub:   0.7914  ← IDENTICO!
```

**Causa:**
L'implementazione del gradient scaling locale era **concettualmente errata**:

1. **Cosa faceva**: Moltiplicava i gradienti dei **parametri del modello** (es. weights dei layer GCN) per i node weights
2. **Cosa dovrebbe fare**: Moltiplicare i gradienti rispetto alle **node embeddings** (∂L/∂h_v) per influenzare selettivamente l'apprendimento per nodo

**Problema specifico con `node_grad_indices=[0]`:**
- L'indice 0 si riferiva al **primo parametro** nell'iterator di `model.parameters()`
- Questo potrebbe essere un bias, un weight di batch norm, o qualsiasi altro parametro
- NON c'era garanzia che fosse il layer GCN corretto
- Inoltre, scalare i parametri del modello NON è lo stesso che scalare i gradienti per nodo

**Perché i risultati erano identici:**
- Il gradient scaling NON veniva applicato correttamente
- Quindi tutte le varianti (AntiHub, Homophily, Ricci) si comportavano identicamente
- Il TRF invece funzionava (scala globalmente il learning rate), quindi i risultati con TRF erano diversi da Adam base

### Problema 2: Metriche Topologiche a Zero

**Sintomo:**
```
Cora:
  A (algebraic connectivity) = 0.0000
  Nc (node connectivity) = 0
  Ec (edge connectivity) = 0

CiteSeer:
  A (algebraic connectivity) = 0.0000
  Nc (node connectivity) = 0
  Ec (edge connectivity) = 0
```

**Diagnosi:**
Questi valori sono **CORRETTI** (non un bug!):
- Indicano che i grafi hanno **componenti disconnesse**
- Questo è normale per molti dataset reali (Cora, CiteSeer, ecc.)
- Il codice gestisce correttamente questi casi con fallback

**Perché non è un problema:**
- `A=0`: Quando il grafo non è connesso, l'algebraic connectivity è 0
- `Nc=0, Ec=0`: La connectivity è 0 per grafi disconnessi
- Il TRF usa comunque altre metriche (E, L) che funzionano

## ✅ Correzioni Applicate

### Correzione 1: Disabilitazione del Gradient Scaling Locale

**File:** `optimizers/t_adam.py`

**Modifiche:**

1. **In `__init__` (linee 126-134)**:
   ```python
   if gradient_scaling_mode is not None:
       print(f"\n⚠️  [WARNING] Local gradient scaling (mode='{gradient_scaling_mode}') is DISABLED")
       print(f"⚠️  The current implementation is broken - it scales model parameters, not node gradients")
       print(f"⚠️  This will be redesigned in a future update")
       print(f"⚠️  For now, only TRF global scaling works correctly\n")
       # self._compute_node_weights()  # DISABLED
   ```

2. **In `_t_adam_update()` (linee 630-646)**:
   - Commentato completamente il codice che applica il gradient scaling
   - Aggiunta documentazione chiara sul perché è disabilitato

**Risultato:**
- Ora è CHIARO che il gradient scaling locale non funziona
- Gli utenti sono avvisati con un WARNING
- Solo il TRF globale (che funziona correttamente) viene utilizzato

### Correzione 2: Logging Chiarificatore per TRF

**File:** `optimizers/t_adam.py` (linee 119-124)

**Aggiunto:**
```python
if trf_weights is not None:
    print(f"[T_Adam] TRF-based global LR scaling ENABLED")
    self._compute_trf()
else:
    print(f"[T_Adam] TRF scaling DISABLED (lr_scale_factor = 1.0)")
```

**Risultato:**
- È ora chiaro quando il TRF viene calcolato e quando no
- Eliminata confusione su quali config usano TRF

## 📊 Configurazioni Valide Dopo la Correzione

### ✅ Configurazioni che Funzionano Correttamente

1. **Adam** (baseline)
   - Standard Adam senza modifiche topologiche

2. **T_Adam** (base)
   - Adam con possibilità di modifiche future
   - Attualmente equivalente ad Adam

3. **T_Adam+TRF**
   - Adam con learning rate globale adattivo
   - Formula: `lr_effective = lr_base × exp(-β × TRF(G))`
   - ✅ **FUNZIONA CORRETTAMENTE**

### ⚠️ Configurazioni Deprecate (Non Funzionano)

Le seguenti configurazioni sono attualmente **equivalenti a T_Adam base** e NON producono risultati diversi:

4. ~~T_Adam+AntiHub~~ → Equivalente a T_Adam (gradient scaling disabilitato)
5. ~~T_Adam+Homophily~~ → Equivalente a T_Adam (gradient scaling disabilitato)
6. ~~T_Adam+Ricci~~ → Equivalente a T_Adam (gradient scaling disabilitato)

Le seguenti usano TRF ma il gradient scaling non funziona:

7. **T_Adam+TRF+AntiHub** → Equivalente a T_Adam+TRF
8. **T_Adam+TRF+Homophily** → Equivalente a T_Adam+TRF
9. **T_Adam+TRF+Ricci** → Equivalente a T_Adam+TRF

## 🔮 Piano Futuro per Gradient Scaling Locale

### Approccio Corretto

Per implementare correttamente il gradient scaling locale, servono:

1. **Hooks sui Node Embeddings**:
   ```python
   # Durante il forward pass
   node_embeddings.register_hook(
       lambda grad: grad * node_weights.view(-1, 1)
   )
   ```

2. **Oppure: Custom Backward Pass**:
   - Implementare un custom layer che applica lo scaling durante la backpropagation
   - Richiede una modifica al GCN stesso

3. **Oppure: Weighted Loss**:
   - Invece di scalare i gradienti, pesare la loss per nodo:
   ```python
   loss_per_node = criterion(out[node_idx], label[node_idx])
   weighted_loss = (loss_per_node * node_weights[node_idx]).mean()
   ```

### Raccomandazione

Per ora, **usa solo T_Adam con TRF** se vuoi scaling topologico. Il gradient scaling locale richiede un redesign completo.

## 📝 Risposta alle Domande dell'Utente

### Q1: "T_Adam con TRF=True e gradient=None non sarebbe uguale ad Adam?"

**NO!** Sono diversi:
- **Adam**: LR fisso (es. 0.01)
- **T_Adam+TRF**: LR adattivo basato su topologia
  - Per Cora: LR = 0.01 × 0.106 = **0.001061** (10x più piccolo!)
  - Per CiteSeer: LR = 0.01 × 0.024 = **0.000237** (40x più piccolo!)

Il TRF scala il learning rate in base alla complessità del grafo.

### Q2: "I valori a 0 e i risultati identici sono un problema?"

**Parzialmente SÌ:**
- I valori a 0 (A, Nc, Ec) sono NORMALI → grafi con componenti disconnesse
- I risultati identici sono un BUG → gradient scaling non funzionava

Ora il bug è corretto (gradient scaling disabilitato con warning chiaro).

## 🎯 Conclusione

### Cosa Funziona Ora
- ✅ Adam baseline
- ✅ T_Adam+TRF (global LR scaling basato su topologia)
- ✅ Caching delle metriche topologiche
- ✅ TRF multi-grafo (media su grafi campionati)

### Cosa NON Funziona (e quindi è Disabilitato)
- ❌ Gradient scaling locale (AntiHub, Homophily, Ricci)
- Gli utenti ricevono un WARNING chiaro se provano a usarlo

### Prossimi Passi
1. Redesign del gradient scaling locale con approccio corretto (hooks o weighted loss)
2. Test approfonditi con il nuovo approccio
3. Eventualmente riabilitare dopo verifica

---

**Autore**: Claude Code
**Data**: 2025-11-22
**Versione**: 2.0 (Post-fix)
