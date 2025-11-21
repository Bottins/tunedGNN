#!/bin/bash
# Quick test script for T_Adam dual scaling modes

echo "=========================================="
echo "T_Adam Dual Scaling Test Script"
echo "=========================================="
echo ""

# Check if dataset is provided
DATASET=${1:-cora}
EPOCHS=${2:-50}

echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo ""

# Test 1: Standard T_Adam (baseline)
echo "1. Testing Standard T_Adam (no scaling)..."
python main.py --dataset $DATASET --task node --epochs $EPOCHS --optimizers T_Adam
echo ""

# Test 2: TRF-based global scaling only
echo "2. Testing TRF-based global scaling..."
python main.py --dataset $DATASET --task node --epochs $EPOCHS --optimizers T_Adam --use_trf
echo ""

# Test 3: Anti-Hub local scaling
echo "3. Testing Anti-Hub local gradient scaling..."
python main.py --dataset $DATASET --task node --epochs $EPOCHS --optimizers T_Adam --gradient_scaling anti_hub
echo ""

# Test 4: Homophily-based local scaling
echo "4. Testing Homophily-based local gradient scaling..."
python main.py --dataset $DATASET --task node --epochs $EPOCHS --optimizers T_Adam --gradient_scaling homophily
echo ""

# Test 5: Ricci Curvature local scaling
echo "5. Testing Ricci Curvature local gradient scaling..."
python main.py --dataset $DATASET --task node --epochs $EPOCHS --optimizers T_Adam --gradient_scaling ricci
echo ""

# Test 6: Full dual scaling (TRF + Anti-Hub)
echo "6. Testing Full Dual Scaling (TRF + Anti-Hub)..."
python main.py --dataset $DATASET --task node --epochs $EPOCHS --optimizers T_Adam --use_trf --gradient_scaling anti_hub
echo ""

echo "=========================================="
echo "All tests completed!"
echo "=========================================="
