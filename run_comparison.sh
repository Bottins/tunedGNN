#!/bin/bash
# Quick launcher for T_Adam Comparison
# Usage: ./run_comparison.sh [mode] [options]

set -e  # Exit on error

# Parse mode
MODE=${1:-full}

case $MODE in
    quick)
        echo "=========================================="
        echo "Quick Test Mode"
        echo "Testing on Cora only, 3 runs, 50 epochs"
        echo "=========================================="
        python Comparison.py --task node --datasets cora --runs 3 --epochs 50
        ;;

    node)
        echo "=========================================="
        echo "Node Classification Mode"
        echo "Testing all node datasets, 5 runs, 200 epochs"
        echo "=========================================="
        python Comparison.py --task node --runs 5 --epochs 200
        ;;

    graph)
        echo "=========================================="
        echo "Graph Classification Mode"
        echo "Testing all graph datasets, 5 runs, 200 epochs"
        echo "=========================================="
        python Comparison.py --task graph --runs 5 --epochs 200
        ;;

    thorough)
        echo "=========================================="
        echo "Thorough Mode"
        echo "Testing all datasets, 10 runs, 300 epochs"
        echo "WARNING: This will take many hours!"
        echo "=========================================="
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python Comparison.py --task both --runs 10 --epochs 300
        else
            echo "Cancelled."
            exit 0
        fi
        ;;

    full)
        echo "=========================================="
        echo "Full Comparison Mode"
        echo "Testing all datasets, 5 runs, 200 epochs"
        echo "This will take several hours on GPU"
        echo "=========================================="
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python Comparison.py --task both --runs 5 --epochs 200
        else
            echo "Cancelled."
            exit 0
        fi
        ;;

    cpu)
        echo "=========================================="
        echo "CPU Mode (Quick Test)"
        echo "Testing on Cora, 3 runs, 50 epochs"
        echo "=========================================="
        python Comparison.py --task node --datasets cora --runs 3 --epochs 50 --device cpu
        ;;

    custom)
        echo "=========================================="
        echo "Custom Mode - Provide your own arguments"
        echo "=========================================="
        shift  # Remove 'custom' from arguments
        python Comparison.py "$@"
        ;;

    help|--help|-h)
        echo "Usage: ./run_comparison.sh [mode]"
        echo ""
        echo "Available modes:"
        echo "  quick      - Quick test on Cora (3 runs, 50 epochs)"
        echo "  node       - Full node classification comparison"
        echo "  graph      - Full graph classification comparison"
        echo "  full       - Complete comparison (node + graph)"
        echo "  thorough   - Thorough study (10 runs, 300 epochs)"
        echo "  cpu        - CPU-only quick test"
        echo "  custom     - Pass custom arguments to Comparison.py"
        echo "  help       - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_comparison.sh quick"
        echo "  ./run_comparison.sh node"
        echo "  ./run_comparison.sh custom --task node --datasets cora citeseer --runs 5"
        echo ""
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Use './run_comparison.sh help' for usage information"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Comparison completed!"
echo "Check ./comparison_results/ for outputs"
echo "=========================================="
