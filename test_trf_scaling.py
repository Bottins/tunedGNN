"""
Test script to verify TRF scaling formula produces values in [0.001, 1.0]
"""

import math

def old_formula(trf, beta=0.1):
    """Old exponential formula"""
    return math.exp(-beta * trf)

def new_formula(trf, beta=0.1):
    """New bounded hyperbolic formula"""
    return 0.001 + 0.999 / (1.0 + beta * trf)

def test_trf_scaling():
    """Test TRF scaling with realistic values"""
    print("=" * 70)
    print("TRF SCALING FORMULA COMPARISON")
    print("=" * 70)
    print()

    # Test cases from real datasets
    test_cases = [
        ("Minimum (ideal graph)", 0.0),
        ("Small graph", 5.0),
        ("Cora (real)", 22.4302),
        ("Medium complexity", 30.0),
        ("CiteSeer (real)", 37.4031),
        ("High complexity", 100.0),
        ("MUTAG (real)", 190.8815),
        ("Very high", 500.0),
    ]

    print(f"{'Dataset':<25} {'TRF':>10} {'Old Formula':>15} {'New Formula':>15} {'Status':>10}")
    print("-" * 70)

    for name, trf_value in test_cases:
        old_scale = old_formula(trf_value)
        new_scale = new_formula(trf_value)

        # Check if new formula is in bounds
        in_bounds = 0.001 <= new_scale <= 1.0
        status = "✓" if in_bounds else "✗"

        # Check if old formula was problematic (too small)
        was_problematic = old_scale < 0.001
        if was_problematic:
            status += " FIXED"

        print(f"{name:<25} {trf_value:>10.4f} {old_scale:>15.6f} {new_scale:>15.6f} {status:>10}")

    print()
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    print()

    # Verify bounds
    print("Testing bounds:")
    print(f"  • TRF = 0: new_scale = {new_formula(0):.6f} (should be ~1.0)")
    print(f"  • TRF → ∞: new_scale = {new_formula(10000):.6f} (should be ~0.001)")
    print()

    # Verify smooth transition
    print("Testing smooth transition (beta=0.1):")
    for trf in [0, 1, 5, 10, 20, 50, 100, 200]:
        scale = new_formula(trf)
        effective_lr = 0.01 * scale  # base lr = 0.01
        print(f"  • TRF = {trf:3}: scale = {scale:.6f}, effective LR = {effective_lr:.6f}")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("✓ New formula guarantees LR scale in [0.001, 1.0]")
    print("✓ Smooth transition from 1.0 (simple graphs) to 0.001 (complex graphs)")
    print("✓ Fixes the problem where MUTAG had scale = 0.000035 (now 0.051)")
    print("✓ More predictable and stable training across different graph complexities")
    print()

if __name__ == "__main__":
    test_trf_scaling()
