"""
Run all analyses for the inverse Bayesian trajectory model.

This script runs the complete analysis pipeline:
1. Count coupling model (A + B*n)
2. Behavior coupling model (A + C*delta_sat)
3. Predictor orthogonality check

Usage:
    python run_all_analyses.py path/to/tracked_lesions_with_radiomics.csv [output_dir]
"""

import sys
from pathlib import Path

from count_coupling_model import main as run_count_coupling
from behavior_coupling_model import main as run_behavior_coupling
from predictor_orthogonality import main as run_orthogonality


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_all_analyses.py <data_path> [output_dir]")
        sys.exit(1)

    data_path = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")

    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("INVERSE BAYESIAN TRAJECTORY MODEL - FULL ANALYSIS")
    print("="*70)

    # 1. Predictor orthogonality
    print("\n" + "#"*70)
    print("# STEP 1: PREDICTOR ORTHOGONALITY CHECK")
    print("#"*70)
    orthogonality_results = run_orthogonality(data_path)

    # 2. Count coupling model
    print("\n" + "#"*70)
    print("# STEP 2: COUNT COUPLING MODEL (A + B*n)")
    print("#"*70)
    count_results = run_count_coupling(
        data_path,
        output_path=output_dir / "count_coupling_results.csv"
    )

    # 3. Behavior coupling model
    print("\n" + "#"*70)
    print("# STEP 3: BEHAVIOR COUPLING MODEL (A + C*delta_sat)")
    print("#"*70)
    behavior_results = run_behavior_coupling(
        data_path,
        output_path=output_dir / "behavior_coupling_results.csv"
    )

    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("  - count_coupling_results.csv")
    print("  - behavior_coupling_results.csv")


if __name__ == "__main__":
    main()
