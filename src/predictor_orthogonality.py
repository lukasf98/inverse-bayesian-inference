"""
Eigenvalue decomposition to check independence of predictors A, B, C.

This analysis checks if the predictor variables (x, n, delta_sat) are orthogonal
or correlated. If correlated, the separate model fits may have confounded effects.

Key metric: Condition number (ratio of largest to smallest eigenvalue)
- kappa < 3: Predictors approximately orthogonal (independent)
- kappa < 10: Moderate correlation
- kappa > 10: Highly correlated (multicollinear)

If predictors are orthogonal, we can fit A+B and A+C models separately
and interpret the effects independently.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from data_utils import load_data, prepare_full_predictor_matrix


FEATURES = ['mean_Iodine', 'mean_VNC', 'mean_70keV', 'mean_High', 'mean_Low']
FEATURE_NAMES = ['Iodine', 'VNC', '70keV', 'High', 'Low']


def eigenvalue_analysis(x, n, delta_sat):
    """
    Perform eigenvalue decomposition on predictor covariance matrix.

    Args:
        x: Array of state values
        n: Array of lesion counts
        delta_sat: Array of satellite behavior values

    Returns:
        Tuple of (eigenvalues, eigenvectors, correlation_matrix)
    """
    # Stack predictors
    X = np.column_stack([x, n, delta_sat])

    # Correlation matrix (same as covariance of standardized data)
    corr_matrix = np.corrcoef(X.T)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors, corr_matrix


def analyze_organ(df, organ='lung'):
    """
    Analyze predictor orthogonality for one organ.

    Args:
        df: DataFrame with lesion data
        organ: 'lung' or 'liver'

    Returns:
        Dictionary with analysis results for each feature
    """
    print(f"\n{'='*70}")
    print(f"{organ.upper()}")
    print(f"{'='*70}")

    data = prepare_full_predictor_matrix(df, FEATURES, organ)
    results = {}

    for i, feat in enumerate(FEATURES):
        feat_data = data[feat]

        if len(feat_data['x']) < 5:
            print(f"\n{FEATURE_NAMES[i]}: Insufficient data ({len(feat_data['x'])} obs)")
            continue

        x = feat_data['x']
        n = feat_data['n']
        delta_sat = feat_data['delta_sat']

        eigenvalues, eigenvectors, corr_matrix = eigenvalue_analysis(x, n, delta_sat)

        print(f"\n{'-'*50}")
        print(f"{FEATURE_NAMES[i]} ({len(x)} observations)")
        print(f"{'-'*50}")

        print("\nCorrelation Matrix:")
        print("              x        n    dx_sat")
        labels = ['x     ', 'n     ', 'dx_sat']
        for j, label in enumerate(labels):
            print(f"  {label}  {corr_matrix[j, 0]:+.3f}   {corr_matrix[j, 1]:+.3f}   {corr_matrix[j, 2]:+.3f}")

        print("\nEigenvalues:")
        for j, ev in enumerate(eigenvalues):
            variance_explained = ev / np.sum(eigenvalues) * 100
            print(f"  L_{j+1} = {ev:.3f} ({variance_explained:.1f}% variance)")

        print("\nEigenvectors (columns = principal components):")
        print("              PC1      PC2      PC3")
        for j, label in enumerate(labels):
            print(f"  {label}  {eigenvectors[j, 0]:+.3f}   {eigenvectors[j, 1]:+.3f}   {eigenvectors[j, 2]:+.3f}")

        # Condition number (ratio of largest to smallest eigenvalue)
        condition_number = eigenvalues[0] / eigenvalues[-1]
        print(f"\nCondition number: {condition_number:.2f}")

        # Interpretation
        print("\nInterpretation:")
        if condition_number < 3:
            print("  -> Predictors are approximately ORTHOGONAL (independent)")
            print("  -> Separate model fits are valid")
            orthogonality = "orthogonal"
        elif condition_number < 10:
            print("  -> Predictors have MODERATE correlation")
            print("  -> Some confounding possible, but likely minor")
            orthogonality = "moderate"
        else:
            print("  -> Predictors are HIGHLY CORRELATED (multicollinear)")
            print("  -> Separate model fits may have confounded effects")
            orthogonality = "collinear"

        # Check pairwise correlations
        r_x_n = corr_matrix[0, 1]
        r_x_sat = corr_matrix[0, 2]
        r_n_sat = corr_matrix[1, 2]

        if abs(r_n_sat) > 0.5:
            print(f"  WARNING: n and dx_sat correlated (r={r_n_sat:.2f})")
            print(f"    -> B (count) and C (behavior) effects may be confounded")

        results[FEATURE_NAMES[i]] = {
            'n_obs': len(x),
            'eigenvalues': eigenvalues.tolist(),
            'condition_number': condition_number,
            'orthogonality': orthogonality,
            'corr_x_n': r_x_n,
            'corr_x_sat': r_x_sat,
            'corr_n_sat': r_n_sat
        }

    return results


def main(data_path):
    """
    Run predictor orthogonality analysis for both organs.

    Args:
        data_path: Path to tracked_lesions_with_radiomics.csv
    """
    print("Loading data...")
    df = load_data(data_path)

    print("\n" + "="*70)
    print("EIGENVALUE DECOMPOSITION OF PREDICTOR SPACE")
    print("Predictors: x (state), n (count), dx_sat (satellite behavior)")
    print("="*70)

    lung_results = analyze_organ(df, organ='lung')
    liver_results = analyze_organ(df, organ='liver')

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
If eigenvalues are approximately equal (L1 ~= L2 ~= L3 ~= 1):
  -> Predictors span orthogonal directions
  -> A, B, C effects are independent
  -> Fitting separate models is valid

If one eigenvalue dominates (L1 >> L2, L3):
  -> Predictors are correlated
  -> Effects may be confounded
  -> Should fit full model (A + B + C) together
""")

    return {'lung': lung_results, 'liver': liver_results}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Check predictor orthogonality')
    parser.add_argument('data_path', type=str, help='Path to tracked_lesions_with_radiomics.csv')

    args = parser.parse_args()
    main(args.data_path)
