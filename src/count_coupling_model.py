"""
Inverse Bayesian inference for lesion count coupling (B parameter).

Model: dx/dt = A*x + B*n

Where:
- x = spectral feature (normalized to baseline)
- n = total number of lesions in that organ at that timepoint
- A = intrinsic rate (self-feedback)
- B = count coupling (how lesion count affects dynamics)

Interpretation:
- B > 0: More lesions correlate with increasing feature values
- B < 0: More lesions correlate with decreasing feature values
"""

import numpy as np
import pandas as pd
import pymc as pm
from pathlib import Path

from data_utils import load_data, prepare_count_coupling_data


FEATURES = ['mean_Iodine', 'mean_VNC', 'mean_70keV', 'mean_High', 'mean_Low']
FEATURE_NAMES = ['Iodine', 'VNC', '70keV', 'High', 'Low']


def run_inference(dx_dt, x, n_lesions, n_samples=2000, n_tune=1000):
    """
    Run Bayesian inference for the count coupling model.

    Args:
        dx_dt: Array of observed derivatives
        x: Array of state values (midpoint of interval)
        n_lesions: Array of lesion counts
        n_samples: Number of posterior samples
        n_tune: Number of tuning samples

    Returns:
        PyMC InferenceData object with posterior samples
    """
    with pm.Model() as model:
        # Priors
        A = pm.Normal('A', mu=0, sigma=1)
        B = pm.Normal('B', mu=0, sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=0.5)

        # Model: dx/dt = A*x + B*n
        mu = A * x + B * n_lesions

        # Likelihood
        likelihood = pm.Normal('obs', mu=mu, sigma=sigma, observed=dx_dt)

        # Sample
        trace = pm.sample(n_samples, tune=n_tune, cores=1,
                         progressbar=True, return_inferencedata=True)

    return trace


def analyze_organ(df, organ='lung'):
    """
    Run count coupling analysis for one organ.

    Args:
        df: DataFrame with lesion data
        organ: 'lung' or 'liver'

    Returns:
        DataFrame with inference results for each spectral feature
    """
    print(f"\n{'='*60}")
    print(f"{organ.upper()} - Count Coupling Model")
    print(f"Model: dx/dt = A*x + B*n")
    print(f"{'='*60}")

    data = prepare_count_coupling_data(df, FEATURES, organ)
    results = []

    for i, feat in enumerate(FEATURES):
        feat_data = data[feat]

        if feat_data['n_observations'] < 5:
            print(f"\n{FEATURE_NAMES[i]}: Skipping ({feat_data['n_observations']} obs)")
            continue

        print(f"\n{'-'*40}")
        print(f"{FEATURE_NAMES[i]}: {feat_data['n_observations']} observations")
        print(f"  n_lesions range: {feat_data['n_lesions'].min():.1f} - {feat_data['n_lesions'].max():.1f}")

        trace = run_inference(
            feat_data['dx_dt'],
            feat_data['x'],
            feat_data['n_lesions'],
            n_samples=2000,
            n_tune=1000
        )

        # Extract posterior statistics
        A_samples = trace.posterior['A'].values.flatten()
        B_samples = trace.posterior['B'].values.flatten()

        A_mean, A_std = np.mean(A_samples), np.std(A_samples)
        B_mean, B_std = np.mean(B_samples), np.std(B_samples)

        A_ci = np.percentile(A_samples, [2.5, 97.5])
        B_ci = np.percentile(B_samples, [2.5, 97.5])

        # Check significance (95% CI excludes zero)
        A_sig = not (A_ci[0] <= 0 <= A_ci[1])
        B_sig = not (B_ci[0] <= 0 <= B_ci[1])

        results.append({
            'organ': organ,
            'feature': FEATURE_NAMES[i],
            'A_mean': A_mean, 'A_std': A_std,
            'A_ci_low': A_ci[0], 'A_ci_high': A_ci[1],
            'A_significant': A_sig,
            'B_mean': B_mean, 'B_std': B_std,
            'B_ci_low': B_ci[0], 'B_ci_high': B_ci[1],
            'B_significant': B_sig,
            'n_obs': feat_data['n_observations']
        })

        A_str = "SIGNIFICANT" if A_sig else "not significant"
        B_str = "SIGNIFICANT" if B_sig else "not significant"

        print(f"  A (intrinsic) = {A_mean:.4f} +/- {A_std:.4f} [{A_ci[0]:.4f}, {A_ci[1]:.4f}] - {A_str}")
        print(f"  B (count)     = {B_mean:.4f} +/- {B_std:.4f} [{B_ci[0]:.4f}, {B_ci[1]:.4f}] - {B_str}")

    return pd.DataFrame(results)


def main(data_path, output_path=None):
    """
    Run count coupling analysis for both organs.

    Args:
        data_path: Path to tracked_lesions_with_radiomics.csv
        output_path: Optional path to save results CSV
    """
    print("Loading data...")
    df = load_data(data_path)

    # Run for both organs
    lung_results = analyze_organ(df, organ='lung')
    liver_results = analyze_organ(df, organ='liver')

    all_results = pd.concat([lung_results, liver_results], ignore_index=True)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - COUNT COUPLING MODEL (B)")
    print("="*60)

    print("\n--- INTRINSIC DYNAMICS (A) ---")
    for organ in ['lung', 'liver']:
        organ_df = all_results[all_results['organ'] == organ]
        sig_A = organ_df[organ_df['A_significant']]
        print(f"\n{organ.upper()}:")
        if len(sig_A) > 0:
            for _, row in sig_A.iterrows():
                direction = "increasing" if row['A_mean'] > 0 else "decreasing"
                print(f"  {row['feature']}: A = {row['A_mean']:.3f} [{row['A_ci_low']:.3f}, {row['A_ci_high']:.3f}] ({direction})")
        else:
            print("  No significant intrinsic dynamics")

    print("\n--- COUNT COUPLING (B) ---")
    for organ in ['lung', 'liver']:
        organ_df = all_results[all_results['organ'] == organ]
        sig_B = organ_df[organ_df['B_significant']]
        print(f"\n{organ.upper()}:")
        if len(sig_B) > 0:
            for _, row in sig_B.iterrows():
                effect = "more lesions -> higher" if row['B_mean'] > 0 else "more lesions -> lower"
                print(f"  {row['feature']}: B = {row['B_mean']:.3f} [{row['B_ci_low']:.3f}, {row['B_ci_high']:.3f}]")
                print(f"    Effect: {effect} {row['feature']}")
        else:
            print("  No significant count coupling")

    if output_path:
        all_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run count coupling analysis')
    parser.add_argument('data_path', type=str, help='Path to tracked_lesions_with_radiomics.csv')
    parser.add_argument('--output', type=str, help='Output CSV path', default=None)

    args = parser.parse_args()
    main(args.data_path, args.output)
