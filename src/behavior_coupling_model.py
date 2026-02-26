"""
Inverse Bayesian inference for satellite behavior coupling (C parameter).

Model: dx/dt = A*x + C*delta_x_sat

Where:
- x = spectral feature (normalized to baseline)
- delta_x_sat = mean normalized change of satellite lesions (other lesions in same organ)
- A = intrinsic rate (self-feedback)
- C = behavior coupling (how satellite behavior affects primary dynamics)

Interpretation:
- C > 0: SYNERGISTIC - when satellites grow, primary tends to grow
- C < 0: ANTAGONISTIC - when satellites grow, primary tends to shrink (competitive)
"""

import numpy as np
import pandas as pd
import pymc as pm
from pathlib import Path

from data_utils import load_data, prepare_behavior_coupling_data


FEATURES = ['mean_Iodine', 'mean_VNC', 'mean_70keV', 'mean_High', 'mean_Low']
FEATURE_NAMES = ['Iodine', 'VNC', '70keV', 'High', 'Low']


def run_inference(dx_dt, x, satellite_change, n_samples=2000, n_tune=1000):
    """
    Run Bayesian inference for the behavior coupling model.

    Args:
        dx_dt: Array of observed derivatives
        x: Array of state values (midpoint of interval)
        satellite_change: Array of mean satellite behavior
        n_samples: Number of posterior samples
        n_tune: Number of tuning samples

    Returns:
        PyMC InferenceData object with posterior samples
    """
    with pm.Model() as model:
        # Priors
        A = pm.Normal('A', mu=0, sigma=1)
        C = pm.Normal('C', mu=0, sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=0.5)

        # Model: dx/dt = A*x + C*delta_sat
        mu = A * x + C * satellite_change

        # Likelihood
        likelihood = pm.Normal('obs', mu=mu, sigma=sigma, observed=dx_dt)

        # Sample
        trace = pm.sample(n_samples, tune=n_tune, cores=1,
                         progressbar=True, return_inferencedata=True)

    return trace


def analyze_organ(df, organ='lung'):
    """
    Run behavior coupling analysis for one organ.

    Args:
        df: DataFrame with lesion data
        organ: 'lung' or 'liver'

    Returns:
        DataFrame with inference results for each spectral feature
    """
    print(f"\n{'='*60}")
    print(f"{organ.upper()} - Behavior Coupling Model")
    print(f"Model: dx/dt = A*x + C*delta_x_sat")
    print(f"{'='*60}")

    data = prepare_behavior_coupling_data(df, FEATURES, organ)
    results = []

    for i, feat in enumerate(FEATURES):
        feat_data = data[feat]

        if feat_data['n_observations'] < 5:
            print(f"\n{FEATURE_NAMES[i]}: Skipping ({feat_data['n_observations']} obs with satellites)")
            continue

        print(f"\n{'-'*40}")
        print(f"{FEATURE_NAMES[i]}: {feat_data['n_observations']} observations")
        print(f"  Satellite change range: {feat_data['satellite_change'].min():.3f} to {feat_data['satellite_change'].max():.3f}")
        print(f"  Mean satellites per obs: {feat_data['n_satellites'].mean():.1f}")

        trace = run_inference(
            feat_data['dx_dt'],
            feat_data['x'],
            feat_data['satellite_change'],
            n_samples=2000,
            n_tune=1000
        )

        # Extract posterior statistics
        A_samples = trace.posterior['A'].values.flatten()
        C_samples = trace.posterior['C'].values.flatten()

        A_mean, A_std = np.mean(A_samples), np.std(A_samples)
        C_mean, C_std = np.mean(C_samples), np.std(C_samples)

        A_ci = np.percentile(A_samples, [2.5, 97.5])
        C_ci = np.percentile(C_samples, [2.5, 97.5])

        # Check significance (95% CI excludes zero)
        A_sig = not (A_ci[0] <= 0 <= A_ci[1])
        C_sig = not (C_ci[0] <= 0 <= C_ci[1])

        results.append({
            'organ': organ,
            'feature': FEATURE_NAMES[i],
            'A_mean': A_mean, 'A_std': A_std,
            'A_ci_low': A_ci[0], 'A_ci_high': A_ci[1],
            'A_significant': A_sig,
            'C_mean': C_mean, 'C_std': C_std,
            'C_ci_low': C_ci[0], 'C_ci_high': C_ci[1],
            'C_significant': C_sig,
            'n_obs': feat_data['n_observations']
        })

        A_str = "SIGNIFICANT" if A_sig else "not significant"
        C_str = "SIGNIFICANT" if C_sig else "not significant"

        # Interpretation
        if C_sig:
            if C_mean > 0:
                interp = "SYNERGISTIC (satellites up -> primary up)"
            else:
                interp = "ANTAGONISTIC (satellites up -> primary down)"
        else:
            interp = "no significant coupling"

        print(f"  A (intrinsic)  = {A_mean:.4f} +/- {A_std:.4f} [{A_ci[0]:.4f}, {A_ci[1]:.4f}] - {A_str}")
        print(f"  C (behavior)   = {C_mean:.4f} +/- {C_std:.4f} [{C_ci[0]:.4f}, {C_ci[1]:.4f}] - {C_str}")
        print(f"  --> {interp}")

    return pd.DataFrame(results)


def main(data_path, output_path=None):
    """
    Run behavior coupling analysis for both organs.

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
    print("SUMMARY - BEHAVIOR COUPLING MODEL (C)")
    print("="*60)
    print("\nKey Question: When satellites change, does primary follow or oppose?")
    print("  C > 0: SYNERGISTIC (move together)")
    print("  C < 0: ANTAGONISTIC (move opposite / competitive)")

    for organ in ['lung', 'liver']:
        organ_df = all_results[all_results['organ'] == organ]
        if len(organ_df) == 0:
            continue

        print(f"\n{organ.upper()}:")
        sig_C = organ_df[organ_df['C_significant']]
        if len(sig_C) > 0:
            for _, row in sig_C.iterrows():
                behavior = "SYNERGISTIC" if row['C_mean'] > 0 else "ANTAGONISTIC"
                print(f"  {row['feature']}: C = {row['C_mean']:.3f} [{row['C_ci_low']:.3f}, {row['C_ci_high']:.3f}] - {behavior}")
        else:
            print("  No significant behavior coupling")

    if output_path:
        all_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run behavior coupling analysis')
    parser.add_argument('data_path', type=str, help='Path to tracked_lesions_with_radiomics.csv')
    parser.add_argument('--output', type=str, help='Output CSV path', default=None)

    args = parser.parse_args()
    main(args.data_path, args.output)
