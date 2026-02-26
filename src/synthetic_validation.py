"""
Synthetic validation of the inverse Bayesian ODE model.

This script validates that the inference method can recover known parameters
when coupling exists. This confirms that findings from real data (e.g., B~0)
are meaningful and not due to method failure.

Model: dx/dt = A*x + B*phi

The validation:
1. Generates synthetic data with known A and B values
2. Runs Bayesian inference to recover parameters
3. Checks if true values fall within 95% credible intervals
"""

import numpy as np
import pymc as pm


# Set random seed for reproducibility
np.random.seed(42)


def generate_synthetic_data(n_lesions=30, n_timepoints=4,
                            true_A=0.5, true_B=0.3,
                            noise_std=0.1):
    """
    Generate synthetic data with known parameters.

    Simulates lesion trajectories following dx/dt = A*x + B*phi
    with Euler integration and adds observation noise.

    Args:
        n_lesions: Number of simulated lesions
        n_timepoints: Number of timepoints per lesion
        true_A: True intrinsic rate parameter
        true_B: True coupling strength parameter
        noise_std: Observation noise standard deviation

    Returns:
        Dictionary with synthetic data arrays
    """
    print(f"Generating synthetic data with A={true_A}, B={true_B}")
    print(f"  {n_lesions} lesions, {n_timepoints} timepoints each")

    data = {
        'lesion_id': [],
        'timepoint': [],
        'x_observed': [],
        'x_true': [],
        'phi': [],
    }

    for lesion in range(n_lesions):
        # Initial conditions (vary between lesions)
        x0 = np.random.uniform(0.8, 1.2)

        # Generate coupling variable trajectory
        phi_trend = np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.5)
        phi_values = phi_trend * np.arange(n_timepoints) + np.random.normal(0, 0.1, n_timepoints)

        # Simulate the ODE: dx/dt = A*x + B*phi
        x_true = [x0]
        for t in range(1, n_timepoints):
            dt = 1.0
            phi_avg = (phi_values[t-1] + phi_values[t]) / 2
            dx = true_A * x_true[-1] + true_B * phi_avg
            x_new = x_true[-1] + dx * dt
            x_true.append(x_new)

        x_true = np.array(x_true)
        x_observed = x_true + np.random.normal(0, noise_std, n_timepoints)

        for t in range(n_timepoints):
            data['lesion_id'].append(lesion)
            data['timepoint'].append(t)
            data['x_observed'].append(x_observed[t])
            data['x_true'].append(x_true[t])
            data['phi'].append(phi_values[t])

    for key in data:
        data[key] = np.array(data[key])

    return data


def run_inverse_bayesian(data, n_samples=2000, n_tune=1000):
    """
    Run inverse Bayesian inference to recover A and B.

    Args:
        data: Dictionary with synthetic data
        n_samples: Number of posterior samples
        n_tune: Number of tuning samples

    Returns:
        PyMC InferenceData object
    """
    print("\nRunning inverse Bayesian inference...")

    lesion_ids = np.unique(data['lesion_id'])

    dx_dt_list = []
    x_list = []
    phi_list = []

    for lesion in lesion_ids:
        mask = data['lesion_id'] == lesion
        x_obs = data['x_observed'][mask]
        phi = data['phi'][mask]

        for t in range(len(x_obs) - 1):
            dx_dt = x_obs[t+1] - x_obs[t]
            x_avg = (x_obs[t] + x_obs[t+1]) / 2
            phi_avg = (phi[t] + phi[t+1]) / 2

            dx_dt_list.append(dx_dt)
            x_list.append(x_avg)
            phi_list.append(phi_avg)

    dx_dt = np.array(dx_dt_list)
    x = np.array(x_list)
    phi = np.array(phi_list)

    print(f"  {len(dx_dt)} data points for inference")

    with pm.Model() as model:
        A = pm.Normal('A', mu=0, sigma=1)
        B = pm.Normal('B', mu=0, sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=0.5)

        mu = A * x + B * phi
        likelihood = pm.Normal('obs', mu=mu, sigma=sigma, observed=dx_dt)

        trace = pm.sample(n_samples, tune=n_tune, cores=1,
                         progressbar=True, return_inferencedata=True)

    return trace


def run_validation():
    """
    Run the full synthetic validation.

    Returns:
        Dictionary with validation results
    """
    TRUE_A = 0.3
    TRUE_B = 0.4

    data = generate_synthetic_data(
        n_lesions=30,
        n_timepoints=4,
        true_A=TRUE_A,
        true_B=TRUE_B,
        noise_std=0.1
    )

    trace = run_inverse_bayesian(data, n_samples=2000, n_tune=1000)

    A_samples = trace.posterior['A'].values.flatten()
    B_samples = trace.posterior['B'].values.flatten()

    A_mean, A_std = np.mean(A_samples), np.std(A_samples)
    B_mean, B_std = np.mean(B_samples), np.std(B_samples)

    A_ci = np.percentile(A_samples, [2.5, 97.5])
    B_ci = np.percentile(B_samples, [2.5, 97.5])

    A_in_ci = A_ci[0] <= TRUE_A <= A_ci[1]
    B_in_ci = B_ci[0] <= TRUE_B <= B_ci[1]

    print("\n" + "="*60)
    print("SYNTHETIC VALIDATION RESULTS")
    print("="*60)
    print(f"\nTrue parameters:")
    print(f"  A = {TRUE_A}")
    print(f"  B = {TRUE_B}")
    print(f"\nRecovered parameters:")
    print(f"  A = {A_mean:.3f} +/- {A_std:.3f}")
    print(f"  B = {B_mean:.3f} +/- {B_std:.3f}")
    print(f"\nRecovery error:")
    print(f"  A error = {abs(A_mean - TRUE_A):.3f} ({abs(A_mean - TRUE_A)/TRUE_A*100:.1f}%)")
    print(f"  B error = {abs(B_mean - TRUE_B):.3f} ({abs(B_mean - TRUE_B)/TRUE_B*100:.1f}%)")
    print(f"\n95% Credible Intervals:")
    print(f"  A: [{A_ci[0]:.3f}, {A_ci[1]:.3f}] - True value {'INSIDE' if A_in_ci else 'OUTSIDE'}")
    print(f"  B: [{B_ci[0]:.3f}, {B_ci[1]:.3f}] - True value {'INSIDE' if B_in_ci else 'OUTSIDE'}")

    if A_in_ci and B_in_ci:
        print("\n" + "="*60)
        print("VALIDATION PASSED: Method successfully recovers known parameters!")
        print("This validates that findings in real data are meaningful.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("VALIDATION WARNING: True values outside credible intervals")
        print("="*60)

    return {
        'true_A': TRUE_A,
        'true_B': TRUE_B,
        'estimated_A': A_mean,
        'estimated_B': B_mean,
        'A_std': A_std,
        'B_std': B_std,
        'A_ci': A_ci.tolist(),
        'B_ci': B_ci.tolist(),
        'A_in_ci': A_in_ci,
        'B_in_ci': B_in_ci
    }


if __name__ == "__main__":
    results = run_validation()
