"""
Data loading and processing utilities for inverse Bayesian trajectory analysis.

This module provides common functions for loading PCCT lesion data
and computing derived variables used in the dynamical models.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_data(data_path):
    """
    Load and prepare lesion tracking data with radiomics features.

    Args:
        data_path: Path to tracked_lesions_with_radiomics.csv

    Returns:
        DataFrame with computed lesion_id and organ columns
    """
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['lesion_id'] = (df['patient_id'] + '_' +
                       df['lesion_type'].str.replace(' ', '_') + '_' +
                       df['lesion_idx'].astype(str))
    df['organ'] = df['lesion_type'].apply(
        lambda x: 'lung' if 'lung' in x.lower() else 'liver'
    )
    return df


def count_lesions_in_organ(df, patient_id, organ, date):
    """
    Count number of lesions in a specific organ at a given date.

    Args:
        df: DataFrame with lesion data
        patient_id: Patient identifier
        organ: 'lung' or 'liver'
        date: Date of observation

    Returns:
        Integer count of unique lesions
    """
    return len(df[(df['patient_id'] == patient_id) &
                  (df['organ'] == organ) &
                  (df['date'] == date)]['lesion_id'].unique())


def compute_satellite_change(df, target_lesion_id, date1, date2, feature, organ):
    """
    Compute mean normalized change of satellite lesions (other lesions in same organ).

    This computes how the "environment" (other lesions) changed between two timepoints,
    which is used as the behavior coupling variable in the C model.

    Args:
        df: DataFrame with lesion data (filtered to specific organ)
        target_lesion_id: The primary lesion being analyzed
        date1: Start date
        date2: End date
        feature: Spectral feature column name (e.g., 'mean_Iodine')
        organ: 'lung' or 'liver'

    Returns:
        Tuple of (mean_change, n_satellites) or (None, 0) if no valid satellites
    """
    patient_id = df[df['lesion_id'] == target_lesion_id]['patient_id'].iloc[0]

    # Get all OTHER lesions in same organ for this patient
    other_lesions = df[(df['patient_id'] == patient_id) &
                       (df['organ'] == organ) &
                       (df['lesion_id'] != target_lesion_id)]

    if len(other_lesions) == 0:
        return None, 0

    satellite_changes = []

    for sat_id in other_lesions['lesion_id'].unique():
        sat_data = other_lesions[other_lesions['lesion_id'] == sat_id]
        sat_t1 = sat_data[sat_data['date'] == date1]
        sat_t2 = sat_data[sat_data['date'] == date2]

        if len(sat_t1) == 0 or len(sat_t2) == 0:
            continue

        val1 = sat_t1[feature].values[0]
        val2 = sat_t2[feature].values[0]

        if val1 == 0 or np.isnan(val1) or np.isnan(val2):
            continue

        # Normalized change
        change = (val2 - val1) / val1
        satellite_changes.append(change)

    if len(satellite_changes) == 0:
        return None, 0

    return np.mean(satellite_changes), len(satellite_changes)


def prepare_count_coupling_data(df, features, organ='lung'):
    """
    Prepare data for the count coupling model (A + B*n).

    Model: dx/dt = A*x + B*n_lesions

    Args:
        df: DataFrame with lesion data
        features: List of spectral feature column names
        organ: 'lung' or 'liver'

    Returns:
        Dictionary with arrays for each feature: dx_dt, x, n_lesions
    """
    organ_df = df[df['organ'] == organ].copy()

    # Compute lesion count per patient-organ-date
    lesion_counts = organ_df.groupby(['patient_id', 'date']).size().reset_index(name='n_lesions')
    organ_df = organ_df.merge(lesion_counts, on=['patient_id', 'date'])

    results = {feat: {'dx_dt': [], 'x': [], 'n_lesions': [], 'lesion_ids': []}
               for feat in features}

    for lesion_id in organ_df['lesion_id'].unique():
        lesion_df = organ_df[organ_df['lesion_id'] == lesion_id].sort_values('date')

        if len(lesion_df) < 2:
            continue

        first_row = lesion_df.iloc[0]
        n_lesions_arr = lesion_df['n_lesions'].values

        for feat in features:
            if first_row[feat] == 0 or np.isnan(first_row[feat]):
                continue

            # Normalize to baseline
            x_norm = lesion_df[feat].values / first_row[feat]

            if np.any(np.isnan(x_norm)):
                continue

            # Compute transitions
            for t in range(len(x_norm) - 1):
                dx_dt = x_norm[t+1] - x_norm[t]
                x_avg = (x_norm[t] + x_norm[t+1]) / 2
                n_avg = (n_lesions_arr[t] + n_lesions_arr[t+1]) / 2

                results[feat]['dx_dt'].append(dx_dt)
                results[feat]['x'].append(x_avg)
                results[feat]['n_lesions'].append(n_avg)
                results[feat]['lesion_ids'].append(lesion_id)

    # Convert to arrays
    for feat in features:
        for key in ['dx_dt', 'x', 'n_lesions']:
            results[feat][key] = np.array(results[feat][key])
        results[feat]['n_observations'] = len(results[feat]['dx_dt'])

    return results


def prepare_behavior_coupling_data(df, features, organ='lung'):
    """
    Prepare data for the behavior coupling model (A + C*delta_sat).

    Model: dx/dt = A*x + C*delta_x_satellite

    Args:
        df: DataFrame with lesion data
        features: List of spectral feature column names
        organ: 'lung' or 'liver'

    Returns:
        Dictionary with arrays for each feature: dx_dt, x, satellite_change
    """
    organ_df = df[df['organ'] == organ].copy()

    results = {feat: {'dx_dt': [], 'x': [], 'satellite_change': [],
                      'lesion_ids': [], 'n_satellites': []}
               for feat in features}

    for lesion_id in organ_df['lesion_id'].unique():
        lesion_data = organ_df[organ_df['lesion_id'] == lesion_id].sort_values('date')

        if len(lesion_data) < 2:
            continue

        dates = lesion_data['date'].values

        for feat in features:
            first_val = lesion_data[feat].iloc[0]
            if first_val == 0 or np.isnan(first_val):
                continue

            x_norm = lesion_data[feat].values / first_val

            for t in range(len(dates) - 1):
                date1 = dates[t]
                date2 = dates[t + 1]

                sat_change, n_sat = compute_satellite_change(
                    organ_df, lesion_id, date1, date2, feat, organ
                )

                if sat_change is None:
                    continue

                dx_dt = x_norm[t + 1] - x_norm[t]
                x_avg = (x_norm[t] + x_norm[t + 1]) / 2

                results[feat]['dx_dt'].append(dx_dt)
                results[feat]['x'].append(x_avg)
                results[feat]['satellite_change'].append(sat_change)
                results[feat]['lesion_ids'].append(lesion_id)
                results[feat]['n_satellites'].append(n_sat)

    # Convert to arrays
    for feat in features:
        for key in ['dx_dt', 'x', 'satellite_change', 'n_satellites']:
            results[feat][key] = np.array(results[feat][key])
        results[feat]['n_observations'] = len(results[feat]['dx_dt'])

    return results


def prepare_full_predictor_matrix(df, features, organ='lung'):
    """
    Prepare predictor matrix with all three variables: x, n, delta_sat.

    Used for eigenvalue decomposition to check predictor orthogonality.

    Args:
        df: DataFrame with lesion data
        features: List of spectral feature column names
        organ: 'lung' or 'liver'

    Returns:
        Dictionary with arrays for each feature: x, n, delta_sat, dx_dt
    """
    organ_df = df[df['organ'] == organ]

    results = {feat: {'x': [], 'n': [], 'delta_sat': [], 'dx_dt': []}
               for feat in features}

    for lesion_id in organ_df['lesion_id'].unique():
        lesion_data = organ_df[organ_df['lesion_id'] == lesion_id].sort_values('date')

        if len(lesion_data) < 2:
            continue

        patient_id = lesion_data['patient_id'].iloc[0]
        dates = lesion_data['date'].values

        for feat in features:
            first_val = lesion_data[feat].iloc[0]
            if first_val == 0 or np.isnan(first_val):
                continue

            x_norm = lesion_data[feat].values / first_val

            for t in range(len(dates) - 1):
                date1 = dates[t]
                date2 = dates[t + 1]

                # x: state variable (average of interval)
                x_avg = (x_norm[t] + x_norm[t + 1]) / 2

                # n: lesion count in organ
                n = count_lesions_in_organ(organ_df, patient_id, organ, date1)

                # delta_sat: satellite behavior
                delta_sat, _ = compute_satellite_change(
                    organ_df, lesion_id, date1, date2, feat, organ
                )

                if delta_sat is None:
                    continue

                # dx/dt: derivative
                dx_dt = x_norm[t + 1] - x_norm[t]

                results[feat]['x'].append(x_avg)
                results[feat]['n'].append(n)
                results[feat]['delta_sat'].append(delta_sat)
                results[feat]['dx_dt'].append(dx_dt)

    # Convert to arrays
    for feat in features:
        for key in results[feat]:
            results[feat][key] = np.array(results[feat][key])

    return results
