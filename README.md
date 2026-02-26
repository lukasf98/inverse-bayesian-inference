# Inverse Bayesian Trajectory Model for PCCT Lesion Dynamics

Code accompanying the paper on spectral CT lesion dynamics analysis using inverse Bayesian inference.

## Overview

This repository implements an inverse Bayesian framework for recovering dynamical parameters from sparse longitudinal photon-counting CT (PCCT) data. The method fits a linear ordinary differential equation (ODE) model:

```
dx/dt = A*x + B*n + C*delta_x_sat
```

Where:
- **x**: Normalized spectral feature (Iodine, VNC, 70keV, High keV, Low keV)
- **A**: Intrinsic dynamics rate (self-feedback)
- **B**: Count coupling (effect of total lesion count in organ)
- **C**: Behavior coupling (effect of satellite lesion behavior)
- **n**: Number of lesions in the organ
- **delta_x_sat**: Mean normalized change of other lesions in the same organ

Due to sparse temporal sampling (typically 2-4 timepoints per lesion), we use finite differences to approximate derivatives and fit separate models for A+B (count coupling) and A+C (behavior coupling).

## Key Files

### Analysis Scripts

- `src/count_coupling_model.py` - Fits the A+B model: dx/dt = A*x + B*n
- `src/behavior_coupling_model.py` - Fits the A+C model: dx/dt = A*x + C*delta_x_sat
- `src/predictor_orthogonality.py` - Eigenvalue decomposition to verify predictor independence
- `src/synthetic_validation.py` - Validates parameter recovery on synthetic data

### Utilities

- `src/data_utils.py` - Data loading and preprocessing functions

## Usage

### Count Coupling Analysis (B parameter)

```bash
python src/count_coupling_model.py path/to/tracked_lesions_with_radiomics.csv --output results_count.csv
```

### Behavior Coupling Analysis (C parameter)

```bash
python src/behavior_coupling_model.py path/to/tracked_lesions_with_radiomics.csv --output results_behavior.csv
```

### Predictor Orthogonality Check

```bash
python src/predictor_orthogonality.py path/to/tracked_lesions_with_radiomics.csv
```

### Synthetic Validation

```bash
python src/synthetic_validation.py
```

## Data Format

The input CSV file (`tracked_lesions_with_radiomics.csv`) should contain:

| Column | Description |
|--------|-------------|
| patient_id | Unique patient identifier |
| date | Scan date (YYYYMMDD format) |
| lesion_type | Type of lesion (e.g., "lung nodule", "liver lesion") |
| lesion_idx | Lesion index within patient |
| mean_Iodine | Mean iodine concentration in lesion ROI |
| mean_VNC | Mean virtual non-contrast value |
| mean_70keV | Mean 70 keV attenuation |
| mean_High | Mean high keV attenuation |
| mean_Low | Mean low keV attenuation |

## Requirements

```
numpy>=1.20
pandas>=1.3
pymc>=5.0
arviz>=0.12
```

## Method Details

### Inverse Problem Formulation

Given sparse temporal observations x(t_i), we approximate the derivative:

```
dx/dt ≈ (x(t_{i+1}) - x(t_i)) / (t_{i+1} - t_i)
```

And use Bayesian linear regression to recover A, B, or C with full uncertainty quantification via Hamiltonian Monte Carlo (HMC) sampling.

### Predictor Independence

Before interpreting A, B, C independently, we verify predictor orthogonality using eigenvalue decomposition of the predictor correlation matrix. The condition number (ratio of largest to smallest eigenvalue) indicates:

- κ < 3: Predictors approximately orthogonal (independent)
- κ < 10: Moderate correlation
- κ > 10: Highly correlated (effects may be confounded)

### Prior Specification

- A, B, C ~ Normal(0, 1)
- σ ~ HalfNormal(0.5)

## Citation

[Paper citation to be added]
