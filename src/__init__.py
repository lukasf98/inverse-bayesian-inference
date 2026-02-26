"""
Inverse Bayesian Trajectory Model for PCCT Lesion Dynamics

This package implements inverse Bayesian inference for recovering dynamical
parameters from sparse longitudinal photon-counting CT data.

Modules:
- data_utils: Data loading and preprocessing
- count_coupling_model: A + B*n model (intrinsic + count effect)
- behavior_coupling_model: A + C*delta_sat model (intrinsic + behavior effect)
- predictor_orthogonality: Eigenvalue decomposition for predictor independence
- synthetic_validation: Method validation on synthetic data
"""

__version__ = "1.0.0"
