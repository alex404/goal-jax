"""JSON schema for the chordal Boltzmann PPC example.

Exchanged between ``run.py`` (computes) and ``plot.py`` (visualizes). All arrays
are plain Python lists. The example tells one story --- a chordal Boltzmann
population code that stays conjugate to a Gaussian latent *while learning data* ---
so there is a single per-family record covering conjugacy, an information-content
sanity check, and the data-learning history.
"""

from typing import TypedDict


class FamilyResult(TypedDict):
    """All diagnostics for one population code (chordal or diagonal).

    Conjugacy: ``psi_curve``/``affine_curve`` are the trained model's Boltzmann
    log-partition along a 1D latent sweep and its affine fit; ``residual_before``/
    ``after`` are the conjugation residual before training (rho=0) vs after.

    Information content: ``decode_rmse`` is exact Bayesian decoding of the latent
    from one spike pattern, ``lesion_rmse`` the no-information control.

    Learning (aligned with ``Results.steps``): mean ``elbo_train``/``elbo_test``
    and conjugation ``conj_r2`` rising as the penalty ramps in. ``model_corr`` is
    the trained model's pairwise correlation matrix and ``corr_offdiag_rmse`` its
    distance to the data's.
    """

    name: str
    psi_curve: list[float]
    affine_curve: list[float]
    residual_before: list[float]
    residual_after: list[float]
    decode_rmse: float
    lesion_rmse: float
    elbo_train: list[float]
    elbo_test: list[float]
    conj_r2: list[float]
    model_corr: list[list[float]]
    corr_offdiag_rmse: float


class Results(TypedDict):
    """Top-level results: the data, the shared axes, and one record per family."""

    n_neurons: int
    n_latent: int
    n_components: int
    lambda_conj: float
    conj_warmup: int
    prior_std: float
    ceiling: float
    z_grid: list[float]
    steps: list[int]
    data_corr: list[list[float]]
    families: dict[str, FamilyResult]
