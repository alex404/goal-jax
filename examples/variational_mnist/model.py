"""Model utilities for variational MNIST.

This module provides:
- Model creation with good defaults
- Conjugation quality metrics (R², Std[residual], ||rho||)
- IWAE bound for amortization gap estimation
"""

# pyright: reportAttributeAccessIssue=false

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from goal.models import (
    BinomialBernoulliMixture,
    CompleteBinomialBernoulliMixture,
    CompletePoissonBernoulliMixture,
    CompleteVariationalBinomialVonMisesMixture,
    PoissonBernoulliMixture,
    VariationalBinomialVonMisesMixture,
    binomial_bernoulli_mixture,
    complete_binomial_bernoulli_mixture,
    complete_poisson_bernoulli_mixture,
    complete_variational_binomial_vonmises_mixture,
    poisson_bernoulli_mixture,
    variational_binomial_vonmises_mixture,
)

# Type alias for mixture models
type MixtureModel = (
    BinomialBernoulliMixture
    | PoissonBernoulliMixture
    | VariationalBinomialVonMisesMixture
    | CompleteBinomialBernoulliMixture
    | CompletePoissonBernoulliMixture
    | CompleteVariationalBinomialVonMisesMixture
)

# Default multiplier for n_samples = multiplier * bas_lat_man.dim
ANALYTICAL_RHO_SAMPLES_MULTIPLIER = 5

# MNIST dimensions
IMG_SIZE = 28
N_OBSERVABLE = IMG_SIZE * IMG_SIZE  # 784

# Default architecture
DEFAULT_N_LATENT = 1024
DEFAULT_N_CLUSTERS = 10
DEFAULT_N_TRIALS = 16


def create_model(
    n_latent: int = DEFAULT_N_LATENT,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    observable_type: Literal["binomial", "poisson"] = "binomial",
    latent_type: Literal["bernoulli", "vonmises"] = "bernoulli",
    n_trials: int = DEFAULT_N_TRIALS,
    complete: bool = False,
) -> MixtureModel:
    """Create a mixture model with the specified observable and latent types.

    Args:
        n_latent: Number of latent units (default: 1024)
        n_clusters: Number of mixture components (default: 10)
        observable_type: Type of observable distribution ("binomial" or "poisson")
        latent_type: Type of latent distribution ("bernoulli" or "vonmises")
        n_trials: Binomial discretization level (default: 16, only used for binomial)
        complete: If True, use complete model variant with full mixture rho correction.
            Standard models have rho_dim = n_latent (or 2*n_latent for VonMises).
            Complete models have rho_dim = n_latent * n_clusters + (n_clusters-1).

    Returns:
        MixtureModel instance
    """
    if observable_type == "poisson":
        if complete:
            return complete_poisson_bernoulli_mixture(
                n_observable=N_OBSERVABLE,
                n_latent=n_latent,
                n_clusters=n_clusters,
            )
        return poisson_bernoulli_mixture(
            n_observable=N_OBSERVABLE,
            n_latent=n_latent,
            n_clusters=n_clusters,
        )
    if latent_type == "vonmises":
        if complete:
            return complete_variational_binomial_vonmises_mixture(
                n_observable=N_OBSERVABLE,
                n_latent=n_latent,
                n_clusters=n_clusters,
                n_trials=n_trials,
            )
        return variational_binomial_vonmises_mixture(
            n_observable=N_OBSERVABLE,
            n_latent=n_latent,
            n_clusters=n_clusters,
            n_trials=n_trials,
        )
    if complete:
        return complete_binomial_bernoulli_mixture(
            n_observable=N_OBSERVABLE,
            n_latent=n_latent,
            n_clusters=n_clusters,
            n_trials=n_trials,
        )
    return binomial_bernoulli_mixture(
        n_observable=N_OBSERVABLE,
        n_latent=n_latent,
        n_clusters=n_clusters,
        n_trials=n_trials,
    )


def compute_conjugation_metrics(
    model: MixtureModel,
    params: Array,
    key: Array,
    n_samples: int = 1000,
) -> tuple[Array, Array, Array, Array]:
    """Compute conjugation quality metrics.

    This measures how well the linear correction rho*s(z) approximates psi_X(eta(z)).

    Args:
        model: The BinomialBernoulliMixture model
        params: Full model parameters (including rho)
        key: JAX random key
        n_samples: Number of samples for estimation

    Returns:
        Tuple of (var_f, std_f, r_squared, rho_norm) where:
        - var_f: Variance of residual f_tilde
        - std_f: Standard deviation of residual f_tilde
        - r_squared: R^2 = 1 - Var[f_tilde]/Var[psi_X]
        - rho_norm: L2 norm of rho parameters
    """
    var_f, std_f, r_squared = model.conjugation_metrics(key, params, n_samples)
    rho = model.conjugation_parameters(params)
    rho_norm = jnp.linalg.norm(rho)
    return var_f, std_f, r_squared, rho_norm


def iwae_bound(
    model: MixtureModel,
    key: Array,
    params: Array,
    x: Array,
    n_importance_samples: int = 1000,
) -> Array:
    """Compute the Importance-Weighted Autoencoder (IWAE) bound.

    IWAE is a tighter bound than ELBO:
        log p(x) >= IWAE(K) >= ELBO

    As K increases, IWAE(K) approaches log p(x).

    IWAE = E[log(1/K * sum_k w_k)] where w_k = p(x,z_k)/q(z_k|x)

    For numerical stability, we use:
        IWAE = logsumexp(log w_k) - log(K)

    Args:
        model: The BinomialBernoulliMixture model
        key: JAX random key
        params: Full model parameters
        x: Single observation
        n_importance_samples: Number of importance samples K

    Returns:
        IWAE bound value (scalar)
    """
    # Get approximate posterior
    q_params = model.approximate_posterior_at(params, x)

    # Sample K z vectors from posterior mixture
    z_samples = model.mix_man.sample(key, q_params, n_importance_samples)

    # Compute log importance weights: log w = log p(x,z) - log q(z|x)
    log_joints = jax.vmap(lambda z: model.log_density(params, x, z))(z_samples)
    log_posteriors = jax.vmap(lambda z: model.mix_man.log_density(q_params, z))(
        z_samples
    )

    log_weights = log_joints - log_posteriors

    # IWAE = logsumexp(log_weights) - log(K)
    return jax.scipy.special.logsumexp(log_weights) - jnp.log(n_importance_samples)


def estimate_amortization_gap(
    model: MixtureModel,
    key: Array,
    params: Array,
    x: Array,
    n_elbo_samples: int = 10,
    n_iwae_samples: int = 1000,
) -> tuple[Array, Array, Array]:
    """Estimate the amortization gap: log p(x) - ELBO.

    The amortization gap measures what we lose from the approximate posterior.
    We estimate it as IWAE(K) - ELBO, which is a lower bound on the true gap.

    Args:
        model: The BinomialBernoulliMixture model
        key: JAX random key
        params: Full model parameters
        x: Single observation
        n_elbo_samples: Number of MC samples for ELBO
        n_iwae_samples: Number of importance samples for IWAE

    Returns:
        Tuple of (gap_estimate, elbo, iwae) where:
        - gap_estimate: IWAE - ELBO (should be >= 0)
        - elbo: ELBO value
        - iwae: IWAE bound value
    """
    key1, key2 = jax.random.split(key)

    elbo = model.elbo_at(key1, params, x, n_elbo_samples)
    iwae = iwae_bound(model, key2, params, x, n_iwae_samples)

    gap_estimate = iwae - elbo
    return gap_estimate, elbo, iwae


def batch_iwae(
    model: MixtureModel,
    key: Array,
    params: Array,
    xs: Array,
    n_importance_samples: int = 1000,
) -> Array:
    """Compute mean IWAE bound over a batch.

    Args:
        model: The BinomialBernoulliMixture model
        key: JAX random key
        params: Full model parameters
        xs: Batch of observations
        n_importance_samples: Number of importance samples

    Returns:
        Mean IWAE over batch
    """
    batch_size = xs.shape[0]
    keys = jax.random.split(key, batch_size)

    iwaes = jax.vmap(
        lambda k, x: iwae_bound(model, k, params, x, n_importance_samples)
    )(keys, xs)

    return jnp.mean(iwaes)


def batch_amortization_gap(
    model: MixtureModel,
    key: Array,
    params: Array,
    xs: Array,
    n_elbo_samples: int = 10,
    n_iwae_samples: int = 1000,
) -> tuple[Array, Array, Array]:
    """Estimate mean amortization gap over a batch.

    Args:
        model: The BinomialBernoulliMixture model
        key: JAX random key
        params: Full model parameters
        xs: Batch of observations
        n_elbo_samples: Number of MC samples for ELBO
        n_iwae_samples: Number of importance samples for IWAE

    Returns:
        Tuple of (mean_gap, mean_elbo, mean_iwae)
    """
    batch_size = xs.shape[0]
    keys = jax.random.split(key, batch_size)

    def compute_single(k: Array, x: Array) -> tuple[Array, Array, Array]:
        return estimate_amortization_gap(
            model, k, params, x, n_elbo_samples, n_iwae_samples
        )

    gaps, elbos, iwaes = jax.vmap(compute_single)(keys, xs)

    return jnp.mean(gaps), jnp.mean(elbos), jnp.mean(iwaes)


def compute_analytical_rho(
    model: MixtureModel,
    hrm_params: Array,
    key: Array,
    n_samples: int | None = None,
) -> tuple[Array, Array, Array]:
    """Compute optimal rho via least squares regression.

    Solves: ρ* = argmin Σ(χ + ρ·s_Y(y) - ψ_X(θ_X + Θ_XY·s_Y(y)))²

    IMPORTANT: n_samples must be >> lat_dim for well-conditioned lstsq.
    Default: 5 * bas_lat_man.dim

    Works for any base latent (Bernoulli, VonMises, etc.) via bas_lat_man.

    Args:
        model: The mixture model (BinomialBernoulliMixture, etc.)
        hrm_params: Harmonium parameters (without rho)
        key: JAX random key
        n_samples: Number of samples for regression (default: 5 * lat_dim)

    Returns:
        Tuple of (rho_star, r_squared, chi) where:
        - rho_star: Optimal rho parameters
        - r_squared: R² goodness of fit (clipped to [-10, 1])
        - chi: Intercept term
    """
    lat_dim = model.bas_lat_man.dim

    # Default: 5x latent dimension for overdetermined system
    if n_samples is None:
        n_samples = ANALYTICAL_RHO_SAMPLES_MULTIPLIER * lat_dim

    # Sample from prior mixture and extract base latent (y) components
    _, _, prior_mixture_params = model.hrm.split_coords(hrm_params)
    z_samples = model.mix_man.sample(key, prior_mixture_params, n_samples)
    # Extract y components (first bas_lat_man.dim dimensions)
    y_samples = z_samples[:, :lat_dim]

    # Get sufficient statistics: shape (n_samples, lat_dim)
    s_y_samples = jax.vmap(model.bas_lat_man.sufficient_statistic)(y_samples)

    # Compute ψ_X at each sample
    obs_bias, int_params, _ = model.hrm.split_coords(hrm_params)
    int_matrix = int_params.reshape(model.n_observable, lat_dim)

    def compute_psi(y: Array) -> Array:
        s_y = model.bas_lat_man.sufficient_statistic(y)
        lkl_nat = obs_bias + int_matrix @ s_y
        return model.obs_man.log_partition_function(lkl_nat)

    psi_values = jax.vmap(compute_psi)(y_samples)

    # Least squares: ψ ≈ χ + ρ·s_Y
    # Design matrix: (n_samples, lat_dim + 1)
    design = jnp.concatenate([jnp.ones((n_samples, 1)), s_y_samples], axis=1)
    coeffs = jnp.linalg.lstsq(design, psi_values, rcond=None)[0]
    chi, rho_star = coeffs[0], coeffs[1:]

    # R² for diagnostics
    predicted = design @ coeffs
    ss_res = jnp.sum((psi_values - predicted) ** 2)
    ss_tot = jnp.sum((psi_values - jnp.mean(psi_values)) ** 2)
    r_squared = 1.0 - ss_res / jnp.maximum(ss_tot, 1e-6)

    return rho_star, jnp.clip(r_squared, -10.0, 1.0), chi
