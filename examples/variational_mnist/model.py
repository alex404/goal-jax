"""Model utilities for variational MNIST.

This module provides:
- Model creation with good defaults
- Conjugation quality metrics (R², Std[residual], ||rho||)
- IWAE bound for amortization gap estimation
"""

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from goal.models import (
    BinomialBernoulliMixture,
    PoissonBernoulliMixture,
    VariationalBinomialVonMisesMixture,
    binomial_bernoulli_mixture,
    poisson_bernoulli_mixture,
    variational_binomial_vonmises_mixture,
)

# Type alias for model union
type MixtureModel = (
    BinomialBernoulliMixture
    | PoissonBernoulliMixture
    | VariationalBinomialVonMisesMixture
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
) -> MixtureModel:
    """Create a mixture model with the specified observable and latent types.

    Args:
        n_latent: Number of latent units (default: 1024)
        n_clusters: Number of mixture components (default: 10)
        observable_type: Type of observable distribution ("binomial" or "poisson")
        latent_type: Type of latent distribution ("bernoulli" or "vonmises")
        n_trials: Binomial discretization level (default: 16, only used for binomial)

    Returns:
        MixtureModel instance
    """
    if observable_type == "poisson":
        return poisson_bernoulli_mixture(
            n_observable=N_OBSERVABLE,
            n_latent=n_latent,
            n_clusters=n_clusters,
        )
    if latent_type == "vonmises":
        return variational_binomial_vonmises_mixture(
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
    rho = model.conjugation_params(params)
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

    # Sample K (y, k) pairs from posterior
    y_samples, k_samples = model.sample_from_posterior(
        key, q_params, n_importance_samples
    )

    # Compute log importance weights: log w = log p(x,y,k) - log q(y,k|x)
    log_joints = jax.vmap(lambda y, k: model.compute_log_joint(params, x, y, k))(
        y_samples, k_samples
    )

    log_posteriors = jax.vmap(lambda y, k: model.compute_log_posterior(q_params, y, k))(
        y_samples, k_samples
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

    # Sample from prior
    _, _, prior_mixture_params = model.hrm.split_coords(hrm_params)
    y_samples, _ = model.sample_from_posterior(key, prior_mixture_params, n_samples)

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


# ============================================================================
# EM-style ELBO optimization functions
# ============================================================================


def compute_positive_phase_stats(
    model: MixtureModel,
    params: Array,
    batch: Array,
) -> Array:
    """Compute positive phase statistics: E_q[s(x,y)|x] averaged over batch.

    For each observation x:
    1. Get approximate posterior q(y,k|x)
    2. Compute E_q[s(x,y)|x] via to_mean

    The positive phase represents the expected sufficient statistics under the
    variational posterior, which corresponds to the "data" term in EM.

    Args:
        model: The mixture model
        params: Full model parameters (including rho)
        batch: Batch of observations, shape (batch_size, obs_dim)

    Returns:
        Average sufficient statistics matching harmonium structure:
        [obs_stats, int_stats, lat_stats]
    """

    def single_positive_stats(x: Array) -> Array:
        # Get approximate posterior q(y,k|x)
        q_params = model.approximate_posterior_at(params, x)

        # Get expected latent stats (mean parameters) via to_mean on the mixture
        # to_mean computes E_q[s_mix(y,k)] for the full mixture structure
        lat_means = model.mix_man.to_mean(q_params)

        # Observable stats are deterministic given x
        obs_stats = model.obs_man.sufficient_statistic(x)

        # Interaction stats using harmonium's int_man for proper embedding
        # This computes s_X(x) ⊗ E[s_Z(z)|x] with proper projections
        int_stats = model.hrm.int_man.outer_product(obs_stats, lat_means)

        # Return full harmonium sufficient statistics
        return model.hrm.join_coords(obs_stats, int_stats, lat_means)

    all_stats = jax.vmap(single_positive_stats)(batch)
    return jnp.mean(all_stats, axis=0)


def compute_negative_phase_stats(
    model: MixtureModel,
    hrm_params: Array,
    key: Array,
    n_samples: int,
) -> Array:
    """Compute negative phase statistics via ancestral sampling from the model.

    Uses ancestral sampling:
    1. Sample (y, k) from prior p(y,k)
    2. Sample x from likelihood p(x|y)
    3. Compute sufficient statistics s(x, (y,k)) and average

    The negative phase represents E_model[s(x,z)], which equals ∇ψ(θ) for
    exponential families. This is the "model" term in EM.

    Args:
        model: The mixture model
        hrm_params: Harmonium parameters (without rho)
        key: JAX random key
        n_samples: Number of samples for estimation

    Returns:
        Average sufficient statistics from model samples, matching harmonium structure
    """
    lat_dim = model.bas_lat_man.dim

    # Sample from prior mixture p(y, k)
    _, _, prior_mix_params = model.hrm.split_coords(hrm_params)
    key1, key2 = jax.random.split(key)
    y_samples, k_samples = model.sample_from_posterior(
        key1, prior_mix_params, n_samples
    )

    # Get observable parameters for sampling
    obs_bias, int_params, _ = model.hrm.split_coords(hrm_params)
    int_matrix = int_params.reshape(model.obs_man.dim, lat_dim)

    # Sample x from likelihood p(x|y) for each y
    def sample_x_given_y(subkey: Array, y: Array) -> Array:
        s_y = model.bas_lat_man.sufficient_statistic(y)
        lkl_params = obs_bias + int_matrix @ s_y
        return model.obs_man.sample(subkey, lkl_params, 1)[0]

    x_keys = jax.random.split(key2, n_samples)
    x_samples = jax.vmap(sample_x_given_y)(x_keys, y_samples)

    # The harmonium's sufficient_statistic expects joint data [x, z] where
    # z is the full mixture data.
    # For CompleteMixture[Bernoullis], the data is (y, k_scalar)
    # where y is the Bernoulli sample (shape: lat_dim) and k is a scalar indicator.
    #
    # We build joint sufficient statistics with the same structure as
    # hrm.sufficient_statistic: [obs_stats, int_stats, lat_stats]

    def compute_joint_stats(x: Array, y: Array, k: Array) -> Array:
        # Observable sufficient statistics
        obs_stats = model.obs_man.sufficient_statistic(x)

        # Build mixture data: concatenate y with k as scalar
        mix_data = jnp.concatenate([y, k[None].astype(y.dtype)])
        lat_stats = model.mix_man.sufficient_statistic(mix_data)

        # Interaction statistics using harmonium's int_man
        # The int_man handles the embedding properly (projects lat_stats)
        int_stats = model.hrm.int_man.outer_product(obs_stats, lat_stats)

        return model.hrm.join_coords(obs_stats, int_stats, lat_stats)

    all_stats = jax.vmap(compute_joint_stats)(x_samples, y_samples, k_samples)
    return jnp.mean(all_stats, axis=0)


def compute_em_elbo_gradient(
    model: MixtureModel,
    hrm_params: Array,
    positive_stats: Array,
    key: Array,
    n_negative_samples: int,
) -> Array:
    """Compute ELBO gradient w.r.t. hrm_params as positive - negative stats.

    For exponential families, the ELBO gradient w.r.t. natural parameters is:
        ∇_θ ELBO = E_q[s(x,y)] - E_model[s(x,y)]
                 = positive_stats - negative_stats

    This is the exact ELBO gradient, just computed via sampling rather than
    through the log-partition function ψ.

    Args:
        model: The mixture model
        hrm_params: Harmonium parameters (without rho)
        positive_stats: Pre-computed from E-step (held fixed during M-step)
        key: JAX random key
        n_negative_samples: Number of samples for negative phase estimation

    Returns:
        Gradient to apply to hrm_params (same shape as hrm_params)
    """
    # Compute negative phase stats (model samples estimate ∇ψ)
    negative_stats = compute_negative_phase_stats(
        model, hrm_params, key, n_negative_samples
    )

    # ELBO gradient = positive - negative
    return positive_stats - negative_stats
