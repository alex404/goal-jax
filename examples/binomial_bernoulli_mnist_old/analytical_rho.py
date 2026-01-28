"""Analytical rho computation for variational conjugation.

This module provides functions to compute the optimal rho parameters analytically
via least-squares regression, rather than learning them. This prevents the optimizer
from "cheating" by flattening the likelihood.

The key insight is that rho* is the optimal linear approximation to psi_X:
    rho* = argmin Σ(rho·s_Y(y) - ψ_X(θ_X + Θ_XY·s_Y(y)))²

We penalize (1-R²) to encourage the likelihood to be linearizable.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


def compute_analytical_rho(
    model: Any,
    hrm_params: Array,
    y_samples: Array,
) -> tuple[Array, Array, Array]:
    """Compute optimal rho via least squares regression.

    Solves: rho* = argmin Σ(chi + rho·s_Y(y) - ψ_X(θ_X + Θ_XY·s_Y(y)))²

    where:
    - s_Y(y) is the sufficient statistic of the base latent
    - ψ_X is the observable log-partition function
    - θ_X is the observable bias
    - Θ_XY is the interaction matrix

    Args:
        model: The BinomialBernoulliMixture model
        hrm_params: Harmonium parameters (without rho)
        y_samples: Samples from the latent space (shape: n_samples, n_latent)

    Returns:
        Tuple of (rho_star, r_squared, chi) where:
        - rho_star: Optimal rho parameters (shape: n_latent)
        - r_squared: R² goodness of fit
        - chi: Intercept term
    """
    # Extract harmonium components
    obs_bias, int_params, _ = model.hrm.split_coords(hrm_params)
    int_matrix = int_params.reshape(model.n_observable, model.n_latent)

    # Compute sufficient statistics for all y samples
    s_y_samples = jax.vmap(model.bas_lat_man.sufficient_statistic)(y_samples)

    # Compute ψ_X at each sample point
    def compute_psi_at_y(y: Array) -> Array:
        s_y = model.bas_lat_man.sufficient_statistic(y)
        lkl_nat = obs_bias + int_matrix @ s_y
        return model.obs_man.log_partition_function(lkl_nat)

    psi_values = jax.vmap(compute_psi_at_y)(y_samples)

    # Build design matrix [1, s_Y(y)]
    n_samples = y_samples.shape[0]
    ones = jnp.ones((n_samples, 1))
    design = jnp.concatenate([ones, s_y_samples], axis=1)

    # Solve least squares: psi ≈ chi + rho^T · s_Y
    # coeffs[0] = chi (intercept), coeffs[1:] = rho
    coeffs = jnp.linalg.lstsq(design, psi_values, rcond=None)[0]
    chi = coeffs[0]
    rho_star = coeffs[1:]

    # Compute R² = 1 - SS_res / SS_tot
    predicted = design @ coeffs
    ss_res = jnp.sum((psi_values - predicted) ** 2)
    ss_tot = jnp.sum((psi_values - jnp.mean(psi_values)) ** 2)

    # Use threshold to avoid numerical instability
    variance_threshold = 1e-6
    raw_r2 = 1.0 - ss_res / jnp.maximum(ss_tot, variance_threshold)
    r_squared = jnp.where(
        ss_tot > variance_threshold,
        jnp.clip(raw_r2, -10.0, 1.0),
        jnp.nan,
    )

    return rho_star, r_squared, chi


def compute_analytical_rho_from_prior(
    key: Array,
    model: Any,
    hrm_params: Array,
    n_samples: int = 100,
) -> tuple[Array, Array, Array]:
    """Sample from the current prior and compute analytical rho.

    This samples from the CURRENT prior distribution (not a fixed distribution)
    so the linear approximation is good in the region where the model operates.

    Args:
        key: JAX random key
        model: The BinomialBernoulliMixture model
        hrm_params: Harmonium parameters (without rho)
        n_samples: Number of samples for regression

    Returns:
        Tuple of (rho_star, r_squared, chi)
    """
    # Get prior mixture parameters
    _, _, prior_mixture_params = model.hrm.split_coords(hrm_params)

    # Sample (y, k) from the prior mixture
    y_samples, _ = model.sample_from_posterior(key, prior_mixture_params, n_samples)

    return compute_analytical_rho(model, hrm_params, y_samples)


def analytical_conjugation_loss(
    key: Array,
    model: Any,
    hrm_params: Array,
    n_samples: int = 100,
) -> tuple[Array, Array, Array]:
    """Compute (1-R²) loss and optimal rho for use in training.

    This is the main function to use in the loss computation. It returns
    the (1-R²) penalty which encourages linearizable ψ_X.

    Args:
        key: JAX random key
        model: The BinomialBernoulliMixture model
        hrm_params: Harmonium parameters (without rho)
        n_samples: Number of samples for regression

    Returns:
        Tuple of (one_minus_r2, rho_star, r_squared) where:
        - one_minus_r2: (1-R²) penalty for loss function
        - rho_star: Optimal rho parameters
        - r_squared: R² goodness of fit
    """
    rho_star, r_squared, _ = compute_analytical_rho_from_prior(
        key, model, hrm_params, n_samples
    )

    # Clamp R² to avoid numerical issues
    r_squared_clamped = jnp.clip(r_squared, -1.0, 1.0)
    one_minus_r2 = 1.0 - r_squared_clamped

    return one_minus_r2, rho_star, r_squared


def compute_var_f_tilde(
    key: Array,
    model: Any,
    hrm_params: Array,
    rho: Array,
    n_samples: int = 100,
) -> tuple[Array, Array]:
    """Compute Var[f̃] where f̃ = rho · s_Y - ψ_X.

    This measures the residual variance after the linear correction.
    Unlike (1-R²), this directly penalizes the absolute variance,
    which provides a stronger learning signal.

    Args:
        key: JAX random key
        model: The BinomialBernoulliMixture model
        hrm_params: Harmonium parameters (without rho)
        rho: The rho parameters (can be analytical or learned)
        n_samples: Number of samples

    Returns:
        Tuple of (var_f_tilde, std_f_tilde)
    """
    # Get prior mixture parameters and sample
    _, _, prior_mixture_params = model.hrm.split_coords(hrm_params)
    y_samples, _ = model.sample_from_posterior(key, prior_mixture_params, n_samples)

    # Extract harmonium components
    obs_bias, int_params, _ = model.hrm.split_coords(hrm_params)
    int_matrix = int_params.reshape(model.n_observable, model.n_latent)

    def f_tilde_at_y(y: Array) -> Array:
        """Compute f̃(y) = rho · s_Y(y) - ψ_X(θ_X + Θ_XY · s_Y(y))"""
        s_y = model.bas_lat_man.sufficient_statistic(y)
        linear_term = jnp.dot(rho, s_y)
        lkl_nat = obs_bias + int_matrix @ s_y
        psi_x = model.obs_man.log_partition_function(lkl_nat)
        return linear_term - psi_x

    f_vals = jax.vmap(f_tilde_at_y)(y_samples)
    var_f = jnp.var(f_vals)
    std_f = jnp.sqrt(var_f)

    return var_f, std_f


def analytical_rho_with_var_penalty(
    key: Array,
    model: Any,
    hrm_params: Array,
    n_samples: int = 100,
) -> tuple[Array, Array, Array, Array]:
    """Compute analytical rho and Var[f̃] penalty.

    This combines:
    1. Analytical rho* via least squares (always optimal)
    2. Var[f̃] penalty to encourage linearizable ψ_X

    The key insight: even with optimal rho, there's residual variance.
    Penalizing this variance forces the model to make ψ_X more linear.

    Args:
        key: JAX random key
        model: The BinomialBernoulliMixture model
        hrm_params: Harmonium parameters (without rho)
        n_samples: Number of samples

    Returns:
        Tuple of (var_f_tilde, rho_star, r_squared, std_f_tilde)
    """
    key1, key2 = jax.random.split(key)

    # Compute analytical rho
    rho_star, r_squared, _ = compute_analytical_rho_from_prior(
        key1, model, hrm_params, n_samples
    )

    # Compute Var[f̃] with the analytical rho
    var_f, std_f = compute_var_f_tilde(key2, model, hrm_params, rho_star, n_samples)

    return var_f, rho_star, r_squared, std_f


def compute_elbo_with_analytical_rho(
    key: Array,
    model: Any,
    hrm_params: Array,
    rho_star: Array,
    batch: Array,
    n_mc_samples: int,
) -> Array:
    """Compute ELBO using analytically-computed rho.

    Args:
        key: JAX random key
        model: The BinomialBernoulliMixture model
        hrm_params: Harmonium parameters (without rho)
        rho_star: Analytically computed optimal rho
        batch: Batch of observations
        n_mc_samples: Number of MC samples for ELBO

    Returns:
        Mean ELBO over the batch
    """
    # Build full params with analytical rho
    params = model.join_coords(rho_star, hrm_params)
    return model.mean_elbo(key, params, batch, n_mc_samples)
