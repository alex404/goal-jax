"""Core functions for structured variational inference.

This module implements the ELBO computation and approximate posterior for
harmoniums with input-dependent recognition models.

Mathematical framework:
- Generative model: p_Z(z; theta_Z), p(x|z) with natural params theta_X + Theta_{XZ} s_Z(z)
- Approximate posterior: hat{theta}_{Z|X}(x) = theta_Z + Theta_{XZ}^T s_X(x) - rho_Z
- ELBO: E_q[log p(x|z)] - D_KL(q_{Z|X=x} || p_Z)

Key insight: Same form as true conjugate posterior, but rho_Z is learnable.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from goal.models import BinomialVonMisesRBM


class StructuredVIState(NamedTuple):
    """Training state for structured variational inference.

    Attributes:
        harmonium_params: Standard harmonium parameters [theta_X, Theta_XZ, theta_Z]
        rho_params: Correction term rho_Z (dim = pst_man.dim)
    """

    harmonium_params: Array
    rho_params: Array


@dataclass
class StructuredVIConfig:
    """Configuration for structured variational inference.

    Attributes:
        n_mc_samples: Number of MC samples for ELBO estimation
        kl_weight: Beta-VAE style weighting on KL term (default 1.0)
        conj_reg_weight: Weight for conjugation error regularizer (default 0.0)
    """

    n_mc_samples: int = 10
    kl_weight: float = 1.0
    conj_reg_weight: float = 0.0


def approximate_posterior_at(
    model: BinomialVonMisesRBM,
    harmonium_params: Array,
    rho_params: Array,
    x: Array,
) -> Array:
    """Compute approximate posterior natural parameters at observation x.

    hat{theta}_{Z|X}(x) = theta_Z + Theta_{XZ}^T s_X(x) - rho_Z

    Args:
        model: The harmonium model
        harmonium_params: Standard harmonium parameters [theta_X, Theta_XZ, theta_Z]
        rho_params: Correction term rho_Z (same shape as pst_man.dim)
        x: Observation (shape: n_observable)

    Returns:
        Natural parameters of approximate posterior q(z|x)
    """
    exact_posterior = model.posterior_at(harmonium_params, x)
    return exact_posterior - rho_params


def compute_reduced_learning_signal(
    model: BinomialVonMisesRBM,
    harmonium_params: Array,
    rho_params: Array,
    z: Array,
) -> Array:
    """Compute the reduced learning signal f_tilde(z, x).

    f_tilde(z, x) = rho_Z . s_Z(z) - psi_X(theta_X + Theta_{XZ} . s_Z(z))

    This is the z-dependent part of the full learning signal f(z, x).
    Any x-dependent terms can be absorbed into the baseline.

    Args:
        model: The harmonium model
        harmonium_params: Harmonium parameters
        rho_params: Correction term rho_Z
        z: Latent sample

    Returns:
        Scalar value of reduced learning signal
    """
    s_z = model.pst_man.sufficient_statistic(z)
    # Linear term: rho . s_Z(z)
    term1 = jnp.dot(rho_params, s_z)
    # Log partition of likelihood: psi_X(theta_X + Theta_{XZ} . s_Z(z))
    lkl_params = model.likelihood_at(harmonium_params, z)
    term2 = model.obs_man.log_partition_function(lkl_params)
    return term1 - term2


def make_elbo_loss_and_grad_fn(
    model: BinomialVonMisesRBM,
    config: StructuredVIConfig,
):
    """Create a JIT-compiled function that returns loss and gradients.

    The ELBO is:
        L(x) = E_q[log p_{XZ}(x, Z)] - E_q[log q(Z|x)]
             = E_q[log p_{XZ}(x, Z)] + H(q)

    Gradients use:
        nabla_theta L = E_q[nabla_theta log p_XZ] + E_q[(f - b) nabla_theta log q]

    where f(z, x) = log p_XZ(x, z) - log q(z|x) is the learning signal.

    Returns:
        A function (key, harm_params, rho_params, batch) -> (loss, harm_grad, rho_grad)
    """

    def single_sample_elbo_and_grads(
        key: Array,
        harm_params: Array,
        rho_params: Array,
        x: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Compute ELBO and gradients for a single observation.

        Uses the score function estimator with optimal baseline.

        The gradient formula is:
            nabla_theta L = E_q[nabla_theta log p_XZ] + E_q[(f - b) nabla_theta log q]

        where f(z, x) = log p_XZ(x, z) - log q(z|x) is the full learning signal.

        Returns:
            Tuple of (elbo, harm_grad, rho_grad, conj_error_at_posterior)
        """
        # Get sufficient statistics
        s_x = model.obs_man.sufficient_statistic(x)

        # Compute approximate posterior parameters and mean
        q_params = approximate_posterior_at(model, harm_params, rho_params, x)
        q_means = model.pst_man.to_mean(q_params)  # hat_mu_{Z|X}(x) = nabla psi_Z(q_params)
        psi_q = model.pst_man.log_partition_function(q_params)

        # Get prior parameters
        _, _, lat_params = model.split_coords(harm_params)

        # Sample z from q (non-differentiable for VonMises)
        z_samples = model.pst_man.sample(key, q_params, config.n_mc_samples)
        s_z_samples = jax.vmap(model.pst_man.sufficient_statistic)(z_samples)

        # === Compute learning signal f_tilde(z) = rho . s_Z(z) - psi_X(lkl(z)) ===
        f_tilde_samples = jax.vmap(
            lambda z: compute_reduced_learning_signal(model, harm_params, rho_params, z)
        )(z_samples)
        f_tilde_mean = jnp.mean(f_tilde_samples)  # baseline b(x)
        f_tilde_centered = f_tilde_samples - f_tilde_mean

        # Conjugation error at this x: Var[f_tilde] under posterior
        conj_err_at_x = jnp.var(f_tilde_samples)

        # Score function: nabla_q_params log q(z|x) = s_Z(z) - mu(x)
        score_samples = s_z_samples - q_means  # (n_samples, lat_dim)

        # === Gradient for rho_Z (pure REINFORCE) ===
        # nabla_rho L = -E_q[(f - b)(s_Z - mu)]  (d_q/d_rho = -I)
        reinforce_rho = -jnp.mean(f_tilde_centered[:, None] * score_samples, axis=0)

        # === Direct gradient E_q[nabla_theta log p_XZ] via autograd ===
        def log_joint_at_z(hp: Array, z: Array) -> Array:
            """log p_{XZ}(x, z) for fixed x and z."""
            _, _, lp = model.split_coords(hp)

            # log p_{X|Z}(x|z)
            lkl_params = model.likelihood_at(hp, z)
            log_likelihood = (
                jnp.dot(s_x, lkl_params)
                - model.obs_man.log_partition_function(lkl_params)
                + model.obs_man.log_base_measure(x)
            )

            # log p_Z(z)
            s_z = model.pst_man.sufficient_statistic(z)
            log_prior = (
                jnp.dot(s_z, lp)
                - model.pst_man.log_partition_function(lp)
                + model.pst_man.log_base_measure(z)
            )

            return log_likelihood + log_prior

        def mean_log_joint(hp: Array) -> Array:
            """E_q[log p_XZ(x, Z)] using samples (stop_gradient on z)."""
            return jnp.mean(
                jax.vmap(lambda z: log_joint_at_z(hp, jax.lax.stop_gradient(z)))(
                    z_samples
                )
            )

        mean_log_p, direct_grad_harm = jax.value_and_grad(mean_log_joint)(harm_params)

        # === REINFORCE correction for harmonium params ===
        # For theta_Z: d_q/d_theta_Z = I, gradient is E[(f-b) * score]
        reinforce_lat = jnp.mean(f_tilde_centered[:, None] * score_samples, axis=0)

        # For Theta_XZ: d_q/d_Theta_XZ gives s_X(x) outer score
        reinforce_int = jnp.mean(
            jax.vmap(lambda fc, sc: fc * model.int_man.outer_product(s_x, sc))(
                f_tilde_centered, score_samples
            ),
            axis=0,
        )

        # For theta_X: no dependence on q, so no REINFORCE term
        reinforce_obs = jnp.zeros(model.obs_man.dim)

        reinforce_grad_harm = model.join_coords(
            reinforce_obs, reinforce_int, reinforce_lat
        )

        # Total harmonium gradient = direct + REINFORCE
        total_grad_harm = direct_grad_harm + reinforce_grad_harm

        # === Compute ELBO value ===
        # ELBO = E_q[log p_XZ] + H(q)
        # H(q) = psi_Z(q_params) - q_params . mu(x)
        entropy_q = psi_q - jnp.dot(q_params, q_means)
        elbo = mean_log_p + entropy_q

        return elbo, total_grad_harm, reinforce_rho, conj_err_at_x

    def conj_error_with_grad(
        key: Array, harm_params: Array, rho_params: Array, n_samples: int = 50
    ) -> tuple[Array, Array, Array]:
        """Compute conjugation error and its gradients.

        ConjErr = Var_z[f_tilde(z)] where z ~ prior

        We compute gradients via autograd by sampling z and computing variance.
        """
        _, _, lat_params = model.split_coords(harm_params)

        # Sample z from prior (stop gradient on samples for proper REINFORCE)
        z_samples = model.pst_man.sample(key, lat_params, n_samples)

        def variance_of_f(hp: Array, rp: Array) -> Array:
            """Compute Var[f_tilde] for fixed samples."""
            f_vals = jax.vmap(
                lambda z: compute_reduced_learning_signal(
                    model, hp, rp, jax.lax.stop_gradient(z)
                )
            )(z_samples)
            return jnp.var(f_vals)

        conj_err, (grad_harm, grad_rho) = jax.value_and_grad(
            variance_of_f, argnums=(0, 1)
        )(harm_params, rho_params)

        return conj_err, grad_harm, grad_rho

    def loss_and_grad(
        key: Array, harm_params: Array, rho_params: Array, batch: Array
    ) -> tuple[Array, Array, Array]:
        """Compute loss and gradients for a batch."""
        batch_size = batch.shape[0]
        key, conj_key = jax.random.split(key)
        keys = jax.random.split(key, batch_size)

        # Compute ELBO and gradients for each sample
        elbos, harm_grads, rho_grads, conj_errs = jax.vmap(
            lambda k, x: single_sample_elbo_and_grads(k, harm_params, rho_params, x)
        )(keys, batch)

        # Average ELBO gradients
        mean_elbo = jnp.mean(elbos)
        mean_harm_grad = jnp.mean(harm_grads, axis=0)
        mean_rho_grad = jnp.mean(rho_grads, axis=0)

        # Conjugation error regularizer (computed once per batch, not per sample)
        if config.conj_reg_weight > 0:
            conj_err, conj_grad_harm, conj_grad_rho = conj_error_with_grad(
                conj_key, harm_params, rho_params
            )
            # Add regularizer gradients: we want to minimize conj_err
            mean_harm_grad = mean_harm_grad - config.conj_reg_weight * conj_grad_harm
            mean_rho_grad = mean_rho_grad - config.conj_reg_weight * conj_grad_rho

        # Loss is negative ELBO (plus conj regularizer, but we track separately)
        loss = -mean_elbo

        return loss, -mean_harm_grad, -mean_rho_grad

    return jax.jit(loss_and_grad)


def compute_elbo_components(
    key: Array,
    model: BinomialVonMisesRBM,
    harmonium_params: Array,
    rho_params: Array,
    x: Array,
    config: StructuredVIConfig,
) -> tuple[Array, Array, Array]:
    """Compute ELBO and its components for monitoring.

    ELBO = E_q[log p(x|z)] - D_KL(q || p_Z)

    Returns:
        Tuple of (elbo, reconstruction_term, kl_term)
    """
    s_x = model.obs_man.sufficient_statistic(x)

    # Get approximate posterior
    q_params = approximate_posterior_at(model, harmonium_params, rho_params, x)
    q_means = model.pst_man.to_mean(q_params)

    # Sample z
    z_samples = model.pst_man.sample(key, q_params, config.n_mc_samples)

    # Reconstruction term: E_q[log p(x|z)]
    def log_likelihood_at_z(z: Array) -> Array:
        lkl_params = model.likelihood_at(harmonium_params, z)
        return (
            jnp.dot(s_x, lkl_params)
            - model.obs_man.log_partition_function(lkl_params)
            + model.obs_man.log_base_measure(x)
        )

    recon = jnp.mean(jax.vmap(log_likelihood_at_z)(z_samples))

    # KL term: D_KL(q || p_Z) = (theta_q - theta_p) . mu_q - psi_q + psi_p
    _, _, prior_params = model.split_coords(harmonium_params)
    param_diff = q_params - prior_params
    psi_q = model.pst_man.log_partition_function(q_params)
    psi_p = model.pst_man.log_partition_function(prior_params)
    kl = jnp.dot(param_diff, q_means) - psi_q + psi_p

    elbo = recon - config.kl_weight * kl

    return elbo, recon, kl


def compute_conjugation_error(
    key: Array,
    model: BinomialVonMisesRBM,
    harmonium_params: Array,
    rho_params: Array,
    n_samples: int = 100,
) -> Array:
    """Compute the conjugation error of the harmonium.

    For a conjugate harmonium, the log-partition function of the likelihood
    should be linear in the latent sufficient statistics:

        psi_X(theta_X + Theta_XZ . s_Z(z)) ~ rho . s_Z(z) + const

    The conjugation error is Var[f_tilde(z)] where:
        f_tilde(z) = rho . s_Z(z) - psi_X(theta_X + Theta_XZ . s_Z(z))

    If perfectly conjugate, f_tilde(z) = const and Var[f_tilde(z)] = 0.

    Args:
        key: JAX random key
        model: The harmonium model
        harmonium_params: Standard harmonium parameters
        rho_params: Conjugation/correction parameters
        n_samples: Number of samples for variance estimation

    Returns:
        Conjugation error (variance of f_tilde(z))
    """
    # Sample z from the prior
    _, _, lat_params = model.split_coords(harmonium_params)
    z_samples = model.pst_man.sample(key, lat_params, n_samples)

    f_vals = jax.vmap(
        lambda z: compute_reduced_learning_signal(model, harmonium_params, rho_params, z)
    )(z_samples)

    return jnp.var(f_vals)


def make_elbo_metrics_fn(
    model: BinomialVonMisesRBM,
    config: StructuredVIConfig,
):
    """Create a JIT-compiled function to compute ELBO metrics over a batch.

    Returns:
        A function (key, harm_params, rho_params, batch) ->
            (mean_elbo, mean_recon, mean_kl, conj_error)
    """

    def batch_metrics(
        key: Array, harm_params: Array, rho_params: Array, batch: Array
    ) -> tuple[Array, Array, Array, Array]:
        """Compute mean ELBO components and conjugation error."""
        batch_size = batch.shape[0]
        key, conj_key = jax.random.split(key)
        keys = jax.random.split(key, batch_size)

        def single_components(k: Array, x: Array) -> tuple[Array, Array, Array]:
            return compute_elbo_components(k, model, harm_params, rho_params, x, config)

        elbos, recons, kls = jax.vmap(single_components)(keys, batch)

        # Compute conjugation error (once, not per sample)
        conj_err = compute_conjugation_error(
            conj_key, model, harm_params, rho_params, n_samples=100
        )

        return jnp.mean(elbos), jnp.mean(recons), jnp.mean(kls), conj_err

    return jax.jit(batch_metrics)


def sample_from_model(
    key: Array,
    model: BinomialVonMisesRBM,
    harmonium_params: Array,
    rho_params: Array,  # noqa: ARG001 - included for API consistency
    n: int,
) -> Array:
    """Sample from the generative model using the prior.

    Uses ancestral sampling:
    1. Sample z ~ p(z) = VonMises(theta_Z)
    2. Sample x ~ p(x|z) = Binomial(n_trials, sigmoid(theta_X + Theta_{XZ} s(z)))

    Note: This samples from the generative model p(x, z), not from the
    approximate posterior. The rho_params don't affect sampling from p.

    Args:
        key: JAX random key
        model: The harmonium model
        harmonium_params: Standard harmonium parameters
        rho_params: Not used for sampling (included for API consistency)
        n: Number of samples

    Returns:
        Samples [x, z] of shape (n, obs_dim + lat_data_dim)
    """
    key1, key2 = jax.random.split(key)

    # Get prior parameters
    _, _, lat_params = model.split_coords(harmonium_params)

    # Sample z from prior p(z) = VonMises(theta_Z)
    z_samples = model.pst_man.sample(key1, lat_params, n)

    # Sample x from likelihood p(x|z) for each z
    def sample_x_given_z(subkey: Array, z: Array) -> Array:
        lkl_params = model.likelihood_at(harmonium_params, z)
        return model.obs_man.sample(subkey, lkl_params, 1)[0]

    x_keys = jax.random.split(key2, n)
    x_samples = jax.vmap(sample_x_given_z)(x_keys, z_samples)

    return jnp.concatenate([x_samples, z_samples], axis=-1)
