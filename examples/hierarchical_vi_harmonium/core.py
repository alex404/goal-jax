"""Core functions for hierarchical variational inference with mixture of VonMises.

This module implements the ELBO computation and approximate posterior for
a hierarchical harmonium with mixture of VonMises latents for clustering.

Mathematical framework:
- Generative model: p(k), p(z|k), p(x|z)
- Latent manifold: Mixture[VonMisesProduct] with params [obs_bias, int_mat, cat_params]
- Approximate posterior: hat{theta}_{ZK|X}(x) = theta_{ZK} + Theta_{X,ZK}^T s_X(x) - [rho_Z, 0, 0]
- ELBO: E_q[log p(x|z)] + E_q[log p(z,k)] - E_q[log q(z,k|x)]

Key insight: rho_Z only affects the VonMises observable bias, not the mixture
interaction or categorical parameters, maintaining the mixture structure.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from goal.models import BinomialVonMisesMixture


class HierarchicalVIState(NamedTuple):
    """Training state for hierarchical variational inference.

    Attributes:
        harmonium_params: Model parameters [theta_X, Theta_XZ, theta_{ZK}]
            where theta_{ZK} are mixture params [obs_bias, int_mat, cat_params]
        rho_params: Correction term rho_Z (dim = VonMisesProduct.dim, not full mixture)
    """

    harmonium_params: Array
    rho_params: Array


@dataclass
class HierarchicalVIConfig:
    """Configuration for hierarchical variational inference.

    Attributes:
        n_mc_samples: Number of MC samples for ELBO estimation
        kl_weight: Beta-VAE style weighting on KL term (default 1.0)
    """

    n_mc_samples: int = 10
    kl_weight: float = 1.0


def approximate_posterior_at(
    model: BinomialVonMisesMixture,
    harmonium_params: Array,
    rho_params: Array,
    x: Array,
) -> Array:
    """Compute approximate posterior mixture parameters at observation x.

    hat{theta}_{ZK|X}(x) = theta_{ZK} + Theta_{X,ZK}^T s_X(x) - [rho_Z, 0, 0]

    The exact posterior (from the harmonium structure) is:
        theta_{ZK} + Theta_{X,ZK}^T s_X(x)

    We subtract rho_Z from the obs_bias (VonMises) component only, leaving
    the mixture interaction matrix and categorical parameters unchanged.

    Args:
        model: The hierarchical harmonium model
        harmonium_params: Model parameters
        rho_params: Correction term rho_Z (shape: VonMisesProduct.dim = 2 * n_latent)
        x: Observation (shape: n_observable)

    Returns:
        Natural parameters of approximate posterior q(z,k|x) as mixture params
    """
    # Get exact posterior from harmonium structure
    exact_posterior = model.posterior_at(harmonium_params, x)

    # Subtract rho from obs_bias only
    obs_bias, int_mat, cat_params = model.pst_man.split_coords(exact_posterior)
    shifted_obs_bias = obs_bias - rho_params
    return model.pst_man.join_coords(shifted_obs_bias, int_mat, cat_params)


def sample_from_mixture(
    key: Array,
    model: BinomialVonMisesMixture,
    mixture_params: Array,
    n_samples: int,
) -> tuple[Array, Array]:
    """Sample (z, k) pairs from the mixture of VonMises.

    First samples k from the categorical, then samples z from the
    corresponding VonMises component.

    Args:
        key: JAX random key
        model: The model
        mixture_params: Mixture natural parameters
        n_samples: Number of samples

    Returns:
        Tuple of (z_samples, k_samples) where:
        - z_samples: shape (n_samples, n_latent) - angles
        - k_samples: shape (n_samples,) - cluster indices
    """
    key_k, key_z = jax.random.split(key)

    # Get cluster probabilities
    _, _, cat_params = model.pst_man.split_coords(mixture_params)
    cat_means = model.pst_man.lat_man.to_mean(cat_params)
    probs = model.pst_man.lat_man.to_probs(cat_means)

    # Sample cluster assignments
    k_samples = jax.random.categorical(key_k, jnp.log(probs), shape=(n_samples,))

    # Get component parameters
    components, _ = model.pst_man.split_natural_mixture(mixture_params)
    comp_params_2d = model.pst_man.cmp_man.to_2d(components)

    # Sample z for each k
    def sample_z_given_k(key: Array, k: Array) -> Array:
        vm_params = comp_params_2d[k]
        return model.vonmises_man.sample(key, vm_params, 1)[0]

    z_keys = jax.random.split(key_z, n_samples)
    z_samples = jax.vmap(sample_z_given_k)(z_keys, k_samples)

    return z_samples, k_samples


def compute_log_joint(
    model: BinomialVonMisesMixture,
    harmonium_params: Array,
    x: Array,
    z: Array,
    k: Array,
) -> Array:
    """Compute log p(x, z, k) = log p(k) + log p(z|k) + log p(x|z).

    Args:
        model: The model
        harmonium_params: Model parameters
        x: Observation
        z: Latent angles (shape: n_latent)
        k: Cluster index (scalar)

    Returns:
        Log joint probability (scalar)
    """
    s_x = model.obs_man.sufficient_statistic(x)

    # Get prior mixture parameters
    _, _, prior_mixture_params = model.split_coords(harmonium_params)

    # log p(k)
    _, _, cat_params = model.pst_man.split_coords(prior_mixture_params)
    cat_means = model.pst_man.lat_man.to_mean(cat_params)
    probs = model.pst_man.lat_man.to_probs(cat_means)
    log_pk = jnp.log(probs[k])

    # log p(z|k) - VonMises for component k
    components, _ = model.pst_man.split_natural_mixture(prior_mixture_params)
    comp_params_2d = model.pst_man.cmp_man.to_2d(components)
    vm_params_k = comp_params_2d[k]
    s_z = model.vonmises_man.sufficient_statistic(z)
    log_pz_given_k = (
        jnp.dot(s_z, vm_params_k)
        - model.vonmises_man.log_partition_function(vm_params_k)
        + model.vonmises_man.log_base_measure(z)
    )

    # log p(x|z)
    lkl_params = model.likelihood_at(harmonium_params, z)
    log_px_given_z = (
        jnp.dot(s_x, lkl_params)
        - model.obs_man.log_partition_function(lkl_params)
        + model.obs_man.log_base_measure(x)
    )

    return log_pk + log_pz_given_k + log_px_given_z


def compute_log_posterior(
    model: BinomialVonMisesMixture,
    posterior_params: Array,
    z: Array,
    k: Array,
) -> Array:
    """Compute log q(z, k | x) from approximate posterior.

    Args:
        model: The model
        posterior_params: Approximate posterior mixture parameters
        z: Latent angles
        k: Cluster index

    Returns:
        Log posterior probability (scalar)
    """
    # log q(k)
    _, _, cat_params = model.pst_man.split_coords(posterior_params)
    cat_means = model.pst_man.lat_man.to_mean(cat_params)
    probs = model.pst_man.lat_man.to_probs(cat_means)
    log_qk = jnp.log(probs[k])

    # log q(z|k) - VonMises for component k
    components, _ = model.pst_man.split_natural_mixture(posterior_params)
    comp_params_2d = model.pst_man.cmp_man.to_2d(components)
    vm_params_k = comp_params_2d[k]
    s_z = model.vonmises_man.sufficient_statistic(z)
    log_qz_given_k = (
        jnp.dot(s_z, vm_params_k)
        - model.vonmises_man.log_partition_function(vm_params_k)
        + model.vonmises_man.log_base_measure(z)
    )

    return log_qk + log_qz_given_k


def make_elbo_loss_and_grad_fn(
    model: BinomialVonMisesMixture,
    config: HierarchicalVIConfig,
):
    """Create a JIT-compiled function that returns loss and gradients.

    The ELBO is:
        L(x) = E_q[log p(x,z,k)] - E_q[log q(z,k|x)]
             = E_q[log p(x,z,k)] + H(q)

    Uses REINFORCE (score function estimator) since VonMises sampling
    is not reparameterizable.

    Returns:
        A function (key, harm_params, rho_params, batch) -> (loss, harm_grad, rho_grad)
    """

    def single_sample_elbo_and_grads(
        key: Array,
        harm_params: Array,
        rho_params: Array,
        x: Array,
    ) -> tuple[Array, Array, Array]:
        """Compute ELBO and gradients for a single observation."""
        # Get approximate posterior
        q_params = approximate_posterior_at(model, harm_params, rho_params, x)

        # Sample (z, k) from posterior
        z_samples, k_samples = sample_from_mixture(
            key, model, q_params, config.n_mc_samples
        )

        # Compute log joint and log posterior for each sample
        log_joints = jax.vmap(
            lambda z, k: compute_log_joint(model, harm_params, x, z, k)
        )(z_samples, k_samples)

        log_posteriors = jax.vmap(
            lambda z, k: compute_log_posterior(model, q_params, z, k)
        )(z_samples, k_samples)

        # Learning signal f = log p(x,z,k) - log q(z,k|x)
        f_samples = log_joints - log_posteriors
        f_mean = jnp.mean(f_samples)
        f_centered = f_samples - f_mean

        # ELBO = E_q[f] (since E_q[log p - log q] = E_q[log p] + H(q))
        elbo = jnp.mean(log_joints) + jnp.mean(-log_posteriors)

        # === Direct gradient for harmonium params ===
        def mean_log_joint(hp: Array) -> Array:
            """E_q[log p(x,z,k)] with samples stopped."""
            return jnp.mean(
                jax.vmap(
                    lambda z, k: compute_log_joint(
                        model, hp, x, jax.lax.stop_gradient(z), jax.lax.stop_gradient(k)
                    )
                )(z_samples, k_samples)
            )

        _, direct_grad_harm = jax.value_and_grad(mean_log_joint)(harm_params)

        # === REINFORCE correction ===
        # Score function for mixture posterior is more complex:
        # nabla_params log q(z,k|x) involves both VonMises and categorical scores

        def compute_score_grad(z: Array, k: Array, f_c: Array) -> tuple[Array, Array]:
            """Compute REINFORCE gradient contribution for one sample."""

            def log_q_fn(hp: Array, rp: Array) -> Array:
                qp = approximate_posterior_at(model, hp, rp, x)
                return compute_log_posterior(model, qp, z, k)

            # Gradient of log q w.r.t. harm_params and rho_params
            grad_hp, grad_rp = jax.grad(log_q_fn, argnums=(0, 1))(
                harm_params, rho_params
            )

            return f_c * grad_hp, f_c * grad_rp

        # Sum REINFORCE contributions
        reinforce_harm, reinforce_rho = jax.vmap(compute_score_grad)(
            z_samples, k_samples, f_centered
        )
        reinforce_grad_harm = jnp.mean(reinforce_harm, axis=0)
        reinforce_grad_rho = jnp.mean(reinforce_rho, axis=0)

        # Total gradients
        total_grad_harm = direct_grad_harm + reinforce_grad_harm
        total_grad_rho = reinforce_grad_rho

        return elbo, total_grad_harm, total_grad_rho

    def loss_and_grad(
        key: Array, harm_params: Array, rho_params: Array, batch: Array
    ) -> tuple[Array, Array, Array]:
        """Compute loss and gradients for a batch."""
        batch_size = batch.shape[0]
        keys = jax.random.split(key, batch_size)

        # Compute ELBO and gradients for each sample
        elbos, harm_grads, rho_grads = jax.vmap(
            lambda k, x: single_sample_elbo_and_grads(k, harm_params, rho_params, x)
        )(keys, batch)

        # Average
        mean_elbo = jnp.mean(elbos)
        mean_harm_grad = jnp.mean(harm_grads, axis=0)
        mean_rho_grad = jnp.mean(rho_grads, axis=0)

        # Loss is negative ELBO
        loss = -mean_elbo

        return loss, -mean_harm_grad, -mean_rho_grad

    return jax.jit(loss_and_grad)


def compute_elbo_components(
    key: Array,
    model: BinomialVonMisesMixture,
    harmonium_params: Array,
    rho_params: Array,
    x: Array,
    config: HierarchicalVIConfig,
) -> tuple[Array, Array, Array]:
    """Compute ELBO and its components for monitoring.

    ELBO = E_q[log p(x|z)] - D_KL(q(z,k|x) || p(z,k))

    Returns:
        Tuple of (elbo, reconstruction_term, kl_term)
    """
    # Get approximate posterior
    q_params = approximate_posterior_at(model, harmonium_params, rho_params, x)

    # Sample (z, k)
    z_samples, k_samples = sample_from_mixture(
        key, model, q_params, config.n_mc_samples
    )

    # Reconstruction: E_q[log p(x|z)]
    s_x = model.obs_man.sufficient_statistic(x)

    def log_likelihood_at_z(z: Array) -> Array:
        lkl_params = model.likelihood_at(harmonium_params, z)
        return (
            jnp.dot(s_x, lkl_params)
            - model.obs_man.log_partition_function(lkl_params)
            + model.obs_man.log_base_measure(x)
        )

    recon = jnp.mean(jax.vmap(log_likelihood_at_z)(z_samples))

    # KL = E_q[log q(z,k|x)] - E_q[log p(z,k)]
    # Get prior mixture parameters
    _, _, prior_mixture_params = model.split_coords(harmonium_params)

    def log_prior_zk(z: Array, k: Array) -> Array:
        """log p(z,k) = log p(k) + log p(z|k)"""
        _, _, cat_params = model.pst_man.split_coords(prior_mixture_params)
        cat_means = model.pst_man.lat_man.to_mean(cat_params)
        probs = model.pst_man.lat_man.to_probs(cat_means)
        log_pk = jnp.log(probs[k])

        components, _ = model.pst_man.split_natural_mixture(prior_mixture_params)
        comp_params_2d = model.pst_man.cmp_man.to_2d(components)
        vm_params_k = comp_params_2d[k]
        s_z = model.vonmises_man.sufficient_statistic(z)
        log_pz_k = (
            jnp.dot(s_z, vm_params_k)
            - model.vonmises_man.log_partition_function(vm_params_k)
            + model.vonmises_man.log_base_measure(z)
        )
        return log_pk + log_pz_k

    log_priors = jax.vmap(log_prior_zk)(z_samples, k_samples)
    log_posts = jax.vmap(lambda z, k: compute_log_posterior(model, q_params, z, k))(
        z_samples, k_samples
    )

    kl = jnp.mean(log_posts - log_priors)

    elbo = recon - config.kl_weight * kl

    return elbo, recon, kl


def make_elbo_metrics_fn(
    model: BinomialVonMisesMixture,
    config: HierarchicalVIConfig,
):
    """Create a JIT-compiled function to compute ELBO metrics over a batch."""

    def batch_metrics(
        key: Array, harm_params: Array, rho_params: Array, batch: Array
    ) -> tuple[Array, Array, Array]:
        """Compute mean ELBO components."""
        batch_size = batch.shape[0]
        keys = jax.random.split(key, batch_size)

        def single_components(k: Array, x: Array) -> tuple[Array, Array, Array]:
            return compute_elbo_components(k, model, harm_params, rho_params, x, config)

        elbos, recons, kls = jax.vmap(single_components)(keys, batch)

        return jnp.mean(elbos), jnp.mean(recons), jnp.mean(kls)

    return jax.jit(batch_metrics)


def get_cluster_assignments(
    model: BinomialVonMisesMixture,
    harmonium_params: Array,
    rho_params: Array,
    x: Array,
) -> Array:
    """Get hard cluster assignment for observation x.

    Returns argmax_k q(k|x).

    Args:
        model: The model
        harmonium_params: Model parameters
        rho_params: Correction parameters
        x: Observation

    Returns:
        Cluster index (scalar integer)
    """
    q_params = approximate_posterior_at(model, harmonium_params, rho_params, x)
    probs = model.get_cluster_probs(q_params)
    return jnp.argmax(probs)


def get_cluster_probs_at(
    model: BinomialVonMisesMixture,
    harmonium_params: Array,
    rho_params: Array,
    x: Array,
) -> Array:
    """Get cluster probabilities q(k|x) for observation x.

    Args:
        model: The model
        harmonium_params: Model parameters
        rho_params: Correction parameters
        x: Observation

    Returns:
        Array of cluster probabilities (shape: n_clusters)
    """
    q_params = approximate_posterior_at(model, harmonium_params, rho_params, x)
    return model.get_cluster_probs(q_params)


def sample_from_model(
    key: Array,
    model: BinomialVonMisesMixture,
    harmonium_params: Array,
    n: int,
) -> tuple[Array, Array, Array]:
    """Sample from the generative model.

    Uses ancestral sampling:
    1. Sample k ~ p(k)
    2. Sample z ~ p(z|k)
    3. Sample x ~ p(x|z)

    Args:
        key: JAX random key
        model: The model
        harmonium_params: Model parameters
        n: Number of samples

    Returns:
        Tuple of (x_samples, z_samples, k_samples)
    """
    key_k, key_x = jax.random.split(key)

    # Get prior mixture parameters
    _, _, prior_mixture_params = model.split_coords(harmonium_params)

    # Sample (z, k) from prior mixture
    z_samples, k_samples = sample_from_mixture(
        jax.random.fold_in(key_k, 0), model, prior_mixture_params, n
    )

    # Sample x given z
    def sample_x_given_z(subkey: Array, z: Array) -> Array:
        lkl_params = model.likelihood_at(harmonium_params, z)
        return model.obs_man.sample(subkey, lkl_params, 1)[0]

    x_keys = jax.random.split(key_x, n)
    x_samples = jax.vmap(sample_x_given_z)(x_keys, z_samples)

    return x_samples, z_samples, k_samples
