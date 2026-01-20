# ruff: noqa: RUF002, RUF003
"""Approximate conjugation for harmoniums.

This module implements approximate conjugation for general harmoniums (including RBMs)
where exact conjugation is not available. The key idea is to introduce trainable
conjugation parameters rho that define a variational prior Q_rho(z) in the latent
exponential family, and minimize D(Q_rho || Q_Z) as a regularizer.

For a harmonium with joint distribution:
    p(x, z) = exp(theta_X . s_X(x) + theta_Z . s_Z(z) + s_X(x) . Theta_XZ . s_Z(z))

The variational prior is:
    Q_rho(z) = exp((theta_Z + rho) . s_Z(z) - psi_Z(theta_Z + rho))

where rho are the learned conjugation parameters.

The combined training objective is:
    L = alpha_cd . D(P_X || Q_X) + alpha_conj . D(Q_rho || Q_Z)

All gradient computations for D(Q_rho || Q_Z) use expectations under Q_rho, which is
easy to sample from since it's in the latent exponential family.

Key insight: When Theta_XZ = 0 and rho = 0, the conjugation is exact (D(Q_rho || Q_Z) = 0).
"""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry.exponential_family.harmonium import GibbsHarmonium


class ConjugatedHarmoniumState(NamedTuple):
    """Training state for approximately conjugated harmonium.

    Attributes:
        harmonium_params: Standard harmonium parameters [theta_X, Theta_XZ, theta_Z]
        conjugation_params: rho - same shape as lat_man.dim
    """

    harmonium_params: Array
    conjugation_params: Array


@dataclass
class ApproximateConjugationConfig:
    """Configuration for approximate conjugation training.

    Attributes:
        alpha_cd: Weight on contrastive divergence objective
        alpha_conj: Weight on conjugation objective D(Q_rho || Q_Z)
        n_conj_samples: Number of samples from Q_rho for gradient estimation
        n_gibbs_conj: Number of Gibbs steps to approximate E_p from Q_rho samples
    """

    alpha_cd: float = 1.0
    alpha_conj: float = 0.1
    n_conj_samples: int = 100
    n_gibbs_conj: int = 5


def initialize_conjugated(
    key: Array,
    model: GibbsHarmonium,  # pyright: ignore[reportMissingTypeArgument]
    data: Array,
    shape: float = 0.01,
    zero_interaction: bool = True,
) -> ConjugatedHarmoniumState:
    """Initialize with Theta_XZ = 0, rho = 0 for exact conjugation at start.

    With Theta_XZ = 0 and rho = 0:
    - The conjugation equation is trivially satisfied: psi_X(theta_X) = const
    - Q_rho(z) = Q_Z(z) exactly (both are exp family with params theta_Z)
    - D(Q_rho || Q_Z) = 0
    """
    harmonium_params = model.initialize_from_sample(key, data, shape=shape)

    if zero_interaction:
        obs_params, _, lat_params = model.split_coords(harmonium_params)
        int_params = jnp.zeros(model.int_man.dim)
        harmonium_params = model.join_coords(obs_params, int_params, lat_params)

    conjugation_params = jnp.zeros(model.pst_man.dim)
    return ConjugatedHarmoniumState(harmonium_params, conjugation_params)


def make_conjugation_gradient_fn(
    model: GibbsHarmonium,  # pyright: ignore[reportMissingTypeArgument]
    n_samples: int,
):
    """Create a JIT-compiled conjugation gradient function for the given model.

    Returns a function: (key, harmonium_params, conjugation_params) -> gradient
    """

    def conjugation_gradient(
        key: Array,
        harmonium_params: Array,
        conjugation_params: Array,
    ) -> Array:
        """Compute d_rho D(Q_rho || Q_Z) = Cov_{Q_rho}(f(z), s_Z(z))."""
        _, _, lat_params = model.split_coords(harmonium_params)
        prior_params = lat_params + conjugation_params
        z_samples = model.pst_man.sample(key, prior_params, n_samples)

        def compute_f(z: Array) -> Array:
            s_z = model.pst_man.sufficient_statistic(z)
            term1 = jnp.dot(s_z, conjugation_params)
            lkl_params = model.likelihood_at(harmonium_params, z)
            term2 = model.obs_man.log_partition_function(lkl_params)
            return term1 - term2

        f_vals = jax.vmap(compute_f)(z_samples)
        s_z_vals = jax.vmap(model.pst_man.sufficient_statistic)(z_samples)

        mean_f = jnp.mean(f_vals)
        mean_s = jnp.mean(s_z_vals, axis=0)
        mean_fs = jnp.mean(f_vals[:, None] * s_z_vals, axis=0)

        return mean_fs - mean_f * mean_s

    return jax.jit(conjugation_gradient)


def make_harmonium_conjugation_gradient_fn(
    model: GibbsHarmonium,  # pyright: ignore[reportMissingTypeArgument]
    n_samples: int,
    n_gibbs: int,
):
    """Create a JIT-compiled harmonium conjugation gradient function.

    Computes the gradient of D(Q_rho || Q_Z) w.r.t. harmonium params.

    The gradient is: E_p[stats] - E_{Q_rho}[conditional_stats]

    where E_p is approximated by running Gibbs chains initialized from Q_rho samples.

    Returns a function: (key, harmonium_params, conjugation_params) -> gradient
    """

    def harmonium_conjugation_gradient(
        key: Array,
        harmonium_params: Array,
        conjugation_params: Array,
    ) -> Array:
        """Gradients of D(Q_rho || Q_Z) w.r.t. harmonium params."""
        # Sample z from Q_rho (variational prior)
        key, sample_key = jax.random.split(key)
        _, _, lat_params = model.split_coords(harmonium_params)
        prior_params = lat_params + conjugation_params
        z_samples = model.pst_man.sample(sample_key, prior_params, n_samples)

        # Run Gibbs chains initialized from Q_rho samples to approximate E_p
        def gibbs_from_z(subkey: Array, z_i: Array) -> Array:
            lkl_params = model.likelihood_at(harmonium_params, z_i)
            x_init = model.obs_man.to_mean(lkl_params)
            state = jnp.concatenate([x_init, z_i])
            return model.gibbs_chain(subkey, harmonium_params, state, n_gibbs)

        key, gibbs_key = jax.random.split(key)
        gibbs_keys = jax.random.split(gibbs_key, n_samples)
        joint_samples = jax.vmap(gibbs_from_z)(gibbs_keys, z_samples)

        obs_dim = model.obs_man.data_dim
        x_p = joint_samples[:, :obs_dim]
        z_p = joint_samples[:, obs_dim:]

        # E_p[sufficient_stats] (approximated via Gibbs)
        obs_stats_p = jnp.mean(
            jax.vmap(model.obs_man.sufficient_statistic)(x_p), axis=0
        )
        lat_stats_p = jnp.mean(
            jax.vmap(model.pst_man.sufficient_statistic)(z_p), axis=0
        )

        def outer_stats(x: Array, z: Array) -> Array:
            obs_s = model.obs_man.sufficient_statistic(x)
            lat_s = model.pst_man.sufficient_statistic(z)
            return model.int_man.outer_product(obs_s, lat_s)

        int_stats_p = jnp.mean(jax.vmap(outer_stats)(x_p, z_p), axis=0)

        # E_{Q_rho}[conditional expectations E[x|z]]
        def expected_x_given_z(z: Array) -> Array:
            lkl_params = model.likelihood_at(harmonium_params, z)
            return model.obs_man.to_mean(lkl_params)

        x_cond = jax.vmap(expected_x_given_z)(z_samples)
        s_z = jax.vmap(model.pst_man.sufficient_statistic)(z_samples)

        obs_stats_qrho = jnp.mean(x_cond, axis=0)
        lat_stats_qrho = jnp.mean(s_z, axis=0)

        def outer_cond(x_mean: Array, sz: Array) -> Array:
            return model.int_man.outer_product(x_mean, sz)

        int_stats_qrho = jnp.mean(jax.vmap(outer_cond)(x_cond, s_z), axis=0)

        # Gradient: E_p - E_{Q_rho}[conditional]
        return model.join_coords(
            obs_stats_p - obs_stats_qrho,
            int_stats_p - int_stats_qrho,
            lat_stats_p - lat_stats_qrho,
        )

    return jax.jit(harmonium_conjugation_gradient)


def make_train_step_fn(
    model: GibbsHarmonium,  # pyright: ignore[reportMissingTypeArgument]
    config: ApproximateConjugationConfig,
    cd_steps: int = 1,
):
    """Create a JIT-compiled training step function.

    Returns a function that performs one training step using scan-friendly signature.
    """
    conj_grad_fn = make_conjugation_gradient_fn(model, config.n_conj_samples)
    harm_conj_grad_fn = make_harmonium_conjugation_gradient_fn(
        model, config.n_conj_samples, config.n_gibbs_conj
    )

    def train_step(
        key: Array,
        harmonium_params: Array,
        conjugation_params: Array,
        batch: Array,
    ) -> tuple[Array, Array]:
        """Single training step returning (harm_grad, conj_grad)."""
        key, k1, k2, k3 = jax.random.split(key, 4)

        # 1. Standard CD gradient
        cd_grad = model.mean_contrastive_divergence_gradient(
            k1, harmonium_params, batch, k=cd_steps
        )

        # 2. Harmonium gradient for conjugation
        conj_grad_harm = harm_conj_grad_fn(k2, harmonium_params, conjugation_params)

        # 3. Conjugation parameter gradient
        conj_grad_rho = conj_grad_fn(k3, harmonium_params, conjugation_params)

        # Combined gradients
        combined_harm_grad = (
            config.alpha_cd * cd_grad + config.alpha_conj * conj_grad_harm
        )
        scaled_conj_grad = config.alpha_conj * conj_grad_rho

        return combined_harm_grad, scaled_conj_grad

    return jax.jit(train_step)


def make_estimate_conjugation_error_fn(
    model: GibbsHarmonium,  # pyright: ignore[reportMissingTypeArgument]
    n_samples: int = 200,
):
    """Create a JIT-compiled conjugation error estimation function."""

    def estimate_error(
        key: Array,
        harmonium_params: Array,
        conjugation_params: Array,
    ) -> Array:
        """Estimate conjugation error as Var[f(z)] under Q_rho."""
        _, _, lat_params = model.split_coords(harmonium_params)
        prior_params = lat_params + conjugation_params
        z_samples = model.pst_man.sample(key, prior_params, n_samples)

        def f_value(z: Array) -> Array:
            s_z = model.pst_man.sufficient_statistic(z)
            term1 = jnp.dot(s_z, conjugation_params)
            lkl_params = model.likelihood_at(harmonium_params, z)
            term2 = model.obs_man.log_partition_function(lkl_params)
            return term1 - term2

        f_vals = jax.vmap(f_value)(z_samples)
        return jnp.var(f_vals)

    return jax.jit(estimate_error)


def make_train_epoch_fn(
    train_step_fn: Any,
    opt_harm_update: Any,
    opt_conj_update: Any,
    batch_size: int,
):
    """Create a JIT-compiled epoch training function.

    Args:
        train_step_fn: JIT-compiled function (key, harm_params, conj_params, batch) -> (harm_grad, conj_grad)
        opt_harm_update: Optimizer update function for harmonium params
        opt_conj_update: Optimizer update function for conjugation params
        batch_size: Batch size for training
    """

    def train_epoch(
        key: Array,
        harmonium_params: Array,
        conjugation_params: Array,
        opt_harm_state: Array,
        opt_conj_state: Array,
        data: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Train one epoch using lax.scan over batches."""
        n_samples = data.shape[0]
        n_batches = n_samples // batch_size

        # Shuffle data
        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, n_samples)
        data_shuffled = data[perm]

        # Reshape into batches
        batches = data_shuffled[: n_batches * batch_size].reshape(
            n_batches, batch_size, -1
        )

        # Generate keys for all batches
        key, batch_keys_key = jax.random.split(key)
        batch_keys = jax.random.split(batch_keys_key, n_batches)

        def scan_body(carry, inputs):
            harm_params, conj_params, opt_h_state, opt_c_state = carry
            batch_key, batch = inputs

            # Compute gradients
            harm_grad, conj_grad = train_step_fn(
                batch_key, harm_params, conj_params, batch
            )

            # Update parameters
            new_opt_h_state, new_harm_params = opt_harm_update(
                opt_h_state, harm_grad, harm_params
            )
            new_opt_c_state, new_conj_params = opt_conj_update(
                opt_c_state, conj_grad, conj_params
            )

            return (new_harm_params, new_conj_params, new_opt_h_state, new_opt_c_state), None

        init_carry = (harmonium_params, conjugation_params, opt_harm_state, opt_conj_state)
        (new_harm, new_conj, new_opt_h, new_opt_c), _ = jax.lax.scan(
            scan_body, init_carry, (batch_keys, batches)
        )

        return new_harm, new_conj, new_opt_h, new_opt_c, key

    return jax.jit(train_epoch)
