"""Binomial-VonMises variational conjugated model.

This module provides a variational inference model for a harmonium with
Binomial observables and VonMises latents.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import DifferentiableVariationalConjugated
from ..base.binomial import Binomials
from ..base.von_mises import VonMisesProduct
from ..harmonium.rbm import BinomialVonMisesRBM


@dataclass(frozen=True)
class BinomialVonMisesVI(
    DifferentiableVariationalConjugated[Binomials, VonMisesProduct]
):
    """Variational conjugated model with Binomial observables and VonMises latents.

    This wraps a BinomialVonMisesRBM harmonium with learnable conjugation
    parameters for variational inference.

    The approximate posterior at observation x is:
        q(z|x) = VonMises(theta_Z + Theta_{XZ}^T s_X(x) - rho)

    where rho is learned to minimize the ELBO.

    Attributes:
        hrm: The underlying BinomialVonMisesRBM harmonium
    """

    hrm: BinomialVonMisesRBM
    """Override with concrete type for type checking."""

    # Convenience accessors

    @property
    def n_observable(self) -> int:
        """Number of observable units."""
        return self.hrm.n_observable

    @property
    def n_latent(self) -> int:
        """Number of latent VonMises units."""
        return self.hrm.n_latent

    @property
    def n_trials(self) -> int:
        """Number of trials for each Binomial."""
        return self.hrm.n_trials

    # Convenience methods

    def observable_means(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] for all observable units."""
        _, hrm_params = self.split_coords(params)
        return self.hrm.observable_means(hrm_params, z)

    def observable_probabilities(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] / n_trials for all observable units."""
        return self.observable_means(params, z) / self.n_trials

    def reconstruct(self, params: Array, x: Array) -> Array:
        """Reconstruct observable via mean-field approximation.

        Uses E[x | E[z|x]] where the expectation is under q(z|x).
        """
        # Get approximate posterior parameters
        q_params = self.approximate_posterior_at(params, x)

        # Compute mean sufficient statistics from VonMises posterior
        z_mean_stats = self.pst_man.to_mean(q_params)

        # Get likelihood parameters using mean sufficient statistics
        _, hrm_params = self.split_coords(params)
        obs_bias, int_params, _ = self.hrm.split_coords(hrm_params)
        int_matrix = int_params.reshape(self.n_observable, 2 * self.n_latent)
        lkl_natural = obs_bias + int_matrix @ z_mean_stats

        return self.obs_man.to_mean(lkl_natural)

    def reconstruction_error(self, params: Array, xs: Array) -> Array:
        """Compute mean squared reconstruction error over a batch."""
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        return jnp.mean((xs - recons) ** 2)

    def normalized_reconstruction_error(self, params: Array, xs: Array) -> Array:
        """Compute MSE on normalized [0, 1] data."""
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        xs_norm = xs / self.n_trials
        recons_norm = recons / self.n_trials
        return jnp.mean((xs_norm - recons_norm) ** 2)

    def get_filters(self, params: Array, img_shape: tuple[int, int]) -> Array:
        """Reshape weight matrix into filter images for visualization."""
        _, hrm_params = self.split_coords(params)
        return self.hrm.get_filters(hrm_params, img_shape)


def binomial_vonmises_vi(
    n_observable: int, n_latent: int, n_trials: int = 16
) -> BinomialVonMisesVI:
    """Create a Binomial-VonMises variational conjugated model.

    Args:
        n_observable: Number of observable units
        n_latent: Number of VonMises latent units
        n_trials: Number of trials for each Binomial

    Returns:
        BinomialVonMisesVI instance
    """
    hrm = BinomialVonMisesRBM(
        n_observable=n_observable, n_latent=n_latent, n_trials=n_trials
    )
    return BinomialVonMisesVI(hrm=hrm)


def make_elbo_loss_and_grad_fn(
    model: BinomialVonMisesVI,
    n_samples: int = 10,
    kl_weight: float = 1.0,
):
    """Create a JIT-compiled function that returns loss and gradients.

    Uses REINFORCE (score function estimator) since VonMises sampling
    is not reparameterizable.

    Args:
        model: The BinomialVonMisesVI model
        n_samples: Number of MC samples for ELBO estimation
        kl_weight: Weight on KL term (1.0 = standard VAE, <1.0 = beta-VAE)

    Returns:
        A function (key, params, batch) -> (loss, grad)
    """

    def single_sample_elbo_and_grads(
        key: Array,
        params: Array,
        x: Array,
    ) -> tuple[Array, Array]:
        """Compute ELBO and gradients for a single observation."""
        # Get sufficient statistics
        s_x = model.obs_man.sufficient_statistic(x)

        # Compute approximate posterior parameters and mean
        q_params = model.approximate_posterior_at(params, x)
        q_means = model.pst_man.to_mean(q_params)
        psi_q = model.pst_man.log_partition_function(q_params)

        # Sample z from q (non-differentiable for VonMises)
        z_samples = model.pst_man.sample(key, q_params, n_samples)
        s_z_samples = jax.vmap(model.pst_man.sufficient_statistic)(z_samples)

        # Compute learning signal f_tilde(z) = rho . s_Z(z) - psi_X(lkl(z))
        f_tilde_samples = jax.vmap(lambda z: model.reduced_learning_signal(params, z))(
            z_samples
        )
        f_tilde_mean = jnp.mean(f_tilde_samples)  # baseline b(x)
        f_tilde_centered = f_tilde_samples - f_tilde_mean

        # Score function: nabla_q_params log q(z|x) = s_Z(z) - mu(x)
        score_samples = s_z_samples - q_means  # (n_samples, lat_dim)

        # Gradient for rho (pure REINFORCE)
        # nabla_rho L = -E_q[(f - b)(s_Z - mu)]  (d_q/d_rho = -I)
        reinforce_rho = -jnp.mean(f_tilde_centered[:, None] * score_samples, axis=0)

        # Direct gradient E_q[nabla_theta log p_XZ] via autograd
        def log_joint_at_z(hp: Array, z: Array) -> Array:
            """log p_{XZ}(x, z) for fixed x and z."""
            _, hrm_params = model.split_coords(hp)
            _, _, lp = model.hrm.split_coords(hrm_params)

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

        mean_log_p, direct_grad = jax.value_and_grad(mean_log_joint)(params)

        # For harmonium params, we also need REINFORCE for lat_params
        # since q depends on lat_params via posterior_at
        hrm_reinforce = jnp.zeros(model.snd_man.dim)

        # lat_params contribution: d_q/d_lat = I (posterior_at adds lat_params)
        reinforce_lat = jnp.mean(f_tilde_centered[:, None] * score_samples, axis=0)

        # Place reinforce_lat in the correct position within hrm params
        obs_dim = model.obs_man.dim
        int_dim = model.hrm.int_man.dim
        hrm_reinforce = hrm_reinforce.at[obs_dim + int_dim :].set(reinforce_lat)

        # int_params contribution: d_q/d_int gives s_X(x) outer score
        reinforce_int = jnp.mean(
            jax.vmap(lambda fc, sc: fc * model.hrm.int_man.outer_product(s_x, sc))(
                f_tilde_centered, score_samples
            ),
            axis=0,
        )
        hrm_reinforce = hrm_reinforce.at[obs_dim : obs_dim + int_dim].set(reinforce_int)

        # Total REINFORCE gradient
        full_reinforce = jnp.concatenate([reinforce_rho, hrm_reinforce])

        # Total gradient = direct + REINFORCE
        total_grad = direct_grad + full_reinforce

        # Compute ELBO value with kl_weight
        # ELBO = E_q[log p(x|z)] - kl_weight * KL
        # = E_q[log p_XZ] + H(q) adjusted for kl_weight
        entropy_q = psi_q - jnp.dot(q_params, q_means)

        # For beta-VAE, we weight the KL term
        # Standard ELBO = E_q[log p_XZ] + H(q) = E_q[log p(x|z)] - KL
        # Beta-VAE ELBO = E_q[log p(x|z)] - kl_weight * KL
        # = E_q[log p_XZ] + H(q) - (kl_weight - 1) * KL
        if kl_weight != 1.0:
            kl = model.elbo_divergence(params, x)
            elbo = mean_log_p + entropy_q - (kl_weight - 1.0) * kl
        else:
            elbo = mean_log_p + entropy_q

        return elbo, total_grad

    def loss_and_grad(key: Array, params: Array, batch: Array) -> tuple[Array, Array]:
        """Compute loss and gradients for a batch."""
        batch_size = batch.shape[0]
        keys = jax.random.split(key, batch_size)

        # Compute ELBO and gradients for each sample
        elbos, grads = jax.vmap(
            lambda k, x: single_sample_elbo_and_grads(k, params, x)
        )(keys, batch)

        # Average
        mean_elbo = jnp.mean(elbos)
        mean_grad = jnp.mean(grads, axis=0)

        # Loss is negative ELBO
        loss = -mean_elbo

        return loss, -mean_grad

    return jax.jit(loss_and_grad)


def make_elbo_metrics_fn(
    model: BinomialVonMisesVI,
    n_samples: int = 10,
    kl_weight: float = 1.0,
):
    """Create a JIT-compiled function to compute ELBO metrics over a batch.

    Args:
        model: The BinomialVonMisesVI model
        n_samples: Number of MC samples for ELBO estimation
        kl_weight: Weight on KL term (1.0 = standard VAE, <1.0 = beta-VAE)

    Returns:
        A function (key, params, batch) -> (mean_elbo, mean_recon, mean_kl, conj_error)
    """

    def batch_metrics(
        key: Array, params: Array, batch: Array
    ) -> tuple[Array, Array, Array, Array]:
        """Compute mean ELBO components and conjugation error."""
        batch_size = batch.shape[0]
        key, conj_key = jax.random.split(key)
        keys = jax.random.split(key, batch_size)

        def single_components(k: Array, x: Array) -> tuple[Array, Array, Array]:
            return model.elbo_components(k, params, x, n_samples, kl_weight)

        elbos, recons, kls = jax.vmap(single_components)(keys, batch)

        # Compute conjugation error (once, not per sample)
        conj_err = model.conjugation_error(conj_key, params, n_samples=100)

        return jnp.mean(elbos), jnp.mean(recons), jnp.mean(kls), conj_err

    return jax.jit(batch_metrics)
