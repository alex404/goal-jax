"""Poisson-VonMises harmonium for toroidal latent discovery.

This module provides a harmonium with Poisson observables and VonMises latents,
suitable for modeling neural spike count data with circular latent structure
(e.g., head direction, spatial tuning, etc.).

The model captures how neurons encode positions on an n-torus through their
spike count tuning curves. Each neuron's firing rate depends on a combination
of circular latent variables through the exponential link:

    rate_i = exp(b_i + W_i @ s(z))

where s(z) = [cos(z_1), sin(z_1), ..., cos(z_n), sin(z_n)] is the von Mises
sufficient statistic for the latent angles z on the n-torus.

Classes:
    PoissonVonMisesHarmonium: Base harmonium with Gibbs sampling support
    VariationalPoissonVonMises: Variational wrapper with learnable rho
"""

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import EmbeddedMap, IdentityEmbedding, Rectangular
from ...geometry.exponential_family.harmonium import DifferentiableHarmonium
from ...geometry.exponential_family.variational import (
    DifferentiableVariationalConjugated,
)
from ..base.poisson import Poissons
from ..base.von_mises import VonMisesProduct


@dataclass(frozen=True)
class PoissonVonMisesHarmonium(DifferentiableHarmonium[Poissons, VonMisesProduct]):
    """Harmonium with Poisson observable units and Von Mises latent units.

    For modeling neural spike counts with circular latent structure.
    Each observable unit is Poisson(rate_i) where rate_i = exp(b_i + W_i @ s(z)).
    Each latent unit is Von Mises with mean and concentration parameters.

    The joint distribution is:

    $$p(x, z) \\propto \\prod_i (rate_i^{x_i} / x_i!)
        \\exp(b^T x + \\theta_z^T s(z) + x^T W s(z))$$

    where:
    - x_i in {0,1,2,...} are observable counts (unbounded)
    - z in [0, 2*pi)^n_latent are circular latent angles
    - s(z) = [cos(z_1), sin(z_1), ..., cos(z_n), sin(z_n)] is the sufficient statistic
    - W is the weight matrix (n_neurons x 2*n_latent)
    - b are observable biases (log baseline rates)
    - theta_z are latent natural parameters

    As a harmonium:
    - obs_man: Poissons(n_neurons) - observable layer (neural populations)
    - pst_man: VonMisesProduct(n_latent) - latent layer (n-torus)
    - int_man: Rectangular weight matrix W

    The conditionals are:
    - p(x_i | z) = Poisson(exp(b_i + W_i @ s(z)))
    - p(z | x) = product of Von Mises with shifted natural parameters

    Attributes:
        n_neurons: Number of observable (Poisson) units
        n_latent: Number of latent (Von Mises) units
    """

    n_neurons: int
    """Number of observable Poisson units (neurons)."""

    n_latent: int
    """Number of latent Von Mises units (dimensions on the torus)."""

    # Properties

    @property
    @override
    def obs_man(self) -> Poissons:
        """Observable layer manifold (Poisson units)."""
        return Poissons(self.n_neurons)

    @property
    @override
    def pst_man(self) -> VonMisesProduct:
        """Latent layer manifold (Von Mises units)."""
        return VonMisesProduct(self.n_latent)

    @property
    @override
    def int_man(self) -> EmbeddedMap[VonMisesProduct, Poissons]:
        """Interaction matrix with identity embeddings (full connectivity).

        The interaction matrix W has shape (n_neurons, 2*n_latent) and connects
        all observable units to all latent Von Mises sufficient statistics.
        """
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(self.pst_man),
            IdentityEmbedding(self.obs_man),
        )

    # Convenience methods

    def observable_rates(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] = exp(b_i + W_i @ s(z)) for all neurons.

        Args:
            params: Harmonium parameters
            z: Latent angles (shape: n_latent)

        Returns:
            Poisson rates for each neuron (shape: n_neurons)
        """
        lkl_params = self.likelihood_at(params, z)
        # For Poisson, mean = exp(natural parameter) = rate
        return self.obs_man.to_mean(lkl_params)

    def latent_sufficient_stats(self, params: Array, x: Array) -> Array:
        """Compute E[s(z) | x] from the posterior parameters.

        Args:
            params: Harmonium parameters
            x: Observable counts (shape: n_neurons)

        Returns:
            Expected sufficient statistics (shape: 2*n_latent)
        """
        post_params = self.posterior_at(params, x)
        return self.pst_man.to_mean(post_params)

    def get_weight_matrix(self, params: Array) -> Array:
        """Extract the interaction weight matrix W.

        Args:
            params: Harmonium parameters

        Returns:
            Weight matrix of shape (n_neurons, 2*n_latent)
        """
        _, int_params, _ = self.split_coords(params)
        return int_params.reshape(self.n_neurons, 2 * self.n_latent)

    def get_baseline_rates(self, params: Array) -> Array:
        """Get baseline firing rates exp(b) when z=0.

        Args:
            params: Harmonium parameters

        Returns:
            Baseline rates for each neuron (shape: n_neurons)
        """
        obs_params, _, _ = self.split_coords(params)
        return jnp.exp(obs_params)


def poisson_vonmises_harmonium(
    n_neurons: int, n_latent: int
) -> PoissonVonMisesHarmonium:
    """Create a Poisson-VonMises Harmonium.

    Factory function for convenient construction.

    Args:
        n_neurons: Number of observable (Poisson) units
        n_latent: Number of latent (Von Mises) units

    Returns:
        PoissonVonMisesHarmonium instance
    """
    return PoissonVonMisesHarmonium(n_neurons=n_neurons, n_latent=n_latent)


@dataclass(frozen=True)
class VariationalPoissonVonMises(
    DifferentiableVariationalConjugated[Poissons, VonMisesProduct]
):
    """Variational conjugated model with Poisson observables and VonMises latents.

    This wraps a PoissonVonMisesHarmonium with learnable conjugation
    parameters for variational inference via ELBO optimization.

    The approximate posterior at observation x is:
        q(z|x) = VonMises(theta_Z + Theta_{XZ}^T s_X(x) - rho)

    where rho is learned to minimize the ELBO (maximize the evidence lower bound).

    Mathematical framework:
    - Generative model: p(z; theta_Z), p(x|z) = Poisson(exp(b + W @ s(z)))
    - Approximate posterior: q(z|x) with natural params theta_Z + W^T x - rho
    - ELBO: E_q[log p(x|z)] - D_KL(q(z|x) || p(z))

    Attributes:
        hrm: The underlying PoissonVonMisesHarmonium
    """

    hrm: PoissonVonMisesHarmonium
    """Override with concrete type for type checking."""

    # Override ELBO computation to handle non-reparameterizable VonMises sampling

    @override
    def elbo_reconstruction_term(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> Array:
        """Compute E_q[log p(x|z)] via Monte Carlo sampling.

        Uses stop_gradient on samples since VonMises sampling (rejection sampling)
        is not reparameterizable. This gives a biased but usable gradient estimator.

        Args:
            key: JAX random key
            params: Full model parameters
            x: Observation
            n_samples: Number of MC samples

        Returns:
            Expected log-likelihood (scalar)
        """
        s_x = self.obs_man.sufficient_statistic(x)
        q_params = self.approximate_posterior_at(params, x)

        # Sample z from q(z|x) - use stop_gradient since VonMises sampling
        # uses rejection sampling which can't be differentiated through
        z_samples = jax.lax.stop_gradient(
            self.pst_man.sample(key, q_params, n_samples)
        )

        def log_likelihood_at_z(z: Array) -> Array:
            lkl_params = self.likelihood_at(params, z)
            return (
                jnp.dot(s_x, lkl_params)
                - self.obs_man.log_partition_function(lkl_params)
                + self.obs_man.log_base_measure(x)
            )

        return jnp.mean(jax.vmap(log_likelihood_at_z)(z_samples))

    # Convenience accessors

    @property
    def n_neurons(self) -> int:
        """Number of observable Poisson units."""
        return self.hrm.n_neurons

    @property
    def n_latent(self) -> int:
        """Number of latent VonMises units."""
        return self.hrm.n_latent

    # Convenience methods

    def observable_rates(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] for all neurons."""
        _, hrm_params = self.split_coords(params)
        return self.hrm.observable_rates(hrm_params, z)

    def reconstruct(self, params: Array, x: Array) -> Array:
        """Reconstruct observable via mean-field approximation.

        Uses E[x | E[z|x]] where the expectation is under q(z|x).

        Args:
            params: Full model parameters [rho, harmonium_params]
            x: Observable spike counts

        Returns:
            Reconstructed expected rates
        """
        # Get approximate posterior parameters
        q_params = self.approximate_posterior_at(params, x)

        # Compute mean sufficient statistics from VonMises posterior
        z_mean_stats = self.pst_man.to_mean(q_params)

        # Get likelihood parameters using mean sufficient statistics
        _, hrm_params = self.split_coords(params)
        obs_bias, int_params, _ = self.hrm.split_coords(hrm_params)
        int_matrix = int_params.reshape(self.n_neurons, 2 * self.n_latent)
        lkl_natural = obs_bias + int_matrix @ z_mean_stats

        return self.obs_man.to_mean(lkl_natural)

    def reconstruction_error(self, params: Array, xs: Array) -> Array:
        """Compute mean squared reconstruction error over a batch.

        Args:
            params: Full model parameters
            xs: Batch of spike counts (shape: n_samples, n_neurons)

        Returns:
            Mean squared error (scalar)
        """
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        return jnp.mean((xs - recons) ** 2)

    def normalized_reconstruction_error(
        self, params: Array, xs: Array, scale: float = 1.0
    ) -> Array:
        """Compute MSE on normalized data.

        Args:
            params: Full model parameters
            xs: Batch of spike counts
            scale: Scale factor for normalization

        Returns:
            Mean squared error on normalized values
        """
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        xs_norm = xs / scale
        recons_norm = recons / scale
        return jnp.mean((xs_norm - recons_norm) ** 2)

    def get_weight_matrix(self, params: Array) -> Array:
        """Extract the interaction weight matrix W.

        Args:
            params: Full model parameters

        Returns:
            Weight matrix of shape (n_neurons, 2*n_latent)
        """
        _, hrm_params = self.split_coords(params)
        return self.hrm.get_weight_matrix(hrm_params)


def variational_poisson_vonmises(
    n_neurons: int, n_latent: int
) -> VariationalPoissonVonMises:
    """Create a variational Poisson-VonMises model.

    Factory function for convenient construction.

    Args:
        n_neurons: Number of observable (Poisson) units
        n_latent: Number of latent (Von Mises) units

    Returns:
        VariationalPoissonVonMises instance
    """
    hrm = PoissonVonMisesHarmonium(n_neurons=n_neurons, n_latent=n_latent)
    return VariationalPoissonVonMises(hrm=hrm)
