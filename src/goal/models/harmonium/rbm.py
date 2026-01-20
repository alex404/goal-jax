"""Restricted Boltzmann Machine (RBM) implementations.

This module provides:
- RestrictedBoltzmannMachine: Binary-binary RBM (Bernoulli observable, Bernoulli latent)
- BinomialRBM: Binomial-binary RBM (Binomial observable, Bernoulli latent)

RBMs are bipartite undirected graphical models that learn to represent
probability distributions. They are trained using contrastive divergence,
which is provided by the GibbsHarmonium base class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import EmbeddedMap, IdentityEmbedding, Rectangular
from ...geometry.exponential_family.harmonium import GibbsHarmonium
from ..base.binomial import Binomials
from ..base.gaussian.boltzmann import Bernoullis
from ..base.poisson import Poissons


@dataclass(frozen=True)
class RestrictedBoltzmannMachine(GibbsHarmonium[Bernoullis, Bernoullis]):
    """Binary-binary Restricted Boltzmann Machine.

    An RBM is a bipartite undirected graphical model with:
    - Observable layer: n_observable Bernoulli units
    - Latent layer: n_latent Bernoulli units
    - Symmetric weights connecting all observable to all latent units

    The joint distribution is:

    $$p(x, z) \\propto \\exp(x^T W z + b^T x + c^T z)$$

    where:
    - x in {0,1}^n_observable are observable units
    - z in {0,1}^n_latent are latent units
    - W is the weight matrix (n_observable x n_latent)
    - b are observable biases
    - c are latent biases

    As a harmonium:
    - obs_man: Bernoullis(n_observable) - observable layer
    - pst_man: Bernoullis(n_latent) - latent layer (posterior = prior for RBMs)
    - int_man: Rectangular weight matrix W

    The conditionals are products of independent Bernoullis:
    - p(x_i = 1 | z) = sigmoid(b_i + sum_j W_ij z_j)
    - p(z_j = 1 | x) = sigmoid(c_j + sum_i W_ij x_i)

    This makes Gibbs sampling efficient and enables contrastive divergence training.

    Attributes:
        n_observable: Number of observable units
        n_latent: Number of latent units
    """

    n_observable: int
    """Number of observable units."""

    n_latent: int
    """Number of latent units."""

    # Properties

    @property
    @override
    def int_man(self) -> EmbeddedMap[Bernoullis, Bernoullis]:
        """Interaction matrix with identity embeddings (full connectivity).

        The interaction matrix W has shape (n_observable, n_latent) and connects
        all observable units to all latent units.
        """
        obs = Bernoullis(self.n_observable)
        lat = Bernoullis(self.n_latent)
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(lat),
            IdentityEmbedding(obs),
        )

    @property
    def lat_man(self) -> Bernoullis:
        """Latent layer manifold (alias for pst_man)."""
        return self.pst_man

    # Convenience methods

    def observable_probabilities(self, params: Array, z: Array) -> Array:
        """Compute P(x_i = 1 | z) for all observable units.

        Args:
            params: RBM parameters
            z: Latent unit activations (shape: n_latent)

        Returns:
            Probabilities for each observable unit (shape: n_observable)
        """
        lkl_params = self.likelihood_at(params, z)
        return self.obs_man.to_mean(lkl_params)

    def latent_probabilities(self, params: Array, x: Array) -> Array:
        """Compute P(z_j = 1 | x) for all latent units.

        Args:
            params: RBM parameters
            x: Observable unit activations (shape: n_observable)

        Returns:
            Probabilities for each latent unit (shape: n_latent)
        """
        post_params = self.posterior_at(params, x)
        return self.pst_man.to_mean(post_params)

    def reconstruct(self, params: Array, x: Array) -> Array:
        """Reconstruct observable units via one mean-field step.

        Computes E[x|E[z|x]], which is a deterministic reconstruction
        using the mean latent probabilities.

        Args:
            params: RBM parameters
            x: Input observable pattern

        Returns:
            Reconstructed observable pattern (probabilities)
        """
        z_probs = self.latent_probabilities(params, x)
        return self.observable_probabilities(params, z_probs)

    def free_energy(self, params: Array, x: Array) -> Array:
        """Compute the free energy F(x) = -log sum_z exp(-E(x,z)).

        For binary RBMs with energy E(x,z) = -b^T x - c^T z - x^T W z:

        F(x) = -b^T x - sum_j softplus(c_j + sum_i W_ij x_i)

        The free energy is useful for:
        - Computing gradients: dF/dx = -b - W @ sigmoid(c + W^T @ x)
        - Approximating log-likelihood: log p(x) = -F(x) + const

        Args:
            params: RBM parameters
            x: Observable pattern

        Returns:
            Free energy (scalar)
        """
        obs_bias, _, _ = self.split_coords(params)

        # Get latent inputs given x
        post_params = self.posterior_at(params, x)  # c + W^T x

        # F(x) = -b^T x - sum_j softplus(c_j + W_j^T x)
        obs_term = -jnp.dot(obs_bias, x)
        lat_term = -self.pst_man.log_partition_function(post_params)

        return obs_term + lat_term

    def mean_free_energy(self, params: Array, xs: Array) -> Array:
        """Compute mean free energy over a batch of observable patterns.

        Args:
            params: RBM parameters
            xs: Batch of observable patterns (shape: n_samples, n_observable)

        Returns:
            Mean free energy (scalar)
        """
        return jnp.mean(jax.vmap(self.free_energy, in_axes=(None, 0))(params, xs))

    def reconstruction_error(self, params: Array, xs: Array) -> Array:
        """Compute mean squared reconstruction error over a batch.

        Args:
            params: RBM parameters
            xs: Batch of observable patterns (shape: n_samples, n_observable)

        Returns:
            Mean squared error (scalar)
        """
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        return jnp.mean((xs - recons) ** 2)

    def get_filters(self, params: Array, img_shape: tuple[int, int]) -> Array:
        """Reshape weight matrix into filter images for visualization.

        Each column of the weight matrix (corresponding to one latent unit)
        is reshaped into an image.

        Args:
            params: RBM parameters
            img_shape: Shape of each observable image (height, width)

        Returns:
            Array of shape (n_latent, height, width)
        """
        _, int_params, _ = self.split_coords(params)
        weights = int_params.reshape(self.n_observable, self.n_latent)
        return weights.T.reshape(self.n_latent, *img_shape)


def rbm(n_observable: int, n_latent: int) -> RestrictedBoltzmannMachine:
    """Create a binary-binary Restricted Boltzmann Machine.

    Factory function for convenient construction of RBMs.

    Args:
        n_observable: Number of observable units
        n_latent: Number of latent units

    Returns:
        RestrictedBoltzmannMachine instance
    """
    return RestrictedBoltzmannMachine(n_observable=n_observable, n_latent=n_latent)


@dataclass(frozen=True)
class BinomialRBM(GibbsHarmonium[Binomials, Bernoullis]):
    """RBM with Binomial observable units and Bernoulli latent units.

    For modeling count data (e.g., grayscale images as pixel counts).
    Each observable unit is Binomial(n_trials, p_i) where n_trials is fixed.

    The joint distribution is:

    $$p(x, z) \\propto \\prod_i \\binom{n}{x_i} \\exp(x^T W z + b^T x + c^T z)$$

    where:
    - x_i in {0,1,...,n_trials} are observable counts
    - z in {0,1}^n_latent are latent units
    - W is the weight matrix (n_observable x n_latent)
    - b are observable biases
    - c are latent biases

    The key difference from binary RBM:
    - Observable sufficient statistic is still s(x) = x (counts)
    - Log partition includes the n_trials scaling: n * softplus(theta)
    - Non-zero base measure: log binomial coefficient

    As a harmonium:
    - obs_man: Binomials(n_observable, n_trials) - observable layer
    - pst_man: Bernoullis(n_latent) - latent layer
    - int_man: Rectangular weight matrix W

    Attributes:
        n_observable: Number of observable units
        n_latent: Number of latent units
        n_trials: Number of trials for each observable unit (e.g., 255 for grayscale)
    """

    n_observable: int
    """Number of observable units."""

    n_latent: int
    """Number of latent units."""

    n_trials: int = 255
    """Number of trials for each observable unit (e.g., 255 for grayscale)."""

    # Properties

    @property
    @override
    def obs_man(self) -> Binomials:
        """Observable layer manifold (Binomial units)."""
        return Binomials(self.n_observable, self.n_trials)

    @property
    @override
    def pst_man(self) -> Bernoullis:
        """Latent layer manifold (Bernoulli units)."""
        return Bernoullis(self.n_latent)

    @property
    def lat_man(self) -> Bernoullis:
        """Latent layer manifold (alias for pst_man)."""
        return self.pst_man

    @property
    @override
    def int_man(self) -> EmbeddedMap[Bernoullis, Binomials]:
        """Interaction matrix with identity embeddings (full connectivity).

        The interaction matrix W has shape (n_observable, n_latent) and connects
        all observable units to all latent units.
        """
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(self.pst_man),
            IdentityEmbedding(self.obs_man),
        )

    # Convenience methods

    def observable_probabilities(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] / n_trials for all observable units.

        Returns the expected proportion (probability) for each observable,
        not the raw expected count.

        Args:
            params: RBM parameters
            z: Latent unit activations (shape: n_latent)

        Returns:
            Probabilities for each observable unit (shape: n_observable)
        """
        lkl_params = self.likelihood_at(params, z)
        # Mean for Binomial is n*p, we return p (probability)
        means = self.obs_man.to_mean(lkl_params)
        return means / self.n_trials

    def observable_means(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] for all observable units.

        Returns expected counts (not probabilities).

        Args:
            params: RBM parameters
            z: Latent unit activations (shape: n_latent)

        Returns:
            Expected counts for each observable unit (shape: n_observable)
        """
        lkl_params = self.likelihood_at(params, z)
        return self.obs_man.to_mean(lkl_params)

    def latent_probabilities(self, params: Array, x: Array) -> Array:
        """Compute P(z_j = 1 | x) for all latent units.

        Args:
            params: RBM parameters
            x: Observable unit counts (shape: n_observable)

        Returns:
            Probabilities for each latent unit (shape: n_latent)
        """
        post_params = self.posterior_at(params, x)
        return self.pst_man.to_mean(post_params)

    def reconstruct(self, params: Array, x: Array) -> Array:
        """Reconstruct observable units via one mean-field step.

        Computes E[x|E[z|x]], returning expected counts.

        Args:
            params: RBM parameters
            x: Input observable counts

        Returns:
            Reconstructed observable counts (expected values)
        """
        z_probs = self.latent_probabilities(params, x)
        return self.observable_means(params, z_probs)

    def free_energy(self, params: Array, x: Array) -> Array:
        """Compute the free energy F(x) = -log sum_z exp(-E(x,z)).

        For Binomial RBMs:

        F(x) = -b^T x - sum_j softplus(c_j + sum_i W_ij x_i) - sum_i log C(n, x_i)

        Note: The binomial coefficient terms are constant given x, so they don't
        affect gradients w.r.t. parameters, but they do affect the absolute value.

        Args:
            params: RBM parameters
            x: Observable counts

        Returns:
            Free energy (scalar)
        """
        obs_bias, _, _ = self.split_coords(params)

        # Get latent inputs given x
        post_params = self.posterior_at(params, x)  # c + W^T x

        # F(x) = -b^T x - sum_j softplus(c_j + W_j^T x) - sum_i log C(n, x_i)
        obs_term = -jnp.dot(obs_bias, x)
        lat_term = -self.pst_man.log_partition_function(post_params)

        # Base measure contribution (binomial coefficients)
        base_measure_term = -jnp.sum(
            jax.vmap(self.obs_man.rep_man.log_base_measure)(x.reshape(-1, 1))
        )

        return obs_term + lat_term + base_measure_term

    def mean_free_energy(self, params: Array, xs: Array) -> Array:
        """Compute mean free energy over a batch of observable patterns.

        Args:
            params: RBM parameters
            xs: Batch of observable counts (shape: n_samples, n_observable)

        Returns:
            Mean free energy (scalar)
        """
        return jnp.mean(jax.vmap(self.free_energy, in_axes=(None, 0))(params, xs))

    def reconstruction_error(self, params: Array, xs: Array) -> Array:
        """Compute mean squared reconstruction error over a batch.

        Error is computed on raw counts, not normalized.

        Args:
            params: RBM parameters
            xs: Batch of observable counts (shape: n_samples, n_observable)

        Returns:
            Mean squared error (scalar)
        """
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        return jnp.mean((xs - recons) ** 2)

    def normalized_reconstruction_error(self, params: Array, xs: Array) -> Array:
        """Compute mean squared reconstruction error on normalized data.

        Normalizes both input and reconstruction to [0, 1] before computing MSE.

        Args:
            params: RBM parameters
            xs: Batch of observable counts (shape: n_samples, n_observable)

        Returns:
            Mean squared error on normalized values (scalar)
        """
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        xs_norm = xs / self.n_trials
        recons_norm = recons / self.n_trials
        return jnp.mean((xs_norm - recons_norm) ** 2)

    def get_filters(self, params: Array, img_shape: tuple[int, int]) -> Array:
        """Reshape weight matrix into filter images for visualization.

        Each column of the weight matrix (corresponding to one latent unit)
        is reshaped into an image.

        Args:
            params: RBM parameters
            img_shape: Shape of each observable image (height, width)

        Returns:
            Array of shape (n_latent, height, width)
        """
        _, int_params, _ = self.split_coords(params)
        weights = int_params.reshape(self.n_observable, self.n_latent)
        return weights.T.reshape(self.n_latent, *img_shape)


def binomial_rbm(
    n_observable: int, n_latent: int, n_trials: int = 255
) -> BinomialRBM:
    """Create a Binomial-Bernoulli Restricted Boltzmann Machine.

    Factory function for convenient construction of Binomial RBMs.

    Args:
        n_observable: Number of observable units
        n_latent: Number of latent units
        n_trials: Number of trials for each observable (e.g., 255 for grayscale)

    Returns:
        BinomialRBM instance
    """
    return BinomialRBM(
        n_observable=n_observable, n_latent=n_latent, n_trials=n_trials
    )


@dataclass(frozen=True)
class PoissonRBM(GibbsHarmonium[Poissons, Bernoullis]):
    """RBM with Poisson observable units and Bernoulli latent units.

    For modeling count data (e.g., grayscale images as pixel counts).
    Each observable unit is Poisson(rate_i) where rate_i = exp(b_i + W_i^T z).

    The joint distribution is:

    $$p(x, z) \\propto \\prod_i (rate_i^{x_i} / x_i!) \\exp(x^T W z + b^T x + c^T z)$$

    where:
    - x_i in {0,1,2,...} are observable counts (unbounded)
    - z in {0,1}^n_latent are latent units
    - W is the weight matrix (n_observable x n_latent)
    - b are observable biases (log rates when z=0)
    - c are latent biases

    Key difference from Binomial RBM:
    - Observable units are unbounded (no n_trials parameter)
    - For visualization (e.g., images), outputs should be clipped to valid range

    As a harmonium:
    - obs_man: Poissons(n_observable) - observable layer
    - pst_man: Bernoullis(n_latent) - latent layer
    - int_man: Rectangular weight matrix W

    The conditionals are:
    - p(x_i | z) = Poisson(exp(b_i + W_i^T z))
    - p(z_j = 1 | x) = sigmoid(c_j + sum_i W_ij x_i)

    Attributes:
        n_observable: Number of observable units
        n_latent: Number of latent units
    """

    n_observable: int
    """Number of observable units."""

    n_latent: int
    """Number of latent units."""

    # Properties

    @property
    @override
    def obs_man(self) -> Poissons:
        """Observable layer manifold (Poisson units)."""
        return Poissons(self.n_observable)

    @property
    @override
    def pst_man(self) -> Bernoullis:
        """Latent layer manifold (Bernoulli units)."""
        return Bernoullis(self.n_latent)

    @property
    def lat_man(self) -> Bernoullis:
        """Latent layer manifold (alias for pst_man)."""
        return self.pst_man

    @property
    @override
    def int_man(self) -> EmbeddedMap[Bernoullis, Poissons]:
        """Interaction matrix with identity embeddings (full connectivity).

        The interaction matrix W has shape (n_observable, n_latent) and connects
        all observable units to all latent units.
        """
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(self.pst_man),
            IdentityEmbedding(self.obs_man),
        )

    # Convenience methods

    def observable_rates(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] = exp(b_i + W_i^T z) for all observable units.

        Args:
            params: RBM parameters
            z: Latent unit activations (shape: n_latent)

        Returns:
            Poisson rates for each observable unit (shape: n_observable)
        """
        lkl_params = self.likelihood_at(params, z)
        # For Poisson, mean = exp(natural parameter) = rate
        return self.obs_man.to_mean(lkl_params)

    def latent_probabilities(self, params: Array, x: Array) -> Array:
        """Compute P(z_j = 1 | x) for all latent units.

        Args:
            params: RBM parameters
            x: Observable unit counts (shape: n_observable)

        Returns:
            Probabilities for each latent unit (shape: n_latent)
        """
        post_params = self.posterior_at(params, x)
        return self.pst_man.to_mean(post_params)

    def reconstruct(self, params: Array, x: Array) -> Array:
        """Reconstruct observable units via one mean-field step.

        Computes E[x|E[z|x]], returning expected Poisson rates.

        Args:
            params: RBM parameters
            x: Input observable counts

        Returns:
            Reconstructed observable counts (expected rates)
        """
        z_probs = self.latent_probabilities(params, x)
        return self.observable_rates(params, z_probs)

    def free_energy(self, params: Array, x: Array) -> Array:
        """Compute the free energy F(x) = -log sum_z exp(-E(x,z)).

        For Poisson RBMs:

        F(x) = -b^T x - sum_j softplus(c_j + sum_i W_ij x_i) - sum_i log(x_i!)

        The log(x_i!) terms are from the base measure and don't affect gradients
        w.r.t. parameters, but they affect the absolute value.

        Args:
            params: RBM parameters
            x: Observable counts

        Returns:
            Free energy (scalar)
        """
        obs_bias, _, _ = self.split_coords(params)

        # Get latent inputs given x
        post_params = self.posterior_at(params, x)  # c + W^T x

        # F(x) = -b^T x - sum_j softplus(c_j + W_j^T x) - sum_i log(x_i!)
        obs_term = -jnp.dot(obs_bias, x)
        lat_term = -self.pst_man.log_partition_function(post_params)

        # Base measure contribution (log factorial)
        base_measure_term = -jnp.sum(
            jax.vmap(self.obs_man.rep_man.log_base_measure)(x.reshape(-1, 1))
        )

        return obs_term + lat_term + base_measure_term

    def mean_free_energy(self, params: Array, xs: Array) -> Array:
        """Compute mean free energy over a batch of observable patterns.

        Args:
            params: RBM parameters
            xs: Batch of observable counts (shape: n_samples, n_observable)

        Returns:
            Mean free energy (scalar)
        """
        return jnp.mean(jax.vmap(self.free_energy, in_axes=(None, 0))(params, xs))

    def reconstruction_error(self, params: Array, xs: Array) -> Array:
        """Compute mean squared reconstruction error over a batch.

        Error is computed on raw counts.

        Args:
            params: RBM parameters
            xs: Batch of observable counts (shape: n_samples, n_observable)

        Returns:
            Mean squared error (scalar)
        """
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        return jnp.mean((xs - recons) ** 2)

    def normalized_reconstruction_error(
        self, params: Array, xs: Array, scale: float = 255.0
    ) -> Array:
        """Compute mean squared reconstruction error on normalized data.

        Normalizes both input and reconstruction to [0, 1] before computing MSE.
        Useful when comparing to other RBM types on image data.

        Args:
            params: RBM parameters
            xs: Batch of observable counts (shape: n_samples, n_observable)
            scale: Scale factor for normalization (default 255 for grayscale)

        Returns:
            Mean squared error on normalized values (scalar)
        """
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        xs_norm = xs / scale
        recons_norm = recons / scale
        return jnp.mean((xs_norm - recons_norm) ** 2)

    def get_filters(self, params: Array, img_shape: tuple[int, int]) -> Array:
        """Reshape weight matrix into filter images for visualization.

        Each column of the weight matrix (corresponding to one latent unit)
        is reshaped into an image.

        Args:
            params: RBM parameters
            img_shape: Shape of each observable image (height, width)

        Returns:
            Array of shape (n_latent, height, width)
        """
        _, int_params, _ = self.split_coords(params)
        weights = int_params.reshape(self.n_observable, self.n_latent)
        return weights.T.reshape(self.n_latent, *img_shape)


def poisson_rbm(n_observable: int, n_latent: int) -> PoissonRBM:
    """Create a Poisson-Bernoulli Restricted Boltzmann Machine.

    Factory function for convenient construction of Poisson RBMs.

    Args:
        n_observable: Number of observable units
        n_latent: Number of latent units

    Returns:
        PoissonRBM instance
    """
    return PoissonRBM(n_observable=n_observable, n_latent=n_latent)
