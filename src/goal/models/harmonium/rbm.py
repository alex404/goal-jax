"""Restricted Boltzmann Machine (RBM) implementations.

This module provides:
- RestrictedBoltzmannMachine: Binary-binary RBM (Bernoulli observable, Bernoulli latent)
- BinomialRBM: Binomial-binary RBM (Binomial observable, Bernoulli latent)
- PoissonRBM: Poisson-binary RBM (Poisson observable, Bernoulli latent)
- BernoulliVonMisesRBM: Bernoulli-VonMises RBM
- BinomialVonMisesRBM: Binomial-VonMises RBM
- BinomialGaussianHarmonium: Binomial-Gaussian harmonium
- BinomialNormalHarmonium: Binomial-Normal harmonium

RBMs are bipartite undirected graphical models that learn to represent
probability distributions. They are trained using contrastive divergence,
which is provided by the GenerativeHarmonium base class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import EmbeddedMap, IdentityEmbedding, Rectangular
from ...geometry.exponential_family.harmonium import (
    DifferentiableHarmonium,
    GenerativeHarmonium,
)
from ..base.binomial import Binomials
from ..base.categorical import Bernoullis
from ..base.gaussian.generalized import Euclidean
from ..base.gaussian.normal import Normal, diagonal_normal
from ..base.poisson import Poissons
from ..base.von_mises import VonMisesProduct
from .lgm import GeneralizedGaussianLocationEmbedding

# Standalone filter visualization functions


def get_rectangular_filters(
    harmonium: GenerativeHarmonium[Any, Any],
    params: Array,
    img_shape: tuple[int, int],
) -> Array:
    """Reshape interaction weights into filter images for visualization.

    For harmoniums with rectangular interaction (standard case).
    Each latent unit's weights form one filter image.

    Args:
        harmonium: The harmonium model
        params: Model parameters
        img_shape: Shape of each observable image (height, width)

    Returns:
        Array of shape (n_latent, height, width)
    """
    _, int_params, _ = harmonium.split_coords(params)
    n_obs = harmonium.obs_man.dim
    n_lat = harmonium.pst_man.dim
    weights = int_params.reshape(n_obs, n_lat)
    return weights.T.reshape(-1, *img_shape)


def get_vonmises_filters(
    harmonium: GenerativeHarmonium[Any, Any],
    params: Array,
    img_shape: tuple[int, int],
) -> Array:
    """Reshape interaction weights for VonMises latents into filter images.

    Combines cos/sin pairs into magnitude for each latent unit.

    Args:
        harmonium: The harmonium model with VonMises latents
        params: Model parameters
        img_shape: Shape of each observable image (height, width)

    Returns:
        Array of shape (n_latent_units, height, width)
    """
    _, int_params, _ = harmonium.split_coords(params)
    n_obs = harmonium.obs_man.dim
    n_lat_pairs = harmonium.pst_man.dim // 2
    weights = int_params.reshape(n_obs, -1)

    # Combine cos and sin weights for each latent into magnitude
    # weights[:, 2*i] is cos weight, weights[:, 2*i+1] is sin weight
    w_cos = weights[:, 0::2]  # Shape: (n_obs, n_lat_pairs)
    w_sin = weights[:, 1::2]  # Shape: (n_obs, n_lat_pairs)
    magnitudes = jnp.sqrt(w_cos**2 + w_sin**2)  # Shape: (n_obs, n_lat_pairs)

    return magnitudes.T.reshape(n_lat_pairs, *img_shape)


@dataclass(frozen=True)
class RestrictedBoltzmannMachine(DifferentiableHarmonium[Bernoullis, Bernoullis]):
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
        return get_rectangular_filters(self, params, img_shape)


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
class BinomialRBM(DifferentiableHarmonium[Binomials, Bernoullis]):
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
        return get_rectangular_filters(self, params, img_shape)


def binomial_rbm(n_observable: int, n_latent: int, n_trials: int = 255) -> BinomialRBM:
    """Create a Binomial-Bernoulli Restricted Boltzmann Machine.

    Factory function for convenient construction of Binomial RBMs.

    Args:
        n_observable: Number of observable units
        n_latent: Number of latent units
        n_trials: Number of trials for each observable (e.g., 255 for grayscale)

    Returns:
        BinomialRBM instance
    """
    return BinomialRBM(n_observable=n_observable, n_latent=n_latent, n_trials=n_trials)


@dataclass(frozen=True)
class PoissonRBM(DifferentiableHarmonium[Poissons, Bernoullis]):
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
        return get_rectangular_filters(self, params, img_shape)


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


@dataclass(frozen=True)
class BernoulliVonMisesRBM(DifferentiableHarmonium[Bernoullis, VonMisesProduct]):
    """RBM with Bernoulli observable units and Von Mises latent units.

    For modeling binary data with circular/angular latent representations.
    Each observable unit is Bernoulli(sigmoid(b_i + W_i^T s(z))).
    Each latent unit is Von Mises with mean and concentration parameters.

    The joint distribution is:

    $$p(x, z) \\propto \\exp(b^T x + \\theta_z^T s(z) + x^T W s(z))$$

    where:
    - x ∈ {0,1}^n_observable are binary observables
    - z ∈ [0, 2π)^n_latent are circular latent angles
    - s(z) = [cos(z_1), sin(z_1), ..., cos(z_n), sin(z_n)] is the sufficient statistic
    - W is the weight matrix (n_observable x 2*n_latent)
    - b are observable biases
    - θ_z are latent natural parameters

    As a harmonium:
    - obs_man: Bernoullis(n_observable)
    - pst_man: VonMisesProduct(n_latent)
    - int_man: Rectangular weight matrix W

    The conditionals are:
    - p(x_i = 1 | z) = sigmoid(b_i + W_i^T s(z))
    - p(z | x) = product of Von Mises with shifted natural parameters

    Attributes:
        n_observable: Number of observable (binary) units
        n_latent: Number of latent (circular) units
    """

    n_observable: int
    """Number of observable units."""

    n_latent: int
    """Number of latent Von Mises units."""

    # Properties

    @property
    @override
    def obs_man(self) -> Bernoullis:
        """Observable layer manifold (Bernoulli units)."""
        return Bernoullis(self.n_observable)

    @property
    @override
    def pst_man(self) -> VonMisesProduct:
        """Latent layer manifold (Von Mises units)."""
        return VonMisesProduct(self.n_latent)

    @property
    @override
    def int_man(self) -> EmbeddedMap[VonMisesProduct, Bernoullis]:
        """Interaction matrix with identity embeddings (full connectivity).

        The interaction matrix W has shape (n_observable, 2*n_latent) and connects
        all observable units to all latent Von Mises sufficient statistics.
        """
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(self.pst_man),
            IdentityEmbedding(self.obs_man),
        )

    # Convenience methods

    def observable_probabilities(self, params: Array, z: Array) -> Array:
        """Compute P(x_i = 1 | z) for all observable units.

        Args:
            params: RBM parameters
            z: Latent angles (shape: n_latent)

        Returns:
            Probabilities for each observable unit (shape: n_observable)
        """
        lkl_params = self.likelihood_at(params, z)
        return self.obs_man.to_mean(lkl_params)

    def get_filters(self, params: Array, img_shape: tuple[int, int]) -> Array:
        """Reshape weight matrix into filter images for visualization.

        Each pair of columns (corresponding to cos and sin for one latent)
        are combined into a single filter image showing the magnitude.

        Args:
            params: RBM parameters
            img_shape: Shape of each observable image (height, width)

        Returns:
            Array of shape (n_latent, height, width)
        """
        return get_vonmises_filters(self, params, img_shape)


def bernoulli_vonmises_rbm(n_observable: int, n_latent: int) -> BernoulliVonMisesRBM:
    """Create a Bernoulli-VonMises Restricted Boltzmann Machine.

    Factory function for convenient construction of Bernoulli-VonMises RBMs.

    Args:
        n_observable: Number of observable (binary) units
        n_latent: Number of latent (circular) units

    Returns:
        BernoulliVonMisesRBM instance
    """
    return BernoulliVonMisesRBM(n_observable=n_observable, n_latent=n_latent)


@dataclass(frozen=True)
class BinomialVonMisesRBM(DifferentiableHarmonium[Binomials, VonMisesProduct]):
    """RBM with Binomial observable units and Von Mises latent units.

    For modeling count data with circular/angular latent representations.
    Each observable unit is Binomial(n_trials, p_i) where p_i depends on z.
    Each latent unit is Von Mises with mean and concentration parameters.

    The joint distribution is:

    $$p(x, z) \\propto \\prod_i \\binom{n}{x_i} \\exp(b^T x + \\theta_z^T s(z) + x^T W s(z))$$

    where:
    - x_i in {0,1,...,n_trials} are observable counts
    - z in [0, 2*pi)^n_latent are circular latent angles
    - s(z) = [cos(z_1), sin(z_1), ..., cos(z_n), sin(z_n)] is the sufficient statistic
    - W is the weight matrix (n_observable x 2*n_latent)
    - b are observable biases
    - theta_z are latent natural parameters

    As a harmonium:
    - obs_man: Binomials(n_observable, n_trials)
    - pst_man: VonMisesProduct(n_latent)
    - int_man: Rectangular weight matrix W

    Attributes:
        n_observable: Number of observable (count) units
        n_latent: Number of latent (circular) units
        n_trials: Number of trials for each Binomial (e.g., 16 for downsampled grayscale)
    """

    n_observable: int
    """Number of observable units."""

    n_latent: int
    """Number of latent Von Mises units."""

    n_trials: int = 16
    """Number of trials for each Binomial unit."""

    # Properties

    @property
    @override
    def obs_man(self) -> Binomials:
        """Observable layer manifold (Binomial units)."""
        return Binomials(self.n_observable, self.n_trials)

    @property
    @override
    def pst_man(self) -> VonMisesProduct:
        """Latent layer manifold (Von Mises units)."""
        return VonMisesProduct(self.n_latent)

    @property
    @override
    def int_man(self) -> EmbeddedMap[VonMisesProduct, Binomials]:
        """Interaction matrix with identity embeddings (full connectivity).

        The interaction matrix W has shape (n_observable, 2*n_latent) and connects
        all observable units to all latent Von Mises sufficient statistics.
        """
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(self.pst_man),
            IdentityEmbedding(self.obs_man),
        )

    # Convenience methods

    def observable_means(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] for all observable units.

        Returns expected counts (not probabilities).

        Args:
            params: RBM parameters
            z: Latent angles (shape: n_latent)

        Returns:
            Expected counts for each observable unit (shape: n_observable)
        """
        lkl_params = self.likelihood_at(params, z)
        return self.obs_man.to_mean(lkl_params)

    def observable_probabilities(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] / n_trials for all observable units.

        Returns the expected proportion (probability) for each observable.

        Args:
            params: RBM parameters
            z: Latent angles (shape: n_latent)

        Returns:
            Probabilities for each observable unit (shape: n_observable)
        """
        return self.observable_means(params, z) / self.n_trials

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

        Each pair of columns (corresponding to cos and sin for one latent)
        are combined into a single filter image showing the magnitude.

        Args:
            params: RBM parameters
            img_shape: Shape of each observable image (height, width)

        Returns:
            Array of shape (n_latent, height, width)
        """
        return get_vonmises_filters(self, params, img_shape)


def binomial_vonmises_rbm(
    n_observable: int, n_latent: int, n_trials: int = 16
) -> BinomialVonMisesRBM:
    """Create a Binomial-VonMises Restricted Boltzmann Machine.

    Factory function for convenient construction of Binomial-VonMises RBMs.

    Args:
        n_observable: Number of observable (count) units
        n_latent: Number of latent (circular) units
        n_trials: Number of trials for each Binomial (e.g., 16 for downsampled grayscale)

    Returns:
        BinomialVonMisesRBM instance
    """
    return BinomialVonMisesRBM(
        n_observable=n_observable, n_latent=n_latent, n_trials=n_trials
    )


@dataclass(frozen=True)
class BinomialGaussianHarmonium(DifferentiableHarmonium[Binomials, Euclidean]):
    """Harmonium with Binomial observable units and standard Gaussian latent units.

    This model uses standard Gaussians (unit variance) for the latent layer,
    where the interaction affects only the mean of the Gaussians, similar to
    a Linear Gaussian Model (LGM).

    For modeling count data with continuous latent representations.
    Each observable unit is Binomial(n_trials, p_i) where p_i depends on z.
    Each latent unit is N(mu, 1) where mu is determined by the interaction.

    The joint distribution is:

    $$p(x, z) \\propto \\prod_i \\binom{n}{x_i} \\exp(b^T x + \\theta_z^T z + x^T W z - ||z||^2/2)$$

    where:
    - x_i in {0,1,...,n_trials} are observable counts
    - z in R^n_latent are continuous latent variables
    - W is the weight matrix (n_observable x n_latent)
    - b are observable biases
    - theta_z are latent natural parameters (means when z has unit variance)

    Key insight: The Euclidean class represents standard Gaussians where:
    - Sufficient statistic: s(z) = z (identity)
    - Natural parameter: theta = mu (the mean)
    - Log partition: psi(theta) = 0.5 ||theta||^2 + const

    So the interaction W only affects the mean of the Gaussian latents,
    exactly like in an LGM where the interaction is only on the location.

    As a harmonium:
    - obs_man: Binomials(n_observable, n_trials)
    - pst_man: Euclidean(n_latent) - standard Gaussians
    - int_man: Rectangular weight matrix W

    The conditionals are:
    - p(x_i | z) = Binomial(n_trials, sigmoid(b_i + W_i^T z))
    - p(z | x) = N(theta_z + W^T x, I) - Gaussian with shifted mean, unit variance

    Attributes:
        n_observable: Number of observable (count) units
        n_latent: Number of latent (Gaussian) units
        n_trials: Number of trials for each Binomial
    """

    n_observable: int
    """Number of observable units."""

    n_latent: int
    """Number of latent Gaussian units."""

    n_trials: int = 16
    """Number of trials for each Binomial unit."""

    # Properties

    @property
    @override
    def obs_man(self) -> Binomials:
        """Observable layer manifold (Binomial units)."""
        return Binomials(self.n_observable, self.n_trials)

    @property
    @override
    def pst_man(self) -> Euclidean:
        """Latent layer manifold (standard Gaussian units)."""
        return Euclidean(self.n_latent)

    @property
    @override
    def int_man(self) -> EmbeddedMap[Euclidean, Binomials]:
        """Interaction matrix with identity embeddings (full connectivity).

        The interaction matrix W has shape (n_observable, n_latent) and connects
        all observable units to all latent Gaussian means.
        """
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(self.pst_man),
            IdentityEmbedding(self.obs_man),
        )

    # Convenience methods

    def observable_means(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] for all observable units.

        Returns expected counts (not probabilities).

        Args:
            params: Harmonium parameters
            z: Latent values (shape: n_latent)

        Returns:
            Expected counts for each observable unit (shape: n_observable)
        """
        lkl_params = self.likelihood_at(params, z)
        return self.obs_man.to_mean(lkl_params)

    def observable_probabilities(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] / n_trials for all observable units.

        Returns the expected proportion (probability) for each observable.

        Args:
            params: Harmonium parameters
            z: Latent values (shape: n_latent)

        Returns:
            Probabilities for each observable unit (shape: n_observable)
        """
        return self.observable_means(params, z) / self.n_trials

    def normalized_reconstruction_error(self, params: Array, xs: Array) -> Array:
        """Compute mean squared reconstruction error on normalized data.

        Normalizes both input and reconstruction to [0, 1] before computing MSE.

        Args:
            params: Harmonium parameters
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
            params: Harmonium parameters
            img_shape: Shape of each observable image (height, width)

        Returns:
            Array of shape (n_latent, height, width)
        """
        return get_rectangular_filters(self, params, img_shape)


def binomial_gaussian_harmonium(
    n_observable: int, n_latent: int, n_trials: int = 16
) -> BinomialGaussianHarmonium:
    """Create a Binomial-Gaussian Harmonium.

    Factory function for convenient construction of Binomial-Gaussian harmoniums
    with standard Gaussian (unit variance) latent units.

    Args:
        n_observable: Number of observable (count) units
        n_latent: Number of latent (Gaussian) units
        n_trials: Number of trials for each Binomial

    Returns:
        BinomialGaussianHarmonium instance
    """
    return BinomialGaussianHarmonium(
        n_observable=n_observable, n_latent=n_latent, n_trials=n_trials
    )


@dataclass(frozen=True)
class BinomialNormalHarmonium(DifferentiableHarmonium[Binomials, Normal]):
    """Harmonium with Binomial observable units and diagonal Normal latent units.

    This model uses diagonal Gaussians (independent latent dimensions with learned
    variances) for the latent layer. Following the LGM pattern, the interaction
    affects only the mean of the Gaussians - the variance/precision parameters
    are not coupled to the observables.

    For modeling count data with continuous latent representations.
    Each observable unit is Binomial(n_trials, p_i) where p_i depends on z.
    Each latent unit is N(mu_i, sigma_i^2) with diagonal covariance.

    The joint distribution is:

    $$p(x, z) \\propto \\prod_i \\binom{n}{x_i}
        \\exp(b^T x + \\theta_z^{loc,T} z - 0.5 z^T \\Sigma^{-1} z + x^T W z)$$

    where:
    - x_i in {0,1,...,n_trials} are observable counts
    - z in R^n_latent are continuous latent variables
    - W is the weight matrix (n_observable x n_latent) affecting only means
    - b are observable biases
    - theta_z = [theta_loc, theta_prec] are latent natural parameters

    Key insight: Using GeneralizedGaussianLocationEmbedding, the interaction
    W only affects the location (mean) parameters of the Normal, not the
    precision/variance. This is analogous to Linear Gaussian Models.

    As a harmonium:
    - obs_man: Binomials(n_observable, n_trials)
    - pst_man: Normal(n_latent, Diagonal()) - diagonal covariance Gaussians
    - int_man: EmbeddedMap with location embedding (interactions on means only)

    The conditionals are:
    - p(x_i | z) = Binomial(n_trials, sigmoid(b_i + W_i^T z))
    - p(z | x) = N(mu_post, Sigma) where mu_post = mu_prior + Sigma @ W^T @ x

    Attributes:
        n_observable: Number of observable (count) units
        n_latent: Number of latent (Gaussian) units
        n_trials: Number of trials for each Binomial
    """

    n_observable: int
    """Number of observable units."""

    n_latent: int
    """Number of latent Gaussian units."""

    n_trials: int = 16
    """Number of trials for each Binomial unit."""

    # Properties

    @property
    @override
    def obs_man(self) -> Binomials:
        """Observable layer manifold (Binomial units)."""
        return Binomials(self.n_observable, self.n_trials)

    @property
    @override
    def pst_man(self) -> Normal:
        """Latent layer manifold (diagonal Normal)."""
        return diagonal_normal(self.n_latent)

    @property
    @override
    def int_man(self) -> EmbeddedMap[Normal, Binomials]:
        """Interaction matrix that only affects latent means.

        Uses GeneralizedGaussianLocationEmbedding on the latent side so that
        the interaction W only shifts the mean of the Normal distribution,
        not its precision/variance parameters. This follows the LGM pattern.

        The interaction matrix W has shape (n_observable, n_latent).
        """
        return EmbeddedMap(
            Rectangular(),
            GeneralizedGaussianLocationEmbedding(self.pst_man),
            IdentityEmbedding(self.obs_man),
        )

    # Convenience methods

    def observable_means(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] for all observable units.

        Returns expected counts (not probabilities).

        Args:
            params: Harmonium parameters
            z: Latent values (shape: n_latent)

        Returns:
            Expected counts for each observable unit (shape: n_observable)
        """
        lkl_params = self.likelihood_at(params, z)
        return self.obs_man.to_mean(lkl_params)

    def observable_probabilities(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] / n_trials for all observable units.

        Returns the expected proportion (probability) for each observable.

        Args:
            params: Harmonium parameters
            z: Latent values (shape: n_latent)

        Returns:
            Probabilities for each observable unit (shape: n_observable)
        """
        return self.observable_means(params, z) / self.n_trials

    def normalized_reconstruction_error(self, params: Array, xs: Array) -> Array:
        """Compute mean squared reconstruction error on normalized data.

        Normalizes both input and reconstruction to [0, 1] before computing MSE.

        Args:
            params: Harmonium parameters
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
            params: Harmonium parameters
            img_shape: Shape of each observable image (height, width)

        Returns:
            Array of shape (n_latent, height, width)
        """
        return get_rectangular_filters(self, params, img_shape)


def binomial_normal_harmonium(
    n_observable: int, n_latent: int, n_trials: int = 16
) -> BinomialNormalHarmonium:
    """Create a Binomial-Normal Harmonium with diagonal covariance latents.

    Factory function for convenient construction of Binomial-Normal harmoniums
    where the interaction only affects the latent means (LGM-style).

    Args:
        n_observable: Number of observable (count) units
        n_latent: Number of latent (Gaussian) units
        n_trials: Number of trials for each Binomial

    Returns:
        BinomialNormalHarmonium instance
    """
    return BinomialNormalHarmonium(
        n_observable=n_observable, n_latent=n_latent, n_trials=n_trials
    )
