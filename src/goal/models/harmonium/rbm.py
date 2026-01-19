"""Restricted Boltzmann Machine (RBM) implementation.

This module provides binary-binary RBMs (Bernoulli visible, Bernoulli hidden)
implemented as a GibbsHarmonium without conjugation structure.

RBMs are bipartite undirected graphical models that learn to represent
probability distributions over binary vectors. They are trained using
contrastive divergence, which is provided by the GibbsHarmonium base class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import EmbeddedMap, IdentityEmbedding, Rectangular
from ...geometry.exponential_family.harmonium import GibbsHarmonium
from ..base.gaussian.boltzmann import Bernoullis


@dataclass(frozen=True)
class RestrictedBoltzmannMachine(GibbsHarmonium[Bernoullis, Bernoullis]):
    """Binary-binary Restricted Boltzmann Machine.

    An RBM is a bipartite undirected graphical model with:
    - Visible layer: n_visible Bernoulli units
    - Hidden layer: n_hidden Bernoulli units
    - Symmetric weights connecting all visible to all hidden units

    The joint distribution is:

    $$p(v, h) \\propto \\exp(v^T W h + b^T v + c^T h)$$

    where:
    - v ∈ {0,1}^n_visible are visible units
    - h ∈ {0,1}^n_hidden are hidden units
    - W is the weight matrix (n_visible x n_hidden)
    - b are visible biases
    - c are hidden biases

    As a harmonium:
    - obs_man: Bernoullis(n_visible) - visible layer
    - pst_man: Bernoullis(n_hidden) - hidden layer (posterior = prior for RBMs)
    - int_man: Rectangular weight matrix W

    The conditionals are products of independent Bernoullis:
    - p(v_i = 1 | h) = sigmoid(b_i + sum_j W_ij h_j)
    - p(h_j = 1 | v) = sigmoid(c_j + sum_i W_ij v_i)

    This makes Gibbs sampling efficient and enables contrastive divergence training.

    Attributes:
        n_visible: Number of visible (observable) units
        n_hidden: Number of hidden (latent) units
    """

    n_visible: int
    """Number of visible (observable) units."""

    n_hidden: int
    """Number of hidden (latent) units."""

    # Properties

    @property
    @override
    def int_man(self) -> EmbeddedMap[Bernoullis, Bernoullis]:
        """Interaction matrix with identity embeddings (full connectivity).

        The interaction matrix W has shape (n_visible, n_hidden) and connects
        all visible units to all hidden units.
        """
        vis_man = Bernoullis(self.n_visible)
        hid_man = Bernoullis(self.n_hidden)
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(hid_man),
            IdentityEmbedding(vis_man),
        )

    @property
    def vis_man(self) -> Bernoullis:
        """Visible layer manifold (alias for obs_man)."""
        return self.obs_man

    @property
    def hid_man(self) -> Bernoullis:
        """Hidden layer manifold (alias for pst_man)."""
        return self.pst_man

    # Convenience methods

    def visible_probabilities(self, params: Array, h: Array) -> Array:
        """Compute P(v_i = 1 | h) for all visible units.

        Args:
            params: RBM parameters
            h: Hidden unit activations (shape: n_hidden)

        Returns:
            Probabilities for each visible unit (shape: n_visible)
        """
        lkl_params = self.likelihood_at(params, h)
        return self.obs_man.to_mean(lkl_params)

    def hidden_probabilities(self, params: Array, v: Array) -> Array:
        """Compute P(h_j = 1 | v) for all hidden units.

        Args:
            params: RBM parameters
            v: Visible unit activations (shape: n_visible)

        Returns:
            Probabilities for each hidden unit (shape: n_hidden)
        """
        post_params = self.posterior_at(params, v)
        return self.pst_man.to_mean(post_params)

    def reconstruct(self, params: Array, v: Array) -> Array:
        """Reconstruct visible units via one mean-field step.

        Computes E[v|E[h|v]], which is a deterministic reconstruction
        using the mean hidden probabilities.

        Args:
            params: RBM parameters
            v: Input visible pattern

        Returns:
            Reconstructed visible pattern (probabilities)
        """
        h_probs = self.hidden_probabilities(params, v)
        return self.visible_probabilities(params, h_probs)

    def free_energy(self, params: Array, v: Array) -> Array:
        """Compute the free energy F(v) = -log sum_h exp(-E(v,h)).

        For binary RBMs with energy E(v,h) = -b^T v - c^T h - v^T W h:

        F(v) = -b^T v - sum_j softplus(c_j + sum_i W_ij v_i)

        The free energy is useful for:
        - Computing gradients: dF/dv = -b - W @ sigmoid(c + W^T @ v)
        - Approximating log-likelihood: log p(v) ≈ -F(v) + const

        Args:
            params: RBM parameters
            v: Visible pattern

        Returns:
            Free energy (scalar)
        """
        vis_bias, _, _ = self.split_coords(params)

        # Get hidden inputs given v
        post_params = self.posterior_at(params, v)  # c + W^T v

        # F(v) = -b^T v - sum_j softplus(c_j + W_j^T v)
        visible_term = -jnp.dot(vis_bias, v)
        hidden_term = -self.pst_man.log_partition_function(post_params)

        return visible_term + hidden_term

    def mean_free_energy(self, params: Array, vs: Array) -> Array:
        """Compute mean free energy over a batch of visible patterns.

        Args:
            params: RBM parameters
            vs: Batch of visible patterns (shape: n_samples, n_visible)

        Returns:
            Mean free energy (scalar)
        """
        return jnp.mean(jax.vmap(self.free_energy, in_axes=(None, 0))(params, vs))

    def reconstruction_error(self, params: Array, vs: Array) -> Array:
        """Compute mean squared reconstruction error over a batch.

        Args:
            params: RBM parameters
            vs: Batch of visible patterns (shape: n_samples, n_visible)

        Returns:
            Mean squared error (scalar)
        """
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, vs)
        return jnp.mean((vs - recons) ** 2)

    # Parameter utilities

    # TODO: This and join_params are not really in the spirit of the library. They should be factored out and join/split coords should be used directly.
    def split_params(self, params: Array) -> tuple[Array, Array, Array]:
        """Split parameters into visible biases, weights, hidden biases.

        Args:
            params: RBM parameters

        Returns:
            Tuple of (visible_biases, weights, hidden_biases)
            where weights has shape (n_visible, n_hidden)
        """
        vis_bias, int_params, hid_bias = self.split_coords(params)
        weights = int_params.reshape(self.n_visible, self.n_hidden)
        return vis_bias, weights, hid_bias

    def join_params(self, vis_bias: Array, weights: Array, hid_bias: Array) -> Array:
        """Join parameters from visible biases, weights, hidden biases.

        Args:
            vis_bias: Visible biases (shape: n_visible)
            weights: Weight matrix (shape: n_visible, n_hidden)
            hid_bias: Hidden biases (shape: n_hidden)

        Returns:
            Combined RBM parameters
        """
        int_params = weights.ravel()
        return self.join_coords(vis_bias, int_params, hid_bias)

    def get_filters(self, params: Array, img_shape: tuple[int, int]) -> Array:
        """Reshape weight matrix into filter images for visualization.

        Each column of the weight matrix (corresponding to one hidden unit)
        is reshaped into an image.

        Args:
            params: RBM parameters
            img_shape: Shape of each visible image (height, width)

        Returns:
            Array of shape (n_hidden, height, width)
        """
        _, weights, _ = self.split_params(params)
        return weights.T.reshape(self.n_hidden, *img_shape)


# TODO: Even though visible/hidden language is fairly standard for RBMs, in the rest of the library I'm using observable/latent, so lets switch back to that.
def rbm(n_visible: int, n_hidden: int) -> RestrictedBoltzmannMachine:
    """Create a binary-binary Restricted Boltzmann Machine.

    Factory function for convenient construction of RBMs.

    Args:
        n_visible: Number of visible units
        n_hidden: Number of hidden units

    Returns:
        RestrictedBoltzmannMachine instance
    """
    return RestrictedBoltzmannMachine(n_visible=n_visible, n_hidden=n_hidden)
