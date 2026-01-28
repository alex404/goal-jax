"""Poisson-Bernoulli Mixture harmonium for clustering.

This module defines a harmonium with Poisson observables and a mixture of
Bernoulli latents, suitable for unsupervised clustering tasks like MNIST.

Unlike the Binomial version, Poisson observables have unbounded support,
which may avoid ceiling effects and provide different learning dynamics.
"""

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import (
    DifferentiableVariationalHierarchicalMixture,
    EmbeddedMap,
    IdentityEmbedding,
    LinearEmbedding,
    Rectangular,
)
from ...geometry.exponential_family.harmonium import DifferentiableConjugated
from ..base.categorical import Bernoullis
from ..base.poisson import Poissons
from .binomial_bernoulli_mixture import MixtureBernoulliEmbedding
from .mixture import CompleteMixture


@dataclass(frozen=True)
class PoissonBernoulliHarmonium(
    DifferentiableConjugated[
        Poissons,
        CompleteMixture[Bernoullis],
        CompleteMixture[Bernoullis],
    ]
):
    """Harmonium with Poisson observables and mixture of Bernoulli latents.

    This is the underlying harmonium for PoissonBernoulliMixture.

    The joint distribution factors as:
        p(x, y, k) = p(k) * p(y|k) * p(x|y)

    where X perp K | Y (conditional independence via MixtureBernoulliEmbedding).

    Key difference from BinomialBernoulliHarmonium:
    - Poisson has unbounded support (counts can be any non-negative integer)
    - Log-partition is exp(theta) instead of n*softplus(theta)
    - No n_trials parameter needed
    """

    n_observable: int
    """Number of observable units (pixels)."""

    n_latent: int
    """Number of Bernoulli latent units."""

    n_clusters: int
    """Number of mixture components (clusters)."""

    @property
    @override
    def obs_man(self) -> Poissons:
        """Observable manifold: product of Poissons."""
        return Poissons(self.n_observable)

    @property
    def bernoulli_man(self) -> Bernoullis:
        """Bernoulli product manifold for latents."""
        return Bernoullis(self.n_latent)

    @property
    @override
    def pst_man(self) -> CompleteMixture[Bernoullis]:
        """Posterior latent manifold: mixture of Bernoulli products."""
        return CompleteMixture(self.bernoulli_man, self.n_clusters)

    @property
    @override
    def prr_man(self) -> CompleteMixture[Bernoullis]:
        """Prior latent manifold (same as posterior for symmetric structure)."""
        return self.pst_man

    @property
    @override
    def pst_prr_emb(
        self,
    ) -> LinearEmbedding[CompleteMixture[Bernoullis], CompleteMixture[Bernoullis]]:
        """Embedding from posterior to prior latent space (identity for symmetric case)."""
        return IdentityEmbedding(self.pst_man)

    @property
    @override
    def int_man(self) -> EmbeddedMap[CompleteMixture[Bernoullis], Poissons]:
        """Interaction matrix connecting Poissons to Bernoullis only (not categorical).

        Uses MixtureBernoulliEmbedding on the domain side to ensure X perp K | Y.
        The interaction W has shape (n_observable, n_latent).
        """
        mix_obs_emb = MixtureBernoulliEmbedding(self.pst_man)
        return EmbeddedMap(
            Rectangular(),
            mix_obs_emb,  # Domain: embeds Bernoullis into mixture
            IdentityEmbedding(self.obs_man),  # Codomain: full Poissons
        )

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute conjugation parameters using grid-based regression.

        For Poisson-Bernoulli harmoniums, we approximate:
            chi + rho^T * s_Y(y) approx psi_X(theta_X + Theta_XY * s_Y(y))

        where psi_X is the Poisson log-partition function (sum of exp(theta_i)).

        The conjugation parameters are embedded into mixture space,
        only affecting the observable (Bernoulli) bias component.
        """
        obs_params, int_params = self.lkl_fun_man.split_coords(lkl_params)

        # Reshape interaction to matrix
        int_matrix = int_params.reshape(self.n_observable, self.n_latent)

        # Create grid of latent values for regression
        # For Bernoulli, sufficient statistics are just the values in [0, 1]
        n_grid = 50

        # Sample random Bernoulli vectors for grid-based regression
        key = jax.random.PRNGKey(0)  # Fixed seed for reproducibility
        y_grid = jax.random.uniform(key, shape=(n_grid, self.n_latent))

        def compute_psi_at_y(y: Array) -> Array:
            """Compute log partition of likelihood at y."""
            s_y = self.bernoulli_man.sufficient_statistic(y)
            lkl_nat = obs_params + int_matrix @ s_y
            return self.obs_man.log_partition_function(lkl_nat)

        def get_sufficient_stat(y: Array) -> Array:
            return self.bernoulli_man.sufficient_statistic(y)

        psi_values = jax.vmap(compute_psi_at_y)(y_grid)
        s_y_grid = jax.vmap(get_sufficient_stat)(y_grid)

        # Linear regression: psi approx chi + rho^T * s_y
        # Design matrix: [1, s_y]
        ones = jnp.ones((n_grid, 1))
        design = jnp.concatenate([ones, s_y_grid], axis=1)

        # Solve least squares: (D^T D)^{-1} D^T psi
        coeffs = jnp.linalg.lstsq(design, psi_values, rcond=None)[0]
        rho_ber = coeffs[1:]  # Skip the intercept (chi)

        # Embed rho into mixture space (only affects obs_bias)
        mix_obs_emb = MixtureBernoulliEmbedding(self.pst_man)
        return mix_obs_emb.embed(rho_ber)


@dataclass(frozen=True)
class PoissonBernoulliMixture(
    DifferentiableVariationalHierarchicalMixture[Poissons, Bernoullis]
):
    """Variational Poisson-Bernoulli Mixture for clustering.

    Model structure:
    - Observable: Poissons (e.g., MNIST pixels as counts)
    - Base latent: Bernoullis (binary hidden units)
    - Mixture: Categorical over K clusters

    Joint: p(x, y, k) = p(k) * p(y|k) * p(x|y)

    Uses variational inference with learnable rho parameters to approximate
    the intractable posterior q(y,k|x).

    Key difference from BinomialBernoulliMixture:
    - No n_trials parameter (Poisson has unbounded support)
    - Different log-partition function behavior

    Attributes:
        hrm: The underlying PoissonBernoulliHarmonium
    """

    hrm: PoissonBernoulliHarmonium
    """The underlying harmonium model."""

    @property
    @override
    def n_categories(self) -> int:
        """Number of mixture components."""
        return self.hrm.n_clusters

    @property
    @override
    def bas_lat_man(self) -> Bernoullis:
        """Base latent manifold (Bernoullis)."""
        return self.hrm.bernoulli_man

    # Convenience accessors

    @property
    def n_observable(self) -> int:
        """Number of observable units."""
        return self.hrm.n_observable

    @property
    def n_latent(self) -> int:
        """Number of Bernoulli latent units."""
        return self.hrm.n_latent

    @property
    def n_clusters(self) -> int:
        """Number of mixture components."""
        return self.hrm.n_clusters

    # Convenience methods

    def observable_means(self, params: Array, y: Array) -> Array:
        """Compute E[x_i | y] for all observable units.

        Args:
            params: Full model parameters
            y: Latent Bernoulli values (shape: n_latent)

        Returns:
            Expected counts for each observable unit (shape: n_observable)
        """
        lkl_params = self.likelihood_at(params, y)
        return self.obs_man.to_mean(lkl_params)

    def reconstruct(self, params: Array, x: Array) -> Array:
        """Reconstruct observable via mean-field approximation.

        Computes E[x | E[y|x]], using the mean of the posterior Bernoullis
        (marginalizing over clusters) to compute expected observables.

        Args:
            params: Full model parameters
            x: Input observable counts

        Returns:
            Reconstructed observable counts (expected values)
        """
        # Get posterior mixture parameters
        post_params = self.approximate_posterior_at(params, x)

        # Get Bernoulli mean sufficient statistics (from the mixture marginal)
        # For the marginal, we use the mean over all components weighted by probs
        probs = self.get_cluster_probs(post_params)

        # Get component Bernoulli means
        components, _ = self.mix_man.split_natural_mixture(post_params)
        comp_params_2d = self.mix_man.cmp_man.to_2d(components)

        # Compute mean sufficient statistics for each component
        def component_mean_stats(comp_params: Array) -> Array:
            return self.bas_lat_man.to_mean(comp_params)

        comp_mean_stats = jax.vmap(component_mean_stats)(comp_params_2d)

        # Weighted average of mean sufficient statistics
        y_mean_stats = jnp.einsum("k,ki->i", probs, comp_mean_stats)

        # Compute likelihood parameters
        hrm_params = self.generative_params(params)
        obs_bias_hrm, int_params, _ = self.hrm.split_coords(hrm_params)
        int_matrix = int_params.reshape(self.n_observable, self.n_latent)
        lkl_natural = obs_bias_hrm + int_matrix @ y_mean_stats

        return self.obs_man.to_mean(lkl_natural)

    def reconstruction_error(self, params: Array, xs: Array) -> Array:
        """Compute mean squared reconstruction error over a batch."""
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        return jnp.mean((xs - recons) ** 2)

    def normalized_reconstruction_error(
        self, params: Array, xs: Array, scale: float = 255.0
    ) -> Array:
        """Compute MSE on normalized [0, 1] data.

        Args:
            params: Full model parameters
            xs: Batch of observations
            scale: The scaling factor used for the data (default 255.0 for raw pixels)

        Returns:
            Mean squared error on [0, 1] normalized data
        """
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        xs_norm = xs / scale
        recons_norm = recons / scale
        return jnp.mean((xs_norm - recons_norm) ** 2)

    def get_filters(self, params: Array, img_shape: tuple[int, int]) -> Array:
        """Reshape weight matrix into filter images for visualization.

        Args:
            params: Full model parameters
            img_shape: Shape of each observable image (height, width)

        Returns:
            Array of shape (n_latent, height, width)
        """
        hrm_params = self.generative_params(params)
        _, int_params, _ = self.hrm.split_coords(hrm_params)
        weights = int_params.reshape(self.n_observable, self.n_latent)

        # Each column is a filter
        filters = []
        for i in range(self.n_latent):
            filters.append(weights[:, i].reshape(*img_shape))

        return jnp.stack(filters)

    def get_cluster_assignments(self, params: Array, x: Array) -> Array:
        """Get hard cluster assignment for observation x.

        Returns argmax_k q(k|x).

        Args:
            params: Full model parameters
            x: Observation

        Returns:
            Cluster index (scalar integer)
        """
        q_params = self.approximate_posterior_at(params, x)
        probs = self.get_cluster_probs(q_params)
        return jnp.argmax(probs)


def poisson_bernoulli_mixture(
    n_observable: int,
    n_latent: int,
    n_clusters: int,
) -> PoissonBernoulliMixture:
    """Create a Poisson-Bernoulli Mixture model for clustering.

    Factory function for convenient construction.

    Args:
        n_observable: Number of observable units (pixels)
        n_latent: Number of Bernoulli latent units
        n_clusters: Number of mixture components (clusters)

    Returns:
        PoissonBernoulliMixture instance
    """
    hrm = PoissonBernoulliHarmonium(
        n_observable=n_observable,
        n_latent=n_latent,
        n_clusters=n_clusters,
    )
    return PoissonBernoulliMixture(hrm=hrm)
