"""Variational hierarchical mixture models (X ↔ Z ↔ K).

This module provides hierarchical mixture models where:
- X: Observable variables (Binomial or Poisson counts)
- Z: Intermediate latent variables (Bernoulli or VonMises)
- K: Categorical mixture indicator

These models have the joint distribution:
    p(x, z, k) = p(k) * p(z|k) * p(x|z)

where X ⊥ K | Z (conditional independence).

Classes:
- BinomialMixtureBernoulliHarmonium: Binomial observables, mixture of Bernoulli latents
- BinomialBernoulliMixture: Variational wrapper for BinomialMixtureBernoulliHarmonium
- PoissonMixtureBernoulliHarmonium: Poisson observables, mixture of Bernoulli latents
- PoissonBernoulliMixture: Variational wrapper for PoissonMixtureBernoulliHarmonium
- BinomialMixtureVonMisesHarmonium: Binomial observables, mixture of VonMises latents
- VariationalBinomialVonMisesMixture: Variational wrapper for BinomialMixtureVonMisesHarmonium

Embeddings:
- MixtureBernoulliEmbedding: Maps to observable bias of Bernoulli mixture
- MixtureObservableEmbedding: Generic embedding for mixture observable biases
"""

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import (
    Differentiable,
    DifferentiableVariationalHierarchicalMixture,
    EmbeddedMap,
    IdentityEmbedding,
    LinearEmbedding,
    Rectangular,
)
from ...geometry.exponential_family.harmonium import DifferentiableConjugated
from ..base.binomial import Binomials
from ..base.categorical import Bernoullis
from ..base.poisson import Poissons
from ..base.von_mises import VonMisesProduct
from ..harmonium.mixture import CompleteMixture


### Embeddings ###


@dataclass(frozen=True)
class MixtureBernoulliEmbedding(
    LinearEmbedding[Bernoullis, CompleteMixture[Bernoullis]]
):
    """Embedding that maps to the observable bias of a Bernoulli mixture.

    This embedding implements the conditional independence X ⊥ K | Y by ensuring
    that the interaction matrix only affects the Bernoulli (observable) component
    of the mixture, not the categorical (latent) component.

    For a mixture with parameters [obs_bias, int_mat, cat_params]:
    - embed: maps obs_params -> [obs_params, 0, 0]
    - project: extracts obs_bias from full mixture params
    - translate: adds to obs_bias only, leaving int_mat and cat_params unchanged
    """

    mixture: CompleteMixture[Bernoullis]
    """The mixture manifold containing the Bernoullis."""

    @property
    @override
    def sub_man(self) -> Bernoullis:
        """The submanifold is the observable component of the mixture."""
        return self.mixture.obs_man

    @property
    @override
    def amb_man(self) -> CompleteMixture[Bernoullis]:
        """The ambient manifold is the full mixture."""
        return self.mixture

    @override
    def embed(self, coords: Array) -> Array:
        """Embed observable params into mixture space with zeros for other components."""
        int_zeros = jnp.zeros(self.mixture.int_man.dim)
        cat_zeros = jnp.zeros(self.mixture.lat_man.dim)
        return self.mixture.join_coords(coords, int_zeros, cat_zeros)

    @override
    def project(self, coords: Array) -> Array:
        """Project mixture params to observable component."""
        obs_bias, _, _ = self.mixture.split_coords(coords)
        return obs_bias

    @override
    def translate(self, p_coords: Array, q_coords: Array) -> Array:
        """Translate obs_bias by q_coords, leaving int_mat and cat_params unchanged."""
        obs_bias, int_mat, cat_params = self.mixture.split_coords(p_coords)
        new_obs_bias = obs_bias + q_coords
        return self.mixture.join_coords(new_obs_bias, int_mat, cat_params)


@dataclass(frozen=True)
class MixtureObservableEmbedding[Observable: Differentiable](
    LinearEmbedding[Observable, CompleteMixture[Observable]]
):
    """Embedding that maps to the observable bias of a mixture, excluding interaction/categorical.

    This embedding implements the conditional independence X ⊥ K | Z by ensuring
    that the interaction matrix only affects the VonMises (observable) component
    of the mixture, not the categorical (latent) component.

    For a mixture with parameters [obs_bias, int_mat, cat_params]:
    - embed: maps obs_params -> [obs_params, 0, 0]
    - project: extracts obs_bias from full mixture params
    - translate: adds to obs_bias only, leaving int_mat and cat_params unchanged
    """

    mixture: CompleteMixture[Observable]
    """The mixture manifold containing the observable."""

    @property
    @override
    def sub_man(self) -> Observable:
        """The submanifold is the observable component of the mixture."""
        return self.mixture.obs_man

    @property
    @override
    def amb_man(self) -> CompleteMixture[Observable]:
        """The ambient manifold is the full mixture."""
        return self.mixture

    @override
    def embed(self, coords: Array) -> Array:
        """Embed observable params into mixture space with zeros for other components."""
        int_zeros = jnp.zeros(self.mixture.int_man.dim)
        cat_zeros = jnp.zeros(self.mixture.lat_man.dim)
        return self.mixture.join_coords(coords, int_zeros, cat_zeros)

    @override
    def project(self, coords: Array) -> Array:
        """Project mixture params to observable component."""
        obs_bias, _, _ = self.mixture.split_coords(coords)
        return obs_bias

    @override
    def translate(self, p_coords: Array, q_coords: Array) -> Array:
        """Translate obs_bias by q_coords, leaving int_mat and cat_params unchanged."""
        obs_bias, int_mat, cat_params = self.mixture.split_coords(p_coords)
        new_obs_bias = obs_bias + q_coords
        return self.mixture.join_coords(new_obs_bias, int_mat, cat_params)


### Binomial-Bernoulli Hierarchical Models ###


@dataclass(frozen=True)
class BinomialMixtureBernoulliHarmonium(
    DifferentiableConjugated[
        Binomials,
        CompleteMixture[Bernoullis],
        CompleteMixture[Bernoullis],
    ]
):
    """Harmonium with Binomial observables and mixture of Bernoulli latents.

    This is the underlying harmonium for BinomialBernoulliMixture.

    The joint distribution factors as:
        p(x, y, k) = p(k) * p(y|k) * p(x|y)

    where X ⊥ K | Y (conditional independence via MixtureBernoulliEmbedding).
    """

    n_observable: int
    """Number of observable units (pixels)."""

    n_latent: int
    """Number of Bernoulli latent units."""

    n_clusters: int
    """Number of mixture components (clusters)."""

    n_trials: int = 16
    """Number of trials for each Binomial."""

    @property
    @override
    def obs_man(self) -> Binomials:
        """Observable manifold: product of Binomials."""
        return Binomials(self.n_observable, self.n_trials)

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
    ) -> LinearEmbedding[
        CompleteMixture[Bernoullis], CompleteMixture[Bernoullis]
    ]:
        """Embedding from posterior to prior latent space (identity for symmetric case)."""
        return IdentityEmbedding(self.pst_man)

    @property
    @override
    def int_man(self) -> EmbeddedMap[CompleteMixture[Bernoullis], Binomials]:
        """Interaction matrix connecting Binomials to Bernoullis only (not categorical).

        Uses MixtureBernoulliEmbedding on the domain side to ensure X ⊥ K | Y.
        The interaction W has shape (n_observable, n_latent).
        """
        mix_obs_emb = MixtureBernoulliEmbedding(self.pst_man)
        return EmbeddedMap(
            Rectangular(),
            mix_obs_emb,  # Domain: embeds Bernoullis into mixture
            IdentityEmbedding(self.obs_man),  # Codomain: full Binomials
        )

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute conjugation parameters using grid-based regression.

        For Binomial-Bernoulli harmoniums, we approximate:
            chi + rho^T * s_Y(y) ≈ psi_X(theta_X + Theta_XY * s_Y(y))

        where psi_X is the Binomial log-partition function.

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

        # Linear regression: psi ≈ chi + rho^T * s_y
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
class BinomialBernoulliMixture(
    DifferentiableVariationalHierarchicalMixture[Binomials, Bernoullis]
):
    """Variational Binomial-Bernoulli Mixture for clustering.

    Model structure:
    - Observable: Binomials (e.g., MNIST pixels in [0, n_trials])
    - Base latent: Bernoullis (binary hidden units)
    - Mixture: Categorical over K clusters

    Joint: p(x, y, k) = p(k) * p(y|k) * p(x|y)

    Uses variational inference with learnable rho parameters to approximate
    the intractable posterior q(y,k|x).

    Attributes:
        hrm: The underlying BinomialMixtureBernoulliHarmonium
    """

    hrm: BinomialMixtureBernoulliHarmonium
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

    @property
    def n_trials(self) -> int:
        """Number of trials for each Binomial."""
        return self.hrm.n_trials

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

    def normalized_reconstruction_error(self, params: Array, xs: Array) -> Array:
        """Compute MSE on normalized [0, 1] data."""
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        xs_norm = xs / self.n_trials
        recons_norm = recons / self.n_trials
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


def binomial_bernoulli_mixture(
    n_observable: int,
    n_latent: int,
    n_clusters: int,
    n_trials: int = 16,
) -> BinomialBernoulliMixture:
    """Create a Binomial-Bernoulli Mixture model for clustering.

    Factory function for convenient construction.

    Args:
        n_observable: Number of observable units (pixels)
        n_latent: Number of Bernoulli latent units
        n_clusters: Number of mixture components (clusters)
        n_trials: Number of trials for each Binomial

    Returns:
        BinomialBernoulliMixture instance
    """
    hrm = BinomialMixtureBernoulliHarmonium(
        n_observable=n_observable,
        n_latent=n_latent,
        n_clusters=n_clusters,
        n_trials=n_trials,
    )
    return BinomialBernoulliMixture(hrm=hrm)


### Poisson-Bernoulli Hierarchical Models ###


@dataclass(frozen=True)
class PoissonMixtureBernoulliHarmonium(
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

    where X ⊥ K | Y (conditional independence via MixtureBernoulliEmbedding).

    Key difference from BinomialMixtureBernoulliHarmonium:
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

        Uses MixtureBernoulliEmbedding on the domain side to ensure X ⊥ K | Y.
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
            chi + rho^T * s_Y(y) ≈ psi_X(theta_X + Theta_XY * s_Y(y))

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

        # Linear regression: psi ≈ chi + rho^T * s_y
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
        hrm: The underlying PoissonMixtureBernoulliHarmonium
    """

    hrm: PoissonMixtureBernoulliHarmonium
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
    hrm = PoissonMixtureBernoulliHarmonium(
        n_observable=n_observable,
        n_latent=n_latent,
        n_clusters=n_clusters,
    )
    return PoissonBernoulliMixture(hrm=hrm)


### Binomial-VonMises Hierarchical Models ###


@dataclass(frozen=True)
class BinomialMixtureVonMisesHarmonium(
    DifferentiableConjugated[
        Binomials,
        CompleteMixture[VonMisesProduct],
        CompleteMixture[VonMisesProduct],
    ]
):
    """Harmonium with Binomial observables and mixture of VonMises latents.

    This model enables MNIST digit clustering by learning:
    - Observable biases theta_X for each pixel
    - Interaction weights W connecting pixels to VonMises latents
    - Mixture of VonMises prior p(z|k) for different cluster components
    - Categorical prior p(k) over clusters

    The joint distribution factors as:
        p(x, z, k) = p(k) * p(z|k) * p(x|z)

    where X ⊥ K | Z (conditional independence via MixtureObservableEmbedding).

    Attributes:
        n_observable: Number of Binomial observable units (pixels)
        n_latent: Number of VonMises latent units
        n_clusters: Number of mixture components (clusters)
        n_trials: Number of trials for each Binomial
    """

    n_observable: int
    """Number of observable units (pixels)."""

    n_latent: int
    """Number of VonMises latent units."""

    n_clusters: int
    """Number of mixture components (clusters)."""

    n_trials: int = 16
    """Number of trials for each Binomial."""

    # Properties

    @property
    @override
    def obs_man(self) -> Binomials:
        """Observable manifold: product of Binomials."""
        return Binomials(self.n_observable, self.n_trials)

    @property
    def vonmises_man(self) -> VonMisesProduct:
        """VonMises product manifold for latents."""
        return VonMisesProduct(self.n_latent)

    @property
    @override
    def pst_man(self) -> CompleteMixture[VonMisesProduct]:
        """Posterior latent manifold: mixture of VonMises products."""
        return CompleteMixture(self.vonmises_man, self.n_clusters)

    @property
    @override
    def prr_man(self) -> CompleteMixture[VonMisesProduct]:
        """Prior latent manifold (same as posterior for symmetric structure)."""
        return self.pst_man

    @property
    @override
    def pst_prr_emb(
        self,
    ) -> LinearEmbedding[
        CompleteMixture[VonMisesProduct], CompleteMixture[VonMisesProduct]
    ]:
        """Embedding from posterior to prior latent space (identity for symmetric case)."""
        return IdentityEmbedding(self.pst_man)

    @property
    @override
    def int_man(self) -> EmbeddedMap[CompleteMixture[VonMisesProduct], Binomials]:
        """Interaction matrix connecting Binomials to VonMises only (not categorical).

        Uses MixtureObservableEmbedding on the domain side to ensure X ⊥ K | Z.
        The interaction W has shape (n_observable, 2 * n_latent).
        """
        mix_obs_emb = MixtureObservableEmbedding(self.pst_man)
        return EmbeddedMap(
            Rectangular(),
            mix_obs_emb,  # Domain: embeds VonMises into mixture
            IdentityEmbedding(self.obs_man),  # Codomain: full Binomials
        )

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute conjugation parameters using grid-based regression.

        For Binomial-VonMises harmoniums, we approximate:
            chi + rho^T * s_Z(z) ≈ psi_X(theta_X + Theta_XZ * s_Z(z))

        where psi_X is the Binomial log-partition function.

        The conjugation parameters are embedded into mixture space,
        only affecting the observable (VonMises) bias component.

        Args:
            lkl_params: Likelihood parameters [obs_bias, int_mat]

        Returns:
            Conjugation parameters in mixture space
        """
        obs_params, int_params = self.lkl_fun_man.split_coords(lkl_params)

        # Reshape interaction to matrix
        int_matrix = int_params.reshape(self.n_observable, 2 * self.n_latent)

        # Create grid of latent angles for regression
        n_grid = 50

        # Sample random VonMises vectors for grid-based regression
        key = jax.random.PRNGKey(0)  # Fixed seed for reproducibility
        z_grid = jax.random.uniform(
            key, shape=(n_grid, self.n_latent), minval=0, maxval=2 * jnp.pi
        )

        def compute_psi_at_z(z: Array) -> Array:
            """Compute log partition of likelihood at z."""
            s_z = self.vonmises_man.sufficient_statistic(z)
            lkl_nat = obs_params + int_matrix @ s_z
            return self.obs_man.log_partition_function(lkl_nat)

        def get_sufficient_stat(z: Array) -> Array:
            return self.vonmises_man.sufficient_statistic(z)

        psi_values = jax.vmap(compute_psi_at_z)(z_grid)
        s_z_grid = jax.vmap(get_sufficient_stat)(z_grid)

        # Linear regression: psi ≈ chi + rho^T * s_z
        # Design matrix: [1, s_z]
        ones = jnp.ones((n_grid, 1))
        design = jnp.concatenate([ones, s_z_grid], axis=1)

        # Solve least squares: (D^T D)^{-1} D^T psi
        coeffs = jnp.linalg.lstsq(design, psi_values, rcond=None)[0]
        rho_vm = coeffs[1:]  # Skip the intercept (chi)

        # Embed rho into mixture space (only affects obs_bias)
        mix_obs_emb = MixtureObservableEmbedding(self.pst_man)
        return mix_obs_emb.embed(rho_vm)

    # Convenience methods

    def get_cluster_probs(self, mixture_params: Array) -> Array:
        """Extract cluster probabilities from mixture parameters.

        Args:
            mixture_params: Natural parameters of the mixture

        Returns:
            Array of shape (n_clusters,) with cluster probabilities
        """
        _, _, cat_params = self.pst_man.split_coords(mixture_params)
        cat_means = self.pst_man.lat_man.to_mean(cat_params)
        return self.pst_man.lat_man.to_probs(cat_means)

    def get_vonmises_bias(self, mixture_params: Array) -> Array:
        """Extract VonMises bias from mixture parameters.

        Args:
            mixture_params: Natural parameters of the mixture

        Returns:
            VonMises natural parameters (shape: 2 * n_latent)
        """
        obs_bias, _, _ = self.pst_man.split_coords(mixture_params)
        return obs_bias

    def get_component_params(self, mixture_params: Array, k: int) -> Array:
        """Get VonMises parameters for a specific mixture component.

        Args:
            mixture_params: Natural parameters of the mixture
            k: Component index (0 to n_clusters-1)

        Returns:
            VonMises natural parameters for component k
        """
        components, _ = self.pst_man.split_natural_mixture(mixture_params)
        return self.pst_man.cmp_man.get_replicate(components, k)

    def observable_means(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] for all observable units.

        Args:
            params: Model parameters
            z: Latent angles (shape: n_latent)

        Returns:
            Expected counts for each observable unit (shape: n_observable)
        """
        lkl_params = self.likelihood_at(params, z)
        return self.obs_man.to_mean(lkl_params)

    def reconstruct(self, params: Array, x: Array) -> Array:
        """Reconstruct observable via mean-field approximation.

        Computes E[x | E[z|x]], using the mean of the posterior VonMises
        (marginalizing over clusters) to compute expected observables.

        Args:
            params: Model parameters
            x: Input observable counts

        Returns:
            Reconstructed observable counts (expected values)
        """
        # Get posterior mixture parameters
        post_params = self.posterior_at(params, x)

        # Get VonMises mean sufficient statistics (from the mixture marginal)
        # For the marginal, we use the mean over all components weighted by probs
        _, _, cat_params = self.pst_man.split_coords(post_params)
        cat_means = self.pst_man.lat_man.to_mean(cat_params)
        probs = self.pst_man.lat_man.to_probs(cat_means)

        # Get component VonMises means
        components, _ = self.pst_man.split_natural_mixture(post_params)
        comp_params_2d = self.pst_man.cmp_man.to_2d(components)

        # Compute mean sufficient statistics for each component
        def component_mean_stats(comp_params: Array) -> Array:
            return self.vonmises_man.to_mean(comp_params)

        comp_mean_stats = jax.vmap(component_mean_stats)(comp_params_2d)

        # Weighted average of mean sufficient statistics
        z_mean_stats = jnp.einsum("k,ki->i", probs, comp_mean_stats)

        # Compute likelihood parameters
        obs_bias_hrm, int_params, _ = self.split_coords(params)
        int_matrix = int_params.reshape(self.n_observable, 2 * self.n_latent)
        lkl_natural = obs_bias_hrm + int_matrix @ z_mean_stats

        return self.obs_man.to_mean(lkl_natural)

    def get_filters(self, params: Array, img_shape: tuple[int, int]) -> Array:
        """Reshape weight matrix into filter images for visualization.

        Each pair of columns (corresponding to cos and sin for one latent)
        are combined into a single filter image showing the magnitude.

        Args:
            params: Model parameters
            img_shape: Shape of each observable image (height, width)

        Returns:
            Array of shape (n_latent, height, width)
        """
        _, int_params, _ = self.split_coords(params)
        weights = int_params.reshape(self.n_observable, 2 * self.n_latent)

        # Combine cos and sin weights for each latent into magnitude
        filters = []
        for i in range(self.n_latent):
            w_cos = weights[:, 2 * i]
            w_sin = weights[:, 2 * i + 1]
            magnitude = jnp.sqrt(w_cos**2 + w_sin**2)
            filters.append(magnitude.reshape(*img_shape))

        return jnp.stack(filters)


@dataclass(frozen=True)
class VariationalBinomialVonMisesMixture(
    DifferentiableVariationalHierarchicalMixture[Binomials, VonMisesProduct]
):
    """Variational Binomial-VonMises Mixture for clustering.

    Wraps BinomialMixtureVonMisesHarmonium with learnable rho parameters
    for variational inference via ELBO optimization.
    """

    hrm: BinomialMixtureVonMisesHarmonium

    @property
    @override
    def n_categories(self) -> int:
        return self.hrm.n_clusters

    @property
    @override
    def bas_lat_man(self) -> VonMisesProduct:
        return self.hrm.vonmises_man

    @property
    def n_observable(self) -> int:
        return self.hrm.n_observable

    @property
    def n_latent(self) -> int:
        return self.hrm.n_latent

    @property
    def n_clusters(self) -> int:
        return self.hrm.n_clusters

    @property
    def n_trials(self) -> int:
        return self.hrm.n_trials

    def observable_means(self, params: Array, z: Array) -> Array:
        """Compute E[x_i | z] for all observable units."""
        lkl_params = self.likelihood_at(params, z)
        return self.obs_man.to_mean(lkl_params)

    def reconstruct(self, params: Array, x: Array) -> Array:
        """Reconstruct observable via mean-field approximation."""
        post_params = self.approximate_posterior_at(params, x)
        probs = self.get_cluster_probs(post_params)

        components, _ = self.mix_man.split_natural_mixture(post_params)
        comp_params_2d = self.mix_man.cmp_man.to_2d(components)

        comp_mean_stats = jax.vmap(self.bas_lat_man.to_mean)(comp_params_2d)
        z_mean_stats = jnp.einsum("k,ki->i", probs, comp_mean_stats)

        hrm_params = self.generative_params(params)
        obs_bias_hrm, int_params, _ = self.hrm.split_coords(hrm_params)
        int_matrix = int_params.reshape(self.n_observable, 2 * self.n_latent)
        lkl_natural = obs_bias_hrm + int_matrix @ z_mean_stats

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
        hrm_params = self.generative_params(params)
        _, int_params, _ = self.hrm.split_coords(hrm_params)
        weights = int_params.reshape(self.n_observable, 2 * self.n_latent)

        filters = []
        for i in range(self.n_latent):
            w_cos = weights[:, 2 * i]
            w_sin = weights[:, 2 * i + 1]
            magnitude = jnp.sqrt(w_cos**2 + w_sin**2)
            filters.append(magnitude.reshape(*img_shape))

        return jnp.stack(filters)

    def get_cluster_assignments(self, params: Array, x: Array) -> Array:
        """Get hard cluster assignment for observation x."""
        q_params = self.approximate_posterior_at(params, x)
        probs = self.get_cluster_probs(q_params)
        return jnp.argmax(probs)


def variational_binomial_vonmises_mixture(
    n_observable: int,
    n_latent: int,
    n_clusters: int,
    n_trials: int = 16,
) -> VariationalBinomialVonMisesMixture:
    """Create a Variational Binomial-VonMises Mixture model for clustering."""
    hrm = BinomialMixtureVonMisesHarmonium(
        n_observable=n_observable,
        n_latent=n_latent,
        n_clusters=n_clusters,
        n_trials=n_trials,
    )
    return VariationalBinomialVonMisesMixture(hrm=hrm)


def binomial_vonmises_mixture(
    n_observable: int,
    n_latent: int,
    n_clusters: int,
    n_trials: int = 16,
) -> BinomialMixtureVonMisesHarmonium:
    """Create a Binomial-VonMises Mixture model for clustering.

    Factory function for convenient construction.

    Args:
        n_observable: Number of observable units (pixels)
        n_latent: Number of VonMises latent units
        n_clusters: Number of mixture components (clusters)
        n_trials: Number of trials for each Binomial

    Returns:
        BinomialMixtureVonMisesHarmonium instance
    """
    return BinomialMixtureVonMisesHarmonium(
        n_observable=n_observable,
        n_latent=n_latent,
        n_clusters=n_clusters,
        n_trials=n_trials,
    )
