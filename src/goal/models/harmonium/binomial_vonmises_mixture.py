"""Binomial-VonMises Mixture harmonium for clustering.

This module defines a harmonium with Binomial observables and a mixture of
VonMises latents, suitable for unsupervised clustering tasks like MNIST.
"""

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import (
    Differentiable,
    EmbeddedMap,
    IdentityEmbedding,
    LinearEmbedding,
    Rectangular,
)
from ...geometry.exponential_family.harmonium import DifferentiableConjugated
from ..base.binomial import Binomials
from ..base.von_mises import VonMisesProduct
from .mixture import CompleteMixture


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


@dataclass(frozen=True)
class BinomialVonMisesMixture(
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


def binomial_vonmises_mixture(
    n_observable: int,
    n_latent: int,
    n_clusters: int,
    n_trials: int = 16,
) -> BinomialVonMisesMixture:
    """Create a Binomial-VonMises Mixture model for clustering.

    Factory function for convenient construction.

    Args:
        n_observable: Number of observable units (pixels)
        n_latent: Number of VonMises latent units
        n_clusters: Number of mixture components (clusters)
        n_trials: Number of trials for each Binomial

    Returns:
        BinomialVonMisesMixture instance
    """
    return BinomialVonMisesMixture(
        n_observable=n_observable,
        n_latent=n_latent,
        n_clusters=n_clusters,
        n_trials=n_trials,
    )
