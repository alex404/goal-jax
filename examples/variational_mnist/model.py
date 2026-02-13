"""Model utilities for variational MNIST.

This module provides:
- ConcreteHarmonium: concrete harmonium for arbitrary observable/latent pairs
- VariationalFullMixture: fully connected X↔Y↔K via CompleteMixtureOfHarmonium
- Model creation with good defaults (hierarchical or full interaction)
- Reconstruction and error utilities
- Conjugation quality metrics (R², Std[residual], ||rho||)
- IWAE bound for amortization gap estimation
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false

from dataclasses import dataclass
from typing import Any, Literal, override

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    Differentiable,
    EmbeddedMap,
    Harmonium,
    IdentityEmbedding,
    ObservableEmbedding,
    Rectangular,
)
from goal.geometry.exponential_family.variational import VariationalConjugated
from goal.geometry.manifold.embedding import LinearEmbedding
from goal.geometry.manifold.linear import LinearMap
from goal.models import (
    Bernoullis,
    Binomials,
    CompleteMixture,
    CompleteMixtureOfHarmonium,
    Poissons,
    VariationalHierarchicalMixture,
)

# MNIST dimensions
IMG_SIZE = 28
N_OBSERVABLE = IMG_SIZE * IMG_SIZE  # 784

# Default architecture
DEFAULT_N_LATENT = 1024
DEFAULT_N_CLUSTERS = 10
DEFAULT_N_TRIALS = 16

# Default multiplier for n_samples = multiplier * bas_lat_man.dim
ANALYTICAL_RHO_SAMPLES_MULTIPLIER = 5


### Concrete harmonium ###


@dataclass(frozen=True)
class ConcreteHarmonium[
    Observable: Differentiable, Latent: Differentiable
](Harmonium[Observable, Latent]):
    """Concrete harmonium coupling an observable to a latent manifold.

    A thin wrapper that stores the interaction map directly.
    """

    _int_man: EmbeddedMap[Latent, Observable]

    @property
    @override
    def int_man(self) -> LinearMap[Latent, Observable]:
        return self._int_man


### Variational full mixture ###


@dataclass(frozen=True)
class VariationalFullMixture[
    Observable: Differentiable, BaseLatent: Differentiable
](
    VariationalConjugated[
        Observable, CompleteMixture[BaseLatent], BaseLatent
    ],
):
    """Fully connected X↔Y↔K via CompleteMixtureOfHarmonium.

    Rho corrects only the BaseLatent (observable) component of the
    CompleteMixture posterior via ObservableEmbedding.  Cluster-specific
    structure is forced through the three-block interaction (xy, xyk, xk).
    """

    hrm: CompleteMixtureOfHarmonium[Observable, BaseLatent]

    @property
    @override
    def rho_emb(self) -> LinearEmbedding[BaseLatent, CompleteMixture[BaseLatent]]:
        return ObservableEmbedding(self.hrm.pst_man)  # pyright: ignore[reportReturnType]

    @property
    def mix_man(self) -> CompleteMixture[BaseLatent]:
        """Complete mixture over base latent."""
        return self.hrm.pst_man  # type: ignore[return-value]

    @property
    def n_categories(self) -> int:
        return self.mix_man.n_categories

    @property
    def bas_lat_man(self) -> BaseLatent:
        return self.mix_man.obs_man  # type: ignore[return-value]

    def get_cluster_probs(self, mixture_params: Array) -> Array:
        """Extract cluster probabilities from mixture parameters."""
        cat_natural = self.mix_man.prior(mixture_params)
        cat_means = self.mix_man.lat_man.to_mean(cat_natural)
        return self.mix_man.lat_man.to_probs(cat_means)

    def prior_entropy(self, params: Array) -> Array:
        """Compute entropy of the prior's marginal cluster distribution."""
        _, hrm_params = self.split_coords(params)
        _, _, prior_mixture_params = self.hrm.split_coords(hrm_params)
        probs = self.get_cluster_probs(prior_mixture_params)
        probs_safe = jnp.clip(probs, 1e-10, 1.0)
        return -jnp.sum(probs_safe * jnp.log(probs_safe))


### Type aliases and factory functions ###

type BinomialBernoulliHierarchical = VariationalHierarchicalMixture[Binomials, Bernoullis]
type PoissonBernoulliHierarchical = VariationalHierarchicalMixture[Poissons, Bernoullis]
type BinomialBernoulliFull = VariationalFullMixture[Binomials, Bernoullis]
type PoissonBernoulliFull = VariationalFullMixture[Poissons, Bernoullis]
type MixtureModel = (
    BinomialBernoulliHierarchical
    | PoissonBernoulliHierarchical
    | BinomialBernoulliFull
    | PoissonBernoulliFull
)


def _hierarchical_harmonium(
    obs_man: Differentiable,
    base_lat_man: Differentiable,
    n_categories: int,
) -> ConcreteHarmonium:  # pyright: ignore[reportMissingTypeArgument]
    """Create a ConcreteHarmonium[Obs, CompleteMixture[Lat]] for hierarchical mode."""
    mix_man = CompleteMixture(base_lat_man, n_categories)
    int_man = EmbeddedMap(
        Rectangular(),
        ObservableEmbedding(mix_man),
        IdentityEmbedding(obs_man),
    )
    return ConcreteHarmonium(int_man)


def _full_base_harmonium(
    obs_man: Differentiable,
    base_lat_man: Differentiable,
) -> ConcreteHarmonium:  # pyright: ignore[reportMissingTypeArgument]
    """Create a ConcreteHarmonium[Obs, Lat] for fully connected mode."""
    int_man = EmbeddedMap(
        Rectangular(),
        IdentityEmbedding(base_lat_man),
        IdentityEmbedding(obs_man),
    )
    return ConcreteHarmonium(int_man)


def create_model(
    n_latent: int = DEFAULT_N_LATENT,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    observable_type: Literal["binomial", "poisson"] = "binomial",
    n_trials: int = DEFAULT_N_TRIALS,
    interaction: Literal["hierarchical", "full"] = "full",
) -> MixtureModel:
    """Create a mixture model with the specified observable type and interaction mode.

    Args:
        n_latent: Number of latent units (default: 1024)
        n_clusters: Number of mixture components (default: 10)
        observable_type: Type of observable distribution ("binomial" or "poisson")
        n_trials: Binomial discretization level (default: 16, only used for binomial)
        interaction: Interaction mode:
            - "hierarchical": X↔(Y,K) with interaction restricted to Y component
            - "full": X↔Y↔K with three-block interaction (xy, xyk, xk)

    Returns:
        MixtureModel instance
    """
    base_lat_man = Bernoullis(n_latent)
    obs_man: Any
    if observable_type == "poisson":
        obs_man = Poissons(N_OBSERVABLE)
    else:
        obs_man = Binomials(N_OBSERVABLE, n_trials)

    if interaction == "full":
        bas_hrm = _full_base_harmonium(obs_man, base_lat_man)
        hrm = CompleteMixtureOfHarmonium(n_categories=n_clusters, bas_hrm=bas_hrm)
        return VariationalFullMixture(hrm=hrm)
    else:
        hrm = _hierarchical_harmonium(obs_man, base_lat_man, n_clusters)
        return VariationalHierarchicalMixture(hrm=hrm)


### Standalone utility functions ###


def reconstruct(model: MixtureModel, params: Array, x: Array) -> Array:
    """Reconstruct an observation via posterior mean."""
    q_params = model.approximate_posterior_at(params, x)
    z_mean_stats = model.pst_man.to_mean(q_params)
    _, hrm_params = model.split_coords(params)
    lkl_fun = model.hrm.likelihood_function(hrm_params)
    lkl_natural = model.hrm.lkl_fun_man(lkl_fun, z_mean_stats)
    return model.obs_man.to_mean(lkl_natural)


def normalized_reconstruction_error(
    model: MixtureModel, params: Array, xs: Array, scale: float = 1.0
) -> Array:
    """Compute normalized MSE between inputs and their reconstructions."""
    recons = jax.vmap(lambda x: reconstruct(model, params, x))(xs)
    return jnp.mean(((xs - recons) / scale) ** 2)


### Conjugation and evaluation utilities ###


def compute_conjugation_metrics(
    model: MixtureModel,
    params: Array,
    key: Array,
    n_samples: int = 1000,
) -> tuple[Array, Array, Array, Array]:
    """Compute conjugation quality metrics.

    Returns:
        Tuple of (var_f, std_f, r_squared, rho_norm)
    """
    var_f, std_f, r_squared = model.conjugation_metrics(key, params, n_samples)
    rho = model.conjugation_parameters(params)
    rho_norm = jnp.linalg.norm(rho)
    return var_f, std_f, r_squared, rho_norm


def iwae_bound(
    model: MixtureModel,
    key: Array,
    params: Array,
    x: Array,
    n_importance_samples: int = 1000,
) -> Array:
    """Compute the Importance-Weighted Autoencoder (IWAE) bound.

    IWAE is a tighter bound than ELBO:
        log p(x) >= IWAE(K) >= ELBO

    Args:
        model: The mixture model
        key: JAX random key
        params: Full model parameters
        x: Single observation
        n_importance_samples: Number of importance samples K

    Returns:
        IWAE bound value (scalar)
    """
    q_params = model.approximate_posterior_at(params, x)
    z_samples = model.pst_man.sample(key, q_params, n_importance_samples)

    log_joints = jax.vmap(lambda z: model.log_density(params, x, z))(z_samples)
    log_posteriors = jax.vmap(lambda z: model.pst_man.log_density(q_params, z))(
        z_samples
    )

    log_weights = log_joints - log_posteriors
    return jax.scipy.special.logsumexp(log_weights) - jnp.log(n_importance_samples)


def estimate_amortization_gap(
    model: MixtureModel,
    key: Array,
    params: Array,
    x: Array,
    n_elbo_samples: int = 10,
    n_iwae_samples: int = 1000,
) -> tuple[Array, Array, Array]:
    """Estimate the amortization gap: log p(x) - ELBO.

    Returns:
        Tuple of (gap_estimate, elbo, iwae)
    """
    key1, key2 = jax.random.split(key)
    elbo = model.elbo_at(key1, params, x, n_elbo_samples)
    iwae = iwae_bound(model, key2, params, x, n_iwae_samples)
    gap_estimate = iwae - elbo
    return gap_estimate, elbo, iwae


def batch_iwae(
    model: MixtureModel,
    key: Array,
    params: Array,
    xs: Array,
    n_importance_samples: int = 1000,
) -> Array:
    """Compute mean IWAE bound over a batch."""
    batch_size = xs.shape[0]
    keys = jax.random.split(key, batch_size)
    iwaes = jax.vmap(
        lambda k, x: iwae_bound(model, k, params, x, n_importance_samples)
    )(keys, xs)
    return jnp.mean(iwaes)


def batch_amortization_gap(
    model: MixtureModel,
    key: Array,
    params: Array,
    xs: Array,
    n_elbo_samples: int = 10,
    n_iwae_samples: int = 1000,
) -> tuple[Array, Array, Array]:
    """Estimate mean amortization gap over a batch.

    Returns:
        Tuple of (mean_gap, mean_elbo, mean_iwae)
    """
    batch_size = xs.shape[0]
    keys = jax.random.split(key, batch_size)

    def compute_single(k: Array, x: Array) -> tuple[Array, Array, Array]:
        return estimate_amortization_gap(
            model, k, params, x, n_elbo_samples, n_iwae_samples
        )

    gaps, elbos, iwaes = jax.vmap(compute_single)(keys, xs)
    return jnp.mean(gaps), jnp.mean(elbos), jnp.mean(iwaes)
