"""Model utilities for variational MNIST.

This module provides:
- ConcreteHarmonium: concrete harmonium for arbitrary observable/latent pairs
- VariationalFullMixture: fully connected X<->Y<->K via CompleteMixtureOfHarmoniums
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
from goal.geometry.exponential_family.variational import (
    VariationalSymmetric,
    conjugation_metrics,
)
from goal.geometry.manifold.map import LinearMap
from goal.models import (
    Bernoullis,
    Binomials,
    ChordalBoltzmann,
    CompleteMixture,
    CompleteMixtureOfHarmoniums,
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
class ConcreteHarmonium[Observable: Differentiable, Latent: Differentiable](
    Harmonium[Observable, Latent]
):
    """Concrete harmonium coupling an observable to a latent manifold.

    A thin wrapper that stores the interaction map directly.
    """

    _int_man: EmbeddedMap[Latent, Observable]

    @property
    @override
    def int_man(self) -> LinearMap[Latent, Observable]:
        return self._int_man


### Concrete mixture of harmoniums ###


@dataclass(frozen=True)
class ConcreteCompleteMixtureOfHarmoniums[
    Observable: Differentiable,
    Posterior: Differentiable,
](CompleteMixtureOfHarmoniums[Observable, Posterior]):
    """Concrete instantiation of CompleteMixtureOfHarmoniums."""

    pass


### Variational full mixture ###


@dataclass(frozen=True)
class VariationalFullMixture[Observable: Differentiable, BaseLatent: Differentiable](
    VariationalSymmetric[Observable, CompleteMixture[BaseLatent], BaseLatent],
):
    """Fully connected X<->Y<->K via CompleteMixtureOfHarmoniums.

    The base ``CompleteMixtureOfHarmoniums`` has a three-block interaction
    (``xy``, ``xyk``, ``xk``), so distinct components carry distinct observable
    offsets. ``conjugation_parameters`` reproduces the analytic mixture
    completion from ``CompleteMixtureOfConjugated.conjugation_parameters``,
    substituting the variationally-learned ``rho`` (in ``BaseLatent``
    shape) for what the analytic case computes from the base harmonium's
    conjugation. ``rho_yz`` collapses to zero (since the variational
    approximation reuses ``rho`` across components) but ``rho_z`` is
    the analytic log-partition difference of the observable across
    component-specific offsets --- this is the contribution that the previous
    ``rho_emb = ObservableEmbedding(...)`` design silently dropped.
    """

    _gen_hrm: CompleteMixtureOfHarmoniums[Observable, BaseLatent]

    @property
    @override
    def gen_hrm(self) -> CompleteMixtureOfHarmoniums[Observable, BaseLatent]:
        return self._gen_hrm

    @property
    @override
    def lat_man(self) -> CompleteMixture[BaseLatent]:
        return self._gen_hrm.pst_man

    @property
    @override
    def cnj_man(self) -> BaseLatent:
        return self.bas_lat_man

    @override
    def conjugation_parameters(self, params: Array, x: Array | None = None) -> Array:
        """Mirror :meth:`CompleteMixtureOfConjugated.conjugation_parameters` with $\\rho_y$ peeled out of ``params``."""
        del x
        _, lkl_params, rho_y = self.split_coords(params)
        x_params, int_params = self.gen_hrm.lkl_fun_man.split_coords(lkl_params)
        xy_params, xyk_params, xk_params = self.gen_hrm.int_man.coord_blocks(int_params)
        del xy_params, xyk_params  # unused under the rho_yi = rho_y approximation

        if self.n_categories == 1:
            return self.prr_man.join_coords(rho_y, jnp.array([]), jnp.array([]))

        xk_2d = xk_params.reshape(-1, self.n_categories - 1)
        psi_base = self.obs_man.log_partition_function(x_params)

        def rho_z_k(x_offset: Array) -> Array:
            return self.obs_man.log_partition_function(x_params + x_offset) - psi_base

        rho_z = jax.vmap(rho_z_k, in_axes=1)(xk_2d)

        # Under the variational approximation rho_yi = rho_y for all components,
        # so rho_yz = rho_yi - rho_y = 0.
        rho_yz_zeros = jnp.zeros(self.prr_man.int_man.dim)

        return self.prr_man.join_coords(rho_y, rho_yz_zeros, rho_z)

    @property
    def mix_man(self) -> CompleteMixture[BaseLatent]:
        """Complete mixture over base latent."""
        return self.gen_hrm.pst_man  # type: ignore[return-value]

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
        prior_mixture_params, _, _ = self.split_coords(params)
        probs = self.get_cluster_probs(prior_mixture_params)
        probs_safe = jnp.clip(probs, 1e-10, 1.0)
        return -jnp.sum(probs_safe * jnp.log(probs_safe))


### Concrete variational hierarchical mixture ###


@dataclass(frozen=True)
class ConcreteVariationalHierarchicalMixture[
    Observable: Differentiable,
    BaseLatent: Differentiable,
](VariationalHierarchicalMixture[Observable, BaseLatent]):
    """Concrete instantiation of VariationalHierarchicalMixture."""

    _gen_hrm: Harmonium[Observable, Any]  # CompleteMixture[BaseLatent] in practice

    @property
    @override
    def gen_hrm(self) -> Harmonium[Observable, Any]:
        return self._gen_hrm


### Type aliases and factory functions ###

type BinomialBernoulliHierarchical = ConcreteVariationalHierarchicalMixture[
    Binomials, Bernoullis
]
type PoissonBernoulliHierarchical = ConcreteVariationalHierarchicalMixture[
    Poissons, Bernoullis
]
type BinomialBernoulliFull = VariationalFullMixture[Binomials, Bernoullis]
type PoissonBernoulliFull = VariationalFullMixture[Poissons, Bernoullis]
type BinomialChordalHierarchical = ConcreteVariationalHierarchicalMixture[
    Binomials, ChordalBoltzmann
]
type PoissonChordalHierarchical = ConcreteVariationalHierarchicalMixture[
    Poissons, ChordalBoltzmann
]
type BinomialChordalFull = VariationalFullMixture[Binomials, ChordalBoltzmann]
type PoissonChordalFull = VariationalFullMixture[Poissons, ChordalBoltzmann]
type MixtureModel = (
    BinomialBernoulliHierarchical
    | PoissonBernoulliHierarchical
    | BinomialBernoulliFull
    | PoissonBernoulliFull
    | BinomialChordalHierarchical
    | PoissonChordalHierarchical
    | BinomialChordalFull
    | PoissonChordalFull
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


def _chain_edges(n: int) -> list[tuple[int, int]]:
    """1D nearest-neighbour edges along a line of ``n`` nodes."""
    return [(i, i + 1) for i in range(n - 1)]


def _build_base_latent(
    n_latent: int,
    latent: Literal["bernoullis", "chordal_boltzmann"],
    latent_graph: Literal["chain"],
) -> Any:
    """Construct the base latent manifold given the requested kind and graph.

    ``latent`` and ``latent_graph`` are both Literal-typed; argparse validates the
    CLI input against the same choices, so there is no runtime fallback path.
    """
    if latent == "bernoullis":
        return Bernoullis(n_latent)
    # latent == "chordal_boltzmann"; only "chain" is a valid latent_graph today.
    _ = latent_graph
    return ChordalBoltzmann.from_edges(n_latent, _chain_edges(n_latent))


def create_model(
    n_latent: int = DEFAULT_N_LATENT,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    observable_type: Literal["binomial", "poisson"] = "binomial",
    n_trials: int = DEFAULT_N_TRIALS,
    interaction: Literal["hierarchical", "full"] = "full",
    latent: Literal["bernoullis", "chordal_boltzmann"] = "bernoullis",
    latent_graph: Literal["chain"] = "chain",
) -> MixtureModel:
    """Create a mixture model with the specified observable type and interaction mode.

    Args:
        n_latent: Number of latent units (default: 1024)
        n_clusters: Number of mixture components (default: 10)
        observable_type: Type of observable distribution ("binomial" or "poisson")
        n_trials: Binomial discretization level (default: 16, only used for binomial)
        interaction: Interaction mode:
            - "hierarchical": X<->(Y,K) with interaction restricted to Y component
            - "full": X<->Y<->K with three-block interaction (xy, xyk, xk)
        latent: Base latent kind. ``"bernoullis"`` (default) is independent
            binary units. ``"chordal_boltzmann"`` is a Boltzmann machine with
            chordal pairwise couplings, with exact junction-tree inference.
        latent_graph: For ``latent="chordal_boltzmann"``, the sparsity pattern
            of the couplings. Currently only ``"chain"`` is supported (1D
            nearest-neighbour, treewidth 1). Ignored for ``latent="bernoullis"``.

    Returns:
        MixtureModel instance
    """
    base_lat_man = _build_base_latent(n_latent, latent, latent_graph)
    obs_man: Any
    if observable_type == "poisson":
        obs_man = Poissons(N_OBSERVABLE)
    else:
        obs_man = Binomials(N_OBSERVABLE, n_trials)

    if interaction == "full":
        bas_hrm = _full_base_harmonium(obs_man, base_lat_man)
        hrm = ConcreteCompleteMixtureOfHarmoniums(
            n_categories=n_clusters, bas_hrm=bas_hrm
        )
        return VariationalFullMixture(_gen_hrm=hrm)
    hrm = _hierarchical_harmonium(obs_man, base_lat_man, n_clusters)
    return ConcreteVariationalHierarchicalMixture(_gen_hrm=hrm)


### Standalone utility functions ###


def reconstruct(model: MixtureModel, params: Array, x: Array) -> Array:
    """Reconstruct an observation via posterior mean."""
    q_params = model.approximate_posterior_at(params, x)
    z_mean_stats = model.pst_man.to_mean(q_params)
    _, lkl, _ = model.split_coords(params)
    lkl_natural = model.gen_hrm.lkl_fun_man(lkl, z_mean_stats)
    return model.obs_man.to_mean(lkl_natural)


def normalized_reconstruction_error(
    model: MixtureModel, params: Array, xs: Array, scale: float = 1.0
) -> Array:
    """Compute normalized MSE between inputs and their reconstructions.

    Reconstructions are computed in fixed-size chunks via ``jax.lax.map`` so
    that latent-prior models with non-trivial ``to_mean`` (e.g. ChordalBoltzmann,
    whose log-partition autodiff retains the full junction-tree scan history)
    do not materialise the full batched backward at once. For Bernoullis the
    chunking is a small constant-factor overhead and the chunk size is large
    enough that XLA still vectorises within a chunk.
    """
    chunk_size = 256

    def recon_one(x: Array) -> Array:
        return reconstruct(model, params, x)

    recons = jax.lax.map(recon_one, xs, batch_size=chunk_size)
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
        Tuple of (var_f, std_f, r_squared, rho_norm) --- ``rho_norm`` is the
        norm of the variationally-learned $\\rho$ (the stored ``BaseLatent``-
        shape correction).
    """
    var_f, std_f, r_squared = conjugation_metrics(model, key, params, n_samples)
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

    log_joints = jax.vmap(lambda z: model.log_density(params, jnp.concatenate([x, z])))(
        z_samples
    )
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
