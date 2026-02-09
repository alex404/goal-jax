"""Poisson-related models: population codes, COM-Poisson, and variational wrappers.

This module provides:
- VariationalPoissonVonMises: Variational wrapper for Poisson-VonMises harmonium
- VonMisesPopulationCode: Population code with Poisson observables and Von Mises stimulus
- CoMPoissonPopulation: Population of COM-Poisson units with shared dispersion
- Poisson mixture type aliases and factory functions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import (
    AnalyticProduct,
    DifferentiableProduct,
    EmbeddedMap,
    IdentityEmbedding,
    LocationShape,
    Product,
    Rectangular,
    TupleEmbedding,
)
from ...geometry.exponential_family.harmonium import (
    Harmonium,
    SymmetricConjugated,
)
from ...geometry.exponential_family.variational import (
    VariationalConjugatedSG,
)
from ..base.poisson import CoMPoisson, CoMShape, Poisson, Poissons
from ..base.von_mises import VonMises, VonMisesProduct
from .mixture import AnalyticMixture, Mixture

### Poisson-VonMises Harmonium ###


@dataclass(frozen=True)
class PoissonVonMisesHarmonium(Harmonium[Poissons, VonMisesProduct]):
    """Harmonium with Poisson observables and VonMises latents."""

    _int_man: EmbeddedMap[VonMisesProduct, Poissons]

    @property
    @override
    def int_man(self) -> EmbeddedMap[VonMisesProduct, Poissons]:
        return self._int_man

    @property
    def n_neurons(self) -> int:
        """Number of Poisson observable neurons."""
        return self.obs_man.n_reps

    @property
    def n_latent(self) -> int:
        """Number of VonMises latent dimensions."""
        return self.pst_man.n_reps

    def observable_rates(self, params: Array, z: Array) -> Array:
        """Compute expected firing rates given latent state z."""
        lkl_params = self.likelihood_at(params, z)
        return self.obs_man.to_mean(lkl_params)

    def get_weight_matrix(self, params: Array) -> Array:
        """Extract interaction matrix as 2D array (n_neurons, 2*n_latent)."""
        _, int_params, _ = self.split_coords(params)
        return int_params.reshape(self.n_neurons, 2 * self.n_latent)


def poisson_vonmises_harmonium(
    n_neurons: int, n_latent: int
) -> PoissonVonMisesHarmonium:
    """Create a Poisson-VonMises Harmonium."""
    obs = Poissons(n_neurons)
    lat = VonMisesProduct(n_latent)
    int_man = EmbeddedMap(Rectangular(), IdentityEmbedding(lat), IdentityEmbedding(obs))
    return PoissonVonMisesHarmonium(int_man)


### Variational Poisson-VonMises ###


@dataclass(frozen=True)
class VariationalPoissonVonMises(
    VariationalConjugatedSG[Poissons, VonMisesProduct, VonMisesProduct]
):
    """Variational conjugated model with Poisson observables and VonMises latents.

    Uses score-function gradient correction since VonMises sampling
    (rejection sampling) is not reparameterizable.

    Attributes:
        hrm: The underlying PoissonVonMisesHarmonium
    """

    hrm: PoissonVonMisesHarmonium

    @property
    @override
    def rho_emb(self) -> IdentityEmbedding[VonMisesProduct]:
        """Identity embedding: rho directly subtracts from posterior params."""
        return IdentityEmbedding(self.hrm.pst_man)

    @property
    def n_neurons(self) -> int:
        """Number of Poisson observable neurons."""
        return self.hrm.n_neurons

    @property
    def n_latent(self) -> int:
        """Number of VonMises latent dimensions."""
        return self.hrm.n_latent

    def get_weight_matrix(self, params: Array) -> Array:
        """Extract interaction matrix as 2D array (n_neurons, 2*n_latent)."""
        _, hrm_params = self.split_coords(params)
        return self.hrm.get_weight_matrix(hrm_params)

    def reconstruct(self, params: Array, x: Array) -> Array:
        """Reconstruct observable via mean-field approximation."""
        q_params = self.approximate_posterior_at(params, x)
        z_mean_stats = self.pst_man.to_mean(q_params)
        _, hrm_params = self.split_coords(params)
        lkl_fun = self.hrm.likelihood_function(hrm_params)
        lkl_natural = self.hrm.lkl_fun_man(lkl_fun, z_mean_stats)
        return self.obs_man.to_mean(lkl_natural)

    def reconstruction_error(self, params: Array, xs: Array) -> Array:
        """Compute mean squared reconstruction error over a batch."""
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        return jnp.mean((xs - recons) ** 2)

    def normalized_reconstruction_error(
        self, params: Array, xs: Array, scale: float = 1.0
    ) -> Array:
        """Compute MSE on normalized data."""
        recons = jax.vmap(self.reconstruct, in_axes=(None, 0))(params, xs)
        xs_norm = xs / scale
        recons_norm = recons / scale
        return jnp.mean((xs_norm - recons_norm) ** 2)


def variational_poisson_vonmises(
    n_neurons: int, n_latent: int
) -> VariationalPoissonVonMises:
    """Create a variational Poisson-VonMises model."""
    hrm = poisson_vonmises_harmonium(n_neurons, n_latent)
    return VariationalPoissonVonMises(hrm=hrm)


### Population Codes ###

type PoissonPopulation = AnalyticProduct[Poisson]


@dataclass(frozen=True)
class VonMisesPopulationCode(SymmetricConjugated[PoissonPopulation, VonMises]):
    """Population code with Poisson observables and Von Mises stimulus.

    Models neural responses where each neuron has a tuning curve over a
    circular stimulus (e.g., orientation).

    Attributes:
        n_neurons: Number of neurons in the population
        n_grid_points: Number of grid points for regression (default 100)
    """

    n_neurons: int
    n_grid_points: int = 100

    @property
    @override
    def lat_man(self) -> VonMises:
        return VonMises()

    @property
    @override
    def obs_man(self) -> PoissonPopulation:
        return AnalyticProduct(Poisson(), self.n_neurons)

    @property
    @override
    def int_man(self) -> EmbeddedMap[VonMises, PoissonPopulation]:
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(self.lat_man),
            IdentityEmbedding(self.obs_man),
        )

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute approximate conjugation parameters via regression."""
        obs_params, int_params = self.lkl_fun_man.split_coords(lkl_params)
        int_matrix = int_params.reshape(self.n_neurons, 2)
        grid = jnp.linspace(0, 2 * jnp.pi, self.n_grid_points, endpoint=False)

        def log_partition_at(z: Array) -> Array:
            suff = self.lat_man.sufficient_statistic(z)
            log_rates = obs_params + int_matrix @ suff
            return jnp.sum(jnp.exp(log_rates))

        log_partitions = jax.vmap(log_partition_at)(grid)
        suff_stats = jax.vmap(self.lat_man.sufficient_statistic)(grid)
        design = jnp.column_stack([jnp.ones(self.n_grid_points), suff_stats])
        coeffs, _, _, _ = jnp.linalg.lstsq(design, log_partitions, rcond=None)
        return coeffs[1:]

    def regression_diagnostics(self, params: Array) -> tuple[Array, Array, Array]:
        """Get regression fit diagnostics for visualization."""
        obs_params, int_params, _ = self.split_coords(params)
        int_matrix = int_params.reshape(self.n_neurons, 2)
        grid = jnp.linspace(0, 2 * jnp.pi, self.n_grid_points, endpoint=False)

        def log_partition_at(z: Array) -> Array:
            suff = self.lat_man.sufficient_statistic(z)
            log_rates = obs_params + int_matrix @ suff
            return jnp.sum(jnp.exp(log_rates))

        actual = jax.vmap(log_partition_at)(grid)
        suff_stats = jax.vmap(self.lat_man.sufficient_statistic)(grid)
        design = jnp.column_stack([jnp.ones(self.n_grid_points), suff_stats])
        coeffs, _, _, _ = jnp.linalg.lstsq(design, actual, rcond=None)
        fitted = design @ coeffs
        return grid, actual, fitted

    def join_tuning_parameters(
        self, gains: Array, preferred: Array, baselines: Array, prior_params: Array
    ) -> Array:
        """Construct population code from tuning curve specification."""
        obs_params = baselines
        int_col_1 = gains * jnp.cos(preferred)
        int_col_2 = gains * jnp.sin(preferred)
        int_params = jnp.stack([int_col_1, int_col_2], axis=1).ravel()
        lkl_params = self.lkl_fun_man.join_coords(obs_params, int_params)
        rho = self.conjugation_parameters(lkl_params)
        lat_params = prior_params - rho
        return self.join_coords(obs_params, int_params, lat_params)

    def split_tuning_parameters(
        self, params: Array
    ) -> tuple[Array, Array, Array, Array]:
        """Extract tuning curve parameters from natural parameters."""
        obs_params, int_params, lat_params = self.split_coords(params)
        baselines = obs_params
        int_matrix = int_params.reshape(self.n_neurons, 2)
        gains = jnp.sqrt(int_matrix[:, 0] ** 2 + int_matrix[:, 1] ** 2)
        preferred = jnp.arctan2(int_matrix[:, 1], int_matrix[:, 0])
        lkl_params = self.lkl_fun_man.join_coords(obs_params, int_params)
        rho = self.conjugation_parameters(lkl_params)
        prior_params = lat_params + rho
        return gains, preferred, baselines, prior_params

    def tuning_curve(self, params: Array, z: Array) -> Array:
        """Evaluate log firing rates at a stimulus."""
        obs_params, int_params, _ = self.split_coords(params)
        int_matrix = int_params.reshape(self.n_neurons, 2)
        suff = self.lat_man.sufficient_statistic(z)
        return obs_params + int_matrix @ suff

    def firing_rates(self, params: Array, z: Array) -> Array:
        """Evaluate firing rates at a stimulus."""
        log_rates = self.tuning_curve(params, z)
        return self.obs_man.to_mean(log_rates)

    def sample_stimulus(self, key: Array, params: Array) -> Array:
        """Sample stimulus z from prior."""
        prior_params = self.prior(params)
        return self.lat_man.sample(key, prior_params, n=1)[0]

    def sample_spikes(self, key: Array, params: Array, z: Array) -> Array:
        """Sample spike counts given a stimulus."""
        log_rates = self.tuning_curve(params, z)
        rates = self.obs_man.to_mean(log_rates)
        return jax.random.poisson(key, rates)

    def sample_joint(
        self, key: Array, params: Array, n: int = 1
    ) -> tuple[Array, Array]:
        """Sample (spike_counts, stimulus) pairs from the joint distribution."""
        key_z, key_x = jax.random.split(key)
        prior_params = self.prior(params)
        zs = self.lat_man.sample(key_z, prior_params, n=n).squeeze(-1)
        keys_x = jax.random.split(key_x, n)
        xs = jax.vmap(lambda k, z: self.sample_spikes(k, params, z))(keys_x, zs)
        return xs, zs

    def posterior_mean(self, params: Array, x: Array) -> Array:
        """Compute posterior mean stimulus given observations."""
        post_params = self.posterior_at(params, x)
        mu, _ = self.lat_man.split_mean_concentration(post_params)
        return mu

    def posterior_concentration(self, params: Array, x: Array) -> Array:
        """Compute posterior concentration given observations."""
        post_params = self.posterior_at(params, x)
        _, kappa = self.lat_man.split_mean_concentration(post_params)
        return kappa

    def log_observable_density(self, params: Array, x: Array) -> Array:
        """Compute log density of the observable distribution p(x)."""
        obs_params, _, _ = self.split_coords(params)
        chi = self.obs_man.log_partition_function(obs_params)
        obs_stats = self.obs_man.sufficient_statistic(x)
        prr = self.prior(params)
        log_density = jnp.dot(obs_params, obs_stats)
        log_density += self.lat_man.log_partition_function(self.posterior_at(params, x))
        log_density -= self.lat_man.log_partition_function(prr) + chi
        return log_density + self.obs_man.log_base_measure(x)

    def observable_density(self, params: Array, x: Array) -> Array:
        """Compute density of the observable distribution p(x)."""
        return jnp.exp(self.log_observable_density(params, x))


def von_mises_population_code(
    n_neurons: int,
    gains: Array,
    preferred: Array,
    baselines: Array,
    prior_mean: float = 0.0,
    prior_concentration: float = 0.0,
) -> tuple[VonMisesPopulationCode, Array]:
    """Create a Von Mises population code with specified tuning curves."""
    model = VonMisesPopulationCode(n_neurons)
    prior_params = model.lat_man.join_mean_concentration(
        prior_mean, prior_concentration
    )
    params = model.join_tuning_parameters(gains, preferred, baselines, prior_params)
    return model, params


### Poisson Mixture Models ###

type PopulationShape = Product[CoMShape]
type PoissonMixture = AnalyticMixture[PoissonPopulation]
type CoMPoissonMixture = Mixture[CoMPoissonPopulation]


@dataclass(frozen=True)
class CoMPoissonPopulation(
    DifferentiableProduct[CoMPoisson], LocationShape[PoissonPopulation, PopulationShape]
):
    """A population of independent Conway-Maxwell-Poisson (COM-Poisson) units."""

    def __init__(self, n_reps: int):
        super().__init__(CoMPoisson(), n_reps)

    @property
    @override
    def fst_man(self) -> PoissonPopulation:
        return AnalyticProduct(Poisson(), n_reps=self.n_reps)

    @property
    @override
    def snd_man(self) -> PopulationShape:
        return Product(CoMShape(), n_reps=self.n_reps)

    @override
    def join_coords(self, fst_coords: Array, snd_coords: Array) -> Array:
        params_matrix = jnp.stack([fst_coords, snd_coords])
        return params_matrix.T.reshape(-1)

    @override
    def split_coords(self, coords: Array) -> tuple[Array, Array]:
        matrix = coords.reshape(self.n_reps, 2).T
        return matrix[0], matrix[1]


@dataclass(frozen=True)
class PopulationLocationEmbedding(
    TupleEmbedding[PoissonPopulation, CoMPoissonPopulation]
):
    """Embedding that projects COM-Poisson to Poisson via location parameters."""

    n_neurons: int

    @property
    @override
    def tup_idx(self) -> int:
        return 0

    @property
    @override
    def amb_man(self) -> CoMPoissonPopulation:
        return CoMPoissonPopulation(n_reps=self.n_neurons)

    @property
    @override
    def sub_man(self) -> PoissonPopulation:
        return AnalyticProduct(Poisson(), n_reps=self.n_neurons)


def poisson_mixture(n_neurons: int, n_components: int) -> PoissonMixture:
    """Create a mixture of independent Poisson populations."""
    pop_man = AnalyticProduct(Poisson(), n_reps=n_neurons)
    return AnalyticMixture(pop_man, n_components)


def com_poisson_mixture(n_neurons: int, n_components: int) -> CoMPoissonMixture:
    """Create a COM-Poisson mixture with shared dispersion parameters."""
    obs_emb = PopulationLocationEmbedding(n_neurons)
    return Mixture(n_components, obs_emb)
