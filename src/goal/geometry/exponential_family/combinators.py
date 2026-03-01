"""Combinators for composing exponential families: location-shape products and replicated (independent) products."""

from __future__ import annotations

from abc import ABC
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.combinators import Pair, Replicated
from .base import (
    Analytic,
    Differentiable,
    ExponentialFamily,
    Generative,
)
from .protocols import StatisticalMoments


class LocationShape[Location: ExponentialFamily, Shape: ExponentialFamily](
    Pair[Location, Shape], ExponentialFamily, ABC
):
    """A product of location and shape exponential families sharing the same data space.

    Used for distributions like the Normal that decompose into a location component (e.g. mean vector) and a shape component (e.g. covariance matrix). The sufficient statistic concatenates both components; the base measure comes from the shape component.
    """

    # Overrides

    @property
    @override
    def data_dim(self) -> int:
        return self.fst_man.data_dim

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        loc_stats = self.fst_man.sufficient_statistic(x)
        shape_stats = self.snd_man.sufficient_statistic(x)
        return self.join_coords(loc_stats, shape_stats)

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        key_loc, key_shp = jax.random.split(key)
        fst_loc = self.fst_man.initialize(key_loc, location, shape)
        shp_loc = self.snd_man.initialize(key_shp, location, shape)
        return self.join_coords(fst_loc, shp_loc)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return self.snd_man.log_base_measure(x)


class Product[M: ExponentialFamily](Replicated[M], ExponentialFamily):
    """Product of ``n_reps`` independent copies of the same exponential family.

    The sufficient statistic and base measure decompose across replicates. If the base family supports ``StatisticalMoments``, the product exposes composed mean and block-diagonal covariance.
    """

    # Overrides

    @property
    @override
    def data_dim(self) -> int:
        return self.rep_man.data_dim * self.n_reps

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        x_reshaped = x.reshape(self.n_reps, -1)
        stats = jax.vmap(self.rep_man.sufficient_statistic)(x_reshaped)
        return stats.reshape(-1)

    @override
    def log_base_measure(self, x: Array) -> Array:
        x_reshaped = x.reshape(self.n_reps, -1)
        return jnp.sum(jax.vmap(self.rep_man.log_base_measure)(x_reshaped))

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        keys = jax.random.split(key, self.n_reps)
        params_2d = jax.vmap(self.rep_man.initialize, in_axes=(0, None, None))(
            keys, location, shape
        )
        return self.to_1d(params_2d)

    @override
    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        keys = jax.random.split(key, self.n_reps)
        # Reshape sample (n_batch, n_reps * data_dim) -> per-replicate samples (n_reps, n_batch, data_dim)
        n_batch = sample.shape[0]
        rep_samples = sample.reshape(n_batch, self.n_reps, self.rep_man.data_dim)
        rep_samples = jnp.moveaxis(rep_samples, 1, 0)

        def init_one(rep_key: Array, rep_sample: Array) -> Array:
            return self.rep_man.initialize_from_sample(
                rep_key, rep_sample, location, shape
            )

        init_params = jax.vmap(init_one)(keys, rep_samples)
        return self.to_1d(init_params)

    # Methods

    def statistical_mean(self, params: Array) -> Array:
        """Compute the mean of the product distribution from its natural parameters.

        Requires the base family to satisfy ``StatisticalMoments``.
        """
        if not isinstance(self.rep_man, StatisticalMoments):
            raise TypeError(
                f"Replicated manifold {type(self.rep_man)} does not support statistical moment computation"
            )
        return self.map(self.rep_man.statistical_mean, params, flatten=True)

    def statistical_covariance(self, params: Array) -> Array:
        """Compute the block-diagonal covariance of the product distribution from its natural parameters.

        Requires the base family to satisfy ``StatisticalMoments``.
        """
        if not isinstance(self.rep_man, StatisticalMoments):
            raise TypeError(
                f"Replicated manifold {type(self.rep_man)} does not support statistical moment computation"
            )

        component_covs = self.map(self.rep_man.statistical_covariance, params)

        # Scalar variances: return diagonal matrix
        if component_covs.size == self.n_reps:
            return jnp.diag(component_covs.ravel())

        # Matrix covariances: build block diagonal
        return jax.scipy.linalg.block_diag(*component_covs)


class GenerativeProduct[M: Generative](Product[M], Generative):
    """Product of generative exponential families, adding independent sampling across replicates."""

    # Overrides

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Draw ``n`` samples from the product distribution given its natural parameters."""
        rep_keys = jax.random.split(key, self.n_reps)
        params_2d = self.to_2d(params)

        def sample_rep(rep_key: Array, rep_params: Array) -> Array:
            return self.rep_man.sample(rep_key, rep_params, n)

        # (n_reps, n, data_dim) -> (n, n_reps * data_dim)
        samples = jax.vmap(sample_rep)(rep_keys, params_2d)
        return jnp.reshape(jnp.moveaxis(samples, 1, 0), (n, -1))


class DifferentiableProduct[M: Differentiable](Differentiable, GenerativeProduct[M]):
    """Product of differentiable exponential families, with log-partition function summed across replicates."""

    # Overrides

    @override
    def log_partition_function(self, params: Array) -> Array:
        """Sum of component log-partition functions at the given natural parameters."""
        return jnp.sum(self.map(self.rep_man.log_partition_function, params))


class AnalyticProduct[M: Analytic](DifferentiableProduct[M], Analytic, ABC):
    """Product of analytic exponential families, with negative entropy summed across replicates."""

    # Overrides

    @override
    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize by delegating to each replicate's initialize_from_sample.

        This override ensures we use the Product version (which respects
        domain-specific initialization in each replicate) rather than
        the generic Analytic version.
        """
        return Product.initialize_from_sample(self, key, sample, location, shape)

    @override
    def negative_entropy(self, means: Array) -> Array:
        """Sum of component negative entropies at the given mean parameters."""
        return jnp.sum(self.map(self.rep_man.negative_entropy, means))
