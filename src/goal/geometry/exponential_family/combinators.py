"""Basic classes for composing exponential families."""

from __future__ import annotations

from abc import ABC
from typing import Protocol, override, runtime_checkable

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

### Protocols ###


@runtime_checkable
class StatisticalMoments(Protocol):
    """Protocol for distributions that can compute statistical moments.

    This is a temporary solution until Python supports proper intersection types that would allow us to express this as ExponentialFamily & StatisticalMoments.
    """

    def statistical_mean(self, params: Array) -> Array:
        """Compute the mean/expected value of the distribution as a 1D Array.

        Args:
            params: Natural parameters

        Returns:
            Mean array
        """
        ...

    def statistical_covariance(self, params: Array) -> Array:
        """Compute the covariance matrix of the distribution as a 2D Array.

        Args:
            params: Natural parameters

        Returns:
            Covariance matrix
        """
        ...


### Classes ###


class LocationShape[Location: ExponentialFamily, Shape: ExponentialFamily](
    Pair[Location, Shape], ExponentialFamily, ABC
):
    """A product exponential family with location and shape parameters.

    This structure captures distributions like the Normal distribution that decompose into a location parameter (e.g. mean) and a shape parameter (e.g. covariance).

    - The components must have matching data dimensions.
    - The sufficient statistic is the concatenation of component sufficient statistics.
    - The log-base measure by default is the log-base measure of the shape component.
    """

    # Overrides

    @property
    @override
    def data_dim(self) -> int:
        """Data dimension must match between location and shape manifolds."""
        assert self.fst_man.data_dim == self.snd_man.data_dim
        return self.fst_man.data_dim

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Sufficient statistic is the concatenation of component sufficient statistics.

        Args:
            x: Data point

        Returns:
            Concatenated sufficient statistics
        """
        loc_stats = self.fst_man.sufficient_statistic(x)
        shape_stats = self.snd_man.sufficient_statistic(x)
        return self.join_coords(loc_stats, shape_stats)

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize location and shape parameters.

        Args:
            key: JAX random key
            location: Location for initialization
            shape: Shape for initialization

        Returns:
            Initialized natural parameters
        """
        key_loc, key_shp = jax.random.split(key)
        fst_loc = self.fst_man.initialize(key_loc, location, shape)
        shp_loc = self.snd_man.initialize(key_shp, location, shape)
        return self.join_coords(fst_loc, shp_loc)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Base measure is assumed to come from the shape parameter (i.e. `snd_man`)."""
        return self.snd_man.log_base_measure(x)


class Product[M: ExponentialFamily](Replicated[M], ExponentialFamily):
    """Replicated manifold for exponential families, representing the product distribution over `n_reps` independent random variables."""

    # Overrides

    @property
    @override
    def data_dim(self) -> int:
        """Data dimension is the product of component data dimensions."""
        return self.rep_man.data_dim * self.n_reps

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Sufficient statistic is the concatenation of replicated component sufficient statistics.

        Args:
            x: Data array

        Returns:
            Sufficient statistics array
        """
        x_reshaped = x.reshape(self.n_reps, -1)
        stats = jax.vmap(self.rep_man.sufficient_statistic)(x_reshaped)
        return stats.reshape(-1)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Base measure is the sum of replicated component base measures.

        Args:
            x: Data array

        Returns:
            Log base measure (scalar)
        """
        x_reshaped = x.reshape(self.n_reps, -1)
        # vmap the base measure computation
        return jnp.sum(jax.vmap(self.rep_man.log_base_measure)(x_reshaped))

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize replicated parameters.

        Args:
            key: JAX random key
            location: Location parameter
            shape: Shape parameter

        Returns:
            Initialized natural parameters
        """
        keys = jax.random.split(key, self.n_reps)
        return jax.vmap(self.rep_man.initialize, in_axes=(0, None, None))(
            keys, location, shape
        ).reshape(-1)

    @override
    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize replicated parameters from sample.

        Args:
            key: JAX random key
            sample: Data sample
            location: Location parameter
            shape: Shape parameter

        Returns:
            Initialized natural parameters
        """
        keys = jax.random.split(key, self.n_reps)
        # sample dimensions: (n_batch, rep_dim)
        # datas dimensions: (n_reps * rep_dim, n_batch)
        datas = sample.T
        # rep_datas dimensions: (n_reps, rep_dim, n_batch)
        rep_datas = datas.reshape(self.n_reps, self.rep_man.data_dim, -1)

        def init_one(rep_key: Array, rep_data: Array) -> Array:
            # rep_sample dimensions: (n_batch, data_dim)
            rep_sample = rep_data.T
            return self.rep_man.initialize_from_sample(
                rep_key, rep_sample, location, shape
            )

        init_params = jax.vmap(init_one)(keys, rep_datas)
        return init_params.reshape(-1)

    def statistical_mean(self, params: Array) -> Array:
        """Compute the mean of the product distribution.

        Args:
            params: Natural parameters

        Returns:
            Vector of means for each component

        Raises:
            TypeError: If replicated manifold doesn't support statistical moments
        """
        if not isinstance(self.rep_man, StatisticalMoments):
            raise TypeError(
                f"Replicated manifold {type(self.rep_man)} does not support statistical moment computation"
            )

        # Map the mean computation across all replicates
        return self.map(self.rep_man.statistical_mean, params).ravel()

    def statistical_covariance(self, params: Array) -> Array:
        """Compute the covariance of the product distribution (block diagonal).

        Args:
            params: Natural parameters

        Returns:
            Block diagonal covariance matrix

        Raises:
            TypeError: If replicated manifold doesn't support statistical moments
        """
        if not isinstance(self.rep_man, StatisticalMoments):
            raise TypeError(
                f"Replicated manifold {type(self.rep_man)} does not support statistical moment computation"
            )

        # Get component covariances
        component_covs = self.map(self.rep_man.statistical_covariance, params)

        # Check if components return scalar variances
        if component_covs.size == self.n_reps:
            # For scalar variances, return diagonal matrix
            return jnp.diag(component_covs.ravel())
        # Build block diagonal matrix for matrix covariances
        rep_dim = self.rep_man.data_dim
        full_dim = self.data_dim
        block_diag = jnp.zeros((full_dim, full_dim))

        # Place component covariances along the diagonal
        for i in range(self.n_reps):
            start_idx = i * rep_dim
            end_idx = start_idx + rep_dim
            block_diag = block_diag.at[start_idx:end_idx, start_idx:end_idx].set(
                component_covs[i]
            )

        return block_diag


class GenerativeProduct[M: Differentiable](Product[M], Generative):
    """Replicated manifold for generative exponential families."""

    # Overrides

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Generate n samples from the product distribution.

        Args:
            key: JAX random key
            params: Natural parameters
            n: Number of samples

        Returns:
            Array of n samples
        """
        rep_keys = jax.random.split(key, self.n_reps)
        params_reshaped = params.reshape(self.n_reps, -1)

        def sample_rep(rep_key: Array, rep_params: Array) -> Array:
            return self.rep_man.sample(rep_key, rep_params, n)

        # samples dimensions: (n_reps, n_batch, data_dim)
        samples = jax.vmap(sample_rep)(rep_keys, params_reshaped)
        # return dimensions: (n_batch, n_reps * data_dim)
        return jnp.reshape(jnp.moveaxis(samples, 1, 0), (n, -1))


class DifferentiableProduct[M: Differentiable](Differentiable, GenerativeProduct[M]):
    """Replicated manifold for differentiable exponential families."""

    # Overrides

    @override
    def log_partition_function(self, params: Array) -> Array:
        """Compute sum of log partition functions.

        Args:
            params: Natural parameters

        Returns:
            Sum of log partition functions (scalar)
        """
        return jnp.sum(self.map(self.rep_man.log_partition_function, params))


class AnalyticProduct[M: Analytic](DifferentiableProduct[M], Analytic, ABC):
    """Replicated manifold for analytic exponential families."""

    # Overrides

    @override
    def negative_entropy(self, means: Array) -> Array:
        """Compute sum of negative entropies.

        Args:
            means: Mean parameters

        Returns:
            Sum of negative entropies (scalar)
        """
        return jnp.sum(self.map(self.rep_man.negative_entropy, means))
