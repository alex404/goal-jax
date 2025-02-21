"""Basic classes for composing exponential families."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Protocol, Self, override, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import (
    Coordinates,
    Point,
)
from ..manifold.combinators import (
    Pair,
    Replicated,
)
from ..manifold.subspace import Subspace
from .base import (
    Analytic,
    Differentiable,
    ExponentialFamily,
    Generative,
    Mean,
    Natural,
)

### Protocols ###


@runtime_checkable
class StatisticalMoments(Protocol):
    """Protocol for distributions that can compute statistical moments.

    This is a temporary solution until Python supports proper intersection types that would allow us to express this as ExponentialFamily & StatisticalMoments.
    """

    def statistical_mean[M: ExponentialFamily](
        self: M, params: Point[Natural, M]
    ) -> Array:
        """Compute the mean/expected value of the distribution as a 1D Array."""
        ...

    def statistical_covariance[M: ExponentialFamily](
        self: M, params: Point[Natural, M]
    ) -> Array:
        """Compute the covariance matrix of the distribution as a 2D Array."""
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
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        """Sufficient statistic is the concatenation of component sufficient statistics."""
        loc_stats = self.fst_man.sufficient_statistic(x)
        shape_stats = self.snd_man.sufficient_statistic(x)
        return self.join_params(loc_stats, shape_stats)

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Point[Natural, Self]:
        """Initialize location and shape parameters."""
        key_loc, key_shp = jax.random.split(key)
        fst_loc = self.fst_man.initialize(key_loc, location, shape)
        shp_loc = self.snd_man.initialize(key_shp, location, shape)
        return self.join_params(fst_loc, shp_loc)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Base measure is sum of component base measures."""
        return self.snd_man.log_base_measure(x)


@dataclass(frozen=True)
class LocationSubspace[
    LS: Any,
    L: ExponentialFamily,
](
    Subspace[LS, L],  # Note: we specify LS directly in Subspace
    ABC,
):
    """Subspace relationship for a product manifold $M \\times N$.

    Type Parameters:

    - `LS`: The combined manifold type. Should be a subclass of `LocationShape[L, S]`.
    - `L`: The location manifold type.
    - `S`: The shape manifold type.

    """

    @override
    def project[C: Coordinates](self, p: Point[C, LS]) -> Point[C, L]:
        first, _ = self.sup_man.split_params(p)
        return first

    @override
    def translate[C: Coordinates](
        self, p: Point[C, LS], q: Point[C, L]
    ) -> Point[C, LS]:
        first, second = self.sup_man.split_params(p)
        return self.sup_man.join_params(first + q, second)


class Product[M: ExponentialFamily](Replicated[M], ExponentialFamily):
    """Replicated manifold for exponential families, representing the product distribution over `n_reps` independent random variables."""

    # Overrides

    @property
    @override
    def data_dim(self) -> int:
        """Data dimension is the product of component data dimensions."""
        return self.rep_man.data_dim * self.n_reps

    @override
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        """Sufficient statistic is the concatenation of replicated component sufficient statistics."""
        x_reshaped = x.reshape(self.n_reps, -1)
        stats = jax.vmap(self.rep_man.sufficient_statistic)(x_reshaped)
        return self.mean_point(stats.array.reshape(-1))

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Base measure is the sum of replicated component base measures."""
        x_reshaped = x.reshape(self.n_reps, -1)
        # vmap the base measure computation
        return jnp.sum(jax.vmap(self.rep_man.log_base_measure)(x_reshaped))

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Point[Natural, Self]:
        """Initialize replicated parameters.

        Generates n_reps independent initializations of the base manifold.
        """
        keys = jax.random.split(key, self.n_reps)

        def init_one(k: Array) -> Array:
            return self.rep_man.initialize(k, location, shape).array

        init_params = jax.vmap(init_one)(keys)
        return self.natural_point(init_params)

    @override
    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Point[Natural, Self]:
        """Initialize replicated parameters from sample.

        Splits sample data into n_reps chunks and initializes each component
        using its corresponding chunk of data.
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
            ).array

        init_params = jax.vmap(init_one)(keys, rep_datas)
        return self.natural_point(init_params)

    def statistical_mean(self, params: Point[Natural, Self]) -> Array:
        """Compute mean for product distribution.

        If the replicated manifold supports statistical moments, returns a vector of means for each component. Otherwise raises TypeError.
        """
        if not isinstance(self.rep_man, StatisticalMoments):
            raise TypeError(
                f"Replicated manifold {type(self.rep_man)} does not support statistical moment computation"
            )

        # Map the mean computation across all replicates

        return self.map(self.rep_man.statistical_mean, params).ravel()  # pyright: ignore[reportArgumentType]

    def statistical_covariance(self, params: Point[Natural, Self]) -> Array:
        """Compute covariance for product distribution."""
        if not isinstance(self.rep_man, StatisticalMoments):
            raise TypeError(
                f"Replicated manifold {type(self.rep_man)} does not support statistical moment computation"
            )

        # Get component covariances
        component_covs = self.map(self.rep_man.statistical_covariance, params)  # pyright: ignore[reportArgumentType]

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


class GenerativeProduct[M: Generative](Product[M], Generative):
    """Replicated manifold for generative exponential families."""

    # Overrides

    @override
    def sample(self, key: Array, params: Point[Natural, Self], n: int = 1) -> Array:
        """Generate n samples from the product distribution."""
        rep_keys = jax.random.split(key, self.n_reps)

        def sample_rep(rep_key: Array, rep_params: Array) -> Array:
            with self.rep_man as rm:
                return rm.sample(rep_key, rm.natural_point(rep_params), n)

        # samples dimensions: (n_reps, n_batch, data_dim)
        samples = jax.vmap(sample_rep)(rep_keys, params.array)
        # return dimensions: (n_batch, n_reps * data_dim)
        return jnp.reshape(jnp.moveaxis(samples, 1, 0), (n, -1))


class DifferentiableProduct[M: Differentiable](Differentiable, GenerativeProduct[M]):
    """Replicated manifold for differentiable exponential families."""

    # Overrides

    @override
    def log_partition_function(self, params: Point[Natural, Self]) -> Array:
        # Reshape instead of split
        return jnp.sum(self.map(self.rep_man.log_partition_function, params))


class AnalyticProduct[M: Analytic](DifferentiableProduct[M], Analytic, ABC):
    """Replicated manifold for analytic exponential families."""

    # Overrides

    @override
    def negative_entropy(self, means: Point[Mean, Self]) -> Array:
        # Reshape instead of split
        return jnp.sum(self.map(self.rep_man.negative_entropy, means))
