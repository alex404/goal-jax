"""Basic classes for composing exponential families."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.manifold import (
    Coordinates,
    Pair,
    Point,
    Replicated,
)
from ..manifold.subspace import Subspace
from .exponential_family import (
    Analytic,
    Differentiable,
    ExponentialFamily,
    Generative,
    Mean,
    Natural,
)


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
    S: ExponentialFamily,
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


class Product[M: ExponentialFamily](Replicated[M], ExponentialFamily, ABC):
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


class GenerativeProduct[M: Generative](Product[M], Generative, ABC):
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


class DifferentiableProduct[M: Differentiable](
    Differentiable, GenerativeProduct[M], ABC
):
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
