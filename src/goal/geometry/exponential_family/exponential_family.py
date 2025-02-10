"""Core definitions for exponential families and their parameterizations.

An exponential family is a collection of probability distributions with densities of the form:

$$p(x; \\theta) = \\mu(x)\\exp(\\theta \\cdot \\mathbf{s}(x) - \\psi(\\theta)),$$

where:

- $\\theta$ are the natural parameters,
- $\\mathbf{s}(x)$ is the sufficient statistic,
- $\\mu(x)$ is the base measure, and
- $\\psi(\\theta)$ is the log partition function.

Exponential families have a rich geometric structure that can be expressed through their dual parameterizations:

1. Natural Parameters ($\\theta$):

    - Define the density directly
    - Connected to maximum likelihood estimation

2. Mean Parameters ($\\eta$):

    - Given by expectations: $\\eta = \\mathbb{E}[\\mathbf{s}(x)]$
    - Connected to moment matching and variational inference

Mappings between the two parameterizations are given by

- $\\eta = \\nabla\\psi(\\theta)$, and
- $\\theta = \\nabla\\phi(\\eta)$,

where $\\phi(\\eta)$ is the negative entropy, which is the convex conjugate of $\\psi(\\theta)$.

This module implements this structure through a hierarchy of classes:

- `ExponentialFamily`: Base class defining sufficient statistics and base measure
- `Differentiable`: Exponential families with analytical log partition function, and support mapping to the mean parameters by autodifferentation.
- `Analytic`: Exponential families with analytical negative entropy, and support mapping to the natural parameters by autodifferentation.

#### Class Hierarchy

![Class Hierarchy](exponential_family.svg)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.manifold import (
    Coordinates,
    Dual,
    Manifold,
    Pair,
    Point,
    Replicated,
    reduce_dual,
)

### Coordinate Systems ###


# Coordinate systems for exponential families
class Natural(Coordinates):
    """Natural parameters $\\theta \\in \\Theta$ defining an exponential family through:

    $$p(x; \\theta) = \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x) - \\psi(\\theta))$$
    """

    ...


type Mean = Dual[Natural]
"""Mean parameters $\\eta \\in \\text{H}$ given by expectations of sufficient statistics:

$$\\eta = \\mathbb{E}_{p(x;\\theta)}[\\mathbf s(x)]$$
"""


### Exponential Families ###


class ExponentialFamily(Manifold, ABC):
    """Base manifold class for exponential families.

    An exponential family is a manifold of probability distributions with densities
    of the form:

    $$
    p(x; \\theta) = \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x) - \\psi(\\theta))
    $$

    where:

    * $\\theta \\in \\Theta$ are the natural parameters,
    * $\\mathbf s(x)$ is the sufficient statistic,
    * $\\mu(x)$ is the base measure, and
    * $\\psi(\\theta)$ is the log partition function.

    The base `ExponentialFamily` class includes methods that fundamentally define an exponential family, namely the base measure and sufficient statistic.
    """

    # Contract

    @property
    @abstractmethod
    def data_dim(self) -> int:
        """Dimension of the data space."""

    @abstractmethod
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        """Convert observation to sufficient statistics.

        Maps a point $x$ to its sufficient statistic $\\mathbf s(x)$ in mean coordinates:

        $$\\mathbf s: \\mathcal{X} \\mapsto \\text{H}$$

        Args:
            x: Array containing a single observation

        Returns:
            Mean coordinates of the sufficient statistics
        """

    @abstractmethod
    def log_base_measure(self, x: Array) -> Array:
        """Compute log of base measure $\\mu(x)$."""
        ...

    # Templates

    def average_sufficient_statistic(self, xs: Array) -> Point[Mean, Self]:
        """Average sufficient statistics of a batch of observations.

        Args:
            xs: Array of shape (batch_size, data_dim) containing : batch of observations

        Returns:
            Mean coordinates of average sufficient statistics
        """

        def _sufficient_statistic(x: Array) -> Array:
            return self.sufficient_statistic(x).array

        return self.mean_point(self.batched_point_mean(_sufficient_statistic, xs))
        # stats = jax.lax.map(self.sufficient_statistic, xs, batch_size=128)
        # return self.mean_point(jnp.mean(stats.array, axis=0))
        # suff_stats = jax.vmap(self.sufficient_statistic)(xs)
        # return self.mean_point(jnp.mean(suff_stats.array, axis=0))

        # first_stat = self.sufficient_statistic(xs[0])
        #
        # def step(carry: Array, x: Array) -> tuple[Array, None]:
        #     suff_stat = self.sufficient_statistic(x)
        #     return carry + suff_stat.array, None
        #
        # total, _ = jax.lax.scan(step, jnp.zeros_like(first_stat.array), xs)
        # return self.mean_point(total / xs.shape[0])

    def mean_point(self, params: Array) -> Point[Mean, Self]:
        """Construct a point in mean coordinates."""
        return self.point(params)

    def check_natural_parameters(self, params: Point[Natural, Self]) -> Array:
        """Check if parameters are valid for this exponential family."""
        return jnp.all(jnp.isfinite(params.array)).astype(jnp.int32)

    def natural_point(self, params: Array) -> Point[Natural, Self]:
        """Construct a point in natural coordinates."""
        return self.point(params)

    def initialize(
        self,
        key: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Point[Natural, Self]:
        """Convenience function to randomly initialize the coordinates of a point based on a mean and a shape parameter --- by default this is a normal distribution, but may be overridden e.g. for bounded parameter spaces."""
        params = jax.random.normal(key, shape=(self.dim,)) * shape + location
        return self.natural_point(params)

    def initialize_from_sample(
        self,
        key: Array,
        sample: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Point[Natural, Self]:
        """Convenience function to initialize a model based on the sample. By default it ignores the sample, but most distributions can do better than this."""
        sample = sample
        return self.initialize(key, location, shape)


class Generative(ExponentialFamily, ABC):
    """An `ExponentialFamily` that supports random sampling.

    Sampling is performed on distributions in natural coordinates. This enables estimation of mean parameters even when closed-form expressions for the log partition function are unavailable.
    """

    # Contract

    @abstractmethod
    def sample(self, key: Array, params: Point[Natural, Self], n: int = 1) -> Array:
        """Generate random samples from the distribution.

        Args:
            params: Parameters in natural coordinates
            n: Number of samples to generate (default=1)

        Returns:
            Array of shape (n, *data_dims) containing samples
        """
        ...

    # Templates

    def stochastic_to_mean(
        self, key: Array, params: Point[Natural, Self], n: int
    ) -> Point[Mean, Self]:
        """Estimate mean parameters via Monte Carlo sampling."""
        samples = self.sample(key, params, n)
        return self.average_sufficient_statistic(samples)


class Differentiable(Generative, ABC):
    """Exponential family with an analytically tractable log-partition function, which permits computing the expecting value of the sufficient statistic, and data-fitting via gradient descent.

    The log partition function $\\psi(\\theta)$ is given by:

    $$
    \\psi(\\theta) = \\log \\int_{\\mathcal{X}} \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x))dx
    $$

    Its gradient at a point in natural coordinates returns that point in mean coordinates:

    $$
    \\eta = \\nabla \\psi(\\theta) = \\mathbb{E}_{p(x;\\theta)}[\\mathbf s(x)]
    $$
    """

    # Contract

    @abstractmethod
    def log_partition_function(self, params: Point[Natural, Self]) -> Array:
        """Compute log partition function $\\psi(\\theta)$."""

    # Templates

    def to_mean(self, params: Point[Natural, Self]) -> Point[Mean, Self]:
        """Convert from natural to mean parameters via $\\eta = \\nabla \\psi(\\theta)$."""
        return self.grad(self.log_partition_function, params)

    def log_density(self, params: Point[Natural, Self], x: Array) -> Array:
        """Compute log density at x.

        $$
        \\log p(x;\\theta) = \\theta \\cdot \\mathbf s(x) + \\log \\mu(x) - \\psi(\\theta)
        $$
        """
        suff_stats = self.sufficient_statistic(x)
        return (
            self.dot(params, suff_stats)
            + self.log_base_measure(x)
            - self.log_partition_function(params)
        )

    def density(self, params: Point[Natural, Self], x: Array) -> Array:
        """Compute density at x.

        $$
        p(x;\\theta) = \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x) - \\psi(\\theta))
        $$
        """
        return jnp.exp(self.log_density(params, x))

    def average_log_density(self, p: Point[Natural, Self], xs: Array) -> Array:
        """Compute average log density over a batch of observations.

        Args:
            p: Parameters in natural coordinates
            xs: Array of shape (batch_size, data_dim) containing batch of observations

        Returns:
            Array of shape (batch_size,) containing average log densities
        """
        log_densities = jax.lax.map(
            lambda x: self.log_density(p, x), xs, batch_size=256
        )
        return jnp.mean(log_densities)

        # return jnp.mean(jax.vmap(self.log_density, in_axes=(None, 0))(p, xs))

        # def step(carry: Array, x: Array) -> tuple[Array, None]:
        #     return carry + self.log_density(p, x), None  # Accumulate sum
        #
        # total_log_likelihood, _ = jax.lax.scan(step, jnp.asarray(0.0), xs)
        # return total_log_likelihood / xs.shape[0]


class Analytic(Differentiable, ABC):
    """An exponential family comprising distributions for which the entropy can be evaluated in closed-form. The negative entropy is the convex conjugate of the log-partition function

    $$
    \\phi(\\eta) = \\sup_{\\theta} \\{\\theta \\cdot \\eta - \\psi(\\theta)\\},
    $$

    and its gradient at a point in mean coordinates returns that point in natural coordinates $\\theta = \\nabla\\phi(\\eta).$

     **NB:** This form of the negative entropy is the convex conjugate of the log-partition function, and does not factor in the base measure of the distribution. It may thus differ from the entropy as traditionally defined.
    """

    # Contract

    @abstractmethod
    def negative_entropy(self, means: Point[Mean, Self]) -> Array:
        """Compute negative entropy $\\phi(\\eta)$."""

    # Overrides

    @override
    def initialize_from_sample(
        self,
        key: Array,
        sample: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Point[Natural, Self]:
        """Initialize a model based on the noisy average sufficient statistics."""
        avg_suff_stat = self.average_sufficient_statistic(sample)
        # add gaussian noise
        noise = jax.random.normal(key, shape=(self.dim,)) * shape + location
        avg_suff_stat = avg_suff_stat + self.point(noise)
        params = self.to_natural(avg_suff_stat)
        return self.natural_point(params.array)

    # Templates

    def to_natural(self, means: Point[Mean, Self]) -> Point[Natural, Self]:
        """Convert mean to natural parameters via $\\theta = \\nabla\\phi(\\eta)$."""
        return reduce_dual(self.grad(self.negative_entropy, means))

    def relative_entropy(
        self, p_means: Point[Mean, Self], q_params: Point[Natural, Self]
    ) -> Array:
        """Compute the entropy of $p$ relative to $q$.

        $D(p \\| q) = \\int p(x) \\log \\frac{p(x)}{q(x)} dx = \\theta \\cdot \\eta - \\psi(\\theta) - \\phi(\\eta)$, where

        - $p(x;\\theta)$ has natural parameters $\\theta$,
        - $q(x;\\eta)$ has mean parameters $\\eta$.
        """
        return (
            self.negative_entropy(p_means)
            + self.log_partition_function(q_params)
            - self.dot(q_params, p_means)
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


class GenerativeReplicated[M: Generative](Replicated[M], Generative, ABC):
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


class DifferentiableReplicated[M: Differentiable](
    Differentiable, GenerativeReplicated[M], ABC
):
    """Replicated manifold for differentiable exponential families."""

    # Overrides

    @override
    def log_partition_function(self, params: Point[Natural, Self]) -> Array:
        # Reshape instead of split
        return jnp.sum(self.map(self.rep_man.log_partition_function, params))


class AnalyticReplicated[M: Analytic](DifferentiableReplicated[M], Analytic, ABC):
    """Replicated manifold for analytic exponential families."""

    # Overrides

    @override
    def negative_entropy(self, means: Point[Mean, Self]) -> Array:
        # Reshape instead of split
        return jnp.sum(self.map(self.rep_man.negative_entropy, means))
