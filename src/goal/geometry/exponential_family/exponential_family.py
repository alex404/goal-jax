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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.manifold import Coordinates, Dual, Manifold, Pair, Point, reduce_dual
from ..manifold.subspace import Subspace

### Coordinate Systems ###


# Coordinate systems for exponential families
class Mean(Coordinates):
    """Mean parameters $\\eta \\in \\text{H}$ given by expectations of sufficient statistics:

    $$\\eta = \\mathbb{E}_{p(x;\\theta)}[\\mathbf s(x)]$$
    """

    ...


type Natural = Dual[Mean]
"""Natural parameters $\\theta \\in \\Theta$ defining an exponential family through:

$$p(x; \\theta) = \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x) - \\psi(\\theta))$$
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

    @property
    @abstractmethod
    def data_dim(self) -> int:
        """Dimension of the data space."""

    @abstractmethod
    def _compute_sufficient_statistic(self, x: Array) -> Array: ...

    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        """Convert observation to sufficient statistics.

        Maps a point $x$ to its sufficient statistic $\\mathbf s(x)$ in mean coordinates:

        $$\\mathbf s: \\mathcal{X} \\mapsto \\text{H}$$

        Args:
            x: Array containing a single observation

        Returns:
            Mean coordinates of the sufficient statistics
        """
        return Point(self._compute_sufficient_statistic(x))

    def _average_sufficient_statistic(self, xs: Array) -> Array:
        return jnp.mean(jax.vmap(self._compute_sufficient_statistic)(xs), axis=0)

    def average_sufficient_statistic(self, xs: Array) -> Point[Mean, Self]:
        """Average sufficient statistics of a batch of observations.

        Args:
            xs: Array of shape (batch_size, data_dim) containing : batch of observations

        Returns:
            Mean coordinates of average sufficient statistics
        """
        return Point(self._average_sufficient_statistic(xs))

    @abstractmethod
    def log_base_measure(self, x: Array) -> Array:
        """Compute log of base measure $\\mu(x)$."""
        ...

    def mean_point(self, params: Array) -> Point[Mean, Self]:
        """Construct a point in mean coordinates."""
        return Point[Mean, Self](jnp.atleast_1d(params))

    def check_natural_parameters(self, params: Point[Natural, Self]) -> Array:
        """Check if parameters are valid for this exponential family."""
        return jnp.all(jnp.isfinite(params.params)).astype(jnp.int32)

    # def natural_point(self, params: Array) -> Point[Natural, Self]:
    #     """Construct a point in natural coordinates."""
    #     p = Point[Natural, Self](jnp.atleast_1d(params))
    #     if __debug__ and not bool(self.check_natural_parameters(p)):
    #         raise ValueError(f"Invalid natural parameters for {type(self)}")
    #     return p

    def natural_point(self, params: Array) -> Point[Natural, Self]:
        """Construct a point in natural coordinates."""
        return Point[Natural, Self](jnp.atleast_1d(params))

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

    @abstractmethod
    def sample(self, key: Array, p: Point[Natural, Self], n: int = 1) -> Array:
        """Generate random samples from the distribution.

        Args:
            p: Parameters in natural coordinates
            n: Number of samples to generate (default=1)

        Returns:
            Array of shape (n, *data_dims) containing samples
        """
        ...

    def stochastic_to_mean(
        self, key: Array, p: Point[Natural, Self], n: int
    ) -> Point[Mean, Self]:
        """Estimate mean parameters via Monte Carlo sampling."""
        samples = self.sample(key, p, n)
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

    @abstractmethod
    def _compute_log_partition_function(self, natural_params: Array) -> Array:
        """Internal method to compute $\\psi(\\theta)$."""
        ...

    def log_partition_function(self, p: Point[Natural, Self]) -> Array:
        """Compute log partition function $\\psi(\\theta)$."""
        return self._compute_log_partition_function(p.params)

    def to_mean(self, p: Point[Natural, Self]) -> Point[Mean, Self]:
        """Convert from natural to mean parameters via $\\eta = \\nabla \\psi(\\theta)$."""
        return reduce_dual(self.grad(self.log_partition_function, p))

    def _log_density(self, params: Array, x: Array) -> Array:
        suff_stats = self._compute_sufficient_statistic(x)
        return (
            jnp.dot(params, suff_stats)
            + self.log_base_measure(x)
            - self._compute_log_partition_function(params)
        )

    def log_density(self, p: Point[Natural, Self], x: Array) -> Array:
        """Compute log density at x.

        $$
        \\log p(x;\\theta) = \\theta \\cdot \\mathbf s(x) + \\log \\mu(x) - \\psi(\\theta)
        $$
        """
        return self._log_density(p.params, x)

    def average_log_density(self, p: Point[Natural, Self], xs: Array) -> Array:
        """Compute average log density over a batch of observations.

        Args:
            p: Parameters in natural coordinates
            xs: Array of shape (batch_size, data_dim) containing batch of observations

        Returns:
            Array of shape (batch_size,) containing average log densities
        """
        return jnp.mean(jax.vmap(self._log_density, in_axes=(None, 0))(p.params, xs))

    def density(self, p: Point[Natural, Self], x: Array) -> Array:
        """Compute density at x.

        $$
        p(x;\\theta) = \\mu(x)\\exp(\\theta \\cdot \\mathbf s(x) - \\psi(\\theta))
        $$
        """
        return jnp.exp(self.log_density(p, x))


class Analytic(Differentiable, ABC):
    """An exponential family comprising distributions for which the entropy can be evaluated in closed-form. The negative entropy is the convex conjugate of the log-partition function

    $$
    \\phi(\\eta) = \\sup_{\\theta} \\{\\theta \\cdot \\eta - \\psi(\\theta)\\},
    $$

    and its gradient at a point in mean coordinates returns that point in natural coordinates $\\theta = \\nabla\\phi(\\eta).$

     **NB:** This form of the negative entropy is the convex conjugate of the log-partition function, and does not factor in the base measure of the distribution. It may thus differ from the entropy as traditionally defined.
    """

    @abstractmethod
    def _compute_negative_entropy(self, mean_params: Array) -> Array:
        """Internal method to compute $\\phi(\\eta)$."""
        ...

    def negative_entropy(self, p: Point[Mean, Self]) -> Array:
        """Compute negative entropy $\\phi(\\eta)$."""
        return self._compute_negative_entropy(p.params)

    def to_natural(self, p: Point[Mean, Self]) -> Point[Natural, Self]:
        """Convert mean to natural parameters via $\\theta = \\nabla\\phi(\\eta)$."""
        return self.grad(self.negative_entropy, p)

    def relative_entropy(self, p: Point[Mean, Self], q: Point[Natural, Self]) -> Array:
        """Compute the entropy of $p$ relative to $q$.

        $D(p \\| q) = \\int p(x) \\log \\frac{p(x)}{q(x)} dx = \\theta \\cdot \\eta - \\psi(\\theta) - \\phi(\\eta)$, where

        - $p(x;\\theta)$ has natural parameters $\\theta$,
        - $q(x;\\eta)$ has mean parameters $\\eta$.
        """
        return (
            self.negative_entropy(p) + self.log_partition_function(q) - self.dot(p, q)
        )

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
        avg_suff_stat = avg_suff_stat + Point(noise)
        params = self.to_natural(avg_suff_stat)
        return self.natural_point(params.params)


class LocationShape[Location: ExponentialFamily, Shape: ExponentialFamily](
    Pair[Location, Shape], ExponentialFamily
):
    """A product exponential family with location and shape parameters.

    This structure captures distributions like the Normal distribution that decompose into a location parameter (e.g. mean) and a shape parameter (e.g. covariance).

    - The components must have matching data dimensions.
    - The sufficient statistic is the concatenation of component sufficient statistics.
    - The log-base measure by default is the log-base measure of the shape component.
    """

    @property
    def data_dim(self) -> int:
        """Data dimension must match between location and shape manifolds."""
        assert self.fst_man.data_dim == self.snd_man.data_dim
        return self.fst_man.data_dim

    def _compute_sufficient_statistic(self, x: Array) -> Array:
        """Concatenate sufficient statistics from location and shape components."""
        loc_stats = self.fst_man._compute_sufficient_statistic(x)
        shape_stats = self.snd_man._compute_sufficient_statistic(x)
        return self.join_params(Point(loc_stats), Point(shape_stats)).params

    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Point[Natural, Self]:
        """Initialize location and shape parameters."""
        key_loc, key_shp = jax.random.split(key)
        fst_loc = self.fst_man.initialize(key_loc, location, shape)
        shp_loc = self.snd_man.initialize(key_shp, location, shape)
        return self.join_params(fst_loc, shp_loc)

    def log_base_measure(self, x: Array) -> Array:
        """Base measure is sum of component base measures."""
        return self.snd_man.log_base_measure(x)


M = TypeVar("M", bound=ExponentialFamily, covariant=True)


@dataclass(frozen=True)
class Replicated(Generic[M], ExponentialFamily):
    man: M
    n_repeats: int

    @property
    def dim(self) -> int:
        return self.man.dim * self.n_repeats

    @property
    def data_dim(self) -> int:
        return self.man.data_dim * self.n_repeats

    def _compute_sufficient_statistic(self, x: Array) -> Array:
        # Use reshape instead of array_split for better performance
        x_reshaped = x.reshape(self.n_repeats, -1)
        # vmap the sufficient statistic computation
        stats = jax.vmap(self.man._compute_sufficient_statistic)(x_reshaped)
        return stats.reshape(-1)  # Flatten output

    def log_base_measure(self, x: Array) -> Array:
        x_reshaped = x.reshape(self.n_repeats, -1)
        # vmap the base measure computation
        return jnp.sum(jax.vmap(self.man.log_base_measure)(x_reshaped))

    def split_params[C: Coordinates](self, p: Point[C, Self]) -> list[Point[C, M]]:
        # Use reshape instead of array_split
        params_matrix = p.params.reshape(self.n_repeats, -1)
        return [Point(row) for row in params_matrix]

    def join_params[C: Coordinates](self, ps: list[Point[C, M]]) -> Point[C, Self]:
        if len(ps) != self.n_repeats:
            raise ValueError(f"Expected {self.n_repeats} points, got {len(ps)}")
        # Stack instead of concatenate
        return Point(jnp.stack([p.params for p in ps]).reshape(-1))

    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Point[Natural, Self]:
        """Initialize replicated parameters.

        Generates n_repeats independent initializations of the base manifold.

        Args:
            key: Random key
            location: Location parameter for initialization
            shape: Shape parameter for random perturbations
        """
        keys = jax.random.split(key, self.n_repeats)
        params = [self.man.initialize(k, location, shape) for k in keys]
        return self.join_params(params)

    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Point[Natural, Self]:
        """Initialize replicated parameters from sample.

        Splits sample data into n_repeats chunks and initializes each component
        using its corresponding chunk of data.

        Args:
            key: Random key
            sample: Sample array of shape (batch_size, data_dim)
            location: Location parameter for initialization
            shape: Shape parameter for random perturbations
        """
        keys = jax.random.split(key, self.n_repeats)
        # Split samples into chunks for each replicate
        samples = sample.reshape(-1, self.n_repeats, self.man.data_dim)
        samples = jnp.moveaxis(
            samples, 1, 0
        )  # Shape: (n_repeats, batch_size, data_dim)

        params = [
            self.man.initialize_from_sample(k, s, location, shape)
            for k, s in zip(keys, samples)
        ]
        return self.join_params(params)


class GenerativeReplicated[M: Generative](Replicated[M], Generative):
    """Replicated manifold for generative exponential families."""

    def sample(self, key: Array, p: Point[Natural, Self], n: int = 1) -> Array:
        params_matrix = p.params.reshape(self.n_repeats, -1)

        def sample_replicate(key: Array, params: Array) -> Array:
            return self.man.sample(key, Point(params), 1)

        def sample_once(key: Array) -> Array:
            rep_keys = jax.random.split(key, self.n_repeats)
            samples = jax.vmap(sample_replicate, in_axes=(0, 0))(
                rep_keys, params_matrix
            )
            return samples.reshape(-1)

        sam_keys = jax.random.split(key, n)
        return jax.vmap(sample_once, in_axes=0)(sam_keys)


class DifferentiableReplicated[M: Differentiable](
    GenerativeReplicated[M], Differentiable
):
    def _compute_log_partition_function(self, natural_params: Array) -> Array:
        # Reshape instead of split
        params_matrix = natural_params.reshape(self.n_repeats, -1)
        # vmap the partition function computation
        return jnp.sum(
            jax.vmap(self.man._compute_log_partition_function)(params_matrix)
        )


class AnalyticReplicated[M: Analytic](DifferentiableReplicated[M], Analytic):
    def _compute_negative_entropy(self, mean_params: Array) -> Array:
        params_matrix = mean_params.reshape(self.n_repeats, -1)
        # vmap the entropy computation
        return jnp.sum(jax.vmap(self.man._compute_negative_entropy)(params_matrix))


class LocationSubspace[Location: ExponentialFamily, Shape: ExponentialFamily](
    Subspace[LocationShape[Location, Shape], Location]
):
    """Subspace relationship for a product manifold $M \\times N$."""

    def __init__(self, lsh_man: LocationShape[Location, Shape]):
        super().__init__(lsh_man, lsh_man.fst_man)

    def project[C: Coordinates](
        self, p: Point[C, LocationShape[Location, Shape]]
    ) -> Point[C, Location]:
        first, _ = self.sup_man.split_params(p)
        return first

    def translate[C: Coordinates](
        self, p: Point[C, LocationShape[Location, Shape]], q: Point[C, Location]
    ) -> Point[C, LocationShape[Location, Shape]]:
        first, second = self.sup_man.split_params(p)
        return self.sup_man.join_params(first + q, second)


class ReplicatedLocationSubspace[Loc: ExponentialFamily, Shp: ExponentialFamily](
    Subspace[Replicated[LocationShape[Loc, Shp]], Replicated[Loc]]
):
    """Subspace relationship that projects only to location parameters of replicated location-shape manifolds.

    For a replicated location-shape manifold with parameters:
    $((l_1,s_1),\\ldots,(l_n,s_n))$

    Projects to just the location parameters:
    $(l_1,\\ldots,l_n)$

    This enables mixture models where components only affect location parameters while
    sharing shape parameters across components.
    """

    def __init__(self, base_location: Loc, base_shape: Shp, n_neurons: int):
        """Initialize subspace relationship.

        Args:
            base_location: Base location manifold
            base_shape: Base shape manifold
            n_neurons: Number of replicated units
        """
        ls_man = LocationShape(base_location, base_shape)
        loc_man = Replicated(base_location, n_neurons)
        sup_man = Replicated(ls_man, n_neurons)
        super().__init__(sup_man, loc_man)

    def project[C: Coordinates](
        self, p: Point[C, Replicated[LocationShape[Loc, Shp]]]
    ) -> Point[C, Replicated[Loc]]:
        """Project to location parameters $(l_1,\\ldots,l_n)$."""
        params_matrix = p.params.reshape(self.sup_man.n_repeats, -1)
        loc_dim = self.sub_man.man.dim
        loc_params = params_matrix[:, :loc_dim]
        return Point(loc_params.reshape(-1))

    def translate[C: Coordinates](
        self,
        p: Point[C, Replicated[LocationShape[Loc, Shp]]],
        q: Point[C, Replicated[Loc]],
    ) -> Point[C, Replicated[LocationShape[Loc, Shp]]]:
        """Update location parameters while preserving shape parameters.

        Given original parameters $((l_1,s_1),\\ldots,(l_n,s_n))$ and new locations
        $(l_1',\\ldots,l_n')$, returns $((l_1',s_1),\\ldots,(l_n',s_n))$.
        """
        params_matrix = p.params.reshape(self.sup_man.n_repeats, -1)
        loc_dim = self.sub_man.man.dim
        q_matrix = q.params.reshape(self.sub_man.n_repeats, -1)
        params_matrix = params_matrix.at[:, :loc_dim].add(q_matrix)
        return Point(params_matrix.reshape(-1))
