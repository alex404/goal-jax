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
from typing import Self

import jax
import jax.numpy as jnp
from jax import Array

from .manifold import Coordinates, Dual, Manifold, Pair, Point, reduce_dual
from .subspace import Subspace

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


@dataclass(frozen=True)
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

    def natural_point(self, params: Array) -> Point[Natural, Self]:
        """Construct a point in natural coordinates."""
        return Point[Natural, Self](jnp.atleast_1d(params))

    def mean_point(self, params: Array) -> Point[Mean, Self]:
        """Construct a point in mean coordinates."""
        return Point[Mean, Self](jnp.atleast_1d(params))


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

    def relative_entropy(self, p: Point[Natural, Self], q: Point[Mean, Self]) -> Array:
        """Compute the entropy of $p$ relative to $q$.

        $D(p \\| q) = \\int p(x) \\log \\frac{p(x)}{q(x)} dx = \\theta \\cdot \\eta - \\psi(\\theta) - \\phi(\\eta)$, where

        - $p(x;\\theta)$ has natural parameters $\\theta$,
        - $q(x;\\eta)$ has mean parameters $\\eta$.
        """
        return (
            self.log_partition_function(p) + self.negative_entropy(q) - self.dot(q, p)
        )


class LocationShape[Location: ExponentialFamily, Shape: ExponentialFamily](
    Pair[Location, Shape], ExponentialFamily
):
    """A product exponential family with location and shape parameters.

    This structure captures distributions like the Normal distribution that decompose into a location parameter (e.g. mean) and a shape parameter (e.g. covariance).

    - The components must have matching data dimensions.
    - The sufficient statistic is the concatenation of component sufficient statistics.
    - The base measure is inherited from the location component since both componentsshoudl share the same base measure.
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

    def log_base_measure(self, x: Array) -> Array:
        """Base measure from location component, as both components should share the same measure."""
        return self.fst_man.log_base_measure(x)

    def shape_initialize(
        self, key: Array, mu: float = 0.0, shp: float = 0.1
    ) -> Point[Natural, Self]:
        """Initialize location and shape parameters."""
        key_loc, key_shp = jax.random.split(key)
        fst_loc = self.fst_man.shape_initialize(key_loc, mu, shp)
        shp_loc = self.snd_man.shape_initialize(key_shp, mu, shp)
        return self.join_params(fst_loc, shp_loc)


@dataclass(frozen=True)
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
