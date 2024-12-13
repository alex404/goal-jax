"""Core exponential families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax import Array

from ..geometry import (
    Backward,
    Coordinates,
    Diagonal,
    ExponentialFamily,
    Identity,
    LocationShape,
    Mean,
    Natural,
    Point,
    PositiveDefinite,
    Scale,
    SquareMap,
    reduce_dual,
)

type FullNormal = Normal[PositiveDefinite]
type DiagonalNormal = Normal[Diagonal]
type IsotropicNormal = Normal[Scale]
type StandardNormal = Normal[Identity]

type FullCovariance = Covariance[PositiveDefinite]
type DiagonalCovariance = Covariance[Diagonal]
type IsotropicCovariance = Covariance[Scale]


@dataclass(frozen=True)
class Euclidean(ExponentialFamily):
    """Euclidean space $\\mathbb{R}^n$ of dimension $n$. Also serves as the location component of a Normal distribution.

    Euclidean space consists of $n$-dimensional real vectors with the standard Euclidean distance metric

    $$d(x,y) = \\sqrt{\\sum_{i=1}^n (x_i - y_i)^2}.$$

    Euclidean space serves as the model space for more general manifolds, which locally resemble $\\mathbb{R}^n$ near each point.

    Args:
        _dim: The dimension $n$ of the space
    """

    _dim: int

    @property
    def dim(self) -> int:
        """Return the dimension of the space."""
        return self._dim

    @property
    def data_dim(self) -> int:
        return self.dim

    def _compute_sufficient_statistic(self, x: Array) -> Array:
        """Identity map on the data."""
        return jnp.atleast_1d(x)

    def log_base_measure(self, x: Array) -> Array:
        """Standard normal base measure including normalizing constant."""
        return -0.5 * self.dim * jnp.log(2 * jnp.pi)


@dataclass(frozen=True)
class Covariance[Rep: PositiveDefinite](SquareMap[Rep, Euclidean], ExponentialFamily):
    """Shape component of a Normal distribution.

    This represents a zero-centered normal distribution with flexible covariance.
    The sufficient statistic is the outer product of the data with itself.
    """

    def __init__(self, data_dim: int, rep: type[Rep]):
        super().__init__(rep(), Euclidean(data_dim))

    @property
    def data_dim(self) -> int:
        return self.shape[0]

    def _compute_sufficient_statistic(self, x: Array) -> Array:
        """Outer product with appropriate covariance structure."""
        x = jnp.atleast_1d(x)
        x_point: Point[Mean, Euclidean] = Point(x)
        return self.outer_product(x_point, x_point).params

    def log_base_measure(self, x: Array) -> Array:
        """Base measure matches location component."""
        return -0.5 * self.data_dim * jnp.log(2 * jnp.pi)

    def shape_initialize(
        self, key: Array, mu: float = 0.0, shp: float = 0.1
    ) -> Point[Natural, Self]:
        """Initialize covariance matrix with random diagonal structure.

        Uses exp(N(mu, shp)) to generate positive diagonal elements.
        """
        diag = jnp.exp(jax.random.normal(key, (self.data_dim,)) * shp + mu)
        return self.from_dense(jnp.diag(diag))


@dataclass(frozen=True)
class Normal[Rep: PositiveDefinite](
    LocationShape[Euclidean, Covariance[Rep]], Backward
):
    """(Multivariate) Normal distributions.

    Parameters:
        data_dim: Dimension of the data space.
        rep: Covariance structure (e.g. `PositiveDefinite`, `Diagonal`, `Scale`).

    Attributes:
        fst_man: Euclidean class. Determines the dimension.
        snd_man: Covariance structure class (e.g. `Scale`, `Diagonal`, `PositiveDefinite`). Determines how the covariance matrix is parameterized.

    The standard expression for the Normal density is

    $$p(x; \\mu, \\Sigma) = (2\\pi)^{-d/2}|\\Sigma|^{-1/2}e^{-\\frac{1}{2}(x-\\mu) \\cdot \\Sigma^{-1} \\cdot (x-\\mu)}.$$

    where

    - $\\mu$ is the mean vector,
    - $\\Sigma$ is the covariance matrix, and
    - $d$ is the dimension of the data.

    As an exponential family, the general form of the Normal family is defined by the base measure $\\mu(x) = -\\frac{d}{2}\\log(2\\pi)$ and sufficient statistic $\\mathbf{s}(x) = (x, x \\otimes x)$. The log-partition function is then given by

    $$\\psi(\\theta_1, \\theta_2) = -\\frac{1}{4}\\theta_1 \\cdot \\theta_2^{-1} \\cdot \\theta - \\frac{1}{2}\\log|-2\\theta_2|,$$

    and the negative entropy is given by

    $$\\psi(\\eta_1, \\eta_2) = -\\frac{1}{2}\\log|\\eta_2 - \\eta_1\\eta_1^T| - \\frac{d}{2}(1 + \\log(2\\pi)).$$

    Consequently, the mean parameters are given simply by the first and second moments $\\eta_1 = \\mu$ and $\\eta_2 = \\mu\\mu^T + \\Sigma$, and the natural parameters are given by $\\theta_1 = \\Sigma^{-1}\\mu$ and $\\theta_2 = -\\frac{1}{2}\\Sigma^{-1}$.

    Finally, we handle different covariance structures (full, diagonal, scalar) through the `LinearOperator` hierarchy, so that the sufficient statistics and parameters are lowerer-dimensional, and the given manifold is a submanifold of the unrestricted family of Normals.
    """

    # Attributes

    def __init__(self, data_dim: int = 1, rep: type[Rep] = PositiveDefinite):
        super().__init__(
            Euclidean(data_dim),
            Covariance(data_dim, rep),
        )

    @property
    def data_dim(self) -> int:
        """Dimension of the data space."""
        return self.fst_man.data_dim

    @property
    def loc_man(self) -> Euclidean:
        """Location manifold."""
        return self.fst_man

    @property
    def cov_man(self) -> Covariance[Rep]:
        """Covariance manifold."""
        return self.snd_man

    # Core class methods

    def _compute_sufficient_statistic(self, x: Array) -> Array:
        x = jnp.atleast_1d(x)

        # Create point in StandardNormal
        x_point: Point[Mean, Euclidean] = Point(x)

        # Compute outer product with appropriate structure
        second_moment: Point[Mean, Covariance[Rep]] = self.snd_man.outer_product(
            x_point, x_point
        )

        # Concatenate components into parameter vector
        return self.join_params(x_point, second_moment).params

    def log_base_measure(self, x: Array) -> Array:
        return -0.5 * self.data_dim * jnp.log(2 * jnp.pi)

    def _compute_log_partition_function(self, natural_params: Array) -> Array:
        loc, precision = self.split_location_precision(Point(natural_params))

        covariance = reduce_dual(self.snd_man.inverse(precision))
        mean = self.snd_man(covariance, loc)

        return 0.5 * (jnp.dot(loc.params, mean.params) - self.snd_man.logdet(precision))

    def _compute_negative_entropy(self, mean_params: Array) -> Array:
        mean, second_moment = self.split_mean_second_moment(Point(mean_params))

        # Compute covariance without forming full matrices
        outer_mean = self.snd_man.outer_product(mean, mean)
        covariance = second_moment - outer_mean
        log_det = self.snd_man.logdet(covariance)

        return -0.5 * (self.data_dim * (1 + jnp.log(2 * jnp.pi)) + log_det)

    def sample(
        self,
        key: Array,
        p: Point[Natural, Self],
        n: int = 1,
    ) -> Array:
        mean_point = self.to_mean(p)
        mean, covariance = self.split_mean_covariance(mean_point)

        # Draw standard normal samples
        key = jnp.asarray(key)
        shape = (n, self.data_dim)
        z = jax.random.normal(key, shape)

        # Transform samples using Cholesky
        samples = self.snd_man.rep.apply_cholesky(
            self.snd_man.shape, covariance.params, z
        )
        return mean.params + samples

    # Additional methods

    def split_params[C: Coordinates](
        self, p: Point[C, Self]
    ) -> tuple[Point[C, Euclidean], Point[C, Covariance[Rep]]]:
        """Split parameters into location and scale components.

        Args:
            p: Point in Normal manifold coordinates

        Returns:
            Tuple of (location vector, covariance matrix)
        """
        loc_params = p.params[: self.data_dim]
        cov_params = p.params[self.data_dim :]

        return Point(loc_params), Point(cov_params)

    def join_mean_covariance(
        self,
        mean: Point[Mean, Euclidean],
        covariance: Point[Mean, Covariance[Rep]],
    ) -> Point[Mean, Self]:
        """Construct a `Point` in `Mean` coordinates from the mean $\\mu$ and covariance $\\Sigma$."""
        # Create the second moment η₂ = μμᵀ + Σ
        outer = self.snd_man.outer_product(mean, mean)
        second_moment = outer + covariance

        return self.join_params(mean, second_moment)

    def split_mean_covariance(
        self, p: Point[Mean, Self]
    ) -> tuple[Point[Mean, Euclidean], Point[Mean, Covariance[Rep]]]:
        """Extract the mean $\\mu$ and covariance $\\Sigma$ from a `Point` in `Mean` coordinates."""
        # Split into μ and η₂
        mean, second_moment = self.split_params(p)

        # Compute Σ = η₂ - μμᵀ
        outer = self.snd_man.outer_product(mean, mean)
        covariance = second_moment - outer

        return mean, covariance

    def split_mean_second_moment(
        self, p: Point[Mean, Self]
    ) -> tuple[Point[Mean, Euclidean], Point[Mean, Covariance[Rep]]]:
        """Split parameters into mean and second-moment components."""
        return self.split_params(p)

    def join_mean_second_moment(
        self, mean: Point[Mean, Euclidean], second_moment: Point[Mean, Covariance[Rep]]
    ) -> Point[Mean, Self]:
        """Join mean and second-moment parameters."""
        return self.join_params(mean, second_moment)

    def split_location_precision(
        self, p: Point[Natural, Self]
    ) -> tuple[Point[Natural, Euclidean], Point[Natural, Covariance[Rep]]]:
        """Join natural location and precision (inverse covariance) parameters. There's a lot of rescaling that has to happen to ensure that the minimal representation of the natural parameters behaves correctly when used either as a vector in a dot product, or as a matrix.


        For a multivariate normal distribution, the natural parameters $(\\theta_1,\\theta_2)$ are related to the standard parameters $(\\mu,\\Sigma)$ by, $\\theta_1 = \\Sigma^{-1}\\mu$ and $\\theta_2 = -\\frac{1}{2}\\Sigma^{-1}$.

        Different matrix representations require different scaling to maintain these relationships:

        1. Diagonal case:
            - No rescaling needed as parameters directly represent diagonal elements
            - $\\theta_2$ diagonal elements already represent $-\\frac{1}{2}\\Sigma^{-1}$

        2. Full (PositiveDefinite) case:
            - Off-diagonal elements appear twice in the matrix but once in parameters
            - For $i \\neq j$, element $\\theta_{2,ij}$ is stored as double its matrix value to account for missing parameters in the dot product
            - When converting to precision $\\Sigma^{-1}$, vector elements corresponding to off-diagonal elements are halved

        3. Scale case:
            - Dot product between second order parameter requires scaling by $\\frac{1}{d}$, stored either in the sufficient statistic or the natural parameters
            - We store it in the sufficient statistic, which requires we divide the natural parameters by $d$ when converting to precision

        Args:
            p: Point in natural coordinates containing concatenated $\\theta_1$ and $\\theta_2$ parameters

        Returns:
            - $\\theta_1$: Natural location parameters $(\\Sigma^{-1}\\mu)$
            - $\\theta_2$: Precision parameters $(\\Sigma^{-1})$
        """
        # First do basic parameter split
        loc, theta2 = self.split_params(p)

        # We need to rescale off-precision params
        if not isinstance(self.snd_man.rep, Diagonal):
            precision_params = theta2.params
            i_diag = (
                jnp.triu_indices(self.data_dim)[0] == jnp.triu_indices(self.data_dim)[1]
            )

            scaled_params = precision_params / 2
            scaled_params = jnp.where(i_diag, scaled_params * 2, scaled_params)
            theta2: Point[Natural, Covariance[Rep]] = Point(scaled_params)

        scl = -2

        if isinstance(self.snd_man.rep, Scale):
            scl = scl / self.data_dim

        return loc, scl * theta2

    def join_location_precision(
        self,
        loc: Point[Natural, Euclidean],
        precision: Point[Natural, Covariance[Rep]],
    ) -> Point[Natural, Self]:
        """Join natural location and precision (inverse covariance) parameters. Inverts the scaling in `split_natural_params`."""
        scl = -0.5

        if isinstance(self.snd_man.rep, Scale):
            scl = self.data_dim * scl

        theta2 = scl * precision.params

        # We need to rescale off-precision params
        if not isinstance(self.snd_man.rep, Diagonal):
            i_diag = (
                jnp.triu_indices(self.data_dim)[0] == jnp.triu_indices(self.data_dim)[1]
            )

            scaled_params = theta2 * 2  # First multiply by -1/2
            scaled_params = jnp.where(i_diag, scaled_params / 2, scaled_params)
            theta2 = scaled_params

        return self.join_params(loc, Point(theta2))

    def transform_rep[TargetRep: PositiveDefinite](
        self, target_man: Normal[TargetRep], p: Point[Natural, Self]
    ) -> Point[Natural, Normal[TargetRep]]:
        """Transform natural parameters to target representation.

        This function converts parameters between different matrix representations
        (Scale, Diagonal, PositiveDefinite). When going from a simpler to more complex
        representation (e.g. Scale to PositiveDefinite), this acts as an embedding.
        When going from complex to simpler (e.g. PositiveDefinite to Scale), this
        acts as a projection.

        Args:
            target_man: Target normal distribution
            p: Parameters in natural coordinates to transform

        Returns:
            Parameters in target representation
        """
        loc, precision = self.split_location_precision(p)
        dense_prec = self.cov_man.to_dense(precision)
        target_prec = target_man.cov_man.from_dense(dense_prec)
        return target_man.join_location_precision(loc, target_prec)

    def standard_normal(self) -> Point[Mean, Self]:
        """Return the standard normal distribution."""
        return self.join_mean_covariance(
            Point(jnp.zeros(self.data_dim)),
            self.cov_man.from_dense(jnp.eye(self.data_dim)),
        )
