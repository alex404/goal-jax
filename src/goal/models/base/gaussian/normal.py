"""Normal distributions as exponential families.

1. Base Components:

    - `Euclidean`: The location component ($\\mathbb{R}^n$)
    - `Covariance`: The shape component with flexible structure

2. Covariance Structures (Type Synonyms):

    - `FullNormal`: Unrestricted positive definite covariance
    - `DiagonalNormal`: Diagonal covariance matrix
    - `IsotropicNormal`: Scalar multiple of identity
    - `StandardNormal`: Unit covariance (identity matrix)

The normal density has the form

$$p(x; \\mu, \\Sigma) = (2\\pi)^{-d/2}|\\Sigma|^{-1/2}\\exp\\left(-\\frac{1}{2}(x-\\mu)^T\\Sigma^{-1}(x-\\mu)\\right)$$.

This can be expressed in exponontial family coordinates as

- Natural parameters: $(\\theta_1, \\theta_2) = (\\Sigma^{-1}\\mu, -\\frac{1}{2}\\Sigma^{-1})$
- Mean parameters: $(\\eta_1, \\eta_2) = (\\mu, \\mu\\mu^T + \\Sigma)$

#### Class Hierarchy

![Class Hierarchy](normal.svg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ....geometry import (
    Analytic,
    Coordinates,
    Diagonal,
    ExponentialFamily,
    Identity,
    LinearMap,
    LocationShape,
    Mean,
    Natural,
    Point,
    PositiveDefinite,
    Scale,
    SquareMap,
    expand_dual,
)
from ....geometry.manifold.base import _Point  # pyright: ignore[reportPrivateUsage]

type FullNormal = Normal[PositiveDefinite]
type DiagonalNormal = Normal[Diagonal]
type IsotropicNormal = Normal[Scale]
type StandardNormal = Normal[Identity]

type FullCovariance = Covariance[PositiveDefinite]
type DiagonalCovariance = Covariance[Diagonal]
type IsotropicCovariance = Covariance[Scale]


# Type Hacks


def cov_to_lin[Rep: PositiveDefinite, C: Coordinates](
    p: Point[C, Covariance[Rep]],
) -> Point[C, LinearMap[Rep, Euclidean, Euclidean]]:
    """Convert a covariance to a linear map."""
    return _Point(p.array)


def lin_to_cov[Rep: PositiveDefinite, C: Coordinates](
    p: Point[C, LinearMap[Rep, Euclidean, Euclidean]],
) -> Point[C, Covariance[Rep]]:
    """Convert a linear map to a covariance."""
    return _Point(p.array)


# Component Classes


@dataclass(frozen=True)
class Euclidean(ExponentialFamily):
    """Euclidean space $\\mathbb{R}^n$ of dimension $n$. Also serves as the location component of a Normal distribution.

    Euclidean space consists of $n$-dimensional real vectors with the standard Euclidean distance metric

    $$d(x,y) = \\sqrt{\\sum_{i=1}^n (x_i - y_i)^2}.$$

    Euclidean space serves as the model space for more general manifolds, which locally resemble $\\mathbb{R}^n$ near each point.

    Args:
        _dim: The dimension $n$ of the space
    """

    # Fields

    _dim: int

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """Return the dimension of the space."""
        return self._dim

    @property
    @override
    def data_dim(self) -> int:
        return self.dim

    @override
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        """Identity map on the data."""
        return self.mean_point(x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Standard normal base measure including normalizing constant."""
        return -0.5 * self.dim * jnp.log(2 * jnp.pi)


@dataclass(frozen=True)
class Covariance[Rep: PositiveDefinite](SquareMap[Rep, Euclidean], ExponentialFamily):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """Shape component of a Normal distribution.

    This represents the covariance structure of a Normal distribution through different matrix representations:

    - `PositiveDefinite`: Full covariance matrix
    - `Diagonal`: diagonal elements
    - `Scale`: Scalar multiple of identity

    The Rep parameter determines both:

    1. How parameters are stored (symmetric matrix, diagonal, scalar)
    2. The effiency of operations (matrix multiply, inversion, etc.)
    """

    # Constructor

    def __init__(self, data_dim: int, rep: type[Rep]):
        super().__init__(rep(), Euclidean(data_dim))

    # Overrides

    @property
    @override
    def data_dim(self) -> int:
        return self.shape[0]

    @override
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        """Outer product with appropriate covariance structure."""
        euclidean = Euclidean(self.data_dim)
        x_point: Point[Mean, Euclidean] = euclidean.mean_point(x)
        return self.outer_product(x_point, x_point)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Base measure matches location component."""
        return -0.5 * self.data_dim * jnp.log(2 * jnp.pi)

    @override
    def check_natural_parameters(self, params: Point[Natural, Self]) -> Array:
        """Check if natural parameters (precision matrix) are valid.

        For covariance/precision matrices, we check:
        1. All parameters are finite
        2. Precision matrix is numerically positive definite
        """
        finite = super().check_natural_parameters(params)
        is_pd = self.is_positive_definite(params)
        return finite & is_pd

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Point[Natural, Self]:
        """Initialize covariance matrix with random perturbation from identity.

        Uses a low-rank perturbation I + LL^T where L has entries N(0, shape/sqrt(dim))
        to ensure reasonable condition numbers. The representation (Scale, Diagonal,
        PositiveDefinite) determines how this matrix is stored and used.

        Args:
            key: Random key
            location: Base location (not currently used)
            shape: Scale of perturbation from identity

        Returns:
            Point in natural coordinates
        """
        noise_scale = shape / jnp.sqrt(self.data_dim)
        big_l = noise_scale * jax.random.normal(key, (self.data_dim, self.data_dim))
        base_cov = jnp.eye(self.data_dim) + big_l @ big_l.T
        return self.from_dense(base_cov)

    # Methods

    def add_jitter(
        self, params: Point[Natural, Self], epsilon: float
    ) -> Point[Natural, Self]:
        """Add epsilon to diagonal elements while preserving matrix structure."""
        return self.map_diagonal(params, lambda x: x + epsilon)


@dataclass(frozen=True)
class Normal[Rep: PositiveDefinite](
    LocationShape[Euclidean, Covariance[Rep]], Analytic
):
    """(Multivariate) Normal distributions.

    Parameters:
        _data_dim: Dimension of the data space.
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

    # Fields

    _data_dim: int
    rep: type[Rep]

    # Overrides

    @property
    @override
    def fst_man(self) -> Euclidean:
        return self.loc_man

    @property
    @override
    def snd_man(self) -> Covariance[Rep]:
        return self.cov_man

    @override
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        x = jnp.atleast_1d(x)

        # Create point in StandardNormal
        x_point: Point[Mean, Euclidean] = self.loc_man.mean_point(x)

        # Compute outer product with appropriate structure
        second_moment: Point[Mean, Covariance[Rep]] = self.snd_man.outer_product(
            x_point, x_point
        )

        # Concatenate components into parameter vector
        return self.join_params(x_point, second_moment)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return -0.5 * self.data_dim * jnp.log(2 * jnp.pi)

    @override
    def log_partition_function(self, params: Point[Natural, Self]) -> Array:
        loc, precision = self.split_location_precision(params)

        covariance = self.snd_man.inverse(precision)
        mean = self.snd_man(covariance, expand_dual(loc))

        return 0.5 * (self.loc_man.dot(loc, mean) - self.snd_man.logdet(precision))

    @override
    def negative_entropy(self, means: Point[Mean, Self]) -> Array:
        mean, second_moment = self.split_mean_second_moment(means)

        # Compute covariance without forming full matrices
        outer_mean = self.snd_man.outer_product(mean, mean)
        covariance = second_moment - outer_mean
        log_det = self.snd_man.logdet(covariance)

        return -0.5 * log_det - 0.5 * self.data_dim

    @override
    def sample(
        self,
        key: Array,
        params: Point[Natural, Self],
        n: int = 1,
    ) -> Array:
        mean_point = self.to_mean(params)
        mean, covariance = self.split_mean_covariance(mean_point)

        # Draw standard normal samples
        shape = (n, self.data_dim)
        z = jax.random.normal(key, shape)

        # Transform samples using Cholesky
        samples = self.snd_man.rep.apply_cholesky(
            self.snd_man.shape, covariance.array, z
        )
        return mean.array + samples

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Point[Natural, Self]:
        """Initialize means with normal and covariance matrix with random diagonal structure."""
        mean = self.loc_man.initialize(key, location, shape)
        cov = self.cov_man.initialize(key, location=location, shape=shape)
        return self.join_location_precision(mean, cov)

    @override
    def initialize_from_sample(
        self,
        key: Array,
        sample: Array,
        location: float = 0.0,  # renamed from location
        shape: float = 0.1,  # renamed from shape
    ) -> Point[Natural, Self]:
        """Initialize Normal parameters from sample data.

        Computes mean and second moments from sample data, then adds regularizing
        noise to avoid degenerate cases. The noise is scaled relative to the
        observed variance to maintain reasonable parameter ranges.

        Args:
            key: Random key
            sample: Sample data to initialize from
            location: Scale for additive noise to mean (relative to observed std dev)
            shape: Scale for multiplicative noise to covariance

        Returns:
            Point in natural coordinates
        """
        # Get average sufficient statistics (mean and second moment)
        avg_stats = self.average_sufficient_statistic(sample)
        mean, second_moment = self.split_mean_second_moment(avg_stats)

        # Add noise to mean based on observed scale
        key_mean, key_cov = jax.random.split(key)
        observed_scale = jnp.sqrt(jnp.diag(self.cov_man.to_dense(second_moment)))
        mean_noise = observed_scale * (
            location + shape * jax.random.normal(key_mean, mean.array.shape)
        )
        mean: Point[Mean, Euclidean] = self.loc_man.mean_point(mean.array + mean_noise)

        # Add multiplicative noise to second moment
        noise = shape * jax.random.normal(key_cov, (self.data_dim, self.data_dim))
        noise = jnp.eye(self.data_dim) + noise @ noise.T / self.data_dim
        noise_matrix: Point[Mean, Covariance[Rep]] = self.cov_man.from_dense(noise)
        second_moment = self.cov_man.mean_point(
            second_moment.array * noise_matrix.array
        )

        # Join parameters and convert to natural coordinates
        mean_params = self.join_mean_second_moment(mean, second_moment)
        return self.to_natural(mean_params)

    @override
    def check_natural_parameters(self, params: Point[Natural, Self]) -> Array:
        """Check if natural parameters are valid.

        Delegates to Covariance check after extracting precision component.
        """
        _, precision = self.split_location_precision(params)
        return self.cov_man.check_natural_parameters(precision)

    # Methods

    @property
    def loc_man(self) -> Euclidean:
        """Location manifold."""
        return Euclidean(self._data_dim)

    @property
    def cov_man(self) -> Covariance[Rep]:
        """Covariance manifold."""
        return Covariance(self._data_dim, self.rep)

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
        """Join natural location and precision (inverse covariance) parameters. There's some subtle rescaling that has to happen to ensure that the minimal representation of the natural parameters behaves correctly when used either as a vector in a dot product, or as a precision matrix.

        For a multivariate normal distribution, the natural parameters $(\\theta_1,\\theta_2)$ are related to the standard parameters $(\\mu,\\Sigma)$ by, $\\theta_1 = \\Sigma^{-1}\\mu$ and $\\theta_2 = -\\frac{1}{2}\\Sigma^{-1}$. matrix representations require different scaling to maintain these relationships:

        1. Diagonal case:
            - No additional rescaling needed as parameters directly represent diagonal elements

        2. Full (PositiveDefinite) case:
            - Off-diagonal elements appear twice in the matrix but once in parameters
            - For $i \\neq j$, element $\\theta_{2,ij}$ is stored as double its matrix value to account for missing parameters in the dot product
            - When converting to precision $\\Sigma^{-1}$, vector elements corresponding to off-diagonal elements are halved

        3. Scale case:
            - The exponential family dot product has to be scaled by $\\frac{1}{d}$
            - This scales needs to be ``stored'' in either in the sufficient statistic or the natural parameters
            - We store it in the sufficient statistic (hence its defined as an average), which requires that we divide the natural parameters by $d$ when converting to precision

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
            precision_params = theta2.array
            i_diag = (
                jnp.triu_indices(self.data_dim)[0] == jnp.triu_indices(self.data_dim)[1]
            )

            scaled_params = precision_params / 2
            scaled_params = jnp.where(i_diag, scaled_params * 2, scaled_params)
            theta2: Point[Natural, Covariance[Rep]] = self.cov_man.natural_point(
                scaled_params
            )

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

        theta2 = scl * precision.array

        # We need to rescale off-precision params
        if not isinstance(self.snd_man.rep, Diagonal):
            i_diag = (
                jnp.triu_indices(self.data_dim)[0] == jnp.triu_indices(self.data_dim)[1]
            )

            scaled_params = theta2 * 2  # First multiply by -1/2
            scaled_params = jnp.where(i_diag, scaled_params / 2, scaled_params)
            theta2 = scaled_params

        return self.join_params(loc, self.cov_man.natural_point(theta2))

    def embed_rep[TargetRep: PositiveDefinite](
        self, trg_man: Normal[TargetRep], p: Point[Natural, Self]
    ) -> Point[Natural, Normal[TargetRep]]:
        """Embed natural parameters into a more complex representation.

        This converts parameters from a simpler to more complex representation (e.g. Scale to PositiveDefinite) by treating the simpler structure as a special case of the more complex one. For example, a diagonal matrix can be embedded as a full matrix with zeros off the diagonal.

        Args:
            target_man: Target normal distribution (must have more complex structure)
            p: Parameters in natural coordinates to embed

        Returns:
            Parameters embedded in target representation

        Raises:
            TypeError: If trying to embed into a simpler representation
        """
        # Embedding in natural coordinates means embedding precision matrix
        loc, prs = self.split_location_precision(p)
        _, trg_prs = self.cov_man.embed_rep(prs, type(trg_man.cov_man.rep))
        return trg_man.join_location_precision(loc, lin_to_cov(trg_prs))

    def project_rep[TargetRep: PositiveDefinite](
        self, trg_man: Normal[TargetRep], p: Point[Mean, Self]
    ) -> Point[Mean, Normal[TargetRep]]:
        """Project mean parameters to a simpler representation.

        This converts parameters from a more complex to simpler representation (e.g. PositiveDefinite to Diagonal) by discarding components not representable in the simpler structure.

        For example, a full matrix can be projected to a diagonal one. In Mean coordinates this this corresponds to the information (moment matching) projection.

        Args:
            target_man: Target normal distribution (must have simpler structure)
            p: Parameters in mean coordinates to project

        Returns:
            Parameters projected to target representation

        Raises:
            TypeError: If trying to project to a more complex representation
        """
        mean, second_moment = self.split_mean_second_moment(p)
        _, target_second = self.cov_man.project_rep(
            second_moment, type(trg_man.cov_man.rep)
        )
        return trg_man.join_mean_second_moment(mean, lin_to_cov(target_second))

    def regularize_covariance(
        self, p: Point[Mean, Self], jitter: float = 0, min_var: float = 0
    ) -> Point[Mean, Self]:
        """Regularize covariance matrix to ensure numerical stability and reasonable variances.

        This method applies two forms of regularization to the covariance matrix:
        1. A minimum variance constraint that prevents any dimension from having
           variance below a specified threshold
        2. A jitter term that adds a small positive value to all diagonal elements,
           improving numerical stability

        The regularization preserves the correlation structure while ensuring the
        covariance matrix remains well-conditioned.

        Args:
            p: Point in mean coordinates containing location and covariance
            min_var: Minimum allowed variance for any dimension
            jitter: Small positive value added to diagonal elements

        Returns:
            New point in mean coordinates with regularized covariance
        """
        mean, covariance = self.split_mean_covariance(p)

        def regularize_diagonal(x: Array) -> Array:
            return jnp.maximum(x + jitter, min_var)

        adjusted_covariance = self.cov_man.map_diagonal(covariance, regularize_diagonal)

        return self.join_mean_covariance(mean, adjusted_covariance)

    def standard_normal(self) -> Point[Mean, Self]:
        """Return the standard normal distribution."""
        return self.join_mean_covariance(
            self.loc_man.mean_point(jnp.zeros(self.data_dim)),
            self.cov_man.from_dense(jnp.eye(self.data_dim)),
        )

    def statistical_mean(self, params: Point[Natural, Self]) -> Array:
        mean, _ = self.split_mean_covariance(self.to_mean(params))
        return mean.array

    def statistical_covariance(self, params: Point[Natural, Self]) -> Array:
        _, cov = self.split_mean_covariance(self.to_mean(params))
        return self.snd_man.to_dense(cov)
