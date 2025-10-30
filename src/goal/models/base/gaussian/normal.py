"""This module provides implementations of multivariate normal distributions in an exponential family framework. Each normal is build out of the base components:
    - `Euclidean`: The location component ($\\mathbb{R}^n$)
    - `Covariance`: The shape component with flexible structure

The matrix representation used for the `Covariance` yields the following variants of normal distribution:
    - `FullNormal`: Unrestricted positive definite covariance
    - `DiagonalNormal`: Diagonal covariance matrix
    - `IsotropicNormal`: Scalar multiple of identity
    - `StandardNormal`: Unit covariance (identity matrix)
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
from .generalized import Euclidean, GeneralizedGaussian

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
    return Point(p.array)


def lin_to_cov[Rep: PositiveDefinite, C: Coordinates](
    p: Point[C, LinearMap[Rep, Euclidean, Euclidean]],
) -> Point[C, Covariance[Rep]]:
    """Convert a linear map to a covariance."""
    return Point(p.array)


# Component Classes


class Covariance[Rep: PositiveDefinite](SquareMap[Rep, Euclidean], ExponentialFamily):
    """Shape component of a Normal distribution.

    This represents the covariance structure of a Normal distribution through different matrix representations:
        - `PositiveDefinite`: Full covariance matrix
        - `Diagonal`: diagonal elements
        - `Scale`: Scalar multiple of identity
        - `Identity`: Unit covariance

    The Rep parameter determines both:

    1. How parameters are stored (symmetric matrix, diagonal, scalar)
    2. The effiency of operations (matrix multiply, inversion, etc.)

    As an exponential family:
        - Sufficient statistic: Outer product $s(x) = x \\otimes x$
        - Base measure: $\\mu(x) = -\\frac{n}{2}\\log(2\\pi)$
    """

    # Constructor

    def __init__(self, data_dim: int, rep: type[Rep]):
        super().__init__(rep(), Euclidean(data_dim))

    # Overrides

    @property
    @override
    def data_dim(self) -> int:
        return self.matrix_shape[0]

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
        """Initialize covariance matrix with random perturbation from identity. Uses a low-rank perturbation $I + LL^T$ where $L$ has entries drawn from $N(0, shape/sqrt(dim))$ to ensure reasonable condition numbers."""
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
    GeneralizedGaussian[Covariance[Rep]],
    LocationShape[Euclidean, Covariance[Rep]],
    Analytic,
):
    """(Multivariate) Normal distributions.

    The standard expression for the Normal density is

    $$p(x; \\mu, \\Sigma) = (2\\pi)^{-d/2}|\\Sigma|^{-1/2}e^{-\\frac{1}{2}(x-\\mu) \\cdot \\Sigma^{-1} \\cdot (x-\\mu)},$$

    where

    - $\\mu$ is the mean vector,
    - $\\Sigma$ is the covariance matrix, and
    - $d$ is the dimension of the data.

    As an exponential family:
        - Sufficient statistic: $\\mathbf{s}(x) = (x, x \\otimes x)$
        - Base measure: $\\mu(x) = -\\frac{d}{2}\\log(2\\pi)$
        - Natural parameters: $\\theta_1 = \\Sigma^{-1}\\mu$, $\\theta_2 = -\\frac{1}{2}\\Sigma^{-1}$
        - Mean parameters: $\\eta_1 = \\mu$, $\\eta_2 = \\mu\\mu^T + \\Sigma$
        - Log-partition: $\\psi(\\theta) = -\\frac{1}{4}\\theta_1 \\cdot \\theta_2^{-1} \\cdot \\theta_1 - \\frac{1}{2}\\log|-2\\theta_2|$
        - Negative entropy: $\\phi(\\eta) = -\\frac{1}{2}\\log|\\eta_2 - \\eta_1\\eta_1^T| - \\frac{d}{2}(1 + \\log(2\\pi))$

    Different covariance structures are handled through the `Rep` parameter, providing appropriate trade-offs between flexibility and computational efficiency.
    """

    # Fields

    _data_dim: int

    rep: type[Rep]
    """Covariance representation type."""

    # Overrides

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
        samples = self.snd_man.rep.cholesky_matvec(
            self.snd_man.matrix_shape, covariance.array, z
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
    @override
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
        # Create the second moment $\\eta_2 = \\mu\\mu^T + \\Sigma$
        outer = self.snd_man.outer_product(mean, mean)
        second_moment = outer + covariance

        return self.join_params(mean, second_moment)

    def split_mean_covariance(
        self, p: Point[Mean, Self]
    ) -> tuple[Point[Mean, Euclidean], Point[Mean, Covariance[Rep]]]:
        """Extract the mean $\\mu$ and covariance $\\Sigma$ from a `Point` in `Mean` coordinates."""
        # Split into $\\mu$ and $\\eta_2$
        mean, second_moment = self.split_params(p)

        # Compute $\\Sigma = \\eta_2 - \\mu\\mu^T$
        outer = self.snd_man.outer_product(mean, mean)
        covariance = second_moment - outer

        return mean, covariance

    @override
    def split_mean_second_moment(
        self, params: Point[Mean, Self]
    ) -> tuple[Point[Mean, Euclidean], Point[Mean, Covariance[Rep]]]:
        """Split parameters into mean and second-moment components."""
        return self.split_params(params)

    @override
    def join_mean_second_moment(
        self, mean: Point[Mean, Euclidean], second_moment: Point[Mean, Covariance[Rep]]
    ) -> Point[Mean, Self]:
        """Join mean and second-moment parameters."""
        return self.join_params(mean, second_moment)

    @override
    def split_location_precision(
        self, params: Point[Natural, Self]
    ) -> tuple[Point[Natural, Euclidean], Point[Natural, Covariance[Rep]]]:
        """Join natural location and precision (inverse covariance) parameters. There's some subtle rescaling that has to happen to ensure that the natural parameters behaves correctly when used either as a vector in a dot product, or as a precision matrix.

        For a multivariate normal distribution, the natural parameters $(\\theta_1,\\theta_2)$ are related to the standard parameters $(\\mu,\\Sigma)$ by, $\\theta_1 = \\Sigma^{-1}\\mu$ and $\\theta_2 = -\\frac{1}{2}\\Sigma^{-1}$. matrix representations require different scaling to maintain these relationships:

        1. Diagonal case:
            - No additional rescaling needed as parameters directly represent diagonal elements

        2. Full (PositiveDefinite) case:
            - Off-diagonal elements appear twice in the precision matrix but once in the natural parameters
            - For $i \\neq j$, element $\\theta_{2,ij}$ is stored as double its matrix value to account for missing parameters in the dot product
            - When converting to precision $\\Sigma^{-1}$, vector elements corresponding to off-diagonal elements are halved

        3. Scale case:
            - The exponential family dot product has to be scaled by $\\frac{1}{d}$
            - This scales needs to be stored in either in the sufficient statistic or the natural parameters
            - We store it in the sufficient statistic (hence its defined as an average), which requires that we divide the natural parameters by $d$ when converting to precision
        """
        # First do basic parameter split
        loc, theta2 = self.split_params(params)

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

    @override
    def join_location_precision(
        self,
        location: Point[Natural, Euclidean],
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

        return self.join_params(location, self.cov_man.natural_point(theta2))

    def embed_rep[TargetRep: PositiveDefinite](
        self, trg_man: Normal[TargetRep], p: Point[Natural, Self]
    ) -> Point[Natural, Normal[TargetRep]]:
        """Embed natural parameters into a more complex representation. For example, a diagonal matrix can be embedded as a full matrix with zeros off the diagonal."""
        # Embedding in natural coordinates means embedding precision matrix
        loc, prs = self.split_location_precision(p)
        _, trg_prs = self.cov_man.embed_rep(prs, type(trg_man.cov_man.rep))
        return trg_man.join_location_precision(loc, lin_to_cov(trg_prs))

    def project_rep[TargetRep: PositiveDefinite](
        self, trg_man: Normal[TargetRep], p: Point[Mean, Self]
    ) -> Point[Mean, Normal[TargetRep]]:
        """Project mean parameters to a simpler representation. For example, a full matrix can be projected to a diagonal one. In Mean coordinates this this corresponds to the information (moment matching) projection."""
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
            1. A minimum variance constraint that prevents any dimension from having variance below a specified threshold
            2. A jitter term that adds a small positive value to all diagonal elements, improving numerical stability

        The regularization preserves the correlation structure while ensuring the covariance matrix remains well-conditioned.
        """
        mean, covariance = self.split_mean_covariance(p)

        def regularize_diagonal(x: Array) -> Array:
            return jnp.maximum(x + jitter, min_var)

        adjusted_covariance = self.cov_man.map_diagonal(covariance, regularize_diagonal)

        return self.join_mean_covariance(mean, adjusted_covariance)

    def whiten(
        self,
        given: Point[Mean, Self],
        relative: Point[Mean, Self],
    ) -> Point[Mean, Self]:
        """Whiten a normal distribution relative to another normal distribution.

        Transforms a normal distribution to have standard normal parameters by:
        1. Extracting the distribution's mean and covariance
        2. Scaling the mean: new_mean = precision^(1/2) @ (old_mean - mean)
        3. Transforming the covariance: new_cov = precision^(1/2) @ old_cov @ precision^(1/2)
        where precision^(1/2) is the inverse of the Cholesky decomposition of covariance
        """
        old_mean, old_cov = self.split_mean_covariance(given)
        rel_mean, rel_cov = self.split_mean_covariance(relative)

        new_array, new_matrix = self.cov_man.rep.cholesky_whiten(
            self.cov_man.matrix_shape,
            old_mean.array,
            old_cov.array,
            rel_mean.array,
            rel_cov.array,
        )

        new_mean = self.loc_man.mean_point(new_array)
        new_cov = self.cov_man.mean_point(new_matrix)

        return self.join_mean_covariance(new_mean, new_cov)

    def standard_normal(self) -> Point[Mean, Self]:
        """Return the standard normal distribution."""
        return self.join_mean_covariance(
            self.loc_man.mean_point(jnp.zeros(self.data_dim)),
            self.cov_man.from_dense(jnp.eye(self.data_dim)),
        )

    def statistical_mean(self, params: Point[Natural, Self]) -> Array:
        """Compute the mean of the distribution."""
        mean, _ = self.split_mean_covariance(self.to_mean(params))
        return mean.array

    def statistical_covariance(self, params: Point[Natural, Self]) -> Array:
        """Compute the covariance of the distribution."""
        _, cov = self.split_mean_covariance(self.to_mean(params))
        return self.cov_man.to_dense(cov)

    # GeneralizedGaussian interface implementation

    @property
    @override
    def shp_man(self) -> Covariance[Rep]:
        """Shape component: covariance manifold."""
        return self.cov_man

    @property
    @override
    def fst_man(self) -> Euclidean:
        """First component: location manifold (Euclidean space)."""
        return self.loc_man

    @property
    @override
    def snd_man(self) -> Covariance[Rep]:
        """Second component: shape manifold (covariance structure)."""
        return self.cov_man
