"""This module provides implementations of multivariate normal distributions in an exponential family framework. Each normal is build out of the base components:
- `Euclidean`: The location component ($\\mathbb{R}^n$)
- `Covariance`: The shape component with flexible structure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ....geometry import (
    Analytic,
    Coordinates,
    Diagonal,
    ExponentialFamily,
    LocationShape,
    PositiveDefinite,
    Scale,
    SquareMap,
)
from .generalized import Euclidean, GeneralizedGaussian

# Component Classes


class Covariance(SquareMap[Euclidean], ExponentialFamily):
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

    rep: PositiveDefinite

    def __init__(self, data_dim: int, rep: PositiveDefinite):
        super().__init__(rep, Euclidean(data_dim))

    # Overrides

    @property
    @override
    def data_dim(self) -> int:
        return self.matrix_shape[0]

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Outer product with appropriate covariance structure.

        Returns:
            Sufficient statistic in mean parameters (outer product of x).
        """
        return self.outer_product(x, x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Base measure matches location component."""
        return -0.5 * self.data_dim * jnp.log(2 * jnp.pi)

    @override
    def check_natural_parameters(self, natural_params: Array) -> Array:
        """Check if natural parameters (precision matrix) are valid.

        For covariance/precision matrices, we check:
        1. All parameters are finite
        2. Precision matrix is numerically positive definite

        Args:
            natural_params: Natural parameters (precision matrix).

        Returns:
            Boolean array indicating validity.
        """
        finite = super().check_natural_parameters(natural_params)
        is_pd = self.is_positive_definite(natural_params)
        return finite & is_pd

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize covariance matrix with random perturbation from identity.

        Uses a low-rank perturbation I + LL^T where L has entries drawn from
        N(0, shape/sqrt(dim)) to ensure reasonable condition numbers.

        Returns:
            Natural parameters (covariance matrix).
        """
        noise_scale = shape / jnp.sqrt(self.data_dim)
        big_l = noise_scale * jax.random.normal(key, (self.data_dim, self.data_dim))
        base_cov = jnp.eye(self.data_dim) + big_l @ big_l.T
        return self.from_dense(base_cov)

    # Methods

    def add_jitter(
        self, natural_params: Array, epsilon: float
    ) -> Array:
        """Add epsilon to diagonal elements while preserving matrix structure.

        Args:
            natural_params: Natural parameters (precision matrix).
            epsilon: Value to add to diagonal.

        Returns:
            Natural parameters with jitter added.
        """
        return self.map_diagonal(natural_params, lambda x: x + epsilon)


# Type Hacks


def cov_to_lin[Rep: PositiveDefinite, C: Coordinates](
    params: Array,
) -> Array:
    """Convert a covariance to a linear map."""
    return params


def lin_to_cov[Rep: PositiveDefinite, C: Coordinates](
    params: Array,
) -> Array:
    """Convert a linear map to a covariance."""
    return params


@dataclass(frozen=True)
class Normal(
    GeneralizedGaussian[Covariance],
    LocationShape[Euclidean, Covariance],
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

    rep: PositiveDefinite
    """Covariance representation type."""

    # Overrides

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Compute sufficient statistic (x, x ⊗ x).

        Returns:
            Sufficient statistic in mean parameters.
        """
        x = jnp.atleast_1d(x)

        # Compute outer product with appropriate structure
        second_moment = self.snd_man.outer_product(x, x)

        # Concatenate components into parameter vector
        return self.join_params(x, second_moment)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return -0.5 * self.data_dim * jnp.log(2 * jnp.pi)

    @override
    def log_partition_function(self, natural_params: Array) -> Array:
        """Compute log partition function from natural parameters.

        Args:
            natural_params: Natural parameters (location, precision).

        Returns:
            Log partition function value.
        """
        loc, precision = self.split_location_precision(natural_params)

        covariance = self.snd_man.inverse(precision)
        mean = self.snd_man(covariance, loc)

        return 0.5 * (self.loc_man.dot(loc, mean) - self.snd_man.logdet(precision))

    @override
    def negative_entropy(self, mean_params: Array) -> Array:
        """Compute negative entropy from mean parameters.

        Args:
            mean_params: Mean parameters (mean, second moment).

        Returns:
            Negative entropy value.
        """
        mean, second_moment = self.split_mean_second_moment(mean_params)

        # Compute covariance without forming full matrices
        outer_mean = self.snd_man.outer_product(mean, mean)
        covariance = second_moment - outer_mean
        log_det = self.snd_man.logdet(covariance)

        return -0.5 * log_det - 0.5 * self.data_dim

    @override
    def sample(
        self,
        key: Array,
        natural_params: Array,
        n: int = 1,
    ) -> Array:
        """Sample from the distribution.

        Args:
            key: Random key.
            natural_params: Natural parameters.
            n: Number of samples.

        Returns:
            Samples from the distribution.
        """
        mean_params = self.to_mean(natural_params)
        mean, covariance = self.split_mean_covariance(mean_params)

        # Draw standard normal samples
        shape = (n, self.data_dim)
        z = jax.random.normal(key, shape)

        # Transform samples using Cholesky
        samples = self.snd_man.rep.cholesky_matvec(
            self.snd_man.matrix_shape, covariance, z
        )
        return mean + samples

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize means with normal and covariance matrix with random diagonal structure.

        Returns:
            Natural parameters.
        """
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
    ) -> Array:
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
            Natural parameters.
        """
        # Get average sufficient statistics (mean and second moment)
        avg_stats = self.average_sufficient_statistic(sample)
        mean, second_moment = self.split_mean_second_moment(avg_stats)

        # Add noise to mean based on observed scale
        key_mean, key_cov = jax.random.split(key)
        observed_scale = jnp.sqrt(jnp.diag(self.cov_man.to_dense(second_moment)))
        mean_noise = observed_scale * (
            location + shape * jax.random.normal(key_mean, mean.shape)
        )
        mean = mean + mean_noise

        # Add multiplicative noise to second moment
        noise = shape * jax.random.normal(key_cov, (self.data_dim, self.data_dim))
        noise = jnp.eye(self.data_dim) + noise @ noise.T / self.data_dim
        noise_matrix = self.cov_man.from_dense(noise)
        second_moment = second_moment * noise_matrix

        # Join parameters and convert to natural coordinates
        mean_params = self.join_mean_second_moment(mean, second_moment)
        return self.to_natural(mean_params)

    @override
    def check_natural_parameters(self, natural_params: Array) -> Array:
        """Check if natural parameters are valid.

        Delegates to Covariance check after extracting precision component.

        Args:
            natural_params: Natural parameters to check.

        Returns:
            Boolean array indicating validity.
        """
        _, precision = self.split_location_precision(natural_params)
        return self.cov_man.check_natural_parameters(precision)

    # Methods

    @property
    @override
    def loc_man(self) -> Euclidean:
        """Location manifold."""
        return Euclidean(self._data_dim)

    @property
    def cov_man(self) -> Covariance:
        """Covariance manifold."""
        return Covariance(self._data_dim, self.rep)

    def join_mean_covariance(
        self,
        mean: Array,
        covariance: Array,
    ) -> Array:
        """Construct mean parameters from the mean μ and covariance Σ.

        Args:
            mean: Mean vector (in mean parameters).
            covariance: Covariance matrix (in mean parameters).

        Returns:
            Combined mean parameters.
        """
        # Create the second moment η₂ = μμᵀ + Σ
        outer = self.snd_man.outer_product(mean, mean)
        second_moment = outer + covariance

        return self.join_params(mean, second_moment)

    def split_mean_covariance(
        self, mean_params: Array
    ) -> tuple[Array, Array]:
        """Extract the mean μ and covariance Σ from mean parameters.

        Args:
            mean_params: Mean parameters (mean, second moment).

        Returns:
            Tuple of (mean vector, covariance matrix) in mean parameters.
        """
        # Split into μ and η₂
        mean, second_moment = self.split_params(mean_params)

        # Compute Σ = η₂ - μμᵀ
        outer = self.snd_man.outer_product(mean, mean)
        covariance = second_moment - outer

        return mean, covariance

    @override
    def split_mean_second_moment(
        self, mean_params: Array
    ) -> tuple[Array, Array]:
        """Split parameters into mean and second-moment components.

        Args:
            mean_params: Mean parameters.

        Returns:
            Tuple of (mean vector, second moment) in mean parameters.
        """
        return self.split_params(mean_params)

    @override
    def join_mean_second_moment(
        self, mean: Array, second_moment: Array
    ) -> Array:
        """Join mean and second-moment parameters.

        Args:
            mean: Mean vector (in mean parameters).
            second_moment: Second moment (in mean parameters).

        Returns:
            Combined mean parameters.
        """
        return self.join_params(mean, second_moment)

    @override
    def split_location_precision(
        self, natural_params: Array
    ) -> tuple[Array, Array]:
        """Split natural location and precision (inverse covariance) parameters.

        There's some subtle rescaling that has to happen to ensure that the natural
        parameters behaves correctly when used either as a vector in a dot product,
        or as a precision matrix.

        For a multivariate normal distribution, the natural parameters (θ₁,θ₂) are
        related to the standard parameters (μ,Σ) by, θ₁ = Σ⁻¹μ and θ₂ = -½Σ⁻¹.
        Matrix representations require different scaling to maintain these relationships:

        1. Diagonal case:
            - No additional rescaling needed as parameters directly represent diagonal elements

        2. Full (PositiveDefinite) case:
            - Off-diagonal elements appear twice in the precision matrix but once in
              the natural parameters
            - For i ≠ j, element θ₂,ᵢⱼ is stored as double its matrix value to account
              for missing parameters in the dot product
            - When converting to precision Σ⁻¹, vector elements corresponding to
              off-diagonal elements are halved

        3. Scale case:
            - The exponential family dot product has to be scaled by 1/d
            - This scales needs to be stored in either in the sufficient statistic or
              the natural parameters
            - We store it in the sufficient statistic (hence its defined as an average),
              which requires that we divide the natural parameters by d when converting
              to precision

        Args:
            natural_params: Natural parameters.

        Returns:
            Tuple of (location, precision) in natural parameters.
        """
        # First do basic parameter split
        loc, theta2 = self.split_params(natural_params)

        # We need to rescale off-precision params
        if not isinstance(self.snd_man.rep, Diagonal):
            i_diag = (
                jnp.triu_indices(self.data_dim)[0] == jnp.triu_indices(self.data_dim)[1]
            )

            scaled_params = theta2 / 2
            scaled_params = jnp.where(i_diag, scaled_params * 2, scaled_params)
            theta2 = scaled_params

        scl = -2

        if isinstance(self.snd_man.rep, Scale):
            scl = scl / self.data_dim

        return loc, scl * theta2

    @override
    def join_location_precision(
        self,
        location: Array,
        precision: Array,
    ) -> Array:
        """Join natural location and precision (inverse covariance) parameters.

        Inverts the scaling in split_location_precision.

        Args:
            location: Location vector (in natural parameters).
            precision: Precision matrix (in natural parameters).

        Returns:
            Combined natural parameters.
        """
        scl = -0.5

        if isinstance(self.snd_man.rep, Scale):
            scl = self.data_dim * scl

        theta2 = scl * precision

        # We need to rescale off-precision params
        if not isinstance(self.snd_man.rep, Diagonal):
            i_diag = (
                jnp.triu_indices(self.data_dim)[0] == jnp.triu_indices(self.data_dim)[1]
            )

            scaled_params = theta2 * 2  # First multiply by -1/2
            scaled_params = jnp.where(i_diag, scaled_params / 2, scaled_params)
            theta2 = scaled_params

        return self.join_params(location, theta2)

    def embed_rep(
        self, trg_man: Normal, natural_params: Array
    ) -> Array:
        """Embed natural parameters into a more complex representation.

        For example, a diagonal matrix can be embedded as a full matrix with zeros
        off the diagonal.

        Args:
            trg_man: Target manifold with more complex representation.
            natural_params: Natural parameters to embed.

        Returns:
            Embedded natural parameters.
        """
        # Embedding in natural coordinates means embedding precision matrix
        loc, prs = self.split_location_precision(natural_params)
        _, trg_prs = self.cov_man.embed_rep(prs, trg_man.cov_man.rep)
        return trg_man.join_location_precision(loc, lin_to_cov(trg_prs))

    def project_rep[TargetRep: PositiveDefinite](
        self, trg_man: Normal, mean_params: Array
    ) -> Array:
        """Project mean parameters to a simpler representation.

        For example, a full matrix can be projected to a diagonal one. In Mean
        coordinates this corresponds to the information (moment matching) projection.

        Args:
            trg_man: Target manifold with simpler representation.
            mean_params: Mean parameters to project.

        Returns:
            Projected mean parameters.
        """
        mean, second_moment = self.split_mean_second_moment(mean_params)
        _, target_second = self.cov_man.project_rep(second_moment, trg_man.cov_man.rep)
        return trg_man.join_mean_second_moment(mean, lin_to_cov(target_second))

    def regularize_covariance(
        self, mean_params: Array, jitter: float = 0, min_var: float = 0
    ) -> Array:
        """Regularize covariance matrix to ensure numerical stability and reasonable variances.

        This method applies two forms of regularization to the covariance matrix:
            1. A minimum variance constraint that prevents any dimension from having
               variance below a specified threshold
            2. A jitter term that adds a small positive value to all diagonal elements,
               improving numerical stability

        The regularization preserves the correlation structure while ensuring the
        covariance matrix remains well-conditioned.

        Args:
            mean_params: Mean parameters to regularize.
            jitter: Value to add to diagonal.
            min_var: Minimum variance.

        Returns:
            Regularized mean parameters.
        """
        mean, covariance = self.split_mean_covariance(mean_params)

        def regularize_diagonal(x: Array) -> Array:
            return jnp.maximum(x + jitter, min_var)

        adjusted_covariance = self.cov_man.map_diagonal(covariance, regularize_diagonal)

        return self.join_mean_covariance(mean, adjusted_covariance)

    def whiten(
        self,
        given: Array,
        relative: Array,
    ) -> Array:
        """Whiten a normal distribution relative to another normal distribution.

        Transforms a normal distribution to have standard normal parameters by:
        1. Extracting the distribution's mean and covariance
        2. Scaling the mean: new_mean = precision^(1/2) @ (old_mean - mean)
        3. Transforming the covariance: new_cov = precision^(1/2) @ old_cov @ precision^(1/2)
        where precision^(1/2) is the inverse of the Cholesky decomposition of covariance

        Args:
            given: Mean parameters to whiten.
            relative: Mean parameters of reference distribution.

        Returns:
            Whitened mean parameters.
        """
        old_mean, old_cov = self.split_mean_covariance(given)
        rel_mean, rel_cov = self.split_mean_covariance(relative)

        new_array, new_matrix = self.cov_man.rep.cholesky_whiten(
            self.cov_man.matrix_shape,
            old_mean,
            old_cov,
            rel_mean,
            rel_cov,
        )

        return self.join_mean_covariance(new_array, new_matrix)

    def standard_normal(self) -> Array:
        """Return the standard normal distribution.

        Returns:
            Mean parameters for standard normal (zero mean, identity covariance).
        """
        return self.join_mean_covariance(
            jnp.zeros(self.data_dim),
            self.cov_man.from_dense(jnp.eye(self.data_dim)),
        )

    def statistical_mean(self, natural_params: Array) -> Array:
        """Compute the mean of the distribution.

        Args:
            natural_params: Natural parameters.

        Returns:
            Mean vector.
        """
        mean, _ = self.split_mean_covariance(self.to_mean(natural_params))
        return mean

    def statistical_covariance(self, natural_params: Array) -> Array:
        """Compute the covariance of the distribution.

        Args:
            natural_params: Natural parameters.

        Returns:
            Covariance matrix (as dense array).
        """
        _, cov = self.split_mean_covariance(self.to_mean(natural_params))
        return self.cov_man.to_dense(cov)

    # GeneralizedGaussian interface implementation

    @property
    @override
    def shp_man(self) -> Covariance:
        """Shape component: covariance manifold."""
        return self.cov_man

    @property
    @override
    def fst_man(self) -> Euclidean:
        """First component: location manifold (Euclidean space)."""
        return self.loc_man

    @property
    @override
    def snd_man(self) -> Covariance:
        """Second component: shape manifold (covariance structure)."""
        return self.cov_man
