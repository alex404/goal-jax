"""Multivariate normal distributions as exponential families with flexible covariance structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ....geometry import (
    Analytic,
    Diagonal,
    ExponentialFamily,
    Identity,
    LocationShape,
    PositiveDefinite,
    Scale,
    SquareMap,
)
from .generalized import Euclidean, GeneralizedGaussian


class Covariance[Rep: PositiveDefinite](SquareMap[Euclidean], ExponentialFamily):
    """Shape component of a Normal distribution.

    This represents the covariance structure of a Normal distribution through different matrix representations:
        - `PositiveDefinite`: Full covariance matrix
        - `Diagonal`: diagonal elements
        - `Scale`: Scalar multiple of identity
        - `Identity`: Unit covariance

    As an exponential family:
        - Sufficient statistic: Outer product $s(x) = x \\otimes x$
        - Base measure: $\\mu(x) = -\\frac{n}{2}\\log(2\\pi)$
    """

    # Fields

    rep: Rep

    def __init__(self, data_dim: int, rep: Rep):
        super().__init__(rep, Euclidean(data_dim))

    # Overrides

    @property
    @override
    def data_dim(self) -> int:
        return self.matrix_shape[0]

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Outer product with appropriate covariance structure."""
        return self.outer_product(x, x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return -0.5 * self.data_dim * jnp.log(2 * jnp.pi)

    @override
    def check_natural_parameters(self, params: Array) -> Array:
        """Check finiteness and positive definiteness of precision matrix."""
        finite = super().check_natural_parameters(params)
        is_pd = self.is_positive_definite(params)
        return finite & is_pd

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize with random perturbation from identity via low-rank $I + LL^T$."""
        noise_scale = shape / jnp.sqrt(self.data_dim)
        big_l = noise_scale * jax.random.normal(key, (self.data_dim, self.data_dim))
        base_cov = jnp.eye(self.data_dim) + big_l @ big_l.T
        return self.from_matrix(base_cov)

    # Methods

    def add_jitter(self, params: Array, epsilon: float) -> Array:
        """Add epsilon to diagonal elements while preserving matrix structure."""
        return self.map_diagonal(params, lambda x: x + epsilon)


@dataclass(frozen=True)
class Normal[Rep: PositiveDefinite](
    GeneralizedGaussian[Euclidean, Covariance[Rep]],
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

    rep: Rep
    """Covariance representation type."""

    # Overrides

    @property
    @override
    def loc_man(self) -> Euclidean:
        """Location manifold."""
        return Euclidean(self._data_dim)

    @property
    @override
    def shp_man(self) -> Covariance[Rep]:
        """Shape component: covariance manifold."""
        return self.cov_man

    @property
    @override
    def fst_man(self) -> Euclidean:
        """First component: location manifold."""
        return self.loc_man

    @property
    @override
    def snd_man(self) -> Covariance[Rep]:
        """Second component: covariance manifold."""
        return self.cov_man

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Compute sufficient statistic $(x, x \\otimes x)$."""
        x = jnp.atleast_1d(x)
        second_moment = self.snd_man.outer_product(x, x)
        return self.join_coords(x, second_moment)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return -0.5 * self.data_dim * jnp.log(2 * jnp.pi)

    @override
    def log_partition_function(self, params: Array) -> Array:
        """Compute log partition function from natural parameters."""
        loc, precision = self.split_location_precision(params)
        covariance = self.snd_man.inverse(precision)
        mean = self.snd_man(covariance, loc)
        return 0.5 * (jnp.dot(loc, mean) - self.snd_man.logdet(precision))

    @override
    def negative_entropy(self, means: Array) -> Array:
        """Compute negative entropy from mean parameters."""
        mean, second_moment = self.split_mean_second_moment(means)
        outer_mean = self.snd_man.outer_product(mean, mean)
        covariance = second_moment - outer_mean
        log_det = self.snd_man.logdet(covariance)
        return -0.5 * log_det - 0.5 * self.data_dim

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Sample from the normal distribution at the given natural parameters."""
        means = self.to_mean(params)
        mean, covariance = self.split_mean_covariance(means)
        shape = (n, self.data_dim)
        z = jax.random.normal(key, shape)
        samples = self.snd_man.rep.cholesky_matvec(
            self.snd_man.matrix_shape, covariance, z
        )
        return mean + samples

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize with random mean and perturbed identity covariance."""
        mean = self.loc_man.initialize(key, location, shape)
        cov = self.cov_man.initialize(key, location=location, shape=shape)
        return self.join_location_precision(mean, cov)

    @override
    def initialize_from_sample(
        self,
        key: Array,
        sample: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Array:
        """Initialize from sample data with regularizing noise scaled relative to observed variance."""
        # Get average sufficient statistics (mean and second moment)
        avg_stats = self.average_sufficient_statistic(sample)
        mean, second_moment = self.split_mean_second_moment(avg_stats)

        # Compute variance for proper scaling: Var(x) = E[x²] - E[x]²
        second_moment_diag = self.cov_man.get_diagonal(second_moment)
        variance_diag = jnp.maximum(second_moment_diag - mean**2, 1e-6)
        observed_scale = jnp.sqrt(variance_diag)

        # Add noise to mean based on observed standard deviation
        key_mean, key_cov = jax.random.split(key)
        mean_noise = observed_scale * (
            location + shape * jax.random.normal(key_mean, mean.shape)
        )
        mean = mean + mean_noise

        # Add multiplicative noise to second moment
        noise = shape * jax.random.normal(key_cov, (self.data_dim, self.data_dim))
        noise = jnp.eye(self.data_dim) + noise @ noise.T / self.data_dim
        noise_matrix = self.cov_man.from_matrix(noise)
        second_moment = second_moment * noise_matrix

        # Ensure second_moment > mean² to maintain positive variance
        min_second_moment = self.cov_man.from_matrix(jnp.diag(mean**2 + 1e-6))
        second_moment = jnp.maximum(second_moment, min_second_moment)

        # Join parameters and convert to natural coordinates
        means = self.join_mean_second_moment(mean, second_moment)
        return self.to_natural(means)

    @override
    def check_natural_parameters(self, params: Array) -> Array:
        """Delegate to Covariance check after extracting precision component."""
        _, precision = self.split_location_precision(params)
        return self.cov_man.check_natural_parameters(precision)

    @override
    def split_location_precision(self, params: Array) -> tuple[Array, Array]:
        """Split natural parameters into location and precision components.

        There's some subtle rescaling that has to happen to ensure that the natural
        parameters behaves correctly when used either as a vector in a dot product,
        or as a precision matrix.

        For a multivariate normal distribution, the natural parameters $(\\theta_1, \\theta_2)$ are
        related to the standard parameters $(\\mu, \\Sigma)$ by $\\theta_1 = \\Sigma^{-1} \\mu$ and
        $\\theta_2 = -1/2 \\Sigma^{-1}$. Matrix representations require different scaling to
        maintain these relationships:

        1. Diagonal case: No additional rescaling needed as parameters directly represent diagonal elements
        2. Full (PositiveDefinite) case: Off-diagonal elements appear twice in the precision matrix but once in the natural parameters. For $i \\neq j$, element $\\theta_{2,ij}$ is stored as double its matrix value. When converting to precision, off-diagonal elements are halved.
        3. Scale case: The exponential family dot product must be scaled by $1/d$, stored in the sufficient statistic (hence defined as an average), requiring division of natural parameters by $d$ when converting to precision.
        """
        # First do basic parameter split
        loc, theta2 = self.split_coords(params)
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
    def join_location_precision(self, location: Array, precision: Array) -> Array:
        """Inverse of split_location_precision."""
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

        return self.join_coords(location, theta2)

    @override
    def split_mean_second_moment(self, means: Array) -> tuple[Array, Array]:
        """Split mean parameters into mean and second-moment components."""
        return self.split_coords(means)

    @override
    def join_mean_second_moment(self, mean: Array, second_moment: Array) -> Array:
        """Join mean and second-moment components into mean parameters."""
        return self.join_coords(mean, second_moment)

    # Methods

    @property
    def cov_man(self) -> Covariance[Rep]:
        """Covariance manifold."""
        return Covariance(self._data_dim, self.rep)

    def join_mean_covariance(self, mean: Array, covariance: Array) -> Array:
        """Construct mean parameters from $\\mu$ and $\\Sigma$ by computing $\\eta_2 = \\mu\\mu^T + \\Sigma$."""
        outer = self.snd_man.outer_product(mean, mean)
        second_moment = outer + covariance
        return self.join_coords(mean, second_moment)

    def split_mean_covariance(self, means: Array) -> tuple[Array, Array]:
        """Extract $\\mu$ and $\\Sigma = \\eta_2 - \\mu\\mu^T$ from mean parameters."""
        mean, second_moment = self.split_coords(means)
        outer = self.snd_man.outer_product(mean, mean)
        covariance = second_moment - outer
        return mean, covariance

    def embed_rep[TargetRep: PositiveDefinite](
        self, trg_man: Normal[TargetRep], params: Array
    ) -> Array:
        """Embed natural parameters into a more complex representation (e.g., diagonal to full)."""
        loc, prs = self.split_location_precision(params)
        _, trg_prs = self.cov_man.embed_rep(prs, trg_man.cov_man.rep)
        return trg_man.join_location_precision(loc, trg_prs)

    def project_rep[TargetRep: PositiveDefinite](
        self, trg_man: Normal[TargetRep], means: Array
    ) -> Array:
        """Project mean parameters to a simpler representation (moment matching projection)."""
        mean, second_moment = self.split_mean_second_moment(means)
        _, target_second = self.cov_man.project_rep(second_moment, trg_man.cov_man.rep)
        return trg_man.join_mean_second_moment(mean, target_second)

    def regularize_covariance(
        self, means: Array, jitter: float = 0, min_var: float = 0
    ) -> Array:
        """Apply minimum variance constraint and diagonal jitter to the covariance in mean parameters."""
        mean, covariance = self.split_mean_covariance(means)

        def regularize_diagonal(x: Array) -> Array:
            return jnp.maximum(x + jitter, min_var)

        adjusted_covariance = self.cov_man.map_diagonal(covariance, regularize_diagonal)

        return self.join_mean_covariance(mean, adjusted_covariance)

    def relative_whiten(
        self,
        given_means: Array,
        relative_means: Array,
    ) -> Array:
        """Whiten mean parameters of one normal distribution relative to another using $\\Sigma_{\\mathrm{rel}}^{-1/2}$."""
        old_mean, old_cov = self.split_mean_covariance(given_means)
        rel_mean, rel_cov = self.split_mean_covariance(relative_means)

        new_array, new_matrix = self.cov_man.rep.cholesky_whiten(
            self.cov_man.matrix_shape,
            old_mean,
            old_cov,
            rel_mean,
            rel_cov,
        )

        return self.join_mean_covariance(new_array, new_matrix)

    def standard_normal(self) -> Array:
        """Return mean parameters for the standard normal (zero mean, identity covariance)."""
        return self.join_mean_covariance(
            jnp.zeros(self.data_dim),
            self.cov_man.from_matrix(jnp.eye(self.data_dim)),
        )

    def statistical_mean(self, params: Array) -> Array:
        """Extract the mean vector from natural parameters."""
        mean, _ = self.split_mean_covariance(self.to_mean(params))
        return mean

    def statistical_covariance(self, params: Array) -> Array:
        """Extract the covariance matrix (as dense array) from natural parameters."""
        _, cov = self.split_mean_covariance(self.to_mean(params))
        return self.cov_man.to_matrix(cov)


# Type aliases for common Normal specializations

type FullNormal = Normal[PositiveDefinite]
"""Normal distribution with full covariance matrix."""

type DiagonalNormal = Normal[Diagonal]
"""Normal distribution with diagonal covariance (independent dimensions)."""

type IsotropicNormal = Normal[Scale]
"""Normal distribution with isotropic covariance (scalar multiple of identity)."""

type StandardNormal = Normal[Identity]
"""Normal distribution with identity covariance (unit variance, independent dimensions)."""


# Factory functions for convenient construction


def full_normal(data_dim: int) -> FullNormal:
    """Create a Normal with full covariance matrix."""
    return Normal(data_dim, PositiveDefinite())


def diagonal_normal(data_dim: int) -> DiagonalNormal:
    """Create a Normal with diagonal covariance."""
    return Normal(data_dim, Diagonal())


def isotropic_normal(data_dim: int) -> IsotropicNormal:
    """Create a Normal with isotropic covariance."""
    return Normal(data_dim, Scale())


def standard_normal(data_dim: int) -> StandardNormal:
    """Create a Normal with identity covariance."""
    return Normal(data_dim, Identity())
