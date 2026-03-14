"""Count distributions as exponential families: Poisson, Conway-Maxwell-Poisson (CoMPoisson), and their components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import (
    Analytic,
    Differentiable,
    ExponentialFamily,
    LocationShape,
    Product,
    TupleEmbedding,
)
from ...geometry.exponential_family.combinators import (
    AnalyticProduct,
    DifferentiableProduct,
)


@dataclass(frozen=True)
class Poisson(Analytic):
    """Standard Poisson distribution for count data with a single rate parameter $\\eta > 0$ where mean equals variance.

    As an exponential family:

    - Natural parameter: $\\theta = \\log(\\eta)$
    - Sufficient statistic: $s(x) = x$
    - Base measure: $\\mu(k) = -\\log(k!)$
    - Log-partition function: $\\psi(\\theta) = e^{\\theta}$
    - Negative entropy: $\\phi(\\eta) = \\eta\\log(\\eta) - \\eta$
    """

    @property
    @override
    def dim(self) -> int:
        return 1

    @property
    @override
    def data_dim(self) -> int:
        return 1

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        return jnp.atleast_1d(x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        k = jnp.asarray(x, dtype=jnp.float32)
        return -_log_factorial(k)

    @override
    def log_partition_function(self, params: Array) -> Array:
        return jnp.squeeze(jnp.exp(params))

    @override
    def negative_entropy(self, means: Array) -> Array:
        rate = means
        return jnp.squeeze(rate * (jnp.log(rate) - 1))

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        means = self.to_mean(params)
        return jax.random.poisson(key, means, shape=(n,))[..., None]

    @override
    def initialize_from_sample(
        self,
        key: Array,
        sample: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Array:
        """Initialize by clipping mean to a small positive value before converting to log rate."""
        avg_suff = self.average_sufficient_statistic(sample)
        clipped_mean = jnp.clip(avg_suff, 0.1, None)
        natural = jnp.log(clipped_mean)
        noise = jax.random.normal(key, shape=(self.dim,)) * shape + location
        return natural + noise

    # Methods

    def statistical_mean(self, params: Array) -> Array:
        return self.to_mean(params).reshape([1])

    def statistical_covariance(self, params: Array) -> Array:
        return self.to_mean(params).reshape([1, 1])


@dataclass(frozen=True)
class CoMShape(ExponentialFamily):
    """Shape component of a CoMPoisson distribution with sufficient statistic $\\log(x!)$, capturing deviations from standard Poisson dispersion."""

    @property
    @override
    def dim(self) -> int:
        return 1

    @property
    @override
    def data_dim(self) -> int:
        return 1

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        return jnp.atleast_1d(_log_factorial(x))

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)


@dataclass(frozen=True)
class CoMPoisson(LocationShape[Poisson, CoMShape], Differentiable):
    """Conway-Maxwell-Poisson distribution generalizing Poisson with flexible dispersion.

    $$p(x; \\mu, \\nu) = \\frac{\\mu^x}{(x!)^\\nu Z(\\mu, \\nu)}$$

    where $\\mu > 0$ is related to the mode, $\\nu > 0$ is the dispersion parameter, and $Z(\\mu, \\nu) = \\sum_{j=0}^{\\infty} \\mu^j / (j!)^\\nu$. Special cases: $\\nu = 1$ (Poisson), $\\nu < 1$ (overdispersed), $\\nu > 1$ (underdispersed), $\\nu \\to \\infty$ (Bernoulli), $\\nu = 0$ (geometric).

    As an exponential family:

    - Natural parameters: $\\theta_1 = \\nu\\log(\\mu)$, $\\theta_2 = -\\nu$
    - Sufficient statistics: $s(x) = (x, \\log(x!))$
    - Log-partition function: log of the normalizing constant $Z$
    """

    # Fields

    window_size: int = 200
    """Fixed number of terms to evaluate in series expansions."""

    # Methods

    def split_mode_dispersion(self, params: Array) -> tuple[Array, Array]:
        """Convert natural parameters to mode $\\mu$ and dispersion $\\nu$."""
        theta1, theta2 = params[0], params[1]
        nu = -theta2
        mu = jnp.exp(-theta1 / theta2)
        return mu, nu

    def join_mode_dispersion(self, mu: Array, nu: Array) -> Array:
        """Convert mode $\\mu$ and dispersion $\\nu$ to natural parameters."""
        theta1 = nu * jnp.log(mu)
        theta2 = -nu
        return jnp.array([theta1, theta2]).ravel()

    def approximate_mean_variance(self, params: Array) -> tuple[Array, Array]:
        """Approximate mean $E(X) \\approx \\mu + 1/(2\\nu) - 1/2$ and variance $\\text{Var}(X) \\approx \\mu / \\nu$."""
        mu, nu = self.split_mode_dispersion(params)
        approx_mean = mu + 1 / (2 * nu) - 0.5
        approx_var = mu / nu
        return approx_mean, approx_var

    def numerical_mean_variance(
        self,
        params: Array,
    ) -> tuple[Array, Array]:
        """Compute mean and variance using window-based numerical summation centered on the mode."""
        # Get mode for window centering
        mu, _ = self.split_mode_dispersion(params)

        # Create fixed window of indices
        mode_shift = jnp.maximum(0, jnp.floor(mu - self.window_size / 2)).astype(
            jnp.int32
        )
        indices = jnp.arange(self.window_size) + mode_shift

        # Compute log probabilities
        log_probs = jax.vmap(self.log_density, in_axes=(None, 0))(params, indices)
        probs = jnp.exp(log_probs)

        # Compute first and second moments
        mean = jnp.sum(indices * probs)
        second_moment = jnp.sum(indices**2 * probs)

        # Compute variance
        variance = second_moment - mean**2

        return mean, variance

    def statistical_mean(self, params: Array) -> Array:
        """Numerical approximation of the mean."""
        mean, _ = self.numerical_mean_variance(params)
        return mean.reshape([1])

    def statistical_covariance(self, params: Array) -> Array:
        """Numerical approximation of the covariance."""
        _, var = self.numerical_mean_variance(params)
        return var.reshape([1, 1])

    # Overrides

    @property
    @override
    def fst_man(self) -> Poisson:
        return Poisson()

    @property
    @override
    def snd_man(self) -> CoMShape:
        return CoMShape()

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.asarray(0.0)

    @override
    def log_partition_function(self, params: Array) -> Array:
        """Evaluate $\\psi(\\theta) = \\log\\sum_j \\exp(\\theta_1 j + \\theta_2 \\log(j!))$ using a fixed window centered on the mode."""
        # Estimate mode
        mu, _ = self.split_mode_dispersion(params)

        # Create fixed window of indices
        base_indices = jnp.arange(self.window_size)

        # Shift window to be centered on mode (rounded down to integer)
        mode_shift = jnp.maximum(0, jnp.floor(mu - self.window_size / 2)).astype(
            jnp.int32
        )
        indices = base_indices + mode_shift

        def _compute_log_partition_terms(index: Array) -> Array:
            return jnp.dot(params, self.sufficient_statistic(index))

        # Compute terms and use log-sum-exp for numerical stability
        log_terms = jax.vmap(_compute_log_partition_terms)(indices)
        return jax.nn.logsumexp(log_terms)

    # TODO: Come up with a better scheme than rejection sampling
    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Generate samples using Algorithm 2 from Benson & Friel (2021)."""
        mu, nu = self.split_mode_dispersion(params)
        mode = jnp.floor(mu)

        # Envelope terms for both Poisson and Geometric cases

        # Underdispersed case (nu >= 1): Poisson envelope
        log_pois_scale = (nu - 1) * (mode * jnp.log(mu) - _log_factorial(mode))

        # Overdispersed case (nu < 1): Geometric envelope
        p_geo = (2 * nu) / (2 * nu * mu + 1 + nu)
        ratio = jnp.floor(mu / ((1 - p_geo) ** (1 / nu)))
        log_geo_scale = (
            -jnp.log(p_geo)
            + nu * ratio * jnp.log(mu)
            - (ratio * jnp.log(1 - p_geo) + nu * _log_factorial(ratio))
        )

        def sample_one(key: Array) -> Array:
            def cond_fn(val: tuple[Array, Array, Array]) -> Array:
                _, _, accept = val
                return jnp.logical_not(accept)

            def body_fn(val: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
                key, y, _ = val
                key, key_prop, key_u = jax.random.split(key, 3)

                # Underdispersed case (nu >= 1): Poisson envelope
                pois_y = jax.random.poisson(key_prop, mu)
                log_alpha_pois = nu * (
                    pois_y * jnp.log(mu) - _log_factorial(pois_y)
                ) - (log_pois_scale + pois_y * jnp.log(mu) - _log_factorial(pois_y))

                # Overdispersed case (nu >= 1): Poisson envelope
                u0 = jax.random.uniform(key_prop)
                geo_y = jnp.floor(jnp.log(u0) / jnp.log(1 - p_geo))
                log_alpha_geo = nu * (geo_y * jnp.log(mu) - _log_factorial(geo_y)) - (
                    log_geo_scale + geo_y * jnp.log(1 - p_geo) + jnp.log(p_geo)
                )

                # Select proposal and ratio based on nu
                y = jnp.where(nu >= 1, pois_y, geo_y)
                log_alpha = jnp.where(nu >= 1, log_alpha_pois, log_alpha_geo)

                # Accept/reject step
                u = jax.random.uniform(key_u)
                accept = jnp.asarray(jnp.log(u) <= log_alpha).reshape(())

                return key, y.reshape(()), accept

            init_val = (key, jnp.array(0), jnp.asarray(False).reshape(()))
            _, sample, _ = jax.lax.while_loop(cond_fn, body_fn, init_val)
            return sample

        keys = jax.random.split(key, n)
        samples = jax.vmap(sample_one)(keys)
        return samples[..., None]

    @override
    def check_natural_parameters(self, params: Array) -> Array:
        """Check that $\\theta_1$ is finite and $\\theta_2 < 0$."""
        finite = super().check_natural_parameters(params)
        theta2_valid = params[1] < 0
        return finite & theta2_valid

    @override
    def initialize(
        self,
        key: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Array:
        key_mu, key_nu = jax.random.split(key)

        # Ensure mu stays positive by using exp
        mu_init = jnp.exp(jax.random.normal(key_mu) * shape + location)

        # Keep nu in a reasonable range
        nu_init = 1.0 + jnp.abs(jax.random.normal(key_nu)) * shape

        return self.join_mode_dispersion(mu_init, nu_init)

    @override
    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize by estimating mode and dispersion via method of moments from sample mean and variance."""
        # Compute sample statistics
        mean = jnp.mean(sample)
        var = jnp.var(sample)

        # Add noise for regularization
        noise = jax.random.normal(key, shape=(2,)) * shape + location
        mean = mean + noise[0]
        var = var + noise[1]

        a = var
        b = -(mean + 0.5)
        c = 0.5

        nu = (-b + jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)

        mu = var * nu

        # Convert to natural parameters
        return self.join_mode_dispersion(mu, nu)


class Poissons(AnalyticProduct[Poisson]):
    """Product of $n$ independent Poisson distributions for modeling unbounded count data."""

    def __init__(self, n_neurons: int):
        super().__init__(Poisson(), n_neurons)

    @property
    def n_neurons(self) -> int:
        """Number of Poisson units."""
        return self.n_reps


type PopulationShape = Product[CoMShape]


@dataclass(frozen=True)
class CoMPoissons(
    DifferentiableProduct[CoMPoisson], LocationShape[Poissons, PopulationShape]
):
    """A population of independent Conway-Maxwell-Poisson (COM-Poisson) units."""

    def __init__(self, n_reps: int):
        super().__init__(CoMPoisson(), n_reps)

    @property
    @override
    def fst_man(self) -> Poissons:
        return Poissons(self.n_reps)

    @property
    @override
    def snd_man(self) -> PopulationShape:
        return Product(CoMShape(), n_reps=self.n_reps)

    @override
    def join_coords(self, fst_coords: Array, snd_coords: Array) -> Array:
        params_matrix = jnp.stack([fst_coords, snd_coords])
        return params_matrix.T.reshape(-1)

    @override
    def split_coords(self, coords: Array) -> tuple[Array, Array]:
        matrix = coords.reshape(self.n_reps, 2).T
        return matrix[0], matrix[1]


@dataclass(frozen=True)
class PopulationLocationEmbedding(
    TupleEmbedding[Poissons, CoMPoissons]
):
    """Embedding that projects COM-Poisson to Poisson via location parameters."""

    n_neurons: int

    @property
    @override
    def tup_idx(self) -> int:
        return 0

    @property
    @override
    def amb_man(self) -> CoMPoissons:
        return CoMPoissons(n_reps=self.n_neurons)

    @property
    @override
    def sub_man(self) -> Poissons:
        return Poissons(self.n_neurons)


def _log_factorial(k: Array) -> Array:
    return jax.lax.lgamma(k.astype(float) + 1)
