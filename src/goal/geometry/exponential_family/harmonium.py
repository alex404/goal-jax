"""Core definitions for harmonium models.

A harmonium is a type of probabilistic graphical model over observable and latent variables with exponential family densities of the form

$$p(x,z) \\propto \\exp(\\theta_X \\cdot \\mathbf{s}_X(x) + \\theta_Z \\cdot \\mathbf{s}_Z(z) + \\mathbf{s}_X(x) \\cdot \\Theta_{XZ} \\cdot \\mathbf{s}_Z(z)).$$

Data points of a harmonium are represented by the concatenation of observable and latent variables $x$ and $z$:

```python
# Observable data point from 2D observable space (e.g. points in R²)
x = jnp.array([1.2, -0.5])  # shape: (2,)
# Latent data point from a 3D latent space (e.g. 3 binary variables)
z = jnp.array([1.0, 0.0, 1.0])  # shape: (3,)
# harmonium data point
xz = jnp.concatenate([x, z])  # shape: (5,)
```

The module provides a hierarchy of harmonium models with increasing structure:

- `Harmonium`: Core exponential family structure with observable and latent variables.
- `DifferentiableLatent`: Supports differentiation of latent variables.
- `Conjugated`: Enables analytical computation of marginals $p(x)$ and $p(z)$ of $p(x, z)$.
- `GenerativeConjugated`: Harmoniums that can be sampled from.
- `DifferentiableConjugated`: Harmoniums with an analytical log-partition function.
- `AnalyticConjugated`: Harmoniums with an analytic negative entropy.

#### Class Hierarchy

![Class Hierarchy](harmonium.svg)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import Point
from ..manifold.combinators import Triple
from ..manifold.linear import AffineMap, LinearMap
from ..manifold.matrix import MatrixRep
from ..manifold.subspace import Subspace
from ..manifold.util import batched_mean
from .base import (
    Analytic,
    Differentiable,
    ExponentialFamily,
    Generative,
    Mean,
    Natural,
)


@dataclass(frozen=True)
class Harmonium[
    Rep: MatrixRep,
    Observable: ExponentialFamily,
    SubObservable: ExponentialFamily,
    SubLatent: ExponentialFamily,
    Latent: ExponentialFamily,
](
    ExponentialFamily,
    Triple[Observable, LinearMap[Rep, SubLatent, SubObservable], Latent],
    ABC,
):
    """An exponential family harmonium is a product of two exponential families.

    The joint distribution of a harmonium takes the form

    $$
    p(x,z) \\propto e^{\\theta_X \\cdot \\mathbf s_X(x) + \\theta_Z \\cdot \\mathbf s_Z(z) + \\mathbf s_X(x)\\cdot \\Theta_{XZ} \\cdot \\mathbf s_Z(z)},
    $$

    where:

    - $\\theta_X$ are the observable biases,
    - $\\theta_Z$ are the latent biases,
    - $\\Theta_{XZ}$ is the interaction matrix, and
    - $\\mathbf s_X(x)$ and $\\mathbf s_Z(z)$ are sufficient statistics of the observable and latent exponential families, respectively.
    """

    # Contract

    @property
    @abstractmethod
    def int_rep(self) -> Rep:
        """Interactive subspace of observable sufficient statistics."""

    @property
    @abstractmethod
    def obs_sub(self) -> Subspace[Observable, SubObservable]:
        """Interactive subspace of observable sufficient statistics."""

    @property
    @abstractmethod
    def lat_sub(self) -> Subspace[Latent, SubLatent]:
        """Interactive subspace of latent sufficient statistics."""

    # Overrides

    @property
    @override
    def fst_man(self) -> Observable:
        return self.obs_man

    @property
    @override
    def snd_man(self) -> LinearMap[Rep, SubLatent, SubObservable]:
        return self.int_man

    @property
    @override
    def trd_man(self) -> Latent:
        return self.lat_man

    @property
    @override
    def data_dim(self) -> int:
        """Total dimension of data points."""
        return self.obs_man.data_dim + self.lat_man.data_dim

    @override
    def sufficient_statistic(self, x: Array) -> Point[Mean, Self]:
        """Compute sufficient statistics for joint observation.

        Args:
            x: Array of shape (obs_dim + lat_dim) containing concatenated observable and latent states

        Returns:
            Array of sufficient statistics
        """
        obs_x = x[: self.obs_man.data_dim]
        lat_x = x[self.obs_man.data_dim :]

        obs_stats = self.obs_man.sufficient_statistic(obs_x)
        lat_stats = self.lat_man.sufficient_statistic(lat_x)
        int_stats = self.int_man.outer_product(
            self.obs_sub.project(obs_stats), self.lat_sub.project(lat_stats)
        )

        return self.join_params(obs_stats, int_stats, lat_stats)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Compute log base measure for joint observation.

        Args:
            x: Array of shape (obs_dim + lat_dim) containing concatenated observable and latent states

        Returns:
            Log base measure at x
        """
        obs_x = x[..., self.obs_man.data_dim :]
        lat_x = x[self.obs_man.data_dim :]

        return self.obs_man.log_base_measure(obs_x) + self.lat_man.log_base_measure(
            lat_x
        )

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Point[Natural, Self]:
        """Initialize harmonium parameters.

        Observable and latent biases are initialized using their respective
        initialization strategies. The interaction matrix is initialized with
        random entries scaled by 1/sqrt(dim) to maintain reasonable magnitudes
        for the interactions.

        Args:
            key: Random key
            location: Location parameter for bias initialization
            shape: Shape parameter for random perturbations
        """
        keys = jax.random.split(key, 3)

        # Initialize biases using component initialization
        obs_params = self.obs_man.initialize(keys[0], location, shape)
        lat_params = self.lat_man.initialize(keys[1], location, shape)

        # Initialize interaction matrix with appropriate scaling
        obs_dim = self.obs_sub.sub_man.dim
        lat_dim = self.lat_sub.sub_man.dim
        scaling = shape / jnp.sqrt(obs_dim * lat_dim)

        noise = scaling * jax.random.normal(keys[2], shape=(obs_dim, lat_dim))
        int_params: Point[Natural, LinearMap[Rep, SubLatent, SubObservable]] = (
            self.int_man.point(self.int_man.rep.from_dense(noise))
        )

        return self.join_params(obs_params, int_params, lat_params)

    @override
    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Point[Natural, Self]:
        """Initialize harmonium using sample data for observable biases.

        Uses sample data to initialize observable biases, while latent biases and interaction matrix use standard initialization.

        Args:
            key: Random key
            sample: Sample data for observable initialization
            location: Location parameter for random initialization
            shape: Shape parameter for random initialization
        """
        keys = jax.random.split(key, 3)

        # Initialize observable biases from data
        obs_params = self.obs_man.initialize_from_sample(
            keys[0], sample, location, shape
        )

        # Use standard initialization for everything else
        lat_params = self.lat_man.initialize(keys[1], location, shape)

        # Initialize interaction matrix with appropriate scaling
        obs_dim = self.obs_sub.sub_man.dim
        lat_dim = self.lat_sub.sub_man.dim
        scaling = shape / jnp.sqrt(obs_dim * lat_dim)

        noise = scaling * jax.random.normal(keys[2], shape=(obs_dim, lat_dim))
        int_params: Point[Natural, LinearMap[Rep, SubLatent, SubObservable]] = (
            self.int_man.point(self.int_man.rep.from_dense(noise))
        )

        return self.join_params(obs_params, int_params, lat_params)

    # Templates

    @property
    def obs_man(self) -> Observable:
        """Manifold of observable biases."""
        return self.obs_sub.sup_man

    @property
    def lat_man(self) -> Latent:
        """Manifold of latent biases."""
        return self.lat_sub.sup_man

    @property
    def int_man(self) -> LinearMap[Rep, SubLatent, SubObservable]:
        """Manifold of interaction matrices."""
        return LinearMap(self.int_rep, self.lat_sub.sub_man, self.obs_sub.sub_man)

    @property
    def lkl_man(self) -> AffineMap[Rep, SubLatent, SubObservable, Observable]:
        """Manifold of conditional posterior distributions $p(z \\mid x)$."""
        return AffineMap(self.int_man.rep, self.lat_sub.sub_man, self.obs_sub)

    @property
    def pst_man(self) -> AffineMap[Rep, SubObservable, SubLatent, Latent]:
        """Manifold of conditional posterior distributions $p(z \\mid x)$."""
        return AffineMap(self.int_man.rep, self.obs_sub.sub_man, self.lat_sub)

    def likelihood_function(
        self, params: Point[Natural, Self]
    ) -> Point[Natural, AffineMap[Rep, SubLatent, SubObservable, Observable]]:
        """Natural parameters of the likelihood distribution as an affine function.

        The affine map is $\\eta \\to \\theta_X + \\Theta_{XZ} \\cdot \\eta$.
        """
        obs_params, int_params, _ = self.split_params(params)
        return self.lkl_man.join_params(obs_params, int_params)

    def posterior_function(
        self, params: Point[Natural, Self]
    ) -> Point[Natural, AffineMap[Rep, SubObservable, SubLatent, Latent]]:
        """Natural parameters of the posterior distribution as an affine function.

        The affine map is $\\eta \\to \\theta_Z + \\eta \\cdot \\Theta_{XZ}$.
        """
        _, int_params, lat_params = self.split_params(params)
        int_mat_t = self.int_man.transpose(int_params)
        return self.pst_man.join_params(lat_params, int_mat_t)

    def likelihood_at(
        self, params: Point[Natural, Self], z: Array
    ) -> Point[Natural, Observable]:
        """Compute natural parameters of likelihood distribution $p(x \\mid z)$ at a given $z$.

        Given a latent state $z$ with sufficient statistics $s(z)$, computes natural parameters $\\theta_X$ of the conditional distribution:

        $$p(x \\mid z) \\propto \\exp(\\theta_X \\cdot s_X(x) + s_X(x) \\cdot \\Theta_{XZ} \\cdot s_Z(z))$$
        """
        mz = self.lat_sub.sub_man.sufficient_statistic(z)
        return self.lkl_man(self.likelihood_function(params), mz)

    def posterior_at(
        self, params: Point[Natural, Self], x: Array
    ) -> Point[Natural, Latent]:
        """Compute natural parameters of posterior distribution $p(z \\mid x)$.

        Given an observation $x$ with sufficient statistics $s(x)$, computes natural
        parameters $\\theta_Z$ of the conditional distribution:

        $$p(z \\mid x) \\propto \\exp(\\theta_Z \\cdot s_Z(z) + s_X(x) \\cdot \\Theta_{XZ} \\cdot s_Z(z))$$
        """
        mx = self.obs_sub.sub_man.sufficient_statistic(x)
        return self.pst_man(self.posterior_function(params), mx)


class Conjugated[
    Rep: MatrixRep,
    Observable: ExponentialFamily,
    SubObservable: ExponentialFamily,
    SubLatent: ExponentialFamily,
    Latent: ExponentialFamily,
](Harmonium[Rep, Observable, SubObservable, SubLatent, Latent], ABC):
    """A harmonium with for which the prior $p(z)$ is conjugate to the posterior $p(x \\mid z)$."""

    # Contract

    @abstractmethod
    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rep, SubLatent, SubObservable, Observable]
        ],
    ) -> tuple[Array, Point[Natural, Latent]]:
        """Compute conjugation parameters for the harmonium."""

    # Templates

    def prior(self, params: Point[Natural, Self]) -> Point[Natural, Latent]:
        """Natural parameters of the prior distribution $p(z)$."""
        obs_params, int_params, lat_params = self.split_params(params)
        lkl_params = self.lkl_man.join_params(obs_params, int_params)
        _, rho = self.conjugation_parameters(lkl_params)
        return lat_params + rho

    def split_conjugated(
        self, params: Point[Natural, Self]
    ) -> tuple[
        Point[Natural, AffineMap[Rep, SubLatent, SubObservable, Observable]],
        Point[Natural, Latent],
    ]:
        """Split conjugated harmonium into likelihood and prior."""
        obs_params, int_params, lat_params = self.split_params(params)
        lkl_params = self.lkl_man.join_params(obs_params, int_params)
        _, rho = self.conjugation_parameters(lkl_params)
        return lkl_params, lat_params + rho

    def join_conjugated(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rep, SubLatent, SubObservable, Observable]
        ],
        prior_params: Point[Natural, Latent],
    ) -> Point[Natural, Self]:
        """Join likelihood and prior parameters into a conjugated harmonium."""
        _, rho = self.conjugation_parameters(lkl_params)
        lat_params = prior_params - rho
        obs_params, int_params = self.lkl_man.split_params(lkl_params)
        return self.join_params(obs_params, int_params, lat_params)


# TODO: Add stochastic expectation step
class GenerativeConjugated[
    Rep: MatrixRep,
    Observable: Generative,
    SubObservable: ExponentialFamily,
    SubLatent: ExponentialFamily,
    Latent: Generative,
](Conjugated[Rep, Observable, SubObservable, SubLatent, Latent], Generative, ABC):
    """A conjugated harmonium that supports sampling.

    Sampling from a conjugated harmonium reduces to

    1. Sampling from the marginal latent distribution $p(z)$, and then
    2. Sampling from the conditional likelihood $p(x \\mid z)$.
    """

    # Overrides

    @override
    def sample(self, key: Array, params: Point[Natural, Self], n: int = 1) -> Array:
        """Generate samples from the harmonium distribution.

        Args:
            key: PRNG key for sampling
            params: Parameters in natural coordinates
            n: Number of samples to generate

        Returns:
            Array of shape (n, data_dim) containing concatenated observable and latent states
        """
        # Split up the sampling key
        key1, key2 = jax.random.split(key)

        # Sample from adjusted latent distribution p(z)
        nat_prior = self.prior(params)
        z_sample = self.lat_man.sample(key1, nat_prior, n)

        # Vectorize sampling from conditional distributions
        x_params = jax.vmap(self.likelihood_at, in_axes=(None, 0))(params, z_sample)

        # Sample from conditionals p(x|z) in parallel
        x_sample = jax.vmap(self.obs_man.sample, in_axes=(0, 0, None))(
            jax.random.split(key2, n), x_params, 1
        ).reshape((n, -1))

        # Concatenate samples along data dimension
        return jnp.concatenate([x_sample, z_sample], axis=-1)

    # Templates

    def observable_sample(
        self, key: Array, params: Point[Natural, Self], n: int = 1
    ) -> Array:
        xzs = self.sample(key, params, n)
        return xzs[:, : self.obs_man.data_dim]


class DifferentiableLatent[
    Rep: MatrixRep,
    Observable: Generative,
    SubObservable: ExponentialFamily,
    SubLatent: ExponentialFamily,
    Latent: Differentiable,
](GenerativeConjugated[Rep, Observable, SubObservable, SubLatent, Latent], ABC):
    """A harmonium with differentiable latent exponential family."""

    # Templates

    def infer_missing_expectations(
        self,
        params: Point[Natural, Self],
        x: Array,
    ) -> Point[Mean, Self]:
        """Compute joint expectations for a single observation.

        For an observation x and current parameters $\\theta = $(\\theta_X, \\theta_Z, \\theta_{XZ})$, we

        1. Compute sufficient statistics $\\mathbf s_X(x)$,
        2. Get natural parameters of $p(z \\mid x)$ given by $\\theta_Z + \\mathbf s_X(x) \\cdot \\Theta_{XZ}$,
        3. Convert to mean parameters $\\mathbb E[\\mathbf s_Z(z) \\mid x]$, and
        4. Form joint expectations $(\\mathbf s_X(x), \\mathbb E[\\mathbf s_Z(z) \\mid x], \\mathbf s_X(x) \\otimes \\mathbb E[\\mathbf s_Z(z) \\mid x])$.

        Args:
            x: Single observation of shape (obs_dim,)
            params: Current harmonium parameters in natural coordinates

        Returns:
            Joint expectations in mean coordinates for this observation
        """
        # Get sufficient statistics of observation
        obs_stats: Point[Mean, Observable] = self.obs_man.sufficient_statistic(x)

        # Get posterior parameters for this observation
        post_map = self.posterior_function(params)
        lat_params: Point[Natural, Latent] = self.pst_man(
            post_map, self.obs_sub.project(obs_stats)
        )

        # Convert to mean parameters (expected sufficient statistics)
        lat_means: Point[Mean, Latent] = self.lat_man.to_mean(lat_params)

        # Form interaction term via outer product
        int_means: Point[Mean, LinearMap[Rep, SubLatent, SubObservable]] = (
            self.int_man.outer_product(
                self.obs_sub.project(obs_stats), self.lat_sub.project(lat_means)
            )
        )

        # Join parameters into harmonium point
        return self.join_params(obs_stats, int_means, lat_means)

    def expectation_step(
        self: Self,
        params: Point[Natural, Self],
        xs: Array,
        batch_size: int = 256,
    ) -> Point[Mean, Self]:
        """Compute average joint expectations over a batch of observations.

        Args:
            params: Current harmonium parameters in natural coordinates
            xs: Batch of observations of shape (batch_size, obs_dim)

        Returns:
            Average joint expectations in mean coordinates
        """

        def _infer_missing_expectations(x: Array) -> Array:
            return self.infer_missing_expectations(params, x).array

        return self.mean_point(
            batched_mean(_infer_missing_expectations, xs, batch_size)
        )


class DifferentiableConjugated[
    Rep: MatrixRep,
    Observable: Generative,
    SubObservable: ExponentialFamily,
    SubLatent: ExponentialFamily,
    Latent: Differentiable,
](
    DifferentiableLatent[Rep, Observable, SubObservable, SubLatent, Latent],
    Differentiable,
    ABC,
):
    """A conjugated harmonium with an analytical log-partition function."""

    # Overrides

    @override
    def log_partition_function(self, params: Point[Natural, Self]) -> Array:
        """Compute the log partition function using conjugation parameters.

        The log partition function can be written as:

        $$\\psi(\\theta) = \\psi_Z(\\theta_Z + \\rho) + \\chi$$

        where $\\psi_Z$ is the log partition function of the latent exponential family.
        """
        # Split parameters
        obs_params, int_params, lat_params = self.split_params(params)
        lkl_params = self.lkl_man.join_params(obs_params, int_params)

        # Get conjugation parameters
        chi, rho = self.conjugation_parameters(lkl_params)

        # Compute adjusted latent parameters
        adjusted_lat = lat_params + rho

        # Use latent family's partition function
        return self.lat_man.log_partition_function(adjusted_lat) + chi

    # Templates
    def log_observable_density(self, params: Point[Natural, Self], x: Array) -> Array:
        """Compute log density of the observable distribution $p(x)$.

        For a conjugated harmonium, the observable density can be computed as:

        $$
        \\log p(x) = \\theta_X \\cdot s_X(x) + \\psi_Z(\\theta_Z + \\mathbf s_X(x) \\cdot \\Theta_{XZ}) - \\psi_Z(\\theta_Z + \\rho) - \\chi + \\log \\mu_X(x)
        $$

        where:

        - $(\\chi, \\rho)$ are the conjugation parameters,
        - $\\psi_Z$ is the latent log-partition function, and
        - $\\mu_X$ is the observable base measure.

        Args:
            params: Parameters in natural coordinates
            x: Observable data point

        Returns:
            Log density at x
        """
        obs_params, int_params, lat_bias = self.split_params(params)
        lkl_params = self.lkl_man.join_params(obs_params, int_params)

        chi, rho = self.conjugation_parameters(lkl_params)
        obs_stats = self.obs_man.sufficient_statistic(x)

        log_density = self.obs_man.dot(obs_params, obs_stats)
        log_density += self.lat_man.log_partition_function(self.posterior_at(params, x))
        log_density -= self.lat_man.log_partition_function(lat_bias + rho) + chi

        return log_density + self.obs_man.log_base_measure(x)

    def average_log_observable_density(
        self, params: Point[Natural, Self], xs: Array, batch_size: int = 2048
    ) -> Array:
        """Compute average log density over a batch of observations."""

        def _log_density(x: Array) -> Array:
            return self.log_observable_density(params, x)

        return batched_mean(_log_density, xs, batch_size)

    def observable_density(self, params: Point[Natural, Self], x: Array) -> Array:
        """Compute density of the observable distribution $p(x)$."""
        return jnp.exp(self.log_observable_density(params, x))


class AnalyticConjugated[
    Rep: MatrixRep,
    Observable: Generative,
    SubObservable: ExponentialFamily,
    SubLatent: ExponentialFamily,
    Latent: Analytic,
](
    DifferentiableConjugated[Rep, Observable, SubObservable, SubLatent, Latent],
    Analytic,
    ABC,
):
    """A conjugated harmonium with an analytically tractable negative entropy.

    AnalyticConjugated harmoniums have:

    1. Conjugate structure (analytical marginals),
    2. Differentiable structure (analytical log partition function), and
    3. Analytic structure (analytical entropy).
    """

    # Contract

    @abstractmethod
    def to_natural_likelihood(
        self, params: Point[Mean, Self]
    ) -> Point[Natural, AffineMap[Rep, SubLatent, SubObservable, Observable]]:
        """Given a harmonium density in mean coordinates, returns the natural parameters of the likelihood."""

    # Overrides

    @override
    def to_natural(self, means: Point[Mean, Self]) -> Point[Natural, Self]:
        """Convert from mean to natural parameters.

        Uses `to_natural_likelihood` for observable and interaction components, then computes latent natural parameters using conjugation structure:

        $\\theta_Z = \\nabla \\phi_Z(\\eta_Z) - \\rho$,

        where $\\rho$ are the conjugation parameters of the likelihood.
        """
        lkl_params = self.to_natural_likelihood(means)
        mean_lat = self.split_params(means)[2]
        nat_lat = self.lat_man.to_natural(mean_lat)
        return self.join_conjugated(lkl_params, nat_lat)

    @override
    def negative_entropy(self, means: Point[Mean, Self]) -> Array:
        """Compute negative entropy using $\\phi(\\eta) = \\eta \\cdot \\theta - \\psi(\\theta)$, $\\theta = \\nabla \\phi(\\eta)$."""
        params = self.to_natural(means)

        log_partition = self.log_partition_function(params)
        return self.dot(params, means) - log_partition

    # Templates

    def expectation_maximization(
        self, params: Point[Natural, Self], xs: Array
    ) -> Point[Natural, Self]:
        """Perform a single iteration of the EM algorithm."""
        q = self.expectation_step(params, xs)
        return self.to_natural(q)
