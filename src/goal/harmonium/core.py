"""Core definitions for harmonium models - product exponential families combining observable and latent variables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax import Array

from ..exponential_family import (
    Backward,
    ExponentialFamily,
    Forward,
    Generative,
    Mean,
    Natural,
)
from ..manifold import Coordinates, Point, expand_dual
from ..transforms import AffineMap, LinearMap, MatrixRep


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Harmonium[R: MatrixRep, O: ExponentialFamily, L: ExponentialFamily](
    ExponentialFamily
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

    Args:
        obs_man: The observable exponential family
        lat_man: The latent exponential family
        inter_man: Representation of the interaction matrix
    """

    obs_man: O
    """The observable exponential family"""

    lat_man: L
    """The latent exponential family"""

    inter_man: LinearMap[R, L, O]
    """The manifold of interaction matrices"""

    @property
    def dim(self) -> int:
        """Total dimension includes biases + interaction parameters."""
        obs_dim = self.obs_man.dim
        lat_dim = self.lat_man.dim
        return obs_dim + lat_dim + (obs_dim * lat_dim)

    @property
    def data_dim(self) -> int:
        """Data dimension includes observable and latent variables."""
        return self.obs_man.data_dim + self.lat_man.data_dim

    @property
    def like_man(self) -> AffineMap[R, L, O]:
        """Manifold of conditional likelihood distributions $p(x \\mid z)$."""
        return AffineMap(self.inter_man.rep, self.lat_man, self.obs_man)

    @property
    def post_man(self) -> AffineMap[R, O, L]:
        """Manifold of conditional posterior distributions $p(z \\mid x)$."""
        return AffineMap(self.inter_man.rep, self.obs_man, self.lat_man)

    def split_params[C: Coordinates](
        self, p: Point[C, Self]
    ) -> tuple[Point[C, O], Point[C, L], Point[C, LinearMap[R, L, O]]]:
        """Split harmonium parameters into observable bias, interaction, and latent bias."""
        obs_dim = self.obs_man.dim
        lat_dim = self.lat_man.dim
        obs_params = p.params[:obs_dim]
        lat_params = p.params[obs_dim : obs_dim + lat_dim]
        int_params = p.params[obs_dim + lat_dim :]
        return Point(obs_params), Point(lat_params), Point(int_params)

    def join_params[C: Coordinates](
        self,
        obs_bias: Point[C, O],
        lat_bias: Point[C, L],
        int_mat: Point[C, LinearMap[R, L, O]],
    ) -> Point[C, Self]:
        """Join component parameters into a harmonium density."""
        return Point(
            jnp.concatenate([obs_bias.params, lat_bias.params, int_mat.params])
        )

    def _compute_sufficient_statistic(self, x: Array) -> Array:
        """Compute sufficient statistics for joint observation.

        Args:
            x: Array of shape (..., obs_dim + lat_dim) containing concatenated observable and latent states

        Returns:
            Array of sufficient statistics
        """
        obs_x = x[..., self.obs_man.data_dim :]
        lat_x = x[self.obs_man.data_dim :]

        obs_bias = self.obs_man.sufficient_statistic(obs_x)
        lat_bias = self.lat_man.sufficient_statistic(lat_x)
        interaction = self.inter_man.outer_product(obs_bias, lat_bias)

        return self.join_params(obs_bias, lat_bias, interaction).params

    def log_base_measure(self, x: Array) -> Array:
        """Compute log base measure for joint observation.

        Args:
            x: Array of shape (..., obs_dim + lat_dim) containing concatenated observable and latent states

        Returns:
            Log base measure at x
        """
        obs_x = x[..., self.obs_man.data_dim :]
        lat_x = x[self.obs_man.data_dim :]

        return self.obs_man.log_base_measure(obs_x) + self.lat_man.log_base_measure(
            lat_x
        )

    def likelihood_function(
        self, p: Point[Natural, Self]
    ) -> Point[Natural, AffineMap[R, L, O]]:
        """Natural parameters of the likelihood distribution as an affine function.

        The affine map is $\\eta \\to \\theta_X + \\Theta_{XZ} \\cdot \\eta$.
        """
        obs_bias, _, int_mat = self.split_params(p)
        return self.like_man.join_params(obs_bias, int_mat)

    def posterior_function(
        self, p: Point[Natural, Self]
    ) -> Point[Natural, AffineMap[R, O, L]]:
        """Natural parameters of the posterior distribution as an affine function.

        The affine map is $\\eta \\to \\theta_Z + \\eta \\cdot \\Theta_{XZ}^T$.
        """
        _, lat_bias, int_mat = self.split_params(p)
        int_mat_t = self.inter_man.transpose(int_mat)
        return self.post_man.join_params(lat_bias, int_mat_t)

    def likelihood_at(self, p: Point[Natural, Self], z: Array) -> Point[Natural, O]:
        """Compute natural parameters of likelihood distribution $p(x \\mid z)$ at a given $z$.

        Given a latent state $z$ with sufficient statistics $s(z)$, computes natural parameters $\\theta_X$ of the conditional distribution:

        $$p(x \\mid z) \\propto \\exp(\\theta_X \\cdot s_X(x) + s_X(x) \\cdot \\Theta_{XZ} \\cdot s_Z(z))$$
        """
        mz = expand_dual(self.lat_man.sufficient_statistic(z))
        return self.like_man(self.likelihood_function(p), mz)

    def posterior_at(self, p: Point[Natural, Self], x: Array) -> Point[Natural, L]:
        """Compute natural parameters of posterior distribution $p(z \\mid x)$.

        Given an observation $x$ with sufficient statistics $s(x)$, computes natural
        parameters $\\theta_Z$ of the conditional distribution:

        $$p(z \\mid x) \\propto \\exp(\\theta_Z \\cdot s_Z(z) + s_X(x) \\cdot \\Theta_{XZ} \\cdot s_Z(z))$$
        """
        mx = expand_dual(self.obs_man.sufficient_statistic(x))
        return self.post_man(self.posterior_function(p), mx)


class BackwardLatent[R: MatrixRep, O: ExponentialFamily, L: Backward](
    Harmonium[R, O, L]
):
    """A harmonium with differentiable latent exponential family."""

    def infer_missing_expectations(
        self,
        p: Point[Natural, Self],
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
            p: Current harmonium parameters in natural coordinates

        Returns:
            Joint expectations in mean coordinates for this observation
        """
        # Get sufficient statistics of observation
        mx: Point[Mean, O] = self.obs_man.sufficient_statistic(x)

        # Get posterior parameters for this observation
        post_map = self.posterior_function(p)
        post_nat: Point[Natural, L] = self.post_man(post_map, expand_dual(mx))

        # Convert to mean parameters (expected sufficient statistics)
        mz: Point[Mean, L] = self.lat_man.to_mean(post_nat)

        # Form interaction term via outer product
        mxz: Point[Mean, LinearMap[R, L, O]] = self.inter_man.outer_product(mx, mz)

        # Join parameters into harmonium point
        return self.join_params(mx, mz, mxz)

    def expectation_step(
        self: Self,
        p: Point[Natural, Self],
        xs: Array,
    ) -> Point[Mean, Self]:
        """Compute average joint expectations over a batch of observations.

        Args:
            p: Current harmonium parameters in natural coordinates
            xs: Batch of observations of shape (batch_size, obs_dim)

        Returns:
            Average joint expectations in mean coordinates
        """
        # Apply vmap to get batch of expectations
        batch_expectations = jax.vmap(
            self.infer_missing_expectations, in_axes=(None, 0)
        )(p, xs)

        # Take mean of the underlying parameter arrays and wrap in a Point
        return Point(jnp.mean(batch_expectations.params, axis=0))


class Conjugated[R: MatrixRep, O: ExponentialFamily, L: ExponentialFamily](
    Harmonium[R, O, L], ABC
):
    """A harmonium with conjugation structure enabling analytical decomposition of the log-partition function and exact Bayesian inference of latent states."""

    @abstractmethod
    def conjugation_parameters(
        self, obs_bias: Point[Natural, O], int_mat: Point[Natural, LinearMap[R, L, O]]
    ) -> tuple[Array, Point[Natural, L]]:
        """Compute conjugation parameters for the harmonium."""
        ...

    def prior(self, p: Point[Natural, Self]) -> Point[Natural, L]:
        """Natural parameters of the prior distribution $p(z)$."""
        obs_bias, lat_bias, int_mat = self.split_params(p)
        _, rho = self.conjugation_parameters(obs_bias, int_mat)
        return lat_bias + rho

    def split_conjugated(
        self, p: Point[Natural, Self]
    ) -> tuple[Point[Natural, AffineMap[R, L, O]], Point[Natural, L]]:
        """Split conjugated harmonium into likelihood and prior."""
        obs_bias, lat_bias, int_mat = self.split_params(p)
        _, rho = self.conjugation_parameters(obs_bias, int_mat)
        return self.like_man.join_params(obs_bias, int_mat), lat_bias + rho

    def join_conjugated(
        self,
        like_params: Point[Natural, AffineMap[R, L, O]],
        prior_params: Point[Natural, L],
    ) -> Point[Natural, Self]:
        """Join likelihood and prior parameters into a conjugated harmonium."""
        obs_bias, int_mat = self.like_man.split_params(like_params)
        _, rho = self.conjugation_parameters(obs_bias, int_mat)
        lat_bias = prior_params - rho
        return self.join_params(obs_bias, lat_bias, int_mat)


class GenerativeConjugated[
    R: MatrixRep,
    O: Generative,
    L: Generative,
](Conjugated[R, O, L], Generative):
    """A conjugated harmonium that supports sampling through its latent distribution.

    The sampling process leverages the conjugate structure to:
    1. Sample from the marginal latent distribution $p(z)$
    2. Sample from the conditional likelihood $p(x \\mid z)$
    """

    def sample(self, key: Array, p: Point[Natural, Self], n: int = 1) -> Array:
        """Generate samples from the harmonium distribution.

        Args:
            key: PRNG key for sampling
            p: Parameters in natural coordinates
            n: Number of samples to generate

        Returns:
            Array of shape (n, data_dim) containing concatenated observable and latent states
        """
        # Split up the sampling key
        key1, key2 = jax.random.split(key)

        # Sample from adjusted latent distribution p(z)
        nat_prior = self.prior(p)
        z_sample = self.lat_man.sample(key1, nat_prior, n)

        # Vectorize sampling from conditional distributions
        x_params = jax.vmap(self.likelihood_at, in_axes=(None, 0))(p, z_sample)

        # Sample from conditionals p(x|z) in parallel
        x_sample = jax.vmap(self.obs_man.sample, in_axes=(0, 0, None))(
            jax.random.split(key2, n), x_params, 1
        ).reshape((n, -1))

        # Concatenate samples along data dimension
        return jnp.concatenate([x_sample, z_sample], axis=-1)

    def observable_sample(
        self, key: Array, p: Point[Natural, Self], n: int = 1
    ) -> Array:
        xzs = self.sample(key, p, n)
        return xzs[:, : self.obs_man.data_dim]


class ForwardConjugated[R: MatrixRep, O: Generative, L: Forward](
    Forward, GenerativeConjugated[R, O, L], ABC
):
    def _compute_log_partition_function(self, natural_params: Array) -> Array:
        """Compute the log partition function using conjugation parameters.

        The log partition function can be written as:

        $$\\psi(\\theta) = \\psi_Z(\\theta_Z + \\rho) + \\chi$$

        where $\\psi_Z$ is the log partition function of the latent exponential family.
        """
        # Split parameters
        obs_bias, lat_bias, int_mat = self.split_params(
            self.natural_point(natural_params)
        )

        # Get conjugation parameters
        chi, rho = self.conjugation_parameters(obs_bias, int_mat)

        # Compute adjusted latent parameters
        adjusted_lat = lat_bias + rho

        # Use latent family's partition function
        return self.lat_man.log_partition_function(adjusted_lat) + chi

    def log_observable_density(self, p: Point[Natural, Self], x: Array) -> Array:
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
            p: Parameters in natural coordinates
            x: Observable data point

        Returns:
            Log density at x
        """
        obs_bias, lat_bias, int_mat = self.split_params(p)
        chi, rho = self.conjugation_parameters(obs_bias, int_mat)
        obs_stats = self.obs_man.sufficient_statistic(x)

        log_density = self.obs_man.dot(obs_bias, obs_stats)
        log_density += self.lat_man.log_partition_function(self.posterior_at(p, x))
        log_density -= self.lat_man.log_partition_function(lat_bias + rho) + chi

        return log_density + self.obs_man.log_base_measure(x)

    def average_log_observable_density(
        self, p: Point[Natural, Self], xs: Array
    ) -> Array:
        """Compute average log density over a batch of observations."""
        return jnp.mean(
            jax.vmap(self.log_observable_density, in_axes=(None, 0))(p, xs), axis=0
        )

    def observable_density(self, p: Point[Natural, Self], x: Array) -> Array:
        """Compute density of the observable distribution $p(x)$."""
        return jnp.exp(self.log_observable_density(p, x))


class BackwardConjugated[R: MatrixRep, O: Generative, L: Backward](
    ForwardConjugated[R, O, L], BackwardLatent[R, O, L], Backward, ABC
):
    """A conjugated harmonium with invertible likelihood parameterization."""

    @abstractmethod
    def to_natural_likelihood(
        self, p: Point[Mean, Self]
    ) -> tuple[Point[Natural, O], Point[Natural, LinearMap[R, L, O]]]:
        """Map mean to natural parameters for observable and interaction components."""
        ...

    def to_natural(self, p: Point[Mean, Self]) -> Point[Natural, Self]:
        """Convert from mean to natural parameters.

        Uses `to_natural_likelihood` for observable and interaction components, then computes latent natural parameters using conjugation structure:

        $\\theta_Z = \\nabla \\phi_Z(\\eta_Z) - \\rho$

        where $\\rho$ are the conjugation parameters of the likelihood.
        """
        # Get natural likelihood parameters
        nat_obs, nat_int = self.to_natural_likelihood(p)

        # Get conjugation parameters from likelihood
        _, rho = self.conjugation_parameters(nat_obs, nat_int)

        # Split mean parameters
        _, mean_lat, _ = self.split_params(p)

        # Get latent natural parameters
        nat_lat = self.lat_man.to_natural(mean_lat) - rho

        # Join parameters
        return self.join_params(nat_obs, nat_lat, nat_int)

    def _compute_negative_entropy(self, mean_params: Array) -> Array:
        """Compute negative entropy using $\\phi(\\eta) = \\eta \\cdot \\theta - \\psi(\\theta)$, $\\theta = \\nabla \\phi(\\eta)$."""
        mean_point = self.mean_point(mean_params)
        nat_point = self.to_natural(mean_point)

        log_partition = self.log_partition_function(nat_point)
        return self.dot(nat_point, mean_point) - log_partition

    def expectation_maximization(
        self, p: Point[Natural, Self], xs: Array
    ) -> Point[Natural, Self]:
        """Perform a single iteration of the EM algorithm."""
        # E-step: Compute expectations
        q = self.expectation_step(p, xs)
        return self.to_natural(q)
