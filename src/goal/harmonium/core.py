"""Core definitions for harmonium models - product exponential families combining observable and latent variables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax import Array

from ..exponential_family import (
    ClosedForm,
    Differentiable,
    ExponentialFamily,
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
    def dimension(self) -> int:
        """Total dimension includes biases + interaction parameters."""
        obs_dim = self.obs_man.dimension
        lat_dim = self.lat_man.dimension
        return obs_dim + lat_dim + (obs_dim * lat_dim)

    @property
    def data_dimension(self) -> int:
        """Data dimension includes observable and latent variables."""
        return self.obs_man.data_dimension + self.lat_man.data_dimension

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
        obs_dim = self.obs_man.dimension
        lat_dim = self.lat_man.dimension
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
        obs_x = x[..., self.obs_man.data_dimension :]
        lat_x = x[self.obs_man.data_dimension :]

        obs_bias = self.obs_man.sufficient_statistic(obs_x)
        lat_bias = self.lat_man.sufficient_statistic(lat_x)
        interaction = self.inter_man.outer_product(lat_bias, obs_bias)

        return self.join_params(obs_bias, lat_bias, interaction).params

    def log_base_measure(self, x: Array) -> Array:
        """Compute log base measure for joint observation.

        Args:
            x: Array of shape (..., obs_dim + lat_dim) containing concatenated observable and latent states

        Returns:
            Log base measure at x
        """
        obs_x = x[..., self.obs_man.data_dimension :]
        lat_x = x[self.obs_man.data_dimension :]

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


class DifferentiableLatentHarmonium[R: MatrixRep, O: ExponentialFamily, L: ClosedForm](
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
        2. Get natural parameters of $p(z \\mid x)$ given by $\\theta_Z + \\mathbf s_X(x) · \\Theta_{XZ}$,
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
        post_func = self.posterior_function(p)
        post_nat: Point[Natural, L] = self.post_man(post_func, expand_dual(mx))

        # Convert to mean parameters (expected sufficient statistics)
        mz: Point[Mean, L] = self.lat_man.to_mean(post_nat)

        # Form interaction term via outer product
        mxz: Point[Mean, LinearMap[R, L, O]] = self.inter_man.outer_product(mz, mx)

        # Join parameters into harmonium point
        return self.join_params(mx, mz, mxz)

    def expectation_step(
        self,
        p: Point[Natural, Self],
        xs: Array,
    ) -> Point[Mean, Self]:
        """Compute average joint expectations over a batch of observations."""
        return jax.vmap(self.infer_missing_expectations, in_axes=(None, 0))(p, xs)


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


class DifferentiableConjugated[R: MatrixRep, O: ExponentialFamily, L: Differentiable](
    Differentiable, Conjugated[R, O, L], ABC
):
    def _compute_log_partition_function(self, natural_params: Array) -> Array:
        """Compute the log partition function using conjugation parameters.

        The log partition function can be written as:

        $$\\psi(\\theta) = \\psi_Z(\\theta_Z + \\rho_Z) + \\rho_0$$

        where $\\psi_Z$ is the log partition function of the latent exponential family.
        """
        # Split parameters
        obs_bias, lat_bias, int_mat = self.split_params(
            self.natural_point(natural_params)
        )

        # Get conjugation parameters
        rho_0, rho_z = self.conjugation_parameters(obs_bias, int_mat)

        # Compute adjusted latent parameters
        adjusted_lat = lat_bias + rho_z

        # Use latent family's partition function
        return self.lat_man.log_partition_function(adjusted_lat) + rho_0


class ClosedFormConjugated[R: MatrixRep, O: ExponentialFamily, L: ClosedForm](
    DifferentiableConjugated[R, O, L], ClosedForm, ABC
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

        Uses `to_natural_likelihood` for observable and interaction components,
        then computes latent natural parameters using conjugation structure:

            $\\theta_Z = \\nabla \\phi_Z(\\eta_Z) - \\rho_Z

        where $\\rho_Z$ are the conjugation parameters of the likelihood.
        """
        # Get natural likelihood parameters
        nat_obs, nat_int = self.to_natural_likelihood(p)

        # Get conjugation parameters from likelihood
        _, rho_z = self.conjugation_parameters(nat_obs, nat_int)

        # Split mean parameters
        _, mean_lat, _ = self.split_params(p)

        # Get latent natural parameters
        nat_lat = self.lat_man.to_natural(mean_lat) - rho_z

        # Join parameters
        return self.join_params(nat_obs, nat_lat, nat_int)

    def _compute_negative_entropy(self, mean_params: Array) -> Array:
        """Compute negative entropy using $\\phi(\\eta) = \\eta \\cdot \\theta - \\psi(\\theta)$, $\\theta = \\nabla \\phi(\\eta)$."""
        mean_point = self.mean_point(mean_params)
        nat_point = self.to_natural(mean_point)

        log_partition = self.log_partition_function(nat_point)
        return jnp.dot(mean_params, nat_point.params) - log_partition
