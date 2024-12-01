"""Core definitions for harmonium models - product exponential families combining observable and latent variables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax import Array

from goal.distributions import Categorical
from goal.exponential_family import Differentiable, ExponentialFamily, Natural
from goal.linear import LinearMap, MatrixRep, Rectangular
from goal.manifold import Coordinates, Point


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
        inter_man: Representation of the interaction matrix
        obs_man: The observable exponential family
        lat_man: The latent exponential family
    """

    inter_man: LinearMap[R, L, O]

    obs_man: O
    """The observable exponential family"""

    lat_man: L
    """The latent exponential family"""

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
            x: Array of shape (..., obs_dim + lat_dim) containing concatenated
               observable and latent states

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
            x: Array of shape (..., obs_dim + lat_dim) containing concatenated
               observable and latent states

        Returns:
            Log base measure at x
        """
        obs_x = x[..., self.obs_man.data_dimension :]
        lat_x = x[self.obs_man.data_dimension :]

        return self.obs_man.log_base_measure(obs_x) + self.lat_man.log_base_measure(
            lat_x
        )


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


@dataclass(frozen=True)
class CategoricalMixtureHarmonium[O: Differentiable](
    DifferentiableConjugated[Rectangular, O, Categorical]
):
    """A mixture model implemented as a harmonium with categorical latent variables.

    The joint distribution takes the form:

    $$p(x,z) \\propto e^{\\theta_x \\cdot s_x(x)} e^{\\theta_z \\cdot s_z} e^{s_x(x) \\cdot \\Theta_{xz} \\cdot s_z}$$

    where $s_z$ is the one-hot encoding of the categorical variable z.

    This can be rewritten as:

    $$p(x,z=k) \\propto e^{\\theta_x \\cdot s_x(x)} e^{\\theta_k} e^{s_x(x) \\cdot \\theta_{k}}$$

    where $\\theta_k$ and $\\theta_{k}$ are the k-th components of $\\theta_z$ and $\\Theta_{xz}$ respectively.
    """

    def conjugation_parameters(
        self,
        obs_bias: Point[Natural, O],
        int_mat: Point[Natural, LinearMap[Rectangular, Categorical, O]],
    ) -> tuple[Array, Point[Natural, Categorical]]:
        """Compute conjugation parameters for categorical mixture.

        For a categorical mixture model:

        - $\\rho_0 = \\psi(\\theta_x)$
        - $\\rho_z = \\{\\psi(\\theta_x + \\theta_{xz,k}) - \\rho_0\\}_k$
        """
        # Compute base term from observable bias
        rho_0 = self.obs_man.log_partition_function(obs_bias)

        # Get component parameter vectors as columns of interaction matrix
        int_cols = self.inter_man.to_columns(int_mat)

        # Compute adjusted partition function for each component
        rho_z_components = []
        for comp_params in int_cols:
            # For each component k, compute psi(theta_x + theta_xz_k)
            adjusted_obs = obs_bias + comp_params
            comp_term = self.obs_man.log_partition_function(adjusted_obs) - rho_0
            rho_z_components.append(comp_term)

        return rho_0, Point(jnp.array(rho_z_components))
