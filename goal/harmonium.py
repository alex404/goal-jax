"""Core definitions for harmonium models - product exponential families combining observable and latent variables."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from goal.exponential_family import ExponentialFamily
from goal.linear import LinearMap, MatrixRep
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
        self, p: Point[C, Harmonium[R, L, O]]
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
    ) -> Point[C, Harmonium[R, L, O]]:
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
