"""Core definitions for harmonium models - product exponential families combining observable and latent variables."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from goal.exponential_family import ExponentialFamily
from goal.linear import Rectangular
from goal.manifold import C, Point

OBS = TypeVar("OBS", bound=ExponentialFamily)
LAT = TypeVar("LAT", bound=ExponentialFamily)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Harmonium(ExponentialFamily, Generic[OBS, LAT]):
    """An exponential family harmonium combining observable and latent distributions.

    A harmonium combines two exponential families into a joint distribution:
        - Observable family O defining p(x)
        - Latent family L defining p(z)

    The joint distribution takes the form:
        p(x,z) ∝ exp(θ_x · s(x) + θ_z · s(z) + s(x)^T W s(z))

    where:
        - θ_x are the observable biases
        - θ_z are the latent biases
        - W is the interaction matrix
        - s(x), s(z) are sufficient statistics

    Args:
        observable: The observable exponential family
        latent: The latent exponential family
    """

    observable: OBS
    """The observable exponential family"""

    latent: LAT
    """The latent exponential family"""

    @property
    def dimension(self) -> int:
        """Total dimension includes biases + interaction parameters."""
        obs_dim = self.observable.dimension
        lat_dim = self.latent.dimension
        return obs_dim + lat_dim + (obs_dim * lat_dim)

    def _split_params(
        self, p: Point[C, "Harmonium[LAT,OBS]"]
    ) -> tuple[Array, Rectangular, Array]:
        """Split harmonium parameters into observable bias, interaction, and latent bias.

        Args:
            p: Point on the harmonium manifold

        Returns:
            Triple of (observable_bias, interaction_matrix, latent_bias)
        """
        obs_dim = self.observable.dimension
        lat_dim = self.latent.dimension
        obs_params = p.params[:obs_dim]
        lat_params = p.params[obs_dim : obs_dim + lat_dim]
        int_params = p.params[obs_dim + lat_dim :].reshape(obs_dim, lat_dim)
        return obs_params, Rectangular.from_matrix(int_params), lat_params

    def _join_params(
        self, obs: Array, interaction: Rectangular, lat: Array
    ) -> Array:
        """Join parameters into a single array.

        Args:
            obs: Observable bias parameters
            interaction: Interaction matrix
            lat: Latent bias parameters

        Returns:
            Concatenated parameter array
        """
        return jnp.concatenate([obs.ravel(), lat.ravel(), interaction.matrix.ravel()])

    def _compute_sufficient_statistic(self, x: ArrayLike) -> Array:
        """Compute sufficient statistics for joint observation.

        Args:
            x: Array of shape (..., obs_dim + lat_dim) containing concatenated
               observable and latent variables

        Returns:
            Array of sufficient statistics
        """
        obs_dim = self.observable.dimension
        obs = x[..., :obs_dim]
        lat = x[..., obs_dim:]

        obs_stats = self.observable._compute_sufficient_statistic(obs)
        lat_stats = self.latent._compute_sufficient_statistic(lat)
        interaction = Rectangular.from_matrix(jnp.outer(obs_stats, lat_stats))

        return self._join_params(obs_stats, interaction, lat_stats)

    def log_base_measure(self, x: ArrayLike) -> Array:
        """Compute log base measure for joint observation.

        Args:
            x: Array of shape (..., obs_dim + lat_dim) containing concatenated
               observable and latent variables

        Returns:
            Log base measure value
        """
        obs_dim = self.observable.dimension
        obs = x[..., :obs_dim]
        lat = x[..., obs_dim:]

        return self.observable.log_base_measure(obs) + self.latent.log_base_measure(lat)
