"""Hierarchical mixture of Gaussians (HMoG) implemented as a hierarchical harmonium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, TypeVar

import jax
import jax.numpy as jnp
from jax import Array

from ..geometry import (
    AffineMap,
    BackwardConjugated,
    ComposedSubspace,
    LinearMap,
    LocationSubspace,
    Mean,
    Natural,
    Point,
    PositiveDefinite,
    Rectangular,
    TripleSubspace,
)
from .lgm import LinearGaussianModel
from .mixture import Mixture
from .normal import Euclidean, FullNormal, Normal

Rep = TypeVar("Rep", bound=PositiveDefinite)


@dataclass(frozen=True)
class HierarchicalMixtureOfGaussians[ObsRep: PositiveDefinite](
    BackwardConjugated[
        Rectangular,
        Normal[ObsRep],
        Euclidean,
        Euclidean,
        Mixture[FullNormal],
    ],
):
    """A hierarchical mixture of Gaussians implemented as a three-layer harmonium.

    Structure:
    - Observable layer: Gaussian with covariance structure specified by ObsRep
    - First latent layer: Mixture of full-covariance Gaussians
    - Second latent layer: Mixture model over parameters of first latent layer

    Args:
        obs_dim: Dimension of observable space
        lat_dim: Dimension of latent Gaussian components
        n_components: Number of mixture components
        n_subcomponents: Number of submixture components
    """

    # Lower harmonium between X and Y
    lwr_man: LinearGaussianModel[ObsRep]

    def __init__(
        self, obs_dim: int, obs_rep: type[ObsRep], lat_dim: int, n_components: int
    ):
        obs_man = Normal(obs_dim, obs_rep)
        lwr_lat_man = Normal(lat_dim)
        lwr_man = LinearGaussianModel(obs_dim, obs_rep, lat_dim)
        object.__setattr__(self, "lwr_man", lwr_man)
        int_man = LinearMap(
            Rectangular(),
            lwr_lat_man.loc_man,
            obs_man.loc_man,
        )
        lat_man = Mixture(lwr_lat_man, n_components)
        obs_sub = LocationSubspace(obs_man)
        lat_sub = ComposedSubspace(
            TripleSubspace(lat_man), LocationSubspace(lwr_lat_man)
        )
        super().__init__(
            obs_man,
            int_man,
            lat_man,
            obs_sub,
            lat_sub,
        )

    @property
    def upr_man(
        self,
    ) -> Mixture[FullNormal]:
        return self.trd_man

    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rectangular, Euclidean, Euclidean, Normal[ObsRep]]
        ],
    ) -> tuple[
        Array,
        Point[
            Natural,
            Mixture[FullNormal],
        ],
    ]:
        """Compute lower conjugation parameters for the hierarchical model."""
        chi, rho0 = self.lwr_man.conjugation_parameters(lkl_params)
        with self.upr_man as um:
            rho = um.join_params(
                rho0, Point(jnp.zeros(um.int_man.dim)), Point(jnp.zeros(um.lat_man.dim))
            )

        # Combine the conjugation parameters
        return chi, rho

    def to_natural_likelihood(
        self, p: Point[Mean, Self]
    ) -> Point[Natural, AffineMap[Rectangular, Euclidean, Euclidean, Normal[ObsRep]]]:
        """Use mean parameters to compute natural parameters for the lower likelihood."""

        obs_means, lwr_int_means, lat_means = self.split_params(p)
        lwr_lat_means = self.upr_man.split_params(lat_means)[0]
        lwr_means = self.lwr_man.join_params(obs_means, lwr_int_means, lwr_lat_means)

        # Combine the likelihoods
        return self.lwr_man.to_natural_likelihood(lwr_means)

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
        yz_sample = self.lat_man.sample(key1, nat_prior, n)
        y_sample = yz_sample[:, : self.lwr_man.lat_man.data_dim]

        # Vectorize sampling from conditional distributions
        x_params = jax.vmap(self.likelihood_at, in_axes=(None, 0))(p, y_sample)

        # Sample from conditionals p(x|z) in parallel
        x_sample = jax.vmap(self.obs_man.sample, in_axes=(0, 0, None))(
            jax.random.split(key2, n), x_params, 1
        ).reshape((n, -1))

        # Concatenate samples along data dimension
        return jnp.concatenate([x_sample, yz_sample], axis=-1)
