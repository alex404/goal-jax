"""Core definitions for graphical harmonium models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, Union

import jax.numpy as jnp
from jax import Array

from ..geometry.exponential_family import ExponentialFamily, Mean, Natural
from ..geometry.harmonium import BackwardConjugated
from ..geometry.linear import AffineMap
from ..geometry.manifold import Point

# Type alias for nodes in the graph
Node = Union[BackwardConjugated, "BackwardGraph"]


@dataclass(frozen=True)
class BackwardGraph[H1: BackwardConjugated, H2: Node](BackwardConjugated):
    """A recursive graph of harmoniums where each layer is either a harmonium or another graph.

    Structure consists of:
    - A lower harmonium H1
    - An upper layer H2 which can be either a harmonium or another graph
    Together they form p(x,y,z) = p(x|y)p(y|z)p(z) or deeper hierarchies.
    """

    harmoniums: tuple[H1, H2]

    @classmethod
    def build[H1: BackwardConjugated, H2: Node](
        cls,
        obs_man: ExponentialFamily,
        h1_and_lat1: tuple[H1, ExponentialFamily],
        h2_and_lat2: tuple[H2, ExponentialFamily],
    ) -> BackwardGraph[H1, H2]:
        """Build a backward graph recursively.

        Args:
            obs_man: Bottom observable manifold
            h1_and_lat1: Lower (harmonium, latent manifold) pair
            h2_and_lat2: Upper (harmonium/graph, latent manifold) pair
        """
        h1, lat1 = h1_and_lat1
        h2, lat2 = h2_and_lat2

        # Verify connections
        assert isinstance(h1.obs_man, type(obs_man))
        assert isinstance(h1.lat_man, type(lat1))
        assert isinstance(h2.obs_man, type(lat1))
        assert isinstance(h2.lat_man, type(lat2))

        return cls((h1, h2))

    @property
    def lwr_man(self) -> H1:
        """The lower harmonium layer."""
        return self.harmoniums[0]

    @property
    def upr_man(self) -> H2:
        """The upper layer (harmonium or graph)."""
        return self.harmoniums[1]

    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rep, SubLowLatentY, SubObservable, Observable]
        ],
    ) -> tuple[Array, Point[Natural, H2]]:
        """Compute conjugation parameters for the graph.

        Follows recursive structure where:
        1. Get conjugation parameters for lower harmonium
        2. Use those to adjust parameters of upper layer
        3. Get conjugation parameters for adjusted upper layer
        """
        chi, rho0 = self.lwr_man.conjugation_parameters(lkl_params)
        with self.upr_man as um:
            rho = um.join_params(
                rho0, Point(jnp.zeros(um.int_man.dim)), Point(jnp.zeros(um.lat_man.dim))
            )
        return chi, rho

    def to_natural_likelihood(
        self, p: Point[Mean, Self]
    ) -> Point[Natural, AffineMap[Rep, SubLowLatentY, SubObservable, Observable]]:
        """Convert mean parameters to natural parameters for lower likelihood.

        Process:
        1. Split parameters into observable, interaction, and latent components
        2. Extract lower latent means from upper layer
        3. Join parameters for lower harmonium
        4. Convert to natural parameters
        """
        obs_means, lwr_int_means, lat_means = self.split_params(p)
        lwr_lat_means = self.upr_man.split_params(lat_means)[0]
        lwr_means = self.lwr_man.join_params(obs_means, lwr_int_means, lwr_lat_means)
        return self.lwr_man.to_natural_likelihood(lwr_means)
