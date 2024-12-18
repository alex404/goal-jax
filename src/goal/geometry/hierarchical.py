"""Core definitions for hierarchical harmoniums.

A hierarchical harmonium is a BackwardConjugated harmonium where the latent manifold is itself a BackwardConjugated harmonium. This allows for deep hierarchical structure where each layer is connected through conjugate distributions.

The basic structure is:

- A lower harmonium $p(x|y)$ between observable $x$ and first latent $y$
- An upper harmonium $p(y|z)$ between first latent $y$ and second latent $z$
- Together they form a joint distribution $p(x,y,z) = p(x|y)p(y|z)p(z)$

Key algorithms (conjugation, natural parameters, sampling) are implemented recursively.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import jax
import jax.numpy as jnp
from jax import Array

from .exponential_family import ExponentialFamily, Generative, Mean, Natural
from .harmonium import BackwardConjugated
from .linear import AffineMap
from .manifold import Point
from .rep.matrix import MatrixRep


class HierarchicalBackward[
    Rep: MatrixRep,
    Observable: Generative,
    SubObservable: ExponentialFamily,
    SubLatent: ExponentialFamily,
    Latent: Any,
](BackwardConjugated[Rep, Observable, SubObservable, SubLatent, Latent], ABC):
    """Class for hierarchical harmoniums with deep conjugate structure.

    This class provides the algorithms needed for hierarchical harmoniums while ensuring proper typing and conjugate relationships between layers. It subclasses BackwardConjugated to maintain the exponential family structure while adding hierarchical capabilities.

    Type Parameters
    --------------
    Rep : MatrixRep
        Matrix representation for interaction terms
    Observable : Generative
        Observable manifold type that supports sampling
    SubObservable : ExponentialFamily
        Interactive subspace of observable manifold
    SubLatent : ExponentialFamily
        Interactive subspace of latent manifold
    Latent : Any
        Type of latent manifold (must be a BackwardConjugated at runtime)

    Properties
    ----------
    lwr_man : BackwardConjugated
        Lower harmonium between observable and first latent layer
    lat_man : BackwardConjugated
        Upper harmonium between latent layers (inherited from BackwardConjugated)

    Notes
    -----
    While the Latent type parameter is Any for typing flexibility, at runtime we verify that it is actually a BackwardConjugated through __post_init__. This ensures proper hierarchical structure while avoiding complex generic typing.
    """

    def __post_init__(self):
        # Check that the latent manifold is a BackwardConjugated
        if not isinstance(self.lat_man, BackwardConjugated):
            raise TypeError(
                f"Latent manifold must be a BackwardConjugated, got {type(self.lat_man)}"
            )

    @property
    @abstractmethod
    def lwr_man(
        self,
    ) -> BackwardConjugated[Rep, Observable, SubObservable, SubLatent, Any]: ...

    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rep, SubLatent, SubObservable, Observable]
        ],
    ) -> tuple[Array, Point[Natural, Latent]]:
        """Compute conjugation parameters recursively."""
        chi, rho0 = self.lwr_man.conjugation_parameters(lkl_params)
        with self.lat_man as um:
            rho = um.join_params(
                rho0, Point(jnp.zeros(um.int_man.dim)), Point(jnp.zeros(um.lat_man.dim))
            )
        return chi, rho

    def to_natural_likelihood(
        self,
        p: Point[Mean, Self],
    ) -> Point[Natural, AffineMap[Rep, SubLatent, SubObservable, Observable]]:
        """Convert mean parameters to natural parameters for lower likelihood."""
        obs_means, lwr_int_means, lat_means = self.split_params(p)
        lwr_lat_means = self.lat_man.split_params(lat_means)[0]
        lwr_means = self.lwr_man.join_params(obs_means, lwr_int_means, lwr_lat_means)
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
