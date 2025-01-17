"""Core definitions for hierarchical harmoniums.

A hierarchical harmonium is a Conjugated harmonium where the latent manifold is itself a Conjugated harmonium. This allows for deep hierarchical structure where each layer is connected through conjugate distributions.

The basic structure is:

- A lower harmonium $p(x|y)$ between observable $x$ and first latent $y$
- An upper harmonium $p(y|z)$ between first latent $y$ and second latent $z$
- Together they form a joint distribution $p(x,y,z) = p(x|y)p(y|z)p(z)$

Key algorithms (conjugation, natural parameters, sampling) are implemented recursively.


#### Class Hierarchy

![Class Hierarchy](hierarchical.svg)
"""

from __future__ import annotations

from typing import Any, Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.linear import AffineMap, LinearMap
from ..manifold.manifold import Point
from ..manifold.matrix import MatrixRep
from ..manifold.subspace import Subspace
from .exponential_family import ExponentialFamily, Generative, Mean, Natural
from .harmonium import AnalyticConjugated, Conjugated, DifferentiableConjugated

# @dataclass(frozen=True)
# class ComposedSubspace[Super: Manifold, Mid: Manifold, Sub: Manifold](
#     Subspace[Super, Sub]
# ):
#     """Composition of two subspace relationships.
#     Given subspaces $S: \\mathcal{M} \\to \\mathcal{L}$ and $T: \\mathcal{L} \\to \\mathcal{N}$, forms their composition $(T \\circ S): \\mathcal{M} \\to \\mathcal{N}$ where:
#
#     - Projection: $\\pi_{T \\circ S}(p) = \\pi_T(\\pi_S(p))$
#     - Translation: $\\tau_{T \\circ S}(p,q) = \\tau_S(p, \\tau_T(0,q))$
#
#     The variance requirements of Subspace (contravariant Super, covariant Sub) ensure that composition is well-typed when the middle types are compatible.
#     """
#
#     sup_sub: Subspace[Super, Mid]
#     sub_sub: Subspace[Mid, Sub]
#
#     @property
#     @override
#     def sup_man(self) -> Super:
#         return self.sup_sub.sup_man
#
#     @property
#     @override
#     def sub_man(self) -> Sub:
#         return self.sub_sub.sub_man
#
#     @override
#     def project[C: Coordinates](self, p: Point[C, Super]) -> Point[C, Sub]:
#         mid = self.sup_sub.project(p)
#         return self.sub_sub.project(mid)
#
#     @override
#     def translate[C: Coordinates](
#         self, p: Point[C, Super], q: Point[C, Sub]
#     ) -> Point[C, Super]:
#         mid_zero: Point[C, Mid] = Point(jnp.zeros(self.sup_sub.sub_man.dim))
#         mid = self.sub_sub.translate(mid_zero, q)
#         return self.sup_sub.translate(p, mid)
#
#
# @dataclass(frozen=True)
# class TripleSubspace[First: Manifold, Second: Manifold, Third: Manifold](
#     Subspace[Triple[First, Second, Third], First]
# ):
#     """Subspace relationship for a product manifold $\\mathcal M \\times \\mathcal N \\times \\mathcal O$."""
#
#     def __init__(self, thr_man: Triple[First, Second, Third]):
#         super().__init__(thr_man, thr_man.fst_man)
#
#     @override
#     def project[C: Coordinates](
#         self, p: Point[C, Triple[First, Second, Third]]
#     ) -> Point[C, First]:
#         first, _, _ = self.sup_man.split_params(p)
#         return first
#
#     @override
#     def translate[C: Coordinates](
#         self, p: Point[C, Triple[First, Second, Third]], q: Point[C, First]
#     ) -> Point[C, Triple[First, Second, Third]]:
#         first, second, third = self.sup_man.split_params(p)
#         return self.sup_man.join_params(first + q, second, third)


class DifferentiableHierarchical[
    Rep: MatrixRep,
    Observable: Generative,
    SubObservable: ExponentialFamily,
    SubLatent: ExponentialFamily,
    Latent: Any,
](DifferentiableConjugated[Rep, Observable, SubObservable, SubLatent, Latent]):
    """Class for hierarchical harmoniums with deep conjugate structure.

    This class provides the algorithms needed for hierarchical harmoniums while ensuring proper typing and conjugate relationships between layers. It subclasses DifferentiableConjugated to maintain the exponential family structure while adding hierarchical capabilities.

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
        Type of latent manifold (must be a DifferentiableConjugated at runtime)

    Properties
    ----------
    lwr_man : DifferentiableConjugated
        Lower harmonium between observable and first latent layer
    lat_man : DifferentiableConjugated
        Upper harmonium between latent layers (inherited from DifferentiableConjugated)

    Notes
    -----
    While the Latent type parameter is Any for typing flexibility, at runtime we verify that it is actually a DifferentiableConjugated through __post_init__. This ensures proper hierarchical structure while avoiding complex generic typing.
    """

    _lwr_man: Conjugated[Rep, Observable, SubObservable, SubLatent, Any]
    upr_man: Latent
    upr_obs_sub: Subspace[Latent, SubObservable]

    def __post_init__(self):
        # Check that the latent manifold is a DifferentiableConjugated
        if not isinstance(self.lat_man, DifferentiableConjugated):
            raise TypeError(
                f"Latent manifold must be a DifferentiableConjugated, got {type(self.lat_man)}"
            )
        if not isinstance(self._lwr_man, DifferentiableConjugated):
            raise TypeError(
                f"Bottom layer must be a DifferentiableConjugated, got {type(self._lwr_man)}"
            )

    @property
    def lwr_man(
        self,
    ) -> DifferentiableConjugated[Rep, Observable, SubObservable, SubLatent, Any]:
        return self._lwr_man  # pyright: ignore[reportReturnType]

    @property
    @override
    def fst_man(self) -> Observable:
        return self.lwr_man.obs_man

    @property
    @override
    def snd_man(self) -> LinearMap[Rep, SubLatent, SubObservable]:
        return self.lwr_man.int_man

    @property
    @override
    def trd_man(self) -> Latent:
        return self.upr_man

    @property
    @override
    def obs_sub(self) -> Subspace[Observable, SubObservable]:
        return self.lwr_man.obs_sub

    @property
    @override
    def lat_sub(self) -> Subspace[Latent, SubLatent]:
        return ComposedSubspace(TripleSubspace(self.upr_man), self.upr_obs_sub)

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
        yz_sample = self.lat_man.sample(key1, nat_prior, n)
        y_sample = yz_sample[:, : self.lwr_man.lat_man.data_dim]

        # Vectorize sampling from conditional distributions
        x_params = jax.vmap(self.likelihood_at, in_axes=(None, 0))(params, y_sample)

        # Sample from conditionals p(x|z) in parallel
        x_sample = jax.vmap(self.obs_man.sample, in_axes=(0, 0, None))(
            jax.random.split(key2, n), x_params, 1
        ).reshape((n, -1))

        # Concatenate samples along data dimension
        return jnp.concatenate([x_sample, yz_sample], axis=-1)

    @override
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


class AnalyticHierarchical[
    Rep: MatrixRep,
    Observable: Generative,
    SubObservable: ExponentialFamily,
    SubLatent: ExponentialFamily,
    Latent: Any,
](
    AnalyticConjugated[Rep, Observable, SubObservable, SubLatent, Latent],
    DifferentiableHierarchical[Rep, Observable, SubObservable, SubLatent, Latent],
):
    """Class for hierarchical harmoniums with deep conjugate structure.

    This class provides the algorithms needed for hierarchical harmoniums while ensuring proper typing and conjugate relationships between layers. It subclasses AnalyticConjugated to maintain the exponential family structure while adding hierarchical capabilities.

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
        Type of latent manifold (must be a AnalyticConjugated at runtime)

    Properties
    ----------
    lwr_man : AnalyticConjugated
        Lower harmonium between observable and first latent layer
    lat_man : AnalyticConjugated
        Upper harmonium between latent layers (inherited from AnalyticConjugated)

    Notes
    -----
    While the Latent type parameter is Any for typing flexibility, at runtime we verify that it is actually a AnalyticConjugated through __post_init__. This ensures proper hierarchical structure while avoiding complex generic typing.
    """

    def __post_init__(self):
        # Check that the latent manifold is a AnalyticConjugated
        if not isinstance(self.lat_man, AnalyticConjugated):
            raise TypeError(
                f"Latent manifold must be a AnalyticConjugated, got {type(self.lat_man)}"
            )

    @property
    @override
    def lwr_man(
        self,
    ) -> AnalyticConjugated[Rep, Observable, SubObservable, SubLatent, Any]:
        return self._lwr_man  # pyright: ignore[reportReturnType]

    @override
    def to_natural_likelihood(
        self,
        params: Point[Mean, Self],
    ) -> Point[Natural, AffineMap[Rep, SubLatent, SubObservable, Observable]]:
        """Convert mean parameters to natural parameters for lower likelihood."""
        obs_means, lwr_int_means, lat_means = self.split_params(params)
        lwr_lat_means = self.lat_man.split_params(lat_means)[0]
        lwr_means = self.lwr_man.join_params(obs_means, lwr_int_means, lwr_lat_means)
        return self.lwr_man.to_natural_likelihood(lwr_means)
