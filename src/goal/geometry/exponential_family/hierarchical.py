"""Core definitions for hierarchical harmoniums.

A hierarchical harmonium is a Conjugated harmonium where the latent manifold is itself a Conjugated harmonium. This allows for deep hierarchical structure where each layer is connected through conjugate distributions. Note that this class pushes the boundary of python typing, and so requires a bit of verbose fiddling to behave somewhat correctly.

The basic structure is:

- A lower harmonium $p(x|y)$ between observable $x$ and first latent $y$
- An upper harmonium $p(y|z)$ between first latent $y$ and second latent $z$
- Together they form a joint distribution $p(x,y,z) = p(x|y)p(y|z)p(z)$

Key algorithms (conjugation, natural parameters, sampling) are implemented recursively.


#### Class Hierarchy

![Class Hierarchy](hierarchical.svg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import Coordinates, Point
from ..manifold.embedding import LinearComposedEmbedding, LinearEmbedding
from ..manifold.linear import AffineMap
from ..manifold.matrix import MatrixRep
from .base import (
    Analytic,
    Differentiable,
    ExponentialFamily,
    Mean,
    Natural,
)
from .harmonium import (
    AnalyticConjugated,
    DifferentiableConjugated,
    Harmonium,
)

### Helper Classes ###


@dataclass(frozen=True)
class ObservableEmbedding[
    Rep: MatrixRep,
    Observable: ExponentialFamily,
    IntObservable: ExponentialFamily,
    IntLatent: ExponentialFamily,
    Latent: ExponentialFamily,
](
    LinearEmbedding[
        Harmonium[Rep, Observable, IntObservable, IntLatent, Latent], Observable
    ]
):
    """Subspace relationship for a product manifold $\\mathcal M \\times \\mathcal N \\times \\mathcal O$."""

    # Fields

    hrm_man: Harmonium[Rep, Observable, IntObservable, IntLatent, Latent]

    # Overrides

    @property
    @override
    def sup_man(self) -> Harmonium[Rep, Observable, IntObservable, IntLatent, Latent]:
        return self.hrm_man

    @property
    @override
    def sub_man(self) -> Observable:
        return self.hrm_man.obs_man

    @override
    def project(
        self,
        p: Point[Mean, Harmonium[Rep, Observable, IntObservable, IntLatent, Latent]],
    ) -> Point[Mean, Observable]:
        first, _, _ = self.sup_man.split_params(p)
        return first

    @override
    def embed(
        self,
        p: Point[Natural, Observable],
    ) -> Point[Natural, Harmonium[Rep, Observable, IntObservable, IntLatent, Latent]]:
        # zero pad
        int_params: Point[Natural, ...] = self.hrm_man.int_man.zeros()
        lat_params: Point[Natural, ...] = self.hrm_man.obs_man.zeros()
        return self.sup_man.join_params(p, int_params, lat_params)

    @override
    def translate[C: Coordinates](
        self,
        p: Point[C, Harmonium[Rep, Observable, IntObservable, IntLatent, Latent]],
        q: Point[C, Observable],
    ) -> Point[C, Harmonium[Rep, Observable, IntObservable, IntLatent, Latent]]:
        first, second, third = self.sup_man.split_params(p)
        return self.sup_man.join_params(first + q, second, third)


### Undirected Harmoniums ###


@dataclass(frozen=True)
class StrongDifferentiableUndirected[
    IntRep: MatrixRep,
    Observable: Differentiable,
    IntObservable: ExponentialFamily,
    IntLatent: ExponentialFamily,
    Latent: Differentiable,
    LowerHarmonium: Differentiable,
    UpperHarmonium: Differentiable,
](
    DifferentiableConjugated[
        IntRep, Observable, IntObservable, IntLatent, UpperHarmonium
    ],
):
    """Class for hierarchical harmoniums with deep conjugate structure.

    This class provides the algorithms needed for hierarchical harmoniums while ensuring proper typing and conjugate relationships between layers. It subclasses `DifferentiableConjugated` to maintain the exponential family structure while adding hierarchical capabilities.

    Type Parameters:

    - `IntRep`: Matrix representation for interaction terms
    - `Observable`: Observable manifold type that supports sampling
    - `IntObservable`: Interactive subspace of observable manifold
    - `IntLatent`: Interactive subspace of latent manifold
    - `Latent`: Complete parameters for shared latent state
    - `LowerHarmonium`: Harmonium between observable and first latent layer. Must be a subtype of `DifferentiableConjugated[IntRep, Observable, IntObservable, IntLatent, Latent]`.
    - `UpperHarmonium`: Latent Harmonium. Requisite structure is enforced by the latent subspaces.

    Notes
    -----
    This class pushes past the boundary of the python type system, and requires a bit of fiddling to behave correctly. If python would support higher kinded types, its type parameters would rather be

        [ IntRep: MatrixRep,
        Observable: Generative,
        IntObservable: ExponentialFamily,
        IntLatent: ExponentialFamily,
        Latent: Differentiable,
        LowerHarmonium: DifferentiableConjugated,
        UpperHarmonium: Differentiable ]

    and `lwr_hrm` would have type

        LowerHarmonium[IntRep, Observable, IntObservable, IntLatent, Latent]

    instead of `LowerHarmonium`.
    """

    # Fields

    lwr_hrm: LowerHarmonium
    upr_hrm: UpperHarmonium

    # Properties

    @property
    def _abc_lwr_hrm(
        self,
    ) -> DifferentiableConjugated[IntRep, Observable, IntObservable, IntLatent, Latent]:
        return self.lwr_hrm  # pyright: ignore[reportReturnType]

    @property
    def lat_sub(self) -> LinearEmbedding[UpperHarmonium, Any]:
        return ObservableEmbedding(self.upr_hrm)  # pyright: ignore[reportReturnType, reportUnknownVariableType, reportArgumentType]

    # Overrides

    @property
    @override
    def int_rep(self) -> IntRep:
        return self._abc_lwr_hrm.snd_man.rep

    @property
    @override
    def int_obs_sub(self) -> LinearEmbedding[Observable, IntObservable]:
        return self._abc_lwr_hrm.int_obs_sub

    @property
    @override
    def int_lat_sub(self) -> LinearEmbedding[UpperHarmonium, IntLatent]:
        return LinearComposedEmbedding(self.lat_sub, self._abc_lwr_hrm.int_lat_sub)

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
        y_sample = yz_sample[:, : self._abc_lwr_hrm.lat_man.data_dim]

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
            Natural, AffineMap[IntRep, IntLatent, IntObservable, Observable]
        ],
    ) -> Point[Natural, UpperHarmonium]:
        """Compute conjugation parameters recursively."""
        rho0 = self._abc_lwr_hrm.conjugation_parameters(lkl_params)
        zero = self.lat_man.natural_point(jnp.zeros(self.lat_man.dim))
        return self.lat_sub.translate(zero, rho0)


class StrongAnalyticUndirected[
    IntRep: MatrixRep,
    Observable: Differentiable,
    IntObservable: ExponentialFamily,
    IntLatent: ExponentialFamily,
    Latent: Analytic,
    LowerHarmonium: Analytic,
    UpperHarmonium: Analytic,
](
    AnalyticConjugated[IntRep, Observable, IntObservable, IntLatent, UpperHarmonium],
    StrongDifferentiableUndirected[
        IntRep,
        Observable,
        IntObservable,
        IntLatent,
        Latent,
        LowerHarmonium,
        UpperHarmonium,
    ],
):
    """Class for hierarchical harmoniums with deep conjugate structure. Adds analytic conjugation capabilities to `DifferentiableHierarchical`."""

    # Overrides

    @property
    @override
    def _abc_lwr_hrm(
        self,
    ) -> AnalyticConjugated[IntRep, Observable, IntObservable, IntLatent, Latent]:
        """Accessor for analytic-typed lower harmonium."""
        return self.lwr_hrm  # pyright: ignore[reportReturnType]

    @override
    def to_natural_likelihood(
        self,
        params: Point[Mean, Self],
    ) -> Point[Natural, AffineMap[IntRep, IntLatent, IntObservable, Observable]]:
        """Convert mean parameters to natural parameters for lower likelihood."""
        obs_means, lwr_int_means, lat_means = self.split_params(params)
        lwr_lat_means = self.lat_sub.project(lat_means)
        lwr_means = self._abc_lwr_hrm.join_params(
            obs_means, lwr_int_means, lwr_lat_means
        )
        return self._abc_lwr_hrm.to_natural_likelihood(lwr_means)


### Front End Classes ###


@dataclass(frozen=True)
class DifferentiableUndirected[
    LowerHarmonium: Differentiable,
    UpperHarmonium: Differentiable,
](
    StrongDifferentiableUndirected[
        Any, Any, Any, Any, Any, LowerHarmonium, UpperHarmonium
    ]
):
    """Class for hierarchical harmoniums with deep conjugate structure. Uses runtime type checking to ensure proper structure. In particular:

    - Lower harmonium must be a subtype of `DifferentiableConjugated`
    - Upper harmonium must be a subtype of `DifferentiableConjugated`
    - The latent manifold of the lower harmonium must match the observable manifold of the upper harmonium.
    """

    # Fields

    def __post_init__(self):
        # Check that the subspaces are compatible - both should be DifferentiableConjugated, and lwr_hrm.lat_man should match upr_hrm.obs_man
        assert isinstance(self.lwr_hrm, DifferentiableConjugated)
        assert isinstance(self.upr_hrm, DifferentiableConjugated)
        assert self.lwr_hrm.lat_man == self.upr_hrm.obs_man


@dataclass(frozen=True)
class AnalyticUndirected[
    LowerHarmonium: Analytic,
    UpperHarmonium: Analytic,
](StrongAnalyticUndirected[Any, Any, Any, Any, Any, LowerHarmonium, UpperHarmonium]):
    """Class for hierarchical harmoniums with deep conjugate structure."""

    # Fields

    lwr_hrm: LowerHarmonium
    upr_hrm: UpperHarmonium

    def __post_init__(self):
        # Check that the subspaces are compatible - both should be DifferentiableConjugated, and lwr_hrm.lat_man should match upr_hrm.obs_man
        assert isinstance(self.lwr_hrm, AnalyticConjugated)
        assert isinstance(self.upr_hrm, AnalyticConjugated)
        assert self.lwr_hrm.lat_man == self.upr_hrm.obs_man


# ### Assymetric Harmoniums ###
#
#
# @dataclass(frozen=True)
# class AsymmetricHarmonium[
#     IntRep: MatrixRep,
#     Observable: Differentiable,
#     IntObservable: ExponentialFamily,
#     IntLatent: ExponentialFamily,
#     PriorLatent: Differentiable,
#     Latent: Differentiable,
#     Undirected: Differentiable,
# ](Pair[AffineMap[IntRep, IntLatent, IntObservable, Observable], PriorLatent]):
#     """Harmoniums represented in conjugated, rather than exponential family form."""
#
#     def __post_init__(self):
#         # Check that the `Harmonium` is a subtype of `DifferentiableConjugated`
#         assert isinstance(self.con_hrm, DifferentiableConjugated)
#
#     # Fields
#
#     con_hrm: Undirected
#     prr_sub: LinearEmbedding[Latent, PriorLatent]
#
#     @property
#     def dif_hrm(
#         self,
#     ) -> DifferentiableConjugated[IntRep, Observable, IntObservable, IntLatent, Latent]:
#         return self.con_hrm  # pyright: ignore[reportReturnType]
#
#     # Overrides
#
#     @property
#     @override
#     def fst_man(self) -> AffineMap[IntRep, IntLatent, IntObservable, Observable]:
#         return self.dif_hrm.lkl_man
#
#     @property
#     @override
#     def snd_man(self) -> PriorLatent:
#         return self.prr_sub.sub_man
#
#     # Templates
#
#     def log_partition_function(
#         self,
#         params: Point[Natural, Self],
#     ) -> Array:
#         """Compute log partition function for the harmonium."""
#         lkl_params, lat_params = self.split_params(params)
#         chi = self.dif_hrm.conjugation_baseline(lkl_params)
#         return chi + self.snd_man.log_partition_function(lat_params)
#
#     def to_mean(self, params: Point[Natural, Self]) -> Point[Mean, Self]:
#         """Convert from natural to mean parameters via $\\eta = \\nabla \\psi(\\theta)$."""
#         return self.grad(self.log_partition_function, params)
#
#     def sample(self, key: Array, params: Point[Natural, Self], n: int = 1) -> Array:
#         """Generate samples from the harmonium distribution.
#
#         Args:
#             key: PRNG key for sampling
#             params: Parameters in natural coordinates
#             n: Number of samples to generate
#
#         Returns:
#             Array of shape (n, data_dim) containing concatenated observable and latent states
#         """
#         # Split up the sampling key
#         key1, key2 = jax.random.split(key)
#
#         # Sample from adjusted latent distribution p(z)
#         lkl_params, lat_params = self.split_params(params)
#         z_sample = self.snd_man.sample(key1, lat_params, n)
#
#         def likelihood_at(y: Array) -> Point[Natural, Observable]:
#             my = self.fst_man.dom_man.sufficient_statistic(y)
#             return self.fst_man(lkl_params, my)
#
#         # Vectorize sampling from conditional distributions
#         x_params = jax.vmap(likelihood_at)(z_sample)
#
#         # Sample from conditionals p(x|z) in parallel
#         x_sample = jax.vmap(self.fst_man.cod_sub.sup_man.sample, in_axes=(0, 0, None))(
#             jax.random.split(key2, n), x_params, 1
#         ).reshape((n, -1))
#
#         # Concatenate samples along data dimension
#         return jnp.concatenate([x_sample, z_sample], axis=-1)
#
#     def to_natural_undirected(
#         self, params: Point[Natural, Self]
#     ) -> Point[Natural, Undirected]:
#         """Convert directed parameters to undirected parameters."""
#         lkl_params, prr_params = self.split_params(params)
#         lat_params0: Point[Natural, ...] = self.dif_hrm.lat_man.zeros()
#         lat_params = self.prr_sub.translate(lat_params0, prr_params)
#         dif_params = self.dif_hrm.join_conjugated(lkl_params, lat_params)
#         return cast(Point[Natural, Undirected], dif_params)
#
#     def from_mean_undirected(self, means: Point[Mean, Undirected]) -> Point[Mean, Self]:
#         """Convert directed parameters to undirected parameters."""
#         dif_means = self.dif_hrm.mean_point(means.array)
#         obs_means, int_means, lat_means0 = self.dif_hrm.split_params(dif_means)
#         lkl_means = self.dif_hrm.lkl_man.join_params(obs_means, int_means)
#         lat_means = self.prr_sub.project(lat_means0)
#         return self.join_params(lkl_means, lat_means)
#
#
# @dataclass(frozen=True)
# class StrongDirectedHarmonium[
#     IntRep: MatrixRep,
#     Observable: Differentiable,
#     IntObservable: ExponentialFamily,
#     IntLatent: ExponentialFamily,
#     PriorLatent: Differentiable,
#     Latent: Differentiable,
#     Undirected: Differentiable,
# ](Pair[AffineMap[IntRep, IntLatent, IntObservable, Observable], PriorLatent]):
#     """Harmoniums represented in conjugated, rather than exponential family form."""
#
#     def __post_init__(self):
#         # Check that the `Harmonium` is a subtype of `DifferentiableConjugated`
#         assert isinstance(self.con_hrm, DifferentiableConjugated)
#
#     # Fields
#
#     con_hrm: Undirected
#     prr_sub: LinearEmbedding[Latent, PriorLatent]
#
#     @property
#     def dif_hrm(
#         self,
#     ) -> DifferentiableConjugated[IntRep, Observable, IntObservable, IntLatent, Latent]:
#         return self.con_hrm  # pyright: ignore[reportReturnType]
#
#     # Overrides
#
#     @property
#     @override
#     def fst_man(self) -> AffineMap[IntRep, IntLatent, IntObservable, Observable]:
#         return self.dif_hrm.lkl_man
#
#     @property
#     @override
#     def snd_man(self) -> PriorLatent:
#         return self.prr_sub.sub_man
#
#     # Templates
#
#     def log_partition_function(
#         self,
#         params: Point[Natural, Self],
#     ) -> Array:
#         """Compute log partition function for the harmonium."""
#         lkl_params, lat_params = self.split_params(params)
#         chi = self.dif_hrm.conjugation_baseline(lkl_params)
#         return chi + self.snd_man.log_partition_function(lat_params)
#
#     def to_mean(self, params: Point[Natural, Self]) -> Point[Mean, Self]:
#         """Convert from natural to mean parameters via $\\eta = \\nabla \\psi(\\theta)$."""
#         return self.grad(self.log_partition_function, params)
#
#     def sample(self, key: Array, params: Point[Natural, Self], n: int = 1) -> Array:
#         """Generate samples from the harmonium distribution.
#
#         Args:
#             key: PRNG key for sampling
#             params: Parameters in natural coordinates
#             n: Number of samples to generate
#
#         Returns:
#             Array of shape (n, data_dim) containing concatenated observable and latent states
#         """
#         # Split up the sampling key
#         key1, key2 = jax.random.split(key)
#
#         # Sample from adjusted latent distribution p(z)
#         lkl_params, lat_params = self.split_params(params)
#         z_sample = self.snd_man.sample(key1, lat_params, n)
#
#         def likelihood_at(y: Array) -> Point[Natural, Observable]:
#             my = self.fst_man.dom_man.sufficient_statistic(y)
#             return self.fst_man(lkl_params, my)
#
#         # Vectorize sampling from conditional distributions
#         x_params = jax.vmap(likelihood_at)(z_sample)
#
#         # Sample from conditionals p(x|z) in parallel
#         x_sample = jax.vmap(self.fst_man.cod_sub.sup_man.sample, in_axes=(0, 0, None))(
#             jax.random.split(key2, n), x_params, 1
#         ).reshape((n, -1))
#
#         # Concatenate samples along data dimension
#         return jnp.concatenate([x_sample, z_sample], axis=-1)
#
#     def to_natural_undirected(
#         self, params: Point[Natural, Self]
#     ) -> Point[Natural, Undirected]:
#         """Convert directed parameters to undirected parameters."""
#         lkl_params, prr_params = self.split_params(params)
#         lat_params0: Point[Natural, ...] = self.dif_hrm.lat_man.zeros()
#         lat_params = self.prr_sub.translate(lat_params0, prr_params)
#         dif_params = self.dif_hrm.join_conjugated(lkl_params, lat_params)
#         return cast(Point[Natural, Undirected], dif_params)
#
#     def from_mean_undirected(self, means: Point[Mean, Undirected]) -> Point[Mean, Self]:
#         """Convert directed parameters to undirected parameters."""
#         dif_means = self.dif_hrm.mean_point(means.array)
#         obs_means, int_means, lat_means0 = self.dif_hrm.split_params(dif_means)
#         lkl_means = self.dif_hrm.lkl_man.join_params(obs_means, int_means)
#         lat_means = self.prr_sub.project(lat_means0)
#         return self.join_params(lkl_means, lat_means)
