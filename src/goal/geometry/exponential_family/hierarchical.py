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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import Coordinates, Manifold, Point
from ..manifold.linear import AffineMap
from ..manifold.matrix import MatrixRep
from ..manifold.subspace import Subspace
from .base import (
    Analytic,
    Differentiable,
    ExponentialFamily,
    Generative,
    Mean,
    Natural,
)
from .harmonium import (
    AnalyticConjugated,
    DifferentiableConjugated,
    Harmonium,
)


@dataclass(frozen=True)
class ComposedSubspace[Super: Manifold, Mid: Manifold, Sub: Manifold](
    Subspace[Super, Sub]
):
    """Composition of two subspace relationships.
    Given subspaces $S: \\mathcal{M} \\to \\mathcal{L}$ and $T: \\mathcal{L} \\to \\mathcal{N}$, forms their composition $(T \\circ S): \\mathcal{M} \\to \\mathcal{N}$ where:

    - Projection: $\\pi_{T \\circ S}(p) = \\pi_T(\\pi_S(p))$
    - Translation: $\\tau_{T \\circ S}(p,q) = \\tau_S(p, \\tau_T(0,q))$
    """

    # Fields

    sup_sub: Subspace[Super, Mid]
    sub_sub: Subspace[Mid, Sub]

    # Overrides

    @property
    @override
    def sup_man(self) -> Super:
        return self.sup_sub.sup_man

    @property
    def mid_man(self) -> Mid:
        return self.sup_sub.sub_man

    @property
    @override
    def sub_man(self) -> Sub:
        return self.sub_sub.sub_man

    @override
    def project[C: Coordinates](self, p: Point[C, Super]) -> Point[C, Sub]:
        mid = self.sup_sub.project(p)
        return self.sub_sub.project(mid)

    @override
    def translate[C: Coordinates](
        self, p: Point[C, Super], q: Point[C, Sub]
    ) -> Point[C, Super]:
        mid_zero: Point[C, Mid] = self.mid_man.point(
            jnp.zeros(self.sup_sub.sub_man.dim)
        )
        mid = self.sub_sub.translate(mid_zero, q)
        return self.sup_sub.translate(p, mid)


class LatentHarmonium[Latent: ExponentialFamily](Protocol):
    def split_params[C: Coordinates](
        self, p: Point[C, Manifold]
    ) -> tuple[Point[C, Latent], Point[C, Manifold], Point[C, Manifold]]: ...

    def join_params[C: Coordinates](
        self,
        first: Point[C, Latent],
        second: Point[C, Manifold],
        third: Point[C, Manifold],
    ) -> Point[C, Manifold]: ...


@dataclass(frozen=True)
class ObservableSubspace[
    Rep: MatrixRep,
    Observable: ExponentialFamily,
    SubObservable: ExponentialFamily,
    SubLatent: ExponentialFamily,
    Latent: ExponentialFamily,
](Subspace[Harmonium[Rep, Observable, SubObservable, SubLatent, Latent], Observable]):
    """Subspace relationship for a product manifold $\\mathcal M \\times \\mathcal N \\times \\mathcal O$."""

    # Fields

    hrm_man: Harmonium[Rep, Observable, SubObservable, SubLatent, Latent]

    # Overrides

    @property
    @override
    def sup_man(self) -> Harmonium[Rep, Observable, SubObservable, SubLatent, Latent]:
        return self.hrm_man

    @property
    @override
    def sub_man(self) -> Observable:
        return self.hrm_man.obs_man

    @override
    def project[C: Coordinates](
        self, p: Point[C, Harmonium[Rep, Observable, SubObservable, SubLatent, Latent]]
    ) -> Point[C, Observable]:
        first, _, _ = self.sup_man.split_params(p)
        return first

    @override
    def translate[C: Coordinates](
        self,
        p: Point[C, Harmonium[Rep, Observable, SubObservable, SubLatent, Latent]],
        q: Point[C, Observable],
    ) -> Point[C, Harmonium[Rep, Observable, SubObservable, SubLatent, Latent]]:
        first, second, third = self.sup_man.split_params(p)
        return self.sup_man.join_params(first + q, second, third)


class StrongDifferentiableHierarchical[
    IntRep: MatrixRep,
    Observable: Generative,
    SubObservable: ExponentialFamily,
    SubLatent: ExponentialFamily,
    MidLatent: Differentiable,
    LowerHarmonium: Differentiable,
    UpperHarmonium: Differentiable,
](
    DifferentiableConjugated[
        IntRep, Observable, SubObservable, SubLatent, UpperHarmonium
    ],
    ABC,
):
    """Class for hierarchical harmoniums with deep conjugate structure.

    This class provides the algorithms needed for hierarchical harmoniums while ensuring proper typing and conjugate relationships between layers. It subclasses `DifferentiableConjugated` to maintain the exponential family structure while adding hierarchical capabilities.

    Type Parameters:

    - `IntRep`: Matrix representation for interaction terms
    - `Observable`: Observable manifold type that supports sampling
    - `SubObservable`: Interactive subspace of observable manifold
    - `SubLatent`: Interactive subspace of latent manifold
    - `MidLatent`: Complete parameters for shared latent state
    - `LowerHarmonium`: Harmonium between observable and first latent layer. Must be a subtype of `DifferentiableConjugated[IntRep, Observable, SubObservable, SubLatent, MidLatent]`.
    - `UpperHarmonium`: Latent Harmonium. Requisite structure is enforced by the latent subspaces.

    Notes
    -----
    This class pushes past the boundary of the python type system, and requires a bit of fiddling to behave correctly. If python would support higher kinded types, its type parameters would rather be

        [ IntRep: MatrixRep,
        Observable: Generative,
        SubObservable: ExponentialFamily,
        SubLatent: ExponentialFamily,
        MidLatent: Differentiable,
        LowerHarmonium: DifferentiableConjugated,
        UpperHarmonium: Differentiable ]

    and `lwr_hrm` would have type

        LowerHarmonium[IntRep, Observable, SubObservable, SubLatent, MidLatent]

    instead of `LowerHarmonium`.
    """

    # Contract

    @property
    @abstractmethod
    def _sup_lat_sub(
        self,
    ) -> Subspace[UpperHarmonium, MidLatent]:
        """Accessor for subspace relationship between upper and mid latent manifolds."""

    @property
    @abstractmethod
    def _upr_hrm(
        self,
    ) -> UpperHarmonium:
        """Accessor for general-typed upper harmonium."""

    @property
    @abstractmethod
    def _lwr_hrm(
        self,
    ) -> LowerHarmonium:
        """Accessor for general-typed lower harmonium."""

    @property
    @abstractmethod
    def _con_lwr_hrm(
        self,
    ) -> DifferentiableConjugated[
        IntRep, Observable, SubObservable, SubLatent, MidLatent
    ]:
        """Accessor for differentiable-typed lower harmonium."""

    # Overrides

    @property
    @override
    def int_rep(self) -> IntRep:
        return self._con_lwr_hrm.snd_man.rep

    @property
    @override
    def obs_sub(self) -> Subspace[Observable, SubObservable]:
        return self._con_lwr_hrm.obs_sub

    @property
    @override
    def lat_sub(self) -> Subspace[UpperHarmonium, SubLatent]:
        return ComposedSubspace(self._sup_lat_sub, self._con_lwr_hrm.lat_sub)

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
        y_sample = yz_sample[:, : self._con_lwr_hrm.lat_man.data_dim]

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
            Natural, AffineMap[IntRep, SubLatent, SubObservable, Observable]
        ],
    ) -> tuple[Array, Point[Natural, UpperHarmonium]]:
        """Compute conjugation parameters recursively."""
        chi, rho0 = self._con_lwr_hrm.conjugation_parameters(lkl_params)
        zero = self.lat_man.natural_point(jnp.zeros(self.lat_man.dim))
        rho = self._sup_lat_sub.translate(zero, rho0)
        return chi, rho


@dataclass(frozen=True)
class DifferentiableHierarchical[
    LowerHarmonium: Differentiable,
    UpperHarmonium: Differentiable,
](
    StrongDifferentiableHierarchical[
        Any, Any, Any, Any, Any, LowerHarmonium, UpperHarmonium
    ]
):
    """Class for hierarchical harmoniums with deep conjugate structure. Uses runtime type checking to ensure proper structure. In particular:

    - Lower harmonium must be a subtype of `DifferentiableConjugated`
    - Upper harmonium must be a subtype of `DifferentiableConjugated`
    - The latent manifold of the lower harmonium must match the observable manifold of the upper harmonium.
    """

    # Fields

    lwr_hrm: LowerHarmonium
    upr_hrm: UpperHarmonium

    def __post_init__(self):
        # Check that the subspaces are compatible - both should be DifferentiableConjugated, and lwr_hrm.lat_man should match upr_hrm.obs_man
        assert isinstance(self.lwr_hrm, DifferentiableConjugated)
        assert isinstance(self.upr_hrm, DifferentiableConjugated)
        assert self.lwr_hrm.lat_man == self.upr_hrm.obs_man

    # Overrides

    @property
    @override
    def _lwr_hrm(self) -> LowerHarmonium:
        return self.lwr_hrm

    @property
    @override
    def _upr_hrm(self) -> UpperHarmonium:
        return self.upr_hrm

    @property
    @override
    def _con_lwr_hrm(
        self,
    ) -> DifferentiableConjugated[Any, Any, Any, Any, Any]:
        return self.lwr_hrm  # pyright: ignore[reportReturnType]

    @property
    @override
    def _sup_lat_sub(self) -> Subspace[UpperHarmonium, Any]:
        return ObservableSubspace(self.upr_hrm)  # pyright: ignore[reportReturnType, reportUnknownVariableType, reportArgumentType]


class StrongAnalyticHierarchical[
    IntRep: MatrixRep,
    Observable: Generative,
    SubObservable: ExponentialFamily,
    SubLatent: ExponentialFamily,
    MidLatent: Analytic,
    LowerHarmonium: Analytic,
    UpperHarmonium: Analytic,
](
    AnalyticConjugated[IntRep, Observable, SubObservable, SubLatent, UpperHarmonium],
    StrongDifferentiableHierarchical[
        IntRep,
        Observable,
        SubObservable,
        SubLatent,
        MidLatent,
        LowerHarmonium,
        UpperHarmonium,
    ],
    ABC,
):
    """Class for hierarchical harmoniums with deep conjugate structure. Adds analytic conjugation capabilities to `DifferentiableHierarchical`."""

    # Overrides

    @property
    @override
    @abstractmethod
    def _con_lwr_hrm(
        self,
    ) -> AnalyticConjugated[IntRep, Observable, SubObservable, SubLatent, MidLatent]:
        """Accessor for analytic-typed lower harmonium."""

    @override
    def to_natural_likelihood(
        self,
        params: Point[Mean, Self],
    ) -> Point[Natural, AffineMap[IntRep, SubLatent, SubObservable, Observable]]:
        """Convert mean parameters to natural parameters for lower likelihood."""
        obs_means, lwr_int_means, lat_means = self.split_params(params)
        lwr_lat_means = self._sup_lat_sub.project(lat_means)
        lwr_means = self._con_lwr_hrm.join_params(
            obs_means, lwr_int_means, lwr_lat_means
        )
        return self._con_lwr_hrm.to_natural_likelihood(lwr_means)


@dataclass(frozen=True)
class AnalyticHierarchical[
    LowerHarmonium: Analytic,
    UpperHarmonium: Analytic,
](StrongAnalyticHierarchical[Any, Any, Any, Any, Any, LowerHarmonium, UpperHarmonium]):
    """Class for hierarchical harmoniums with deep conjugate structure."""

    # Fields

    lwr_hrm: LowerHarmonium
    upr_hrm: UpperHarmonium

    def __post_init__(self):
        # Check that the subspaces are compatible - both should be DifferentiableConjugated, and lwr_hrm.lat_man should match upr_hrm.obs_man
        assert isinstance(self.lwr_hrm, AnalyticConjugated)
        assert isinstance(self.upr_hrm, AnalyticConjugated)
        assert self.lwr_hrm.lat_man == self.upr_hrm.obs_man

    # Overrides

    @property
    @override
    def _lwr_hrm(self) -> LowerHarmonium:
        return self.lwr_hrm

    @property
    @override
    def _upr_hrm(self) -> UpperHarmonium:
        return self.upr_hrm

    @property
    @override
    def _con_lwr_hrm(
        self,
    ) -> AnalyticConjugated[Any, Any, Any, Any, Any]:
        return self.lwr_hrm  # pyright: ignore[reportReturnType]

    @property
    @override
    def _sup_lat_sub(self) -> Subspace[UpperHarmonium, Any]:
        return ObservableSubspace(self.upr_hrm)  # pyright: ignore[reportReturnType, reportUnknownVariableType, reportArgumentType]
