"""Defines subspace relationships between manifolds.

A subspace relationship defines how one manifold $\\mathcal{N}$ can be embedded within another manifold $\\mathcal{M}$, supporting two key operations:

1. Projection $\\pi: \\mathcal{M} \\to \\mathcal{N}$, that takes a point in the larger space and returns its projected components in the subspace.

2. Translation $\\tau: \\mathcal{M} \\times \\mathcal{N} \\to \\mathcal{M}$, that takes a point in the larger space and a point in the subspace and a new point in the larger space shifted by the subspace components.

The module implements several types of subspace relationships:

#### Class Hierarchy

![Class Hierarchy](subspace.svg)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from .base import Coordinates, Dual, Manifold, Point

### Linear Subspaces ###


@dataclass(frozen=True)
class Embedding[Sup: Manifold, Sub: Manifold](ABC):
    """Base class for all subspace relationships."""

    # Contract

    @property
    @abstractmethod
    def sup_man(self) -> Sup:
        """Super-manifold."""

    @property
    @abstractmethod
    def sub_man(self) -> Sub:
        """Sub-manifold."""

    @abstractmethod
    def embed[C: Coordinates](self, p: Point[C, Sub]) -> Point[C, Sup]:
        """Embed a point from the submanifold to the ambient manifold."""

    def pullback[C: Coordinates](
        self, at_point: Point[C, Sub], cotangent_vector: Point[Dual[C], Sup]
    ) -> Point[Dual[C], Sub]:
        """
        Pull back a cotangent vector from the ambient manifold to the submanifold.

        This default implementation uses vector-Jacobian products and works for
        any subspace that defines an embedding function.
        """

        def embed_func(sub_array: Array) -> Array:
            sub_point: Point[C, ...] = self.sub_man.point(sub_array)
            embedded_point = self.embed(sub_point)
            return embedded_point.array

        # Compute VJP function at the current point
        _, vjp_fn = jax.vjp(embed_func, at_point.array)

        # Apply VJP to transform the cotangent vector
        pulled_back_array = vjp_fn(cotangent_vector.array)[0]

        return self.sub_man.point(pulled_back_array)


@dataclass(frozen=True)
class LinearEmbedding[Sup: Manifold, Sub: Manifold](Embedding[Sup, Sub], ABC):
    """Linear subspace with additional operations."""

    # Contract

    @abstractmethod
    def project[C: Coordinates](self, p: Point[C, Sup]) -> Point[C, Sub]:
        """Project a point from the ambient space to the subspace."""

    def translate[C: Coordinates](
        self, p: Point[C, Sup], q: Point[C, Sub]
    ) -> Point[C, Sup]:
        """Translate point by subspace components."""
        return p + self.embed(q)

    # Overrides

    @override
    def pullback[C: Coordinates](
        self, at_point: Point[C, Sub], cotangent_vector: Point[Dual[C], Sup]
    ) -> Point[Dual[C], Sub]:
        """For linear subspaces, pullback is location-independent."""
        return self.project(cotangent_vector)


@dataclass(frozen=True)
class IdentityEmbedding[M: Manifold](LinearEmbedding[M, M]):
    """Identity subspace relationship for a manifold M."""

    man: M

    @property
    @override
    def sup_man(self) -> M:
        return self.man

    @property
    @override
    def sub_man(self) -> M:
        return self.man

    @override
    def project[C: Coordinates](self, p: Point[Dual[C], M]) -> Point[Dual[C], M]:
        return p

    @override
    def embed[C: Coordinates](self, p: Point[C, M]) -> Point[C, M]:
        return p

    @override
    def translate[C: Coordinates](self, p: Point[C, M], q: Point[C, M]) -> Point[C, M]:
        return p + q


@dataclass(frozen=True)
class ComposedEmbedding[
    Sup: Manifold,
    Mid: Manifold,
    Sub: Manifold,
](Embedding[Sup, Sub], ABC):
    """Composition of two subspace relationships.
    Given subspaces $S: \\mathcal{M} \\to \\mathcal{L}$ and $T: \\mathcal{L} \\to \\mathcal{N}$, forms their composition $(T \\circ S): \\mathcal{M} \\to \\mathcal{N}$ where:

    - Projection: $\\pi_{T \\circ S}(p) = \\pi_T(\\pi_S(p))$
    - Translation: $\\tau_{T \\circ S}(p,q) = \\tau_S(p, \\tau_T(0,q))$
    """

    # Fields

    sup_emb: Embedding[Sup, Mid]
    sub_emb: Embedding[Mid, Sub]

    # Overrides

    @property
    @override
    def sup_man(self) -> Sup:
        return self.sup_emb.sup_man

    @property
    def mid_man(self) -> Mid:
        return self.sup_emb.sub_man

    @property
    @override
    def sub_man(self) -> Sub:
        return self.sub_emb.sub_man

    @override
    def embed[C: Coordinates](self, p: Point[C, Sub]) -> Point[C, Sup]:
        mid = self.sub_emb.embed(p)
        return self.sup_emb.embed(mid)

    @override
    def pullback[C: Coordinates](
        self, at_point: Point[C, Sub], cotangent_vector: Point[Dual[C], Sup]
    ) -> Point[Dual[C], Sub]:
        # First, embed the point to get its location in the middle space
        mid_point = self.sub_emb.embed(at_point)

        # Pull back through the upper subspace
        mid_cotangent = self.sup_emb.pullback(mid_point, cotangent_vector)

        # Pull back through the lower subspace
        return self.sub_emb.pullback(at_point, mid_cotangent)


@dataclass(frozen=True)
class LinearComposedEmbedding[
    Sup: Manifold,
    Mid: Manifold,
    Sub: Manifold,
](ComposedEmbedding[Sup, Mid, Sub], LinearEmbedding[Sup, Sub]):
    """Composition of two subspace relationships.
    Given subspaces $S: \\mathcal{M} \\to \\mathcal{L}$ and $T: \\mathcal{L} \\to \\mathcal{N}$, forms their composition $(T \\circ S): \\mathcal{M} \\to \\mathcal{N}$ where:

    - Projection: $\\pi_{T \\circ S}(p) = \\pi_T(\\pi_S(p))$
    - Translation: $\\tau_{T \\circ S}(p,q) = \\tau_S(p, \\tau_T(0,q))$
    """

    # Fields

    sup_emb: LinearEmbedding[Sup, Mid]
    sub_emb: LinearEmbedding[Mid, Sub]

    # Overrides

    @override
    def project[C: Coordinates](self, p: Point[Dual[C], Sup]) -> Point[Dual[C], Sub]:
        mid = self.sup_emb.project(p)
        return self.sub_emb.project(mid)

    @override
    def translate[C: Coordinates](
        self, p: Point[C, Sup], q: Point[C, Sub]
    ) -> Point[C, Sup]:
        mid_zero: Point[C, Mid] = self.mid_man.point(
            jnp.zeros(self.sup_emb.sub_man.dim)
        )
        mid = self.sub_emb.translate(mid_zero, q)
        return self.sup_emb.translate(p, mid)
