"""Defines subspace relationships between manifolds.

A subspace relationship defines how one manifold $\\mathcal{N}$ can be embedded within another manifold $\\mathcal{M}$, supporting two key operations:

1. Projection $\\pi: \\mathcal{M} \\to \\mathcal{N}$, that takes a point in the larger space and returns its projected components in the subspace.

2. Translation $\\tau: \\mathcal{M} \\times \\mathcal{N} \\to \\mathcal{M}$, that takes a point in the larger space and a point in the subspace and a new point in the larger space shifted by the subspace components.

The module implements several types of subspace relationships:

- `IdentitySubspace`: Trivial embedding of a manifold in itself.
- `ComposedSubspace`: Composes two subspace relationships $\\mathcal{L} \\to \\mathcal{M} \\to \\mathcal{N}$.

#### Class Hierarchy

![Class Hierarchy](subspace.svg)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import jax.numpy as jnp

from .base import Coordinates, Manifold, Point

### Linear Subspaces ###


@dataclass(frozen=True)
class Subspace[Super: Manifold, Sub: Manifold](ABC):
    """Defines how a super-manifold $\\mathcal{M}$ can be projected onto a sub-manifold $\\mathcal{N}$.

    Type Parameters:
        Super: Super-manifold type (contravariant).
            A subspace that can handle a more general super-manifold can handle more specific ones. This contravariance is required for composition - if $S(A,B)$ and $T(B',C)$ are subspaces, then $B$ must be a subtype of $B'$ for composition to be valid.

        Sub: Sub-manifold type (covariant).
            A subspace producing elements of a more specific type can be used where one producing more general types is expected. Again, this is necessary for composition - if $S(A,B)$ composes with $T(B',C)$, then $B$ being more specific than $B'$ is safe.

    Class Attributes:
        sup_man: The super-manifold $\\mathcal{M}$
        sub_man: The sub-manifold $\\mathcal{N}$

    A subspace relationship supports two key operations that must preserve the geometric structure:

    1. Projection $\\pi: \\mathcal{M} \\to \\mathcal{N}$ that extracts the subspace components
    2. Translation $\\tau: \\mathcal{M} \\times \\mathcal{N} \\to \\mathcal{M}$ that shifts by subspace elements

    The variance choices for Super and Sub arise naturally from the need to compose subspaces:
    If $S: \\mathcal{A} \\to \\mathcal{B}_1$ and $T: \\mathcal{B}_2 \\to \\mathcal{C}$ are subspaces, then composition $(T \\circ S)$ is valid when $\\mathcal{B}_1 <: \\mathcal{B}_2$.
    """

    @property
    @abstractmethod
    def sup_man(self) -> Super:
        """Super-manifold."""

    @property
    @abstractmethod
    def sub_man(self) -> Sub:
        """Sub-manifold."""

    @abstractmethod
    def project[C: Coordinates](self, p: Point[C, Super]) -> Point[C, Sub]:
        """Project point to subspace components."""
        ...

    @abstractmethod
    def translate[C: Coordinates](
        self, p: Point[C, Super], q: Point[C, Sub]
    ) -> Point[C, Super]:
        """Translate point by subspace components."""
        ...


@dataclass(frozen=True)
class IdentitySubspace[M: Manifold](Subspace[M, M]):
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
    def project[C: Coordinates](self, p: Point[C, M]) -> Point[C, M]:
        return p

    @override
    def translate[C: Coordinates](self, p: Point[C, M], q: Point[C, M]) -> Point[C, M]:
        return p + q


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
