"""Defines subspace relationships between manifolds.

A subspace relationship defines how one manifold $\\mathcal{N}$ can be embedded within another manifold $\\mathcal{M}$, supporting two key operations:

1. Projection $\\pi: \\mathcal{M} \\to \\mathcal{N}$, that takes a point in the larger space and returns its projected components in the subspace.

2. Translation $\\tau: \\mathcal{M} \\times \\mathcal{N} \\to \\mathcal{M}$, that takes a point in the larger space and a point in the subspace and a new point in the larger space shifted by the subspace components.

The module implements several types of subspace relationships:

- `IdentitySubspace`: Trivial embedding of a manifold in itself.
- `PairSubspace`: Projects product manifold $\\mathcal{M} \\times \\mathcal{N}$ to first component.
- `TripleSubspace`: Projects triple product to first component.
- `ComposedSubspace`: Composes two subspace relationships $\\mathcal{L} \\to \\mathcal{M} \\to \\mathcal{N}$.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import jax.numpy as jnp

from .manifold import Coordinates, Manifold, Pair, Point, Triple

### Linear Subspaces ###

Super = TypeVar("Super", bound="Manifold", contravariant=True)
Sub = TypeVar("Sub", bound="Manifold", covariant=True)


@dataclass(frozen=True)
class Subspace(Generic[Super, Sub], ABC):
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

    sup_man: Super
    sub_man: Sub

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

    def __init__(self, man: M):
        super().__init__(man, man)

    def project[C: Coordinates](self, p: Point[C, M]) -> Point[C, M]:
        return p

    def translate[C: Coordinates](self, p: Point[C, M], q: Point[C, M]) -> Point[C, M]:
        return p + q


@dataclass(frozen=True)
class PairSubspace[First: Manifold, Second: Manifold](
    Subspace[Pair[First, Second], First]
):
    """Subspace relationship for a product manifold $\\mathcal M \\times \\mathcal N$."""

    def __init__(self, two_man: Pair[First, Second]):
        super().__init__(two_man, two_man.fst_man)

    def project[C: Coordinates](
        self, p: Point[C, Pair[First, Second]]
    ) -> Point[C, First]:
        first, _ = self.sup_man.split_params(p)
        return first

    def translate[C: Coordinates](
        self, p: Point[C, Pair[First, Second]], q: Point[C, First]
    ) -> Point[C, Pair[First, Second]]:
        first, second = self.sup_man.split_params(p)
        return self.sup_man.join_params(first + q, second)


@dataclass(frozen=True)
class TripleSubspace[First: Manifold, Second: Manifold, Third: Manifold](
    Subspace[Triple[First, Second, Third], First]
):
    """Subspace relationship for a product manifold $\\mathcal M \\times \\mathcal N \\times \\mathcal O$."""

    def __init__(self, thr_man: Triple[First, Second, Third]):
        super().__init__(thr_man, thr_man.fst_man)

    def project[C: Coordinates](
        self, p: Point[C, Triple[First, Second, Third]]
    ) -> Point[C, First]:
        first, _, _ = self.sup_man.split_params(p)
        return first

    def translate[C: Coordinates](
        self, p: Point[C, Triple[First, Second, Third]], q: Point[C, First]
    ) -> Point[C, Triple[First, Second, Third]]:
        first, second, third = self.sup_man.split_params(p)
        return self.sup_man.join_params(first + q, second, third)


@dataclass(frozen=True)
class ComposedSubspace[Super: Manifold, Mid: Manifold, Sub: Manifold](
    Subspace[Super, Sub]
):
    """Composition of two subspace relationships.
    Given subspaces $S: \\mathcal{M} \\to \\mathcal{L}$ and $T: \\mathcal{L} \\to \\mathcal{N}$, forms their composition $(T \\circ S): \\mathcal{M} \\to \\mathcal{N}$ where:

    - Projection: $\\pi_{T \\circ S}(p) = \\pi_T(\\pi_S(p))$
    - Translation: $\\tau_{T \\circ S}(p,q) = \\tau_S(p, \\tau_T(0,q))$

    The variance requirements of Subspace (contravariant Super, covariant Sub) ensure that composition is well-typed when the middle types are compatible.
    """

    sup_sub: Subspace[Super, Mid]
    sub_sub: Subspace[Mid, Sub]

    def __init__(
        self,
        sup_sub: Subspace[Super, Mid],
        sub_sub: Subspace[Mid, Sub],
    ):
        object.__setattr__(self, "sup_sub", sup_sub)
        object.__setattr__(self, "sub_sub", sub_sub)
        super().__init__(sup_sub.sup_man, sub_sub.sub_man)

    def project[C: Coordinates](self, p: Point[C, Super]) -> Point[C, Sub]:
        mid = self.sup_sub.project(p)
        return self.sub_sub.project(mid)

    def translate[C: Coordinates](
        self, p: Point[C, Super], q: Point[C, Sub]
    ) -> Point[C, Super]:
        mid_zero: Point[C, Mid] = Point(jnp.zeros(self.sup_sub.sub_man.dim))
        mid = self.sub_sub.translate(mid_zero, q)
        return self.sup_sub.translate(p, mid)
