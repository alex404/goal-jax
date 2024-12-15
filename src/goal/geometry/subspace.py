"""Defines the subspace relationship between manifolds."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import jax.numpy as jnp

from .manifold import Coordinates, Manifold, Pair, Point, Triple

### Linear Subspaces ###

Super = TypeVar("Super", bound="Manifold", contravariant=True)
Sub = TypeVar("Sub", bound="Manifold")  # , covariant=True)


@dataclass(frozen=True)
class Subspace(Generic[Super, Sub], ABC):
    """Defines how a super-manifold $\\mathcal M$ can be projected onto a sub-manifold $\\mathcal N$.

    A subspace relationship between manifolds allows us to:
    1. Project a point in the super manifold $\\mathcal M$ to its $\\mathcal N$-manifold components
    2. Partially translate a point in $\\mathcal M$ by a point in $\\mathcal N$
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
    """Subspace relationship for a product manifold $M \\times N$."""

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
    """Subspace relationship for a product manifold $M \\times N$."""

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
    """Composition of two subspace relationships."""

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
