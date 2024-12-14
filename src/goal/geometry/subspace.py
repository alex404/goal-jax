"""Defines the subspace relationship between manifolds."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

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
class PairPairSubspace[
    FirstFirst: Manifold,
    FirstSecond: Manifold,
    Second: Manifold,
](Subspace[Pair[Pair[FirstFirst, FirstSecond], Second], FirstFirst]):
    """Subspace relationship for a product manifold $M \\times N$."""

    def __init__(self, prp_man: Pair[Pair[FirstFirst, FirstSecond], Second]):
        super().__init__(prp_man, prp_man.fst_man.fst_man)

    def project[C: Coordinates](
        self, p: Point[C, Pair[Pair[FirstFirst, FirstSecond], Second]]
    ) -> Point[C, FirstFirst]:
        fst_params = self.sup_man.split_params(p)[0]
        return self.sup_man.fst_man.split_params(fst_params)[0]

    def translate[C: Coordinates](
        self,
        p: Point[C, Pair[Pair[FirstFirst, FirstSecond], Second]],
        q: Point[C, FirstFirst],
    ) -> Point[C, Pair[Pair[FirstFirst, FirstSecond], Second]]:
        fst_params, snd_params = self.sup_man.split_params(p)
        with self.sup_man.fst_man as fm:
            fstfst_params, fstsnd_params = fm.split_params(fst_params)
            fst_params1 = fm.join_params(fstfst_params + q, fstsnd_params)
        return self.sup_man.join_params(fst_params1, snd_params)


@dataclass(frozen=True)
class TripleSubspace[First: Manifold, Second: Manifold, Third: Manifold](
    Subspace[Triple[First, Second, Third], First]
):
    """Subspace relationship for a product manifold $M \\times N$."""

    def __init__(self, fst_man: First, snd_man: Second, trd_man: Third):
        super().__init__(Triple(fst_man, snd_man, trd_man), fst_man)

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
