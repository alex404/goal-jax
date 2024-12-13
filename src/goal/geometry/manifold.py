"""Core definitions for parameterized objects and their geometry.

A [Manifold][goal.geometry.manifold.Manifold] is a space that can be locally represented by $\\mathbb R^n$. A [`Point`][goal.geometry.manifold.Point] on the [`Manifold`][goal.geometry.manifold.Manifold] can then be locally represented by their [`Coordinates`][goal.geometry.manifold.Coordinates] in $\\mathbb R^n$.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import jax
import jax.numpy as jnp
from jax import Array


# Coordinate system types
class Coordinates:
    """Base class for coordinate systems.

    In theory, a coordinate system (or chart) $(U, \\phi)$ consists of an open set $U \\subset \\mathcal{M}$ and a homeomorphism $\\phi: U \\to \\mathbb{R}^n$ mapping points to their coordinate representation. In practice, `Coordinates` do not exist at runtime and only help type checking.
    """

    ...


class Dual[C: Coordinates](Coordinates):
    """Dual coordinates to a given coordinate system.

    For a vector space $V$, its dual space $V^*$ consists of linear functionals $\\{V \\mapsto \\mathbb R\\}$.
    Where $V$ has coordinates $\\theta$, $V^*$ has dual coordinates $\\theta^*$.
    """


def reduce_dual[C: Coordinates, M: Manifold](
    p: Point[Dual[Dual[C]], M],
) -> Point[C, M]:
    """Takes a point in the dual of the dual space and returns a point in the original space."""
    return Point(p.params)


def expand_dual[C: Coordinates, M: Manifold](
    p: Point[C, M],
) -> Point[Dual[Dual[C]], M]:
    """Takes a point in the original space and returns a point in the dual of the dual space."""
    return Point(p.params)


@dataclass(frozen=True)
class Manifold(ABC):
    """A manifold $\\mathcal M$ is a topological space that locally resembles $\\mathbb R^n$. A manifold has a geometric structure described by:

    - The dimension $n$ of the manifold,
    - valid coordinate systems,
    - Transition maps between coordinate systems, and
    - Geometric constructions like tangent spaces and metrics.

    In our implementation, a `Manifold` defines operations on `Point`s rather than containing `Point`s itself - it also acts as a "Point factory", and should be used to create points rather than the `Point` constructor itself.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """The dimension of the manifold."""
        ...

    def dot[C: Coordinates](self, p: Point[C, Self], q: Point[Dual[C], Self]) -> Array:
        return jnp.dot(p.params, q.params)


C = TypeVar("C", bound=Coordinates, covariant=True)
M = TypeVar("M", bound=Manifold, covariant=True)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Point(Generic[C, M]):
    """A point $p$ on a manifold $\\mathcal{M}$ in a given coordinate system.

    Points are identified by their coordinates $x \\in \\mathbb{R}^n$ in a particular coordinate chart $(U, \\phi)$. The coordinate space inherits a vector space structure enabling operations like:

    - Addition: $\\phi(p) + \\phi(q)$
    - Scalar multiplication: $\\alpha\\phi(p)$
    - Vector subtraction: $\\phi(p) - \\phi(q)$

    Args:
        params: Coordinate vector $x = \\phi(p) \\in \\mathbb{R}^n$
    """

    params: Array

    def __add__(self, other: Point[C, M]) -> Point[C, M]:
        return Point(self.params + other.params)

    def __sub__(self, other: Point[C, M]) -> Point[C, M]:
        return Point(self.params - other.params)

    def __mul__(self, scalar: float) -> Point[C, M]:
        return Point(scalar * self.params)

    def __rmul__(self, scalar: float) -> Point[C, M]:
        return self.__mul__(scalar)

    # def __truediv__(self, scalar: float) -> Point[C, M]:
    #     return Point(self.params / scalar)

    def __truediv__(self, other: float | Array) -> Point[C, M]:
        return Point(self.params / other)


@dataclass(frozen=True)
class Pair[First: Manifold, Second: Manifold](Manifold):
    """A product manifold combining two component manifolds.

    The product structure allows operations to be performed on each component separately
    while maintaining the joint structure of the manifold.
    """

    fst_man: First
    """First component manifold"""

    snd_man: Second
    """Second component manifold"""

    @property
    def dim(self) -> int:
        """Total dimension is sum of component dimensions."""
        return self.fst_man.dim + self.snd_man.dim

    def split_params[C: Coordinates](
        self, p: Point[C, Self]
    ) -> tuple[Point[C, First], Point[C, Second]]:
        """Split parameters into first and second components."""
        first_params = p.params[: self.fst_man.dim]
        second_params = p.params[self.fst_man.dim :]
        return Point(first_params), Point(second_params)

    def join_params[C: Coordinates](
        self,
        first: Point[C, First],
        second: Point[C, Second],
    ) -> Point[C, Self]:
        """Join component parameters into a single point."""
        return Point(jnp.concatenate([first.params, second.params]))


@dataclass(frozen=True)
class Triple[First: Manifold, Second: Manifold, Third: Manifold](Manifold):
    """A product manifold combining three component manifolds.

    The product structure allows operations to be performed on each component separately
    while maintaining the joint structure of the manifold.
    """

    fst_man: First
    """First component manifold"""

    snd_man: Second
    """Second component manifold"""

    trd_man: Third
    """Third component manifold"""

    @property
    def dim(self) -> int:
        """Total dimension is sum of component dimensions."""
        return self.fst_man.dim + self.snd_man.dim + self.trd_man.dim

    def split_params[C: Coordinates](
        self, p: Point[C, Self]
    ) -> tuple[Point[C, First], Point[C, Second], Point[C, Third]]:
        """Split parameters into first, second, and third components."""
        first_dim = self.fst_man.dim
        second_dim = self.snd_man.dim

        first_params = p.params[:first_dim]
        second_params = p.params[first_dim : first_dim + second_dim]
        third_params = p.params[first_dim + second_dim :]

        return Point(first_params), Point(second_params), Point(third_params)

    def join_params[C: Coordinates](
        self,
        first: Point[C, First],
        second: Point[C, Second],
        third: Point[C, Third],
    ) -> Point[C, Self]:
        """Join component parameters into a single point."""
        return Point(jnp.concatenate([first.params, second.params, third.params]))


### Linear Subspaces ###

Super = TypeVar("Super", bound="Manifold", contravariant=True)
Sub = TypeVar("Sub", bound="Manifold", contravariant=True)


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

    def __init__(self, fst_man: First, snd_man: Second):
        super().__init__(Pair(fst_man, snd_man), fst_man)

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
