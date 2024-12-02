"""Core definitions for parameterized objects and their geometry.

A [Manifold][goal.manifold.Manifold] is a space that can be locally represented by $\\mathbb R^n$. A [`Point`][goal.manifold.Point] on the [`Manifold`][goal.manifold.Manifold] can then be locally represented by their [`Coordinates`][goal.manifold.Coordinates] in $\\mathbb R^n$.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
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


def reduce_double_dual[C: Coordinates, M: Manifold](
    p: Point[Dual[Dual[C]], M],
) -> Point[C, M]:
    return Point(p.params)


def expand_double_dual[C: Coordinates, M: Manifold](
    p: Point[C, M],
) -> Point[Dual[Dual[C]], M]:
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
    def dimension(self) -> int:
        """The dimension of the manifold."""
        ...


@dataclass(frozen=True)
class Euclidean(Manifold):
    """Euclidean space $\\mathbb{R}^n$ of dimension $n$.

    Euclidean space consists of $n$-dimensional real vectors with the standard Euclidean distance metric

    $$d(x,y) = \\sqrt{\\sum_{i=1}^n (x_i - y_i)^2}.$$

    Euclidean space serves as the model space for more general manifolds, which locally resemble $\\mathbb{R}^n$ near each point.

    Args:
        dim: The dimension $n$ of the space
    """

    dim: int

    @property
    def dimension(self) -> int:
        """Return the dimension of the space."""
        return self.dim


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Point[C: Coordinates, M: Manifold]:
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
