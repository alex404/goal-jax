"""Core definitions for parameterized objects and their geometry.

A [Manifold][goal.manifold.Manifold] is a space that can be locally represented by $\\mathbb R^n$. A [`Point`][goal.manifold.Point] on the [`Manifold`][goal.manifold.Manifold] can then be locally represented by their [`Coordinates`][goal.manifold.Coordinates] in $\\mathbb R^n$.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

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

    def dot[C: Coordinates](self, p: Point[Dual[C], Self], q: Point[C, Self]) -> Array:
        return jnp.dot(p.params, q.params)

    def normal_initialize(
        self,
        key: Array,
        mu: float = 0.0,
        sigma: float = 1.0,
    ) -> Point[Any, Self]:
        """Initialize a point with normally distributed parameters."""
        params = jax.random.normal(key, shape=(self.dim,)) * sigma + mu
        return Point(params)

    def uniform_initialize(
        self,
        key: Array,
        low: float = -1.0,
        high: float = 1.0,
    ) -> Point[Coordinates, Self]:
        """Initialize a point with uniformly distributed parameters."""
        params = jax.random.uniform(key, shape=(self.dim,), minval=low, maxval=high)
        return Point(params)


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

    # def __truediv__(self, scalar: float) -> Point[C, M]:
    #     return Point(self.params / scalar)

    def __truediv__(self, other: float | Array) -> Point[C, M]:
        return Point(self.params / other)


@dataclass(frozen=True)
class Product[First: Manifold, Second: Manifold](Manifold):
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
