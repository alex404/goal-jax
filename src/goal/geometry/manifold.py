"""Core definitions for parameterized objects and their geometry.

A [Manifold][goal.geometry.manifold.Manifold] is a space that can be locally represented by $\\mathbb R^n$. A [`Point`][goal.geometry.manifold.Point] on the [`Manifold`][goal.geometry.manifold.Manifold] can then be locally represented by their [`Coordinates`][goal.geometry.manifold.Coordinates] in $\\mathbb R^n$.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generic, Self, TypeVar

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

    For a vector space $V$, its dual space $V^*$ consists of linear functionals $f: V \\to \\mathbb{R}$.
    The duality pairing between $V$ and $V^*$ is given by:

    $$\\langle v, v^* \\rangle = \\sum_i v_i v^*_i$$

    where $v \\in V$ and $v^* \\in V^*$.
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

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: type[BaseException] | None,
    ) -> None:
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """The dimension of the manifold."""
        ...

    def dot[C: Coordinates](self, p: Point[C, Self], q: Point[Dual[C], Self]) -> Array:
        return jnp.dot(p.params, q.params)

    def value_and_grad[C: Coordinates](
        self,
        f: Callable[[Point[C, Self]], Array],
        point: Point[C, Self],
    ) -> tuple[Array, Point[Dual[C], Self]]:
        """Compute value and gradients of loss function.

        Returns gradients in the dual coordinate system to the input point's coordinates.
        """
        value, grads = jax.value_and_grad(f)(point)
        return value, Point[Dual[C], Self](grads)

    def grad[C: Coordinates](
        self,
        f: Callable[[Point[C, Self]], Array],
        point: Point[C, Self],
    ) -> Point[Dual[C], Self]:
        """Compute gradients of loss function.

        Returns gradients in the dual coordinate system to the input point's coordinates.
        """
        return self.value_and_grad(f, point)[1]

    def shape_initialize(
        self,
        key: Array,
        mu: float = 0.0,
        shp: float = 0.1,
    ) -> Point[Any, Self]:
        """Convenience function to randomly initialize the coordinates of a point based on a mean and a shape parameter --- by default this is a normal distribution, but may be overridden e.g. for bounded parameter spaces."""
        params = jax.random.normal(key, shape=(self.dim,)) * shp + mu
        return Point(params)

    def uniform_initialize(
        self,
        key: Array,
        low: float = -1.0,
        high: float = 1.0,
    ) -> Point[Any, Self]:
        """Initialize a point from a uniformly distributed, bounded square in parameter space."""
        params = jax.random.uniform(key, shape=(self.dim,), minval=low, maxval=high)
        return Point(params)


C = TypeVar("C", bound=Coordinates)
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

    def __truediv__(self, other: float | Array) -> Point[C, M]:
        return Point(self.params / other)


@dataclass(frozen=True)
class Pair[First: Manifold, Second: Manifold](Manifold):
    """The manifold given by the Cartesian product between the first and second manifold."""

    fst_man: First
    """First component manifold."""

    snd_man: Second
    """Second component manifold."""

    @property
    def dim(self) -> int:
        """Total dimension is the sum of component dimensions."""
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
    """A product manifold combining three component manifolds."""

    fst_man: First
    """First component manifold."""

    snd_man: Second
    """Second component manifold."""

    trd_man: Third
    """Third component manifold."""

    @property
    def dim(self) -> int:
        """Total dimension is the sum of component dimensions."""
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
