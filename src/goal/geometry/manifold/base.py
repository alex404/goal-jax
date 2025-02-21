"""Core definitions for parameterized objects and their geometry.

A [Manifold][goal.geometry.manifold.manifold.Manifold] is a space that can be locally represented by $\\mathbb R^n$. A [`Point`][goal.geometry.manifold.manifold.Point] on the [`Manifold`][goal.geometry.manifold.manifold.Manifold] can then be locally represented by their [`Coordinates`][goal.geometry.manifold.manifold.Coordinates] in $\\mathbb R^n$.

#### Class Hierarchy

![Class Hierarchy](manifold.svg)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Self

import jax
import jax.numpy as jnp
from jax import Array


# Coordinate system types
class Coordinates:
    """Base class for coordinate systems.

    In theory, a coordinate system (or chart) $(U, \\phi)$ consists of an open set $U \\subset \\mathcal{M}$ and a homeomorphism $\\phi: U \\to \\mathbb{R}^n$ mapping points to their coordinate representation. In practice, `Coordinates` do not exist at runtime and only help type checking.
    """


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
    return _Point(p.array)


def expand_dual[C: Coordinates, M: Manifold](
    p: Point[C, M],
) -> Point[Dual[Dual[C]], M]:
    """Takes a point in the original space and returns a point in the dual of the dual space."""
    return _Point(p.array)


class Manifold(ABC):
    """A manifold $\\mathcal M$ is a topological space that locally resembles $\\mathbb R^n$. A manifold has a geometric structure described by:

    - The dimension $n$ of the manifold,
    - valid coordinate systems,
    - Transition maps between coordinate systems, and
    - Geometric constructions like tangent spaces and metrics.

    In our implementation, a `Manifold` defines operations on `Point`s rather than containing `Point`s itself - it also acts as a "Point factory", and should be used to create points rather than the `Point` constructor itself.
    """

    # Simple context manager

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: type[BaseException] | None,
    ) -> None:
        pass

    # Abstract methods

    @property
    @abstractmethod
    def dim(self) -> int:
        """The dimension of the manifold."""
        ...

    @property
    def coordinates_shape(self) -> list[int]:
        """The shape of the coordinate array."""
        return [self.dim]

    # Templates

    def point[Coords: Coordinates](self, array: Array) -> Point[Coords, Self]:
        # Check size matches dimension
        if array.size != self.dim:
            raise ValueError(
                f"Array size {array.size} doesn't match manifold dimension {self.dim}"
            )
        return _Point(jnp.reshape(array, self.coordinates_shape))

    def dot[C: Coordinates](self, p: Point[C, Self], q: Point[Dual[C], Self]) -> Array:
        return jnp.dot(p.array.ravel(), q.array.ravel())

    def value_and_grad[C: Coordinates](
        self,
        f: Callable[[Point[C, Self]], Array],
        point: Point[C, Self],
    ) -> tuple[Array, Point[Dual[C], Self]]:
        """Compute value and gradients of loss function.

        Returns gradients in the dual coordinate system to the input point's coordinates.
        """
        value, grads = jax.value_and_grad(f)(point)
        return value, grads

    def grad[C: Coordinates](
        self,
        f: Callable[[Point[C, Self]], Array],
        point: Point[C, Self],
    ) -> Point[Dual[C], Self]:
        """Compute gradients of loss function.

        Returns gradients in the dual coordinate system to the input point's coordinates.
        """
        return self.value_and_grad(f, point)[1]

    def uniform_initialize(
        self,
        key: Array,
        low: float = -1.0,
        high: float = 1.0,
    ) -> Point[Any, Self]:
        """Initialize a point from a uniformly distributed, bounded square in parameter space."""
        params = jax.random.uniform(key, shape=(self.dim,), minval=low, maxval=high)
        return _Point(params)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Point[C: Coordinates, M: Manifold]:
    """A point $p$ on a manifold $\\mathcal{M}$ in a given coordinate system.

    Points are identified by their coordinates $x \\in \\mathbb{R}^n$ in a particular coordinate chart $(U, \\phi)$. The coordinate space inherits a vector space structure enabling operations like:

    - Addition: $\\phi(p) + \\phi(q)$
    - Scalar multiplication: $\\alpha\\phi(p)$
    - Vector subtraction: $\\phi(p) - \\phi(q)$

    The constructor is private to prevent direct instantiation of points. Use the `Manifold` to create points instead.

    Args:
        array: Coordinate vector $x = \\phi(p) \\in \\mathbb{R}^n$
    """

    array: Array

    def __array__(self) -> Array:
        """Allow numpy to treat Points as arrays."""
        return self.array

    def __getitem__(self, idx: int) -> Array:
        return self.array[idx]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def __len__(self) -> int:
        return len(self.array)

    def __add__(self, other: Point[C, M]) -> Point[C, M]:
        return _Point(self.array + other.array)

    def __sub__(self, other: Point[C, M]) -> Point[C, M]:
        return _Point(self.array - other.array)

    def __neg__(self) -> Point[C, M]:
        return _Point(-self.array)

    def __mul__(self, scalar: float) -> Point[C, M]:
        return _Point(scalar * self.array)

    def __rmul__(self, scalar: float) -> Point[C, M]:
        return self.__mul__(scalar)

    def __truediv__(self, other: float | Array) -> Point[C, M]:
        return _Point(self.array / other)


type Point[C: Coordinates, M: Manifold] = _Point[C, M]
"""A point on a manifold in a given coordinate system."""
