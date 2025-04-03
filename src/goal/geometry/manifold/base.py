"""This module provides the building blocks for representing geometric objects and their transformations.

In practice, these tools let you create and manipulate points on manifolds, calculate gradients, and implement optimization algorithms.

In theory, this implements the mathematical foundation for information geometry, where statistical models form manifolds with natural geometric structures.
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

    In practice, this class acts as a type tag and doesn't contain any runtime functionality. It helps ensure the type system recognize when certain operations on points are valid or not.

    In theory, a coordinate system (or chart) $(U, \\phi)$ consists of an open set $U \\subset \\mathcal{M}$ and a homeomorphism $\\phi: U \\to \\mathbb{R}^n$ mapping points to their coordinate representation.
    """


class Dual[C: Coordinates](Coordinates):
    """Dual coordinates to a given coordinate system.

    In practice, these coordinates are used for gradients, optimization, and representing linear functionals.

    In theory, for a vector space $V$, its dual space $V^*$ comprises all linear functionals $v^*: V \\to \\mathbb{R}$.
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
    """A space that locally resembles Euclidean space.

    In practice, a Manifold defines operations on points and provides methods to create, transform and optimize over them. It acts as a factory for Point objects.

    In theory, a manifold $\\mathcal M$ is a topological space that locally resembles $\\mathbb R^n$ with a geometric structure described by:

    - The dimension $n$ of the manifold,
    - a collection of valid coordinate systems,
    - transition maps between coordinate systems, and
    - geometric constructions like tangent spaces and metrics.
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
        """Create a point in the manifold in an arbitrary coordinate system --- more specific versions of this method should be preferred where applicable."""
        # Check size matches dimension
        if array.size != self.dim:
            raise ValueError(
                f"Array size {array.size} doesn't match manifold dimension {self.dim}"
            )
        return _Point(jnp.reshape(array, self.coordinates_shape))

    def zeros[Coords: Coordinates](self) -> Point[Coords, Self]:
        """Create a point in an arbitrary coordinate system with coordinates $\\mathbf 0$."""
        return self.point(jnp.zeros(self.coordinates_shape))

    def dot[C: Coordinates](self, p: Point[C, Self], q: Point[Dual[C], Self]) -> Array:
        """Compute the dot product between a point and its dual."""
        return jnp.dot(p.array.ravel(), q.array.ravel())

    def value_and_grad[C: Coordinates](
        self,
        f: Callable[[Point[C, Self]], Array],
        p: Point[C, Self],
    ) -> tuple[Array, Point[Dual[C], Self]]:
        """Compute both value and gradient of a function at a point.

        In practice, this is useful for optimization algorithms that need both the function value and its gradient.

        In theory, this computes $f(p)$ and $\\nabla f(p)$, where the gradient is represented in the dual coordinate system.
        """
        value, grads = jax.value_and_grad(f)(p)
        return value, grads

    def grad[C: Coordinates](
        self,
        f: Callable[[Point[C, Self]], Array],
        p: Point[C, Self],
    ) -> Point[Dual[C], Self]:
        """Compute gradients of loss function.

        Returns gradients in the dual coordinate system to the input point's coordinates.
        """
        return self.value_and_grad(f, p)[1]

    def uniform_initialize(
        self,
        key: Array,
        low: float = -1.0,
        high: float = 1.0,
    ) -> Point[Any, Self]:
        """Initialize a point from a uniformly distributed, bounded square in parameter space."""
        params = jax.random.uniform(key, shape=(self.dim,), minval=low, maxval=high)
        return _Point(params)


type Point[C: Coordinates, M: Manifold] = _Point[C, M]
"""A point on a manifold in a given coordinate system."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Point[C: Coordinates, M: Manifold]:
    """A point on a specific manifold with coordinates.

    In practice, Points behave like arrays with additional type safety and mathematical operations. They're created by Manifold objects rather than directly instantiated.

    In theory, points are identified by their coordinates $x \\in \\mathbb{R}^n$ in a particular coordinate chart $(U, \\phi)$. The coordinate space inherits a vector space structure enabling operations like:

    - Addition: $\\phi(p) + \\phi(q)$
    - Scalar multiplication: $\\alpha\\phi(p)$
    - Vector subtraction: $\\phi(p) - \\phi(q)$
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
