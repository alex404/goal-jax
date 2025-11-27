"""This module provides the building blocks for representing geometric objects and their transformations.

In practice, these tools let you create and manipulate points on manifolds, calculate gradients, and implement optimization algorithms.

In theory, this implements the mathematical foundation for information geometry, where statistical models form manifolds with natural geometric structures.

See the package index for additional mathematical background on manifolds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array


# Coordinate system types (documentation-only type tags)
class Coordinates:
    """Base class for coordinate systems.

    In practice, this class acts as a type tag for documentation purposes.
    It helps clarify which coordinate system an array represents without enforcing it at runtime.

    In theory, a coordinate system (or chart) $(U, \\phi)$ consists of an open set $U \\subset \\mathcal{M}$
    and a homeomorphism $\\phi: U \\to \\mathbb{R}^n$ mapping points to their coordinate representation.
    """


class Dual[C: Coordinates](Coordinates):
    """Dual coordinates to a given coordinate system.

    In practice, these coordinates are used for gradients, optimization, and representing linear functionals.

    In theory, for a vector space $V$, its dual space $V^*$ comprises all linear functionals $v^*: V \\to \\mathbb{R}$.
    """


# Type alias for backward compatibility with code that hasn't been refactored yet
# Point[C, M] is now just Array, with coordinate system tracked via naming conventions
Point = Array


def expand_dual(p: Array) -> Array:
    """Expand dual coordinates (backward compatibility stub).

    This function is a no-op for array-based coordinates.
    Kept for backward compatibility with code using Dual[C] coordinate tracking.
    """
    return p


def reduce_dual(p: Array) -> Array:
    """Reduce dual coordinates (backward compatibility stub).

    This function is a no-op for array-based coordinates.
    Kept for backward compatibility with code using Dual[C] coordinate tracking.
    """
    return p


class Manifold(ABC):
    """A space that locally resembles Euclidean space.

    In practice, a Manifold defines operations on arrays representing points and provides methods
    to create, transform and optimize over them.

    In theory, a manifold $\\mathcal M$ is a topological space that locally resembles $\\mathbb R^n$
    with a geometric structure described by:

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
        """Shape of the flattened coordinate array for points on this manifold.

        In practice, this describes how the parameter array is physically stored.
        Most manifolds use a flat representation `[dim]`, while specialized manifolds
        like `Replicated` may use structured representations.

        Important distinction: `coordinates_shape` describes **storage layout**, while
        properties like `LinearMap.matrix_shape` describe **mathematical structure**.
        These are intentionally separate because:

        - Points must be flat to be joined together in composite manifolds (e.g., harmoniums)
        - Yet manifolds need to carry shape metadata for operations (matrix multiplication, vmapping, etc.)

        For example, a `LinearMap[Rectangular, R^2, R^3]` has:

        - `coordinates_shape = [6]` (stores as a single flat vector)
        - `matrix_shape = (3, 2)` (mathematical interpretation of those 6 parameters)
        """
        return [self.dim]

    # Array operations

    def zeros(self) -> Array:
        """Create an array of zeros with the manifold's dimension."""
        return jnp.zeros(self.coordinates_shape)

    def dot(self, p: Array, q: Array) -> Array:
        """Compute the dot product between two arrays.

        Args:
            p: First array
            q: Second array (typically the dual or gradient)

        Returns:
            Scalar dot product
        """
        return jnp.dot(p.ravel(), q.ravel())

    def value_and_grad(
        self,
        f: Callable[[Array], Array],
        p: Array,
    ) -> tuple[Array, Array]:
        """Compute both value and gradient of a function at a point.

        In practice, this is useful for optimization algorithms that need both the function value
        and its gradient.

        In theory, this computes f(p) and ∇f(p).

        Args:
            f: Function that takes an array and returns a scalar
            p: Point (array) at which to evaluate

        Returns:
            Tuple of (function value, gradient array)
        """
        value, grads = jax.value_and_grad(f)(p)
        return value, grads

    def grad(
        self,
        f: Callable[[Array], Array],
        p: Array,
    ) -> Array:
        """Compute gradients of a function at a point.

        Args:
            f: Function that takes an array and returns a scalar
            p: Point (array) at which to evaluate

        Returns:
            Gradient array
        """
        return self.value_and_grad(f, p)[1]

    def uniform_initialize(
        self,
        key: Array,
        low: float = -1.0,
        high: float = 1.0,
    ) -> Array:
        """Initialize an array from a uniformly distributed, bounded square in parameter space.

        Args:
            key: JAX random key
            low: Lower bound
            high: Upper bound

        Returns:
            Array of random parameters
        """
        return jax.random.uniform(key, shape=(self.dim,), minval=low, maxval=high)
