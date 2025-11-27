"""This module provides the building blocks for representing manifolds and points upon them.

In practice, these tools let you create and manipulate points on manifolds, calculate gradients, and implement optimization algorithms.

In theory, this implements the mathematical foundation for information geometry, where statistical models form manifolds with natural geometric structures.

See the package index for additional mathematical background on manifolds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax import Array


class Manifold(ABC):
    """A space that locally resembles Euclidean space.

    In practice, a Manifold defines operations on arrays representing points and provides methods for manipulating them.

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

    # Array operations

    def zeros(self) -> Array:
        """Create an array of zeros with the manifold's dimension."""
        return jnp.zeros(self.dim)

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
