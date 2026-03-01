"""Base classes for manifolds.

A `Manifold` is a stateless object that bundles operations for interpreting and manipulating flat JAX arrays of a fixed dimension. It holds no data itself --- the arrays it operates on are passed in and returned as plain `Array` values. Subclasses add structure (array manipulation, matrix representations, exponential-family operations) while preserving this pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax import Array


class Manifold(ABC):
    """A stateless bundle of dimension and array operations.

    A ``Manifold`` declares the length $n$ of the flat arrays it works with and provides operations (initialization, coordinate access, geometric computations) that give those arrays meaning. Points on the manifold are plain ``Array`` values --- the manifold object interprets them but does not store them.

    Mathematically, a smooth manifold $\\mathcal M$ of dimension $n$ is a topological space that is locally homeomorphic to $\\mathbb R^n$. A coordinate chart $(U, \\phi)$ maps an open subset $U \\subseteq \\mathcal M$ to $\\mathbb R^n$, and an atlas is a collection of charts covering $\\mathcal M$ with smooth transition maps on overlaps.
    """

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
