"""Core classes for combining and replicating manifolds.

This module provides ways to build complex manifolds from simpler ones, including products, triples, and replicated copies. These combinators let you create parameter spaces for complex models by combining simpler components, making it easy to split, join, and transform parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from .base import Manifold


@dataclass(frozen=True)
class Null(Manifold):
    """A zero-dimensional manifold with no parameters."""

    @property
    @override
    def dim(self) -> int:
        return 0


@dataclass(frozen=True)
class Tuple(Manifold, ABC):
    """This protocol defines a common interface for manifolds like Pair and Triple
    that split parameters into tuples and join them back together.

    Subclasses implement split_params with a specific return type (e.g., tuple[Array, Array]
    for Pair), and join_params with a specific number of parameters.
    """

    @abstractmethod
    def split_params(self, params: Array) -> tuple[Array, ...]:
        """Split parameters into tuple components."""

    @abstractmethod
    def join_params(self, *components: Array) -> Array:
        """Join tuple components into a single array.

        Note: Subclasses implement this with specific numbers of parameters
        (e.g., Pair takes exactly 2, Triple takes exactly 3).
        """


@dataclass(frozen=True)
class Pair[First: Manifold, Second: Manifold](Tuple, ABC):
    """Pair combines two parameter spaces, providing methods to split parameters into their respective components and join them back together. This is useful for models that have separate location and shape parameters, for example.

    In theory, this implements the Cartesian product $\\mathcal M_1 \\times \\mathcal M_2$ of two manifolds, where the dimension is the sum of the component dimensions.
    """

    # Contract

    @property
    @abstractmethod
    def fst_man(self) -> First:
        """First component manifold."""

    @property
    @abstractmethod
    def snd_man(self) -> Second:
        """Second component manifold."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return self.fst_man.dim + self.snd_man.dim

    @override
    def split_params(self, params: Array) -> tuple[Array, Array]:
        """Split parameters into first and second components.

        Args:
            params: Array of concatenated parameters

        Returns:
            Tuple of (first_params, second_params)
        """
        first_params = params[: self.fst_man.dim]
        second_params = params[self.fst_man.dim :]
        return first_params, second_params

    @override
    def join_params(self, first: Array, second: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Join component parameters into a single array.

        Args:
            first: Parameters from first manifold
            second: Parameters from second manifold

        Returns:
            Concatenated array
        """
        return jnp.concatenate([jnp.ravel(first), jnp.ravel(second)])


@dataclass(frozen=True)
class Triple[First: Manifold, Second: Manifold, Third: Manifold](Tuple, ABC):
    """Triple combines three parameter spaces, providing methods to split parameters into their respective components and join them back together."""

    # Contract

    @property
    @abstractmethod
    def fst_man(self) -> First:
        """First component manifold."""

    @property
    @abstractmethod
    def snd_man(self) -> Second:
        """Second component manifold."""

    @property
    @abstractmethod
    def trd_man(self) -> Third:
        """Third component manifold."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """Total dimension is the sum of component dimensions."""
        return self.fst_man.dim + self.snd_man.dim + self.trd_man.dim

    @override
    def split_params(self, params: Array) -> tuple[Array, Array, Array]:
        """Split parameters into first, second, and third components.

        Args:
            params: Array of concatenated parameters

        Returns:
            Tuple of (first_params, second_params, third_params)
        """
        first_dim = self.fst_man.dim
        second_dim = self.snd_man.dim

        first_params = params[:first_dim]
        second_params = params[first_dim : first_dim + second_dim]
        third_params = params[first_dim + second_dim :]

        return (first_params, second_params, third_params)

    @override
    def join_params(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, first: Array, second: Array, third: Array
    ) -> Array:
        """Join component parameters into a single array.

        Args:
            first: Parameters from first manifold
            second: Parameters from second manifold
            third: Parameters from third manifold

        Returns:
            Concatenated array
        """
        return jnp.concatenate([jnp.ravel(first), jnp.ravel(second), jnp.ravel(third)])


@dataclass(frozen=True)
class Replicated[M: Manifold](Manifold):
    """Replicated allows working with collections of parameters each of which lives on the same fundamental manifold, like mixture model components or time series observations. It provides special methods for mapping functions across the copies. In contrast with most manifolds, coordinates are represented as multidimensional (as opposed to flat) arrays.

    In theory, this implements the product manifold $\\mathcal M^n$ consisting of $n$ copies of the same manifold $\\mathcal M$, with special handling for the array structure.
    """

    # Fields

    rep_man: M
    """The base manifold being replicated."""
    n_reps: int
    """Number of copies of the base manifold."""

    # Array operations

    def get_replicate(self, p: Array, idx: Array) -> Array:
        """Get parameters for a specific replicate.

        Args:
            p: Array of replicated parameters
            idx: Index of the replicate to extract

        Returns:
            Array of parameters for that replicate
        """
        return p[idx]

    def map(self, f: Callable[[Array], Array], p: Array) -> Array:
        """Map a function across replicates, returning stacked array results.

        Args:
            f: Function that takes parameters for one replicate
            p: Array of replicated parameters

        Returns:
            Stacked results from applying f to each replicate
        """
        return jax.vmap(f)(p)

    def man_map(self, f: Callable[[Array], Array], p: Array) -> Array:
        """Map a function across replicates, returning array in replicated codomain.

        Args:
            f: Function that takes parameters and returns array
            p: Array of replicated parameters

        Returns:
            Array with results stacked across replicates
        """
        return jax.vmap(f)(p)

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """Total dimension is product of base dimension and number of copies."""
        return self.rep_man.dim * self.n_reps

    @property
    @override
    def coordinates_shape(self) -> list[int]:
        """Shape of the coordinates array is `[n_reps, *rep_man.coordinates_shape]`."""
        return [self.n_reps, *self.rep_man.coordinates_shape]
