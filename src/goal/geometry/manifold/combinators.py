"""Core classes for combining and replicating manifolds.

This module provides ways to build complex manifolds from simpler ones, including products, triples, and replicated copies. These combinators let you create coordinate spaces for complex models by combining simpler components, making it easy to split, join, and transform coordinates.
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
    """A zero-dimensional manifold with no coordinates."""

    @property
    @override
    def dim(self) -> int:
        return 0


@dataclass(frozen=True)
class Tuple(Manifold, ABC):
    """This protocol defines a common interface for manifolds like Pair and Triple that split coordinates into tuples and join them back together.

    Subclasses implement split_coords with a specific return type (e.g., tuple[Array, Array]
    for Pair), and join_coords with a specific number of coordinates.
    """

    @abstractmethod
    def split_coords(self, coords: Array) -> tuple[Array, ...]:
        """Split coordinates into tuple components."""

    @abstractmethod
    def join_coords(self, *components: Array) -> Array:
        """Join tuple components into a single array.

        Note: Subclasses implement this with specific numbers of coordinates
        (e.g., Pair takes exactly 2, Triple takes exactly 3).
        """


@dataclass(frozen=True)
class Pair[First: Manifold, Second: Manifold](Tuple, ABC):
    """Pair combines two coordinate spaces, providing methods to split coordinates into their respective components and join them back together. This is useful for models that have separate location and shape coordinates, for example.

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
    def split_coords(self, coords: Array) -> tuple[Array, Array]:
        """Split coordinates into first and second components.

        Args:
            coords: Array of concatenated coordinates

        Returns:
            Tuple of (first_coords, second_coords)
        """
        first_coords = coords[: self.fst_man.dim]
        second_coords = coords[self.fst_man.dim :]
        return first_coords, second_coords

    @override
    def join_coords(self, fst_coords: Array, snd_coords: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Join component coordinates into a single array.

        Args:
            fst_coords: coordinates from first manifold
            snd_coords: coordinates from second manifold

        Returns:
            Concatenated array
        """
        return jnp.concatenate([fst_coords, snd_coords])


@dataclass(frozen=True)
class Triple[First: Manifold, Second: Manifold, Third: Manifold](Tuple, ABC):
    """Triple combines three coordinate spaces, providing methods to split coordinates into their respective components and join them back together."""

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
    def split_coords(self, coords: Array) -> tuple[Array, Array, Array]:
        """Split coordinates into first, second, and third components.

        Args:
            coords: Array of concatenated coordinates

        Returns:
            Tuple of (fst_coords, snd_coords, trd_coords)
        """
        first_dim = self.fst_man.dim
        second_dim = self.snd_man.dim

        fst_coords = coords[:first_dim]
        snd_coords = coords[first_dim : first_dim + second_dim]
        trd_coords = coords[first_dim + second_dim :]

        return (fst_coords, snd_coords, trd_coords)

    @override
    def join_coords(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, fst_coords: Array, snd_coords: Array, trd_coords: Array
    ) -> Array:
        """Join component coordinates into a single array.

        Args:
            fst_coords: coordinates from first manifold
            snd_coords: coordinates from second manifold
            trd_coords: coordinates from third manifold

        Returns:
            Concatenated array
        """
        return jnp.concatenate([fst_coords, snd_coords, trd_coords])


@dataclass(frozen=True)
class Replicated[M: Manifold](Manifold):
    """Replicated allows working with collections of coordinates each of which lives on the same fundamental manifold, like mixture model components or time series observations. It provides special methods for mapping functions across the copies.

    In theory, this implements the product manifold $\\mathcal M^n$ consisting of $n$ copies of the same manifold $\\mathcal M$, with efficient mapping operations via vmap.

    In practice, each point is stored as a flat array of shape ``[n_reps * rep_man.dim]``, i.e. in row-major order.
    """

    # Fields

    rep_man: M
    """The base manifold being replicated."""
    n_reps: int
    """Number of copies of the base manifold."""

    # Array operations

    def get_replicate(self, coords: Array, idx: int) -> Array:
        """Get coordinates for a specific replicate.

        Args:
            coords: Array of replicated coordinates (flat, shape [n_reps * rep_man.dim])
            idx: Index of the replicate to extract

        Returns:
            Array of coordinates for that replicate (shape [rep_man.dim])
        """
        start = idx * self.rep_man.dim
        end = start + self.rep_man.dim
        return coords[start:end]

    def to_2d(self, coords: Array) -> Array:
        """Convert flat coordinates to 2D array for easier indexing.

        Args:
            coords: Flat array of replicated coordinates (shape ``[n_reps * rep_man.dim]``)

        Returns:
            2D array with shape ``[n_reps, rep_man.dim]``

        Raises:
            ValueError: If input is not 1D or has wrong size
        """
        if coords.ndim != 1:
            raise ValueError(
                f"to_2d expects 1D array, got shape {coords.shape}. Array is already {coords.ndim}D."
            )
        expected_size = self.n_reps * self.rep_man.dim
        if coords.size != expected_size:
            raise ValueError(
                f"to_2d expects array of size {expected_size} (n_reps={self.n_reps} * rep_man.dim={self.rep_man.dim}), got size {coords.size}"
            )
        return coords.reshape([self.n_reps, self.rep_man.dim])

    def to_1d(self, array: Array) -> Array:
        """Convert 2D array to flat coordinates.

        Args:
            array: 2D array with shape ``[n_reps, rep_man.dim]``

        Returns:
            Flat array of replicated coordinates (shape ``[n_reps * rep_man.dim]``)

        Raises:
            ValueError: If input is not 2D or has wrong shape
        """
        if array.ndim != 2:
            raise ValueError(
                f"to_1d expects 2D array, got shape {array.shape}. Array is already {array.ndim}D."
            )
        expected_shape = (self.n_reps, self.rep_man.dim)
        if array.shape != expected_shape:
            raise ValueError(
                f"to_1d expects array of shape {expected_shape}, got {array.shape}"
            )
        return array.ravel()

    def map(
        self,
        f: Callable[[Array], Array],
        coords: Array,
        flatten: bool = False,
    ) -> Array:
        """Map a function across replicates.

        By default, returns stacked 2D results for easier indexing and inspection.
        Use ``flatten=True`` when the result should be flat coordinates on another manifold.

        Args:
            f: Function that takes coordinates for one replicate (shape ``[rep_man.dim]``)
            coords: Flat array of replicated coordinates (shape ``[n_reps * rep_man.dim]``)
            flatten: If True, return flat array ``[n_reps * f_result_dim]``.
                     If False (default), return stacked array ``[n_reps, *f_result_shape]``
                     for easier indexing and inspection.

        Returns:
            Stacked 2D array by default, or flat 1D array if ``flatten=True``
        """
        shaped = coords.reshape([self.n_reps, self.rep_man.dim])
        result = jax.vmap(f)(shaped)
        return result.ravel() if flatten else result

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """Total dimension is product of base dimension and number of copies."""
        return self.rep_man.dim * self.n_reps
