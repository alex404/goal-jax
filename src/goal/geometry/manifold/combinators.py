"""Combinators for building complex manifolds from simpler ones.

Provides product manifolds (`Pair`, `Triple`), homogeneous products (`Replicated`), and the zero-dimensional `Null`. Each combinator stores coordinates as a flat concatenation and provides ``split_coords`` / ``join_coords`` for component access.
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
    """Abstract Cartesian product of manifolds, with coordinates stored as a flat concatenation.

    Mathematically, the Cartesian product $\\mathcal M_1 \\times \\cdots \\times \\mathcal M_k$ has $\\dim = \\sum_i \\dim(\\mathcal M_i)$. Subclasses (``Pair``, ``Triple``) fix the arity and provide typed ``split_coords`` / ``join_coords``.
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
    """Binary Cartesian product, with coordinates stored as ``[fst | snd]``."""

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
        """Split into ``(fst, snd)`` components."""
        first_coords = coords[: self.fst_man.dim]
        second_coords = coords[self.fst_man.dim :]
        return first_coords, second_coords

    @override
    def join_coords(self, fst_coords: Array, snd_coords: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Concatenate component coordinates."""
        return jnp.concatenate([fst_coords, snd_coords])


@dataclass(frozen=True)
class Triple[First: Manifold, Second: Manifold, Third: Manifold](Tuple, ABC):
    """Product of three manifolds, with coordinates stored as ``[fst | snd | trd]``."""

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
        """Split into ``(fst, snd, trd)`` components."""
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
        """Concatenate component coordinates."""
        return jnp.concatenate([fst_coords, snd_coords, trd_coords])


@dataclass(frozen=True)
class Replicated[M: Manifold](Manifold):
    """Homogeneous product of $n$ copies of the same manifold, stored flat as ``[n_reps * rep_man.dim]``.

    Used for collections where every element lives on the same manifold (e.g. mixture components, time series states). The ``map`` method applies a function across copies via ``vmap``.

    Mathematically, the $n$-fold product $\\mathcal M^n = \\mathcal M \\times \\cdots \\times \\mathcal M$ with $\\dim = n \\cdot \\dim(\\mathcal M)$. Unlike ``Tuple``, the homogeneity allows ``vmap``-based operations over the copies.
    """

    # Fields

    rep_man: M
    """The base manifold being replicated."""
    n_reps: int
    """Number of copies of the base manifold."""

    # Array operations

    def get_replicate(self, coords: Array, idx: int) -> Array:
        """Extract the ``idx``-th replicate from flat coordinates."""
        start = idx * self.rep_man.dim
        end = start + self.rep_man.dim
        return coords[start:end]

    def to_2d(self, coords: Array) -> Array:
        """Convert flat coordinates to 2D array with shape ``[n_reps, rep_man.dim]``."""
        return coords.reshape([self.n_reps, self.rep_man.dim])

    def to_1d(self, array: Array) -> Array:
        """Convert 2D array back to flat coordinates."""
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
