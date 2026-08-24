"""Combinators for building complex manifolds from simpler ones.

Provides product manifolds of fixed arity (`Pair`, `Triple`, `Quadruple`), the clique-indexed `CliqueManifold` whose arity is read off a graph, homogeneous products (`Replicated`), and the zero-dimensional `Null`. Each combinator stores coordinates as a flat concatenation and provides ``split_coords`` / ``join_coords`` for component access.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ..algebra.clique import CliqueSet
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
class Quadruple[First: Manifold, Second: Manifold, Third: Manifold, Fourth: Manifold](
    Tuple, ABC
):
    """Product of four manifolds, with coordinates stored as ``[fst | snd | trd | fth]``."""

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

    @property
    @abstractmethod
    def fth_man(self) -> Fourth:
        """Fourth component manifold."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """Total dimension is the sum of component dimensions."""
        return self.fst_man.dim + self.snd_man.dim + self.trd_man.dim + self.fth_man.dim

    @override
    def split_coords(self, coords: Array) -> tuple[Array, Array, Array, Array]:
        """Split into ``(fst, snd, trd, fth)`` components."""
        d1 = self.fst_man.dim
        d2 = self.snd_man.dim
        d3 = self.trd_man.dim

        fst_coords = coords[:d1]
        snd_coords = coords[d1 : d1 + d2]
        trd_coords = coords[d1 + d2 : d1 + d2 + d3]
        fth_coords = coords[d1 + d2 + d3 :]

        return (fst_coords, snd_coords, trd_coords, fth_coords)

    @override
    def join_coords(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        fst_coords: Array,
        snd_coords: Array,
        trd_coords: Array,
        fth_coords: Array,
    ) -> Array:
        """Concatenate component coordinates."""
        return jnp.concatenate([fst_coords, snd_coords, trd_coords, fth_coords])


@dataclass(frozen=True)
class CliqueManifold[Root: Manifold, Cross: Manifold, Deep: Manifold](Tuple, ABC):
    """Product manifold over a graph, laid out as the three spans of one level ascent.

    Coordinates are stored as ``[root | cross | deep]``: the parameters carried by the root
    nodes, the interactions joining the root nodes to the rest of the graph, and everything
    above. The deep span is laid out exactly as the manifold on ``clq_set.ascend_level()``
    lays itself out, so the same split applies again one level up --- recursion over the
    graph is a sequence of these.

    Unlike ``Pair`` and ``Triple``, the components are not arbitrary: the graph says what
    each one is. A span may hold several cliques --- ``deep`` always does past depth two ---
    which is why the three spans are named rather than the individual blocks.
    """

    # Contract

    @property
    @abstractmethod
    def clq_set(self) -> CliqueSet:
        """The graph this manifold is defined on."""

    @property
    @abstractmethod
    def root_man(self) -> Root:
        """Manifold of the parameters carried by the root nodes."""

    @property
    @abstractmethod
    def cross_man(self) -> Cross:
        """Manifold of the interactions joining the root nodes to the rest of the graph."""

    @property
    @abstractmethod
    def deep_man(self) -> Deep:
        """Manifold of everything above the root level."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """Total dimension is the sum of the three span dimensions."""
        return self.root_man.dim + self.cross_man.dim + self.deep_man.dim

    @override
    def split_coords(self, coords: Array) -> tuple[Array, Array, Array]:
        """Split coordinates into the root, cross, and deep spans."""
        root_dim = self.root_man.dim
        cross_dim = self.cross_man.dim
        return (
            coords[:root_dim],
            coords[root_dim : root_dim + cross_dim],
            coords[root_dim + cross_dim :],
        )

    @override
    def join_coords(self, *components: Array) -> Array:
        """Concatenate the root, cross, and deep spans."""
        if len(components) != 3:
            raise ValueError(f"expected 3 spans, got {len(components)}")
        return jnp.concatenate(components)

    # Methods

    def split_level(self, coords: Array) -> tuple[Array, Array, Array]:
        """Split off one level: the root span, the cross span, and the deep span.

        The domain-facing name for :meth:`split_coords`. A graph of depth one has an empty
        cross and deep span.
        """
        return self.split_coords(coords)

    def join_level(self, root: Array, cross: Array, deep: Array) -> Array:
        """Concatenate the root, cross, and deep spans."""
        return self.join_coords(root, cross, deep)


@dataclass(frozen=True)
class Replicated[M: Manifold](Manifold, ABC):
    """Homogeneous product of $n$ copies of the same manifold, stored flat as ``[n_reps * rep_man.dim]``.

    Used for collections where every element lives on the same manifold (e.g. mixture components, time series states). The ``map`` method applies a function across copies via ``vmap``.

    Mathematically, the $n$-fold product $\\mathcal M^n = \\mathcal M \\times \\cdots \\times \\mathcal M$ with $\\dim = n \\cdot \\dim(\\mathcal M)$. Unlike ``Tuple``, the homogeneity allows ``vmap``-based operations over the copies.
    """

    # Contract

    @property
    @abstractmethod
    def rep_man(self) -> M:
        """The base manifold being replicated."""

    @property
    @abstractmethod
    def n_reps(self) -> int:
        """Number of copies of the base manifold."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """Total dimension is product of base dimension and number of copies."""
        return self.rep_man.dim * self.n_reps

    # Methods

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
