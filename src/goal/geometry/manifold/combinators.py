"""Core classes for combining and replicating manifolds.

This module provides ways to build complex manifolds from simpler ones, including products, triples, and replicated copies. These combinators let you create parameter spaces for complex models by combining simpler components, making it easy to split, join, and transform parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Self, override

import jax
import jax.numpy as jnp
from jax import Array

from .base import (
    Coordinates,
    Manifold,
    Point,
)


@dataclass(frozen=True)
class Null(Manifold):
    """A zero-dimensional manifold with no parameters."""

    @property
    @override
    def dim(self) -> int:
        return 0


@dataclass(frozen=True)
class Tuple(Manifold, ABC):
    """This protocol defines a common interface for manifolds like Pair and Triple that split parameters into tuples and join them back together."""

    @abstractmethod
    def split_params[C: Coordinates, M: Manifold](
        self, params: Point[C, M]
    ) -> tuple[Point[C, Any], ...]:
        """Split parameters into tuple components."""

    @abstractmethod
    def join_params[C: Coordinates](self, *components: Point[C, Any]) -> Point[C, Any]:
        """Join tuple components into a single point."""


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
    def split_params[C: Coordinates](
        self, params: Point[C, Self]
    ) -> tuple[Point[C, First], Point[C, Second]]:
        """Split parameters into first and second components."""
        first_params = params.array[: self.fst_man.dim]
        second_params = params.array[self.fst_man.dim :]
        return self.fst_man.point(first_params), self.snd_man.point(second_params)

    @override
    def join_params[C: Coordinates](
        self, first: Point[C, First], second: Point[C, Second]
    ) -> Point[C, Self]:
        """Join component parameters into a single point."""
        return self.point(
            jnp.concatenate([jnp.ravel(first.array), jnp.ravel(second.array)])
        )


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
    def split_params[C: Coordinates](
        self, params: Point[C, Self]
    ) -> tuple[Point[C, First], Point[C, Second], Point[C, Third]]:
        """Split parameters into first, second, and third components."""
        first_dim = self.fst_man.dim
        second_dim = self.snd_man.dim

        first_params = params.array[:first_dim]
        second_params = params.array[first_dim : first_dim + second_dim]
        third_params = params.array[first_dim + second_dim :]

        return (
            self.fst_man.point(first_params),
            self.snd_man.point(second_params),
            self.trd_man.point(third_params),
        )

    @override
    def join_params[C: Coordinates](
        self,
        first: Point[C, First],
        second: Point[C, Second],
        third: Point[C, Third],
    ) -> Point[C, Self]:
        """Join component parameters into a single point."""
        return self.point(
            jnp.concatenate(
                [
                    jnp.ravel(first.array),
                    jnp.ravel(second.array),
                    jnp.ravel(third.array),
                ]
            )
        )


@dataclass(frozen=True)
class Replicated[M: Manifold](Manifold):
    """Replicated allows working with collections of parameters that share the same structure, like mixture model components or time series observations. It provides special methods for mapping functions across the copies. In contrast with most manifolds, coordinates are represented as multidimensional (as opposed to flat) arrays.

    In theory, this implements the product manifold $\\mathcal M^n$ consisting of $n$ copies of the same manifold $\\mathcal M$, with special handling for the array structure.
    """

    # Fields

    rep_man: M
    """The base manifold being replicated."""
    n_reps: int
    """Number of copies of the base manifold."""

    # Templates

    def get_replicate[C: Coordinates](
        self, p: Point[C, Self], idx: Array
    ) -> Point[C, M]:
        """Get parameters for a specific copy."""
        return self.rep_man.point(p.array[idx])

    def map[C: Coordinates](
        self, f: Callable[[Point[C, M]], Array], p: Point[C, Self]
    ) -> Array:
        """Map a function across replicates, returning stacked array results."""

        def array_f(row: Array) -> Array:
            return f(self.rep_man.point(row))

        return jax.vmap(array_f)(p.array)

    def man_map[C: Coordinates, D: Coordinates, N: Manifold](
        self,
        f: Callable[[Point[C, M]], Point[D, N]],
        p: Point[C, Self],
    ) -> Point[D, Replicated[N]]:
        """Map a function across replicates, returning point in replicated codomain."""

        def array_f(row: Array) -> Array:
            return f(self.rep_man.point(row)).array

        return Point(jax.vmap(array_f)(p.array))

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
