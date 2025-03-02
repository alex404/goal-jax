"""Core definitions for parameterized objects and their geometry.

A [Manifold][goal.geometry.manifold.manifold.Manifold] is a space that can be locally represented by $\\mathbb R^n$. A [`Point`][goal.geometry.manifold.manifold.Point] on the [`Manifold`][goal.geometry.manifold.manifold.Manifold] can then be locally represented by their [`Coordinates`][goal.geometry.manifold.manifold.Coordinates] in $\\mathbb R^n$.

#### Class Hierarchy

![Class Hierarchy](manifold.svg)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Self, override

import jax
import jax.numpy as jnp
from jax import Array

from .base import (
    Coordinates,
    Manifold,
    Point,
    _Point,  # pyright: ignore[reportPrivateUsage]
)


@dataclass(frozen=True)
class Null(Manifold):
    """Dummy manifold for use when nothing is being masked."""

    @property
    @override
    def dim(self) -> int:
        return 0


@dataclass(frozen=True)
class Pair[First: Manifold, Second: Manifold](Manifold, ABC):
    """The manifold given by the Cartesian product between the first and second manifold."""

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
        """Total dimension is the sum of component dimensions."""
        return self.fst_man.dim + self.snd_man.dim

    # Templates

    def split_params[C: Coordinates](
        self, params: Point[C, Self]
    ) -> tuple[Point[C, First], Point[C, Second]]:
        """Split parameters into first and second components."""
        first_params = params.array[: self.fst_man.dim]
        second_params = params.array[self.fst_man.dim :]
        return self.fst_man.point(first_params), self.snd_man.point(second_params)

    def join_params[C: Coordinates](
        self,
        first: Point[C, First],
        second: Point[C, Second],
    ) -> Point[C, Self]:
        """Join component_Point parameters into a single point."""
        return self.point(
            jnp.concatenate([jnp.ravel(first.array), jnp.ravel(second.array)])
        )


@dataclass(frozen=True)
class Triple[First: Manifold, Second: Manifold, Third: Manifold](Manifold, ABC):
    """A product manifold combining three component manifolds."""

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

    # Templates

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
    """Manifold representing multiple copies of a base manifold. Unlike standard points defined by flat arrays, the shape of a Point on a replicated manifold is (n_reps, base_dim)."""

    # Fields

    rep_man: M
    """The base manifold being replicated."""
    n_reps: int
    """Number of copies of the base manifold."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """Total dimension is product of base dimension and number of copies."""
        return self.rep_man.dim * self.n_reps

    @property
    @override
    def coordinates_shape(self) -> list[int]:
        """Shape of the coordinate array."""
        return [self.n_reps, *self.rep_man.coordinates_shape]

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
        """Map a function across replicates, returning point in replicated codomain.

        Args:
            f: Function mapping points in base manifold to points in codomain
            p: Point to map over
        """

        def array_f(row: Array) -> Array:
            return f(self.rep_man.point(row)).array

        return _Point(jax.vmap(array_f)(p.array))
