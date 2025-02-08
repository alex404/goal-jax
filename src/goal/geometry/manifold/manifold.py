"""Core definitions for parameterized objects and their geometry.

A [Manifold][goal.geometry.manifold.manifold.Manifold] is a space that can be locally represented by $\\mathbb R^n$. A [`Point`][goal.geometry.manifold.manifold.Point] on the [`Manifold`][goal.geometry.manifold.manifold.Manifold] can then be locally represented by their [`Coordinates`][goal.geometry.manifold.manifold.Coordinates] in $\\mathbb R^n$.

#### Class Hierarchy

![Class Hierarchy](manifold.svg)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Self, override

import jax
import jax.numpy as jnp
from jax import Array


# Coordinate system types
class Coordinates:
    """Base class for coordinate systems.

    In theory, a coordinate system (or chart) $(U, \\phi)$ consists of an open set $U \\subset \\mathcal{M}$ and a homeomorphism $\\phi: U \\to \\mathbb{R}^n$ mapping points to their coordinate representation. In practice, `Coordinates` do not exist at runtime and only help type checking.
    """


class Dual[C: Coordinates](Coordinates):
    """Dual coordinates to a given coordinate system.

    For a vector space $V$, its dual space $V^*$ consists of linear functionals $f: V \\to \\mathbb{R}$.
    The duality pairing between $V$ and $V^*$ is given by:

    $$\\langle v, v^* \\rangle = \\sum_i v_i v^*_i$$

    where $v \\in V$ and $v^* \\in V^*$.
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
    """A manifold $\\mathcal M$ is a topological space that locally resembles $\\mathbb R^n$. A manifold has a geometric structure described by:

    - The dimension $n$ of the manifold,
    - valid coordinate systems,
    - Transition maps between coordinate systems, and
    - Geometric constructions like tangent spaces and metrics.

    In our implementation, a `Manifold` defines operations on `Point`s rather than containing `Point`s itself - it also acts as a "Point factory", and should be used to create points rather than the `Point` constructor itself.
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

    # Templates

    def point[Coords: Coordinates](self, array: Array) -> Point[Coords, Self]:
        """Create a point on the manifold from a coordinate array."""
        # flatten array
        return _Point(jnp.ravel(jnp.atleast_1d(array)))

    def dot[C: Coordinates](self, p: Point[C, Self], q: Point[Dual[C], Self]) -> Array:
        return jnp.dot(p.array, q.array)

    def value_and_grad[C: Coordinates](
        self,
        f: Callable[[Point[C, Self]], Array],
        point: Point[C, Self],
    ) -> tuple[Array, Point[Dual[C], Self]]:
        """Compute value and gradients of loss function.

        Returns gradients in the dual coordinate system to the input point's coordinates.
        """
        value, grads = jax.value_and_grad(f)(point)
        return value, grads

    def grad[C: Coordinates](
        self,
        f: Callable[[Point[C, Self]], Array],
        point: Point[C, Self],
    ) -> Point[Dual[C], Self]:
        """Compute gradients of loss function.

        Returns gradients in the dual coordinate system to the input point's coordinates.
        """
        return self.value_and_grad(f, point)[1]

    def uniform_initialize(
        self,
        key: Array,
        low: float = -1.0,
        high: float = 1.0,
    ) -> Point[Any, Self]:
        """Initialize a point from a uniformly distributed, bounded square in parameter space."""
        params = jax.random.uniform(key, shape=(self.dim,), minval=low, maxval=high)
        return _Point(params)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Point[C: Coordinates, M: Manifold]:
    """A point $p$ on a manifold $\\mathcal{M}$ in a given coordinate system.

    Points are identified by their coordinates $x \\in \\mathbb{R}^n$ in a particular coordinate chart $(U, \\phi)$. The coordinate space inherits a vector space structure enabling operations like:

    - Addition: $\\phi(p) + \\phi(q)$
    - Scalar multiplication: $\\alpha\\phi(p)$
    - Vector subtraction: $\\phi(p) - \\phi(q)$

    The constructor is private to prevent direct instantiation of points. Use the `Manifold` to create points instead.

    Args:
        array: Coordinate vector $x = \\phi(p) \\in \\mathbb{R}^n$
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


type Point[C: Coordinates, M: Manifold] = _Point[C, M]
"""A point on a manifold in a given coordinate system."""


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
        self, p: Point[C, Self]
    ) -> tuple[Point[C, First], Point[C, Second]]:
        """Split parameters into first and second components."""
        first_params = p.array[: self.fst_man.dim]
        second_params = p.array[self.fst_man.dim :]
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
        self, p: Point[C, Self]
    ) -> tuple[Point[C, First], Point[C, Second], Point[C, Third]]:
        """Split parameters into first, second, and third components."""
        first_dim = self.fst_man.dim
        second_dim = self.snd_man.dim

        first_params = p.array[:first_dim]
        second_params = p.array[first_dim : first_dim + second_dim]
        third_params = p.array[first_dim + second_dim :]

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
    """Manifold representing multiple copies of a base manifold."""

    rep_man: M
    """The base manifold being replicated."""
    n_reps: int
    """Number of copies of the base manifold."""

    @property
    @override
    def dim(self) -> int:
        """Total dimension is product of base dimension and number of copies."""
        return self.rep_man.dim * self.n_reps

    @override
    def point[Coords: Coordinates](self, array: Array) -> Point[Coords, Self]:
        """Create point with correct shape for replicated manifold.

        Reshapes flat array into (n_copies, base_dim) matrix.
        """
        return _Point(jnp.reshape(array, (self.n_reps, -1)))

    def get_replicate[C: Coordinates](self, p: Point[C, Self], idx: int) -> Point[C, M]:
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
        codomain: N,
        f: Callable[[Point[C, M]], Point[D, N]],
        p: Point[C, Self],
    ) -> tuple[Replicated[N], Point[D, Replicated[N]]]:
        """Map a function across replicates, returning point in replicated codomain.

        Args:
            f: Function mapping points in base manifold to points in codomain
            codomain: Target manifold for each replicate
            p: Point to map over
        """

        def array_f(row: Array) -> Array:
            return f(self.rep_man.point(row)).array

        mapped = jax.vmap(array_f)(p.array)
        cod_man = Replicated(codomain, self.n_reps)
        return cod_man, cod_man.point(mapped)
