"""Core definitions for parameterized objects. A manifold is a space that can be parameterized by a flat vector of real numbers."""

from abc import ABC
from dataclasses import dataclass
from typing import TypeVar

import jax.numpy as jnp

M = TypeVar("M", bound="Manifold")


@dataclass(frozen=True)
class Manifold(ABC):
    """A class that represents both a mathematical manifold $\\mathcal M$ and a point $p \\in \\mathcal M$ on that manifold.

    A manifold is a topological space that locally resembles Euclidean space and supports
    calculus. Points on the manifold are represented by coordinates in $\\mathbb{R}^n$ through charts.

    As a manifold $\\mathcal M$, subclasses of this class may define:

    - charts (parameterizations) giving local coordinates,
    - transition maps between different charts, or
    - the tangent bundle at a particular point.

    As a point $p \\in \\mathcal M$, instances of this class contain `params` that represent the coordinates of the point in $\\mathbb{R}^n$.
    """

    params: jnp.ndarray

    def dimension(self) -> int:
        """Return dimension of the parameter space."""
        return len(self.params)

    def __add__(self: M, other: M) -> M:
        return type(self)(self.params + other.params)

    def __sub__(self: M, other: M) -> M:
        return type(self)(self.params - other.params)

    def __mul__(self: M, scalar: float) -> M:
        return type(self)(scalar * self.params)

    def __rmul__(self: M, scalar: float) -> M:
        return self.__mul__(scalar)
