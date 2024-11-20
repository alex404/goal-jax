"""Core definitions for parameterized objects and their geometry.

A manifold is a space that can be parameterized by coordinates in R^n. Points on
the manifold are represented by their coordinates in a particular coordinate system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import jax.numpy as jnp


# Coordinate system types
class Coordinates:
    """Base class for coordinate system type markers."""

    pass


# Type variables
C = TypeVar("C", bound=Coordinates)
M = TypeVar("M", bound="Manifold")


@dataclass(frozen=True)
class Point(Generic[C, M]):
    """A point p in a manifold M represented in coordinate system C.

    A point is defined by its coordinates in $\\mathbb R^n$ along with the manifold structure
    that gives meaning to these coordinates. The coordinate system C determines what
    operations are valid on the point.

    Args:
        params: Coordinates of the point in $\\mathbb R^n$
        manifold: The manifold this point belongs to, containing the geometric structure
    """

    params: jnp.ndarray
    manifold: M

    def __add__(self, other: "Point[C, M]") -> "Point[C, M]":
        """Add points in the same coordinate system."""
        assert self.manifold == other.manifold
        return Point(self.params + other.params, self.manifold)

    def __sub__(self, other: "Point[C, M]") -> "Point[C, M]":
        """Subtract points in the same coordinate system."""
        assert self.manifold == other.manifold
        return Point(self.params - other.params, self.manifold)

    def __mul__(self, scalar: float) -> "Point[C, M]":
        """Scalar multiplication in coordinate space."""
        return Point(scalar * self.params, self.manifold)

    def __rmul__(self, scalar: float) -> "Point[C, M]":
        """Right scalar multiplication in coordinate space."""
        return self.__mul__(scalar)


@dataclass(frozen=True)
class Manifold(ABC):
    """A manifold $\\mathcal M$ with associated geometric structure.

    A manifold is a topological space that locally resembles Euclidean space and supports
    calculus. The manifold defines the geometric structure including:

    - Dimension of the parameter space
    - Valid coordinate systems/charts
    - Transition maps between coordinate systems
    - Geometric operations like metrics and geodesics

    Manifolds are immutable and define operations on points rather than containing
    point data themselves.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return dimension of the parameter space."""
        pass
