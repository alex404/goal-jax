from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

import jax.numpy as jnp
from jax import Array


class Coordinates: ...


class Natural(Coordinates): ...


class Mean(Coordinates): ...


C = TypeVar("C", Coordinates, tuple[Coordinates, Coordinates])
C1 = TypeVar("C1", bound=Coordinates)
C2 = TypeVar("C2", bound=Coordinates)
M = TypeVar("M", bound="Manifold")
N = TypeVar("N", bound="Manifold")

type NaturalMap = tuple[Natural, Mean]
type MeanMap = tuple[Mean, Natural]


@dataclass(frozen=True)
class Point(Generic[C, M]):
    params: Array


@dataclass(frozen=True)
class Manifold(ABC):
    dimension: int


@dataclass(frozen=True)
class LinearMap(Manifold, Generic[M, N]):
    """Manifold of linear maps between manifolds."""

    source: N
    target: M

    def __call__(
        self, map_point: Point[tuple[C1, C2], "LinearMap[M, N]"], p: Point[C2, N]
    ) -> Point[C1, M]:
        result = jnp.dot(map_point.params, p.params)
        return Point(result)

    def transpose(
        self, map_point: Point[tuple[C1, C2], "LinearMap[M, N]"]
    ) -> Point[tuple[C2, C1], "LinearMap[N, M]"]:
        return Point(map_point.params.T)
