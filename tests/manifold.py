from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

import jax.numpy as jnp
from jax import Array


class Natural: ...


class Mean: ...


C = TypeVar("C", Natural, Mean)
C1 = TypeVar("C1", Natural, Mean)
C2 = TypeVar("C2", Natural, Mean)
M = TypeVar("M", bound="Manifold")
N = TypeVar("N", bound="Manifold")


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
