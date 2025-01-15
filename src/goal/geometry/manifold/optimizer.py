"""Optimization primitives for exponential families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from optax import (  # pyright: ignore[reportMissingTypeStubs]
    GradientTransformation,
    adam,  # pyright: ignore[reportUnknownVariableType]
    apply_updates,  # pyright: ignore[reportUnknownVariableType]
    sgd,  # pyright: ignore[reportUnknownVariableType]
)

from .manifold import Coordinates, Dual, Manifold, Point

# Create an opaque type for optimizer state
OptState = NewType("OptState", object)


@dataclass(frozen=True)
class Optimizer[C: Coordinates, M: Manifold]:
    """Handles parameter updates using gradients in dual coordinates."""

    optimizer: GradientTransformation

    @classmethod
    def adam(
        cls,
        learning_rate: float = 0.1,
        b1: float = 0.9,
        b2: float = 0.999,
    ) -> Optimizer[C, M]:
        return cls(adam(learning_rate, b1=b1, b2=b2))

    @classmethod
    def sgd(
        cls,
        learning_rate: float = 0.1,
        momentum: float = 0.0,
    ) -> Optimizer[C, M]:
        return cls(sgd(learning_rate, momentum=momentum))

    def init(self, point: Point[Dual[C], M]) -> OptState:
        return OptState(self.optimizer.init(point.array))

    def update(
        self,
        opt_state: OptState,
        grads: Point[C, M],
        point: Point[Dual[C], M],
    ) -> tuple[OptState, Point[Dual[C], M]]:
        updates, new_opt_state = self.optimizer.update(  # pyright: ignore[reportUnknownVariableType]
            grads.array,
            opt_state,  # pyright: ignore[reportArgumentType]
            point.array,
        )
        new_params = apply_updates(point.array, updates)  # pyright: ignore[reportUnknownVariableType]
        return OptState(new_opt_state), Point(new_params)  # pyright: ignore[reportUnknownVariableType, reportArgumentType]
