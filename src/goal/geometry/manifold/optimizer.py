"""Optimization primitives for exponential families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from optax import (  # pyright: ignore[reportMissingTypeStubs]
    GradientTransformation,
    ScalarOrSchedule,  # pyright: ignore[reportUnknownVariableType]
    adamw,  # pyright: ignore[reportUnknownVariableType]
    apply_updates,  # pyright: ignore[reportUnknownVariableType]
    chain,
    clip_by_global_norm,
    sgd,  # pyright: ignore[reportUnknownVariableType]
)

from .base import Coordinates, Dual, Manifold, Point

# Create an opaque type for optimizer state
OptState = NewType("OptState", object)


@dataclass(frozen=True)
class Optimizer[C: Coordinates, M: Manifold]:
    """Handles parameter updates using gradients in dual coordinates."""

    optimizer: GradientTransformation
    opt_man: M

    @classmethod
    def adamw(
        cls,
        man: M,
        learning_rate: ScalarOrSchedule = 0.1,
        b1: float = 0.9,
        b2: float = 0.999,
        weight_decay: float = 0.0001,
    ) -> Optimizer[C, M]:
        """Create AdamW optimizer with optional gradient clipping.

        Args:
            man: Manifold
            learning_rate: Learning rate or schedule
            b1: First moment decay
            b2: Second moment decay
            weight_decay: Weight decay parameter

        Returns:
            Optimizer instance
        """
        opt = adamw(learning_rate, b1=b1, b2=b2, weight_decay=weight_decay)

        return cls(opt, man)

    def init(self, point: Point[C, M]) -> OptState:
        return OptState(self.optimizer.init(point.array))

    @classmethod
    def sgd(
        cls,
        man: M,
        learning_rate: ScalarOrSchedule = 0.1,
        momentum: float = 0.0,
    ) -> Optimizer[C, M]:
        """Create SGD optimizer with optional gradient clipping.

        Args:
            man: Manifold
            learning_rate: Learning rate or schedule
            momentum: Momentum parameter

        Returns:
            Optimizer instance
        """
        opt = sgd(learning_rate, momentum=momentum)

        return cls(opt, man)

    def update(
        self,
        opt_state: OptState,
        grads: Point[Dual[C], M],
        point: Point[C, M],
    ) -> tuple[OptState, Point[C, M]]:
        updates, new_opt_state = self.optimizer.update(  # pyright: ignore[reportUnknownVariableType]
            grads.array,
            opt_state,  # pyright: ignore[reportArgumentType]
            point.array,
        )
        new_params = apply_updates(point.array, updates)  # pyright: ignore[reportUnknownVariableType]
        return OptState(new_opt_state), self.opt_man.point(new_params)  # pyright: ignore[reportUnknownVariableType, reportArgumentType]

    def with_grad_clip(self, max_norm: float) -> Optimizer[C, M]:
        """Add gradient clipping to this optimizer."""
        new_opt = chain(clip_by_global_norm(max_norm), self.optimizer)
        return type(self)(new_opt, self.opt_man)
