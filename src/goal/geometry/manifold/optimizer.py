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
        grad_clip: float | None = None,
    ) -> Optimizer[C, M]:
        """Create AdamW optimizer with optional gradient clipping.

        Args:
            man: Manifold
            learning_rate: Learning rate or schedule
            b1: First moment decay
            b2: Second moment decay
            weight_decay: Weight decay parameter
            grad_clip: Optional global norm clipping value

        Returns:
            Optimizer instance
        """
        if grad_clip is not None:
            # Apply gradient clipping before optimizer
            opt = chain(
                clip_by_global_norm(grad_clip),
                adamw(learning_rate, b1=b1, b2=b2, weight_decay=weight_decay),
            )
        else:
            opt = adamw(learning_rate, b1=b1, b2=b2, weight_decay=weight_decay)

        return cls(opt, man)

    @classmethod
    def sgd(
        cls,
        man: M,
        learning_rate: ScalarOrSchedule = 0.1,
        momentum: float = 0.0,
        grad_clip: float | None = None,
    ) -> Optimizer[C, M]:
        """Create SGD optimizer with optional gradient clipping.

        Args:
            man: Manifold
            learning_rate: Learning rate or schedule
            momentum: Momentum parameter
            grad_clip: Optional global norm clipping value

        Returns:
            Optimizer instance
        """
        if grad_clip is not None:
            # Apply gradient clipping before optimizer
            opt = chain(
                clip_by_global_norm(grad_clip), sgd(learning_rate, momentum=momentum)
            )
        else:
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
