"""This module is a thin, type-safe wrapper around `Optax`. It provides optimization tools tailored to the geometric structure of manifolds, enabling gradient-based optimization that respects the intrinsic properties of the parameter space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from jax import Array
from optax import (
    GradientTransformation,
    ScalarOrSchedule,
    adamw,
    apply_updates,
    chain,
    clip_by_global_norm,
    sgd,
)

from .base import Coordinates, Manifold

# Create an opaque type for optimizer state
OptState = NewType("OptState", object)
"""Represents the internal state of an optimizer. """


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
        """Create AdamW optimizer with optional gradient clipping."""
        opt = adamw(learning_rate, b1=b1, b2=b2, weight_decay=weight_decay)

        return cls(opt, man)

    def init(self, point: Array) -> OptState:
        """Initialize optimizer state for a point.

        Args:
            point: Parameter array

        Returns:
            Optimizer state
        """
        return OptState(self.optimizer.init(point))

    @classmethod
    def sgd(
        cls,
        man: M,
        learning_rate: ScalarOrSchedule = 0.1,
        momentum: float = 0.0,
    ) -> Optimizer[C, M]:
        """Create SGD optimizer with optional gradient clipping.

        Args:
            man: Manifold for optimization
            learning_rate: Learning rate
            momentum: Momentum parameter

        Returns:
            SGD optimizer instance
        """
        opt = sgd(learning_rate, momentum=momentum)
        return cls(opt, man)

    def update(
        self,
        opt_state: OptState,
        grads: Array,
        point: Array,
    ) -> tuple[OptState, Array]:
        """Apply optimizer update to parameters.

        Args:
            opt_state: Current optimizer state
            grads: Gradient array
            point: Current parameter array

        Returns:
            Tuple of (new optimizer state, updated parameters)
        """
        updates, new_opt_state = self.optimizer.update(
            grads,
            opt_state,
            point,
        )
        new_params = apply_updates(point, updates)
        return OptState(new_opt_state), new_params

    def with_grad_clip(self, max_norm: float) -> Optimizer[C, M]:
        """Add gradient clipping to this optimizer."""
        new_opt = chain(clip_by_global_norm(max_norm), self.optimizer)
        return type(self)(new_opt, self.opt_man)
