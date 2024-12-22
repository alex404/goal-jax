"""Optimization primitives for exponential families."""

from dataclasses import dataclass

from optax import GradientTransformation, OptState, apply_updates  # type: ignore

from ..geometry import Manifold, Mean, Natural, Point


@dataclass(frozen=True)
class Optimizer[M: Manifold]:
    """Handles parameter updates using gradients in dual coordinates."""

    optimizer: GradientTransformation

    def init(self, point: Point[Natural, M]) -> OptState:
        return self.optimizer.init(point.params)  # type: ignore

    def update(
        self,
        grads: Point[Mean, M],
        opt_state: OptState,
        point: Point[Natural, M],
    ) -> tuple[Point[Natural, M], OptState]:
        updates, new_opt_state = self.optimizer.update(  # type: ignore
            grads.params, opt_state, point.params
        )
        new_params = apply_updates(point.params, updates)  # type: ignore
        return Point(new_params), new_opt_state  # type: ignore
