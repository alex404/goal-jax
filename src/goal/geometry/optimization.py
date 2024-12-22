"""Optimization primitives for exponential families."""

from typing import Callable, NamedTuple, TypeVar

import optax  # type: ignore
from jax import Array

from ..geometry import Differentiable, Natural, Point

M = TypeVar("M", bound=Differentiable)


class OptState(NamedTuple):
    """Optimization state for exponential families."""

    point: Point[Natural, M]
    opt_state: optax.OptState


def create_optimizer[M: Differentiable](
    manifold: M,
    learning_rate: float = 0.1,
) -> tuple[
    Callable[[Point[Natural, M]], OptState],  # init_fn
    Callable[[Array, OptState], tuple[Point[Natural, M], OptState]],  # update_fn
]:
    """Create an optimizer for a Differentiable exponential family.

    Args:
        manifold: The manifold to optimize over
        learning_rate: Learning rate for the optimizer

    Returns:
        init_fn: Function to initialize optimizer state
        update_fn: Function to update parameters given gradients
    """
    optimizer = optax.adam(learning_rate)

    def init_fn(init_point: Point[Natural, M]) -> OptState:
        opt_state = optimizer.init(init_point.params)
        return OptState(init_point, opt_state)

    def update_fn(grads: Array, state: OptState) -> tuple[Point[Natural, M], OptState]:
        updates, new_opt_state = optimizer.update(
            grads, state.opt_state, state.point.params
        )
        new_params = optax.apply_updates(state.point.params, updates)
        return Point(new_params), OptState(Point(new_params), new_opt_state)

    return init_fn, update_fn
