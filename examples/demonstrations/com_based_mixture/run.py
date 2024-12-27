"""Test script for mixtures of replicated COM-Poisson distributions."""

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    DifferentiableReplicated,
    Natural,
    Optimizer,
    OptState,
    Point,
    reduce_dual,
)
from goal.models import (
    CoMPoisson,
    DifferentiableMixture,
    Poisson,
)


def create_ground_truth_parameters(
    mix_man: DifferentiableMixture[
        DifferentiableReplicated[CoMPoisson], DifferentiableReplicated[Poisson]
    ],
) -> Point[
    Natural,
    DifferentiableMixture[
        DifferentiableReplicated[CoMPoisson], DifferentiableReplicated[Poisson]
    ],
]:
    """Create ground truth parameters for mixture of n_neurons COM-Poisson units."""

    # Define ground truth parameters
    # One set with high means, low dispersion (near-Poisson)
    # One set with low means, high dispersion (overdispersed)
    means1 = [10.0, 8.0, 12.0]  # Example for 3 neurons
    disp1 = [1.2, 1.1, 1.3]  # Near Poisson

    means2 = [3.0, 2.0, 4.0]
    disp2 = [0.5, 0.4, 0.6]  # Overdispersed

    # Create component parameters in mean coordinates
    components = []
    for means, disps in [(means1, disp1), (means2, disp2)]:
        rep_means = jnp.array(means)
        rep_disps = jnp.array(disps)
        # Create mean parameters for this component
        mean_params = mix_man.obs_man.join_mode_dispersion(rep_means, rep_disps)
        components.append(mean_params)

    # Set mixture weights
    weights = Point(jnp.array([0.6, 0.4]))

    # Join into mixture
    mean_mix = mix_man.join_mean_mixture(components, weights)
    return mix_man.to_natural(mean_mix)


def fit_model(
    key: Array,
    mix_man: DifferentiableMixture[
        DifferentiableReplicated[CoMPoisson], DifferentiableReplicated[Poisson]
    ],
    n_steps: int,
    sample: Array,
    learning_rate: float = 0.1,
) -> tuple[Array, Point[Natural, Any], Point[Natural, Any]]:
    """Fit mixture model using gradient descent on observable log likelihood."""

    init_params = mix_man.shape_initialize(key)

    optimizer: Optimizer[Any] = Optimizer.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(init_params)

    def loss(params: Point[Natural, Any]) -> Array:
        return -mix_man.average_log_observable_density(params, sample)

    def grad_step(
        opt_state_and_params: tuple[OptState, Point[Natural, Any]], _: Any
    ) -> tuple[tuple[OptState, Point[Natural, Any]], Array]:
        opt_state, params = opt_state_and_params
        loss_val, grads = mix_man.value_and_grad(loss, params)
        opt_state, params = optimizer.update(opt_state, reduce_dual(grads), params)
        return (opt_state, params), loss_val

    (_, final_params), lls = jax.lax.scan(
        grad_step, (opt_state, init_params), None, length=n_steps
    )
    return lls.ravel(), init_params, final_params
