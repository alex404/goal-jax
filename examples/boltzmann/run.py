"""Boltzmann machine pattern learning and Gibbs sampling convergence."""

import itertools
from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.models.base.gaussian.boltzmann import Boltzmann

from ..shared import example_paths, jax_cli
from .types import BoltzmannPatternResults


def create_ground_truth_model() -> tuple[Boltzmann, Array]:
    """Create ground truth Boltzmann machine with structured correlations."""
    model = Boltzmann(n_neurons=4)
    params = jnp.array(
        [
            0.5,  # bias for unit 0
            1.2,  # interaction (0,1)
            -0.2,  # bias for unit 1
            0.8,  # interaction (0,2)
            0.6,  # interaction (1,2)
            0.3,  # bias for unit 2
            -0.5,  # interaction (0,3)
            -0.3,  # interaction (1,3)
            0.4,  # interaction (2,3)
            -0.1,  # bias for unit 3
        ]
    )
    return model, params


def fit_boltzmann(
    training_data: Array, n_steps: int = 2000, learning_rate: float = 0.01
) -> tuple[Boltzmann, Array, Array]:
    """Fit Boltzmann machine to training data."""
    model = Boltzmann(n_neurons=4)
    key = jax.random.PRNGKey(42)
    init_params = jax.random.normal(key, (model.dim,)) * 0.1

    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(init_params)

    def loss_fn(params: Array) -> Array:
        return -model.average_log_density(params, training_data)

    def step(state: tuple[Any, Any], _: Any) -> tuple[tuple[Any, Any], Array]:
        opt_state, params = state
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (opt_state, params), loss

    (_, final_params), losses = jax.lax.scan(
        step, (opt_state, init_params), None, length=n_steps
    )
    return model, final_params, losses


def evaluate_model(model: Boltzmann, params: Array) -> tuple[Array, Array, Array]:
    """Evaluate model on all possible states."""
    all_states = jnp.array(list(itertools.product([0, 1], repeat=4)))
    log_probs = jax.vmap(model.log_density, in_axes=(None, 0))(params, all_states)
    probs = jnp.exp(log_probs)
    log_partition = model.log_partition_function(params)
    energies = -(log_probs + log_partition)
    return all_states, probs, energies


def test_sampling_convergence(
    key: Array, model: Boltzmann, params: Array
) -> tuple[list[int], list[int], list[list[float]]]:
    """Test Gibbs sampling convergence to exact probabilities."""
    all_states = jnp.array(list(itertools.product([0, 1], repeat=4)))
    exact_probs = jnp.exp(
        jax.vmap(model.log_density, in_axes=(None, 0))(params, all_states)
    )

    sample_sizes = [50, 100, 200, 500]
    thin_levels = [1, 5, 10]
    errors_matrix = []

    for n_samples in sample_sizes:
        sample_errors = []
        for n_thin in thin_levels:
            key, subkey = jax.random.split(key)
            samples = model.sample(
                subkey, params, n=n_samples, n_burnin=50, n_thin=n_thin
            )

            empirical_probs = jnp.zeros(16)
            for k, state in enumerate(all_states):
                matches = jnp.all(
                    samples.astype(jnp.int32) == state.astype(jnp.int32), axis=1
                )
                empirical_probs = empirical_probs.at[k].set(jnp.mean(matches))

            l2_error = jnp.sqrt(jnp.sum((empirical_probs - exact_probs) ** 2))
            sample_errors.append(float(l2_error))
        errors_matrix.append(sample_errors)

    return sample_sizes, thin_levels, errors_matrix


def main():
    jax_cli()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)

    # Ground truth and training data
    gt_model, gt_params = create_ground_truth_model()
    key_sample, key_test = jax.random.split(key)
    training_data = gt_model.sample(key_sample, gt_params, n=1000)

    # Fit model
    model, fitted_params, training_losses = fit_boltzmann(training_data)

    # Evaluate
    all_states, learned_probs, energies = evaluate_model(model, fitted_params)
    _, true_probs, _ = evaluate_model(gt_model, gt_params)

    # Sampling convergence
    sample_sizes, thin_levels, errors_matrix = test_sampling_convergence(
        key_test, model, fitted_params
    )

    state_labels = ["".join(map(str, s)) for s in all_states]

    results = BoltzmannPatternResults(
        all_possible_states=all_states.tolist(),
        true_probabilities=true_probs.tolist(),
        learned_probabilities=learned_probs.tolist(),
        energy_values=energies.tolist(),
        training_losses=training_losses.tolist(),
        sampling_sizes=sample_sizes,
        sampling_thin_levels=thin_levels,
        sampling_errors_matrix=errors_matrix,
        state_labels=state_labels,
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
