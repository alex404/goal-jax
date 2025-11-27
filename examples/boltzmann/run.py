"""Test script for Boltzmann machine pattern learning.

This example demonstrates how Boltzmann machines can learn and reproduce
binary patterns. We create some interpretable 2x2 binary patterns, fit
a Boltzmann machine to learn these patterns, and then visualize:

1. The ground truth patterns and their frequencies
2. The learned model parameters (bias and interactions)
3. The energy landscape over all possible states
4. Generated samples from the learned model

This provides a nice sanity check that the Boltzmann machine implementation
is working correctly and can capture pattern structures in binary data.
"""

import itertools
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import Natural, Optimizer, OptState
from goal.models.base.gaussian.boltzmann import Boltzmann

from ..shared import example_paths, initialize_jax
from .types import BoltzmannPatternResults


def create_ground_truth_model() -> tuple[Boltzmann, Array]:
    """Create a ground truth Boltzmann machine with meaningful correlations."""
    model = Boltzmann(n_neurons=4)

    # Create parameters that encourage some patterns over others
    # Triangular storage (row-major): [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3)]
    ground_truth_params = jnp.array(
        [
            0.5,  # (0,0) - bias for unit 0
            1.2,  # (1,0) - interaction between units 0 and 1
            -0.2,  # (1,1) - bias for unit 1
            0.8,  # (2,0) - interaction between units 0 and 2
            0.6,  # (2,1) - interaction between units 1 and 2
            0.3,  # (2,2) - bias for unit 2
            -0.5,  # (3,0) - interaction between units 0 and 3
            -0.3,  # (3,1) - interaction between units 1 and 3
            0.4,  # (3,2) - interaction between units 2 and 3
            -0.1,  # (3,3) - bias for unit 3
        ]
    )

    params = ground_truth_params
    return model, params


def generate_training_data(
    key: Array,
    ground_truth_model: Boltzmann,
    ground_truth_natural_params: Array,
    n_samples: int = 1000,
) -> Array:
    """Generate training data from ground truth Boltzmann machine."""
    return ground_truth_model.sample(key, ground_truth_natural_params, n=n_samples)


def fit_boltzmann_machine(
    training_data: Array, n_steps: int = 2000, learning_rate: float = 0.01
) -> tuple[Boltzmann, Array, Array]:
    """Fit a Boltzmann machine to the training data using gradient descent.

    Args:
        training_data: Array of shape (n_samples, 4) with binary training data
        n_steps: Number of optimization steps
        learning_rate: Learning rate for optimizer

    Returns:
        model: Fitted Boltzmann machine
        natural_params: Learned parameters in natural coordinates
        losses: Training loss history
    """
    model = Boltzmann(n_neurons=4)

    # Get unique patterns and their frequencies from training data
    unique_patterns, inverse_indices = jnp.unique(
        training_data, axis=0, return_inverse=True
    )
    pattern_counts = jnp.bincount(inverse_indices)

    print(f"Found {len(unique_patterns)} unique patterns in training data")
    print(f"Pattern counts: {pattern_counts}")

    # Initialize parameters at small random values (triangular storage)
    key = jax.random.PRNGKey(42)
    init_params_flat = jax.random.normal(key, (model.dim,)) * 0.1
    init_natural_params = init_params_flat

    # Setup optimizer
    optimizer: Optimizer[Natural, Boltzmann] = Optimizer.adamw(
        model, learning_rate=learning_rate
    )
    opt_state = optimizer.init(init_natural_params)

    def cross_entropy_loss(natural_params: Array) -> Array:
        """Negative log-likelihood loss using exponential family interface."""
        return -model.average_log_density(natural_params, training_data)

    def grad_step(
        opt_state_and_params: tuple[OptState, Array], _: Any
    ) -> tuple[tuple[OptState, Array], Array]:
        # Compute loss and gradients
        opt_state, natural_params = opt_state_and_params
        loss_val, grads = model.value_and_grad(cross_entropy_loss, natural_params)

        # Get updates from optimizer
        opt_state, natural_params = optimizer.update(opt_state, grads, natural_params)

        return (opt_state, natural_params), loss_val

    print(f"Starting optimization with {n_steps} steps...")
    (_, final_natural_params), losses = jax.lax.scan(
        grad_step, (opt_state, init_natural_params), None, length=n_steps
    )

    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")

    # Extract final parameters for inspection
    print(f"Final parameters shape: {final_natural_params.shape}")
    print(f"Final parameters: {final_natural_params}")

    return model, final_natural_params, losses


def evaluate_model(
    model: Boltzmann, natural_params: Array
) -> tuple[Array, Array, Array]:
    """Evaluate the fitted model on all possible 2x2 binary states.

    Returns:
        all_states: All possible 4-bit binary states
        probabilities: Model probability for each state
        energies: Negative log unnormalized probabilities
    """
    # Generate all possible 2x2 binary states
    all_states = jnp.array(list(itertools.product([0, 1], repeat=4)))

    # Use exponential family interface directly
    log_probs = jax.vmap(model.log_density, in_axes=(None, 0))(natural_params, all_states)
    probabilities = jnp.exp(log_probs)

    # Energies are negative log unnormalized probabilities
    log_partition = model.log_partition_function(natural_params)
    energies = -(log_probs + log_partition)

    return all_states, probabilities, energies


def test_sampling_convergence(
    key: Array, model: Boltzmann, natural_params: Array
) -> tuple[list[int], list[int], list[list[float]]]:
    """Test that Gibbs sampling converges to exact probabilities.

    This is a numerical simulation test independent of the learning problem.
    We test how both sample size and number of Gibbs steps affect convergence
    to the exact probabilities computed via the exponential family interface.
    """
    # Get exact probabilities using exponential family interface
    all_states = jnp.array(list(itertools.product([0, 1], repeat=4)))
    exact_probs = jnp.exp(
        jax.vmap(model.log_density, in_axes=(None, 0))(natural_params, all_states)
    )

    # Test different sample sizes and thinning levels (reduced for speed)
    sample_sizes = [50, 100, 200, 500]
    thin_levels = [
        1,
        5,
        10,
    ]  # Different levels of thinning (Gibbs steps between samples)

    errors_matrix = []  # errors_matrix[i][j] = error for sample_sizes[i] and thin_levels[j]

    print("Testing Gibbs sampling convergence...")

    for n_samples in sample_sizes:
        sample_errors = []

        for n_thin in thin_levels:
            # Generate samples using Gibbs sampling
            key, subkey = jax.random.split(key)
            samples = model.sample(
                subkey, natural_params, n=n_samples, n_burnin=50, n_thin=n_thin
            )

            # Compute empirical probabilities
            empirical_probs = jnp.zeros(16)
            for k, state in enumerate(all_states):
                # Convert both to same type for comparison
                matches = jnp.all(
                    samples.astype(jnp.int32) == state.astype(jnp.int32), axis=1
                )
                empirical_probs = empirical_probs.at[k].set(jnp.mean(matches))

            # Compute L2 error
            l2_error = jnp.sqrt(jnp.sum((empirical_probs - exact_probs) ** 2))
            sample_errors.append(float(l2_error))

            print(
                f"  n_samples={n_samples:4d}, thin={n_thin:2d}: L2 error={l2_error:.4f}"
            )

        errors_matrix.append(sample_errors)

    return sample_sizes, thin_levels, errors_matrix


def run_boltzmann_analysis(key: Array) -> BoltzmannPatternResults:
    """Run the complete Boltzmann pattern learning analysis."""

    # 1. Create ground truth model
    ground_truth_model, ground_truth_params = create_ground_truth_model()

    # 2. Generate training data from ground truth
    key, subkey = jax.random.split(key)
    training_data = generate_training_data(
        subkey, ground_truth_model, ground_truth_params, n_samples=1000
    )

    # 3. Fit Boltzmann machine to training data
    model, fitted_params, training_losses = fit_boltzmann_machine(training_data)

    # 4. Evaluate both models on all possible states
    all_states, learned_probs, energies = evaluate_model(model, fitted_params)
    _, true_probs, _ = evaluate_model(ground_truth_model, ground_truth_params)

    # 5. Test Gibbs sampling convergence (numerical simulation test)
    key, subkey = jax.random.split(key)
    sampling_sizes, sampling_thin_levels, sampling_errors_matrix = (
        test_sampling_convergence(subkey, model, fitted_params)
    )

    # 6. Create state labels for visualization
    state_labels = [f"{''.join(map(str, state))}" for state in all_states]

    return BoltzmannPatternResults(
        all_possible_states=all_states.tolist(),
        true_probabilities=true_probs.tolist(),
        learned_probabilities=learned_probs.tolist(),
        energy_values=energies.tolist(),
        training_losses=training_losses.tolist(),
        sampling_sizes=sampling_sizes,
        sampling_thin_levels=sampling_thin_levels,
        sampling_errors_matrix=sampling_errors_matrix,
        state_labels=state_labels,
    )


### Main ###


def main():
    """Run Boltzmann pattern learning analysis."""
    initialize_jax()
    paths = example_paths(__file__)

    key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility

    print("Running Boltzmann pattern learning analysis...")
    results = run_boltzmann_analysis(key)

    print(f"Evaluated {len(results['all_possible_states'])} possible states")
    print(
        f"Learned model with {len(results['learned_probabilities'])} state probabilities"
    )

    # Save results
    paths.save_analysis(results)
    print(f"Results saved to {paths.analysis_path}")


if __name__ == "__main__":
    main()
