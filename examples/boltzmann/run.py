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

from goal.geometry import Natural, Optimizer, OptState, Point
from goal.models.base.gaussian.boltzmann import Boltzmann

from ..shared import example_paths, initialize_jax
from .types import BoltzmannPatternResults


def create_ground_truth_patterns() -> tuple[Array, Array]:
    """Create interpretable 2x2 binary patterns.

    We'll create patterns that have clear structure:
    - Corner patterns (one corner active)
    - Edge patterns (two adjacent units active)
    - Diagonal patterns (opposite corners active)

    Returns:
        patterns: Array of shape (n_patterns, 4) with binary patterns
        weights: Array of shape (n_patterns,) with pattern probabilities
    """
    # Define meaningful 2x2 patterns (flattened to 4D vectors)
    patterns = jnp.array(
        [
            # Corner patterns (more likely)
            [1, 0, 0, 0],  # Top-left corner
            [0, 1, 0, 0],  # Top-right corner
            [0, 0, 1, 0],  # Bottom-left corner
            [0, 0, 0, 1],  # Bottom-right corner
            # Diagonal patterns (medium likelihood)
            [1, 0, 0, 1],  # Main diagonal
            [0, 1, 1, 0],  # Anti-diagonal
            # Edge patterns (lower likelihood)
            [1, 1, 0, 0],  # Top edge
            [0, 0, 1, 1],  # Bottom edge
        ]
    )

    # Assign probabilities (corner patterns more likely)
    weights = jnp.array(
        [
            0.15,
            0.15,
            0.15,
            0.15,  # Corners
            0.10,
            0.10,  # Diagonals
            0.10,
            0.10,
        ]
    )  # Edges

    # Normalize to ensure they sum to 1
    weights = weights / weights.sum()

    return patterns, weights


def generate_training_data(
    key: Array, patterns: Array, weights: Array, n_samples: int = 1000
) -> Array:
    """Generate training data by sampling from the ground truth distribution."""
    # Sample pattern indices according to weights
    pattern_indices = jax.random.choice(key, len(patterns), (n_samples,), p=weights)

    # Return the corresponding patterns
    return patterns[pattern_indices]


def fit_boltzmann_machine(
    training_data: Array, n_steps: int = 500, learning_rate: float = 0.01
) -> tuple[Boltzmann, Point[Natural, Boltzmann], Array]:
    """Fit a Boltzmann machine to the training data using gradient descent.

    Args:
        training_data: Array of shape (n_samples, 4) with binary training data
        n_steps: Number of optimization steps
        learning_rate: Learning rate for optimizer

    Returns:
        model: Fitted Boltzmann machine
        params: Learned parameters in natural coordinates
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
    init_params = model.natural_point(init_params_flat)

    # Setup optimizer
    optimizer: Optimizer[Natural, Boltzmann] = Optimizer.adamw(
        model, learning_rate=learning_rate
    )
    opt_state = optimizer.init(init_params)

    def cross_entropy_loss(params: Point[Natural, Boltzmann]) -> Array:
        """Negative log-likelihood loss using exponential family interface."""
        return -model.average_log_density(params, training_data)

    def grad_step(
        opt_state_and_params: tuple[OptState, Point[Natural, Boltzmann]], _: Any
    ) -> tuple[tuple[OptState, Point[Natural, Boltzmann]], Array]:
        # Compute loss and gradients
        opt_state, params = opt_state_and_params
        loss_val, grads = model.value_and_grad(cross_entropy_loss, params)

        # Get updates from optimizer
        opt_state, params = optimizer.update(opt_state, grads, params)

        return (opt_state, params), loss_val

    print(f"Starting optimization with {n_steps} steps...")
    (_, final_params), losses = jax.lax.scan(
        grad_step, (opt_state, init_params), None, length=n_steps
    )

    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")

    # Extract final parameters for inspection
    print(f"Final parameters shape: {final_params.array.shape}")
    print(f"Final parameters: {final_params.array}")

    return model, final_params, losses


def evaluate_model(
    model: Boltzmann, params: Point[Natural, Boltzmann]
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
    log_probs = jax.vmap(model.log_density, in_axes=(None, 0))(params, all_states)
    probabilities = jnp.exp(log_probs)

    # Energies are negative log unnormalized probabilities
    log_partition = model.log_partition_function(params)
    energies = -(log_probs + log_partition)

    return all_states, probabilities, energies


def compute_ground_truth_probabilities(
    all_states: Array, gt_patterns: Array, gt_weights: Array
) -> Array:
    """Compute ground truth probabilities for all states."""
    gt_probs = jnp.zeros(len(all_states))

    for i, pattern in enumerate(gt_patterns):
        # Find which states match this pattern
        matches = jnp.all(all_states == pattern, axis=1)
        gt_probs = gt_probs + matches * gt_weights[i]

    return gt_probs


def compute_pattern_frequencies(data: Array, patterns: Array) -> Array:
    """Compute frequency of each pattern in the data."""
    frequencies = []
    for pattern in patterns:
        # Convert both to same type for comparison
        matches = jnp.all(data.astype(jnp.int32) == pattern.astype(jnp.int32), axis=1)
        freq = jnp.mean(matches)
        frequencies.append(freq)
    return jnp.array(frequencies)


def test_sampling_convergence(
    key: Array, model: Boltzmann, params: Point[Natural, Boltzmann]
) -> tuple[list[int], list[int], list[list[float]]]:
    """Test that Gibbs sampling converges to exact probabilities.

    This is a numerical simulation test independent of the learning problem.
    We test how both sample size and number of Gibbs steps affect convergence
    to the exact probabilities computed via the exponential family interface.
    """
    # Get exact probabilities using exponential family interface
    all_states = jnp.array(list(itertools.product([0, 1], repeat=4)))
    exact_probs = jnp.exp(
        jax.vmap(model.log_density, in_axes=(None, 0))(params, all_states)
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
                subkey, params, n_samples=n_samples, n_burnin=50, n_thin=n_thin
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

    # 1. Create ground truth patterns
    gt_patterns, gt_weights = create_ground_truth_patterns()

    # 2. Generate training data
    key, subkey = jax.random.split(key)
    training_data = generate_training_data(
        subkey, gt_patterns, gt_weights, n_samples=1000
    )

    # 3. Fit Boltzmann machine
    model, fitted_params, training_losses = fit_boltzmann_machine(training_data)

    # 4. Evaluate model on all possible states
    all_states, learned_probs, energies = evaluate_model(model, fitted_params)
    gt_probs = compute_ground_truth_probabilities(all_states, gt_patterns, gt_weights)

    # 5. Generate samples from fitted model
    key, subkey = jax.random.split(key)
    generated_samples = model.sample(
        subkey, fitted_params, n_samples=1000, n_burnin=500, n_thin=5
    )

    # 6. Extract learned parameters for visualization
    # Create a simple matrix representation for visualization
    n = model.n_neurons
    interaction_matrix = jnp.zeros((n, n))
    triangular_params = fitted_params.array

    # Extract bias terms (diagonal)
    bias_indices = jnp.array([i * (i + 1) // 2 + i for i in range(n)])
    bias_terms = triangular_params[bias_indices]

    # Reconstruct symmetric matrix from triangular storage
    param_idx = 0
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                interaction_matrix = interaction_matrix.at[i, j].set(
                    triangular_params[param_idx]
                )
            else:
                interaction_matrix = interaction_matrix.at[i, j].set(
                    triangular_params[param_idx]
                )
                interaction_matrix = interaction_matrix.at[j, i].set(
                    triangular_params[param_idx]
                )
            param_idx += 1

    # 7. Compute pattern frequencies
    pattern_frequencies = compute_pattern_frequencies(training_data, gt_patterns)
    generated_frequencies = compute_pattern_frequencies(generated_samples, gt_patterns)

    # 8. Test Gibbs sampling convergence (numerical simulation test)
    key, subkey = jax.random.split(key)
    sampling_sizes, sampling_thin_levels, sampling_errors_matrix = (
        test_sampling_convergence(subkey, model, fitted_params)
    )

    # 9. Create state labels for visualization
    state_labels = [f"{''.join(map(str, state))}" for state in all_states]

    return BoltzmannPatternResults(
        ground_truth_patterns=gt_patterns.tolist(),
        ground_truth_weights=gt_weights.tolist(),
        pattern_frequencies=pattern_frequencies.tolist(),
        fitted_bias=bias_terms.tolist(),
        fitted_interactions=interaction_matrix.tolist(),
        all_possible_states=all_states.tolist(),
        true_probabilities=gt_probs.tolist(),
        learned_probabilities=learned_probs.tolist(),
        energy_values=energies.tolist(),
        generated_samples=generated_samples.tolist(),
        generated_frequencies=generated_frequencies.tolist(),
        training_losses=training_losses.tolist(),
        sampling_sizes=sampling_sizes,
        sampling_thin_levels=sampling_thin_levels,
        sampling_errors_matrix=sampling_errors_matrix,
        pattern_grid_size=2,
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

    print(f"Generated {len(results['generated_samples'])} samples")
    print(f"Learned {len(results['fitted_bias'])} bias parameters")
    print(
        f"Learned interaction matrix of size {len(results['fitted_interactions'])}x{len(results['fitted_interactions'][0])}"
    )

    # Save results
    paths.save_analysis(results)
    print(f"Results saved to {paths.analysis_path}")


if __name__ == "__main__":
    main()
