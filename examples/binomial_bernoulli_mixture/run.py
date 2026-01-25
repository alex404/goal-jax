"""Binomial-Bernoulli Mixture training example.

This example demonstrates clustering with a hierarchical model:
- Observable: Binomials (count data)
- Latent: Bernoullis (binary hidden units)
- Mixture: Categorical over K clusters

Uses single-phase variational training with entropy regularization to
prevent cluster collapse.

Usage:
    python -m examples.binomial_bernoulli_mixture.run
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.models import binomial_bernoulli_mixture

from ..shared import example_paths, initialize_jax
from .types import BinomialBernoulliResults

# Configuration
N_OBSERVABLE = 16  # 4x4 grid for easier visualization
N_LATENT = 4  # Fewer hidden units
N_CLUSTERS = 3  # Fewer clusters for easier separation
N_TRIALS = 8  # Binomial trials

N_TRAIN = 500
N_TEST = 100
BATCH_SIZE = 50

N_STEPS = 500
LEARNING_RATE = 1e-3
N_MC_SAMPLES = 5
ENTROPY_WEIGHT = 2.0  # Weight on batch entropy to prevent cluster collapse


def generate_synthetic_data(
    key: Array,
    n_samples: int,
    n_features: int,
    n_clusters: int,
    n_trials: int,
    beta_a: float = 0.2,
    beta_b: float = 0.2,
) -> tuple[Array, Array]:
    """Generate synthetic clustered Binomial data.

    Uses Beta distribution with extreme parameters (0.2, 0.2) to create
    very distinct cluster patterns (probabilities near 0 or 1).

    Args:
        key: Random key
        n_samples: Number of samples
        n_features: Number of features
        n_clusters: Number of clusters
        n_trials: Binomial trials
        beta_a: Beta distribution alpha (lower = more extreme)
        beta_b: Beta distribution beta

    Returns:
        Tuple of (data, labels) where data has values in [0, n_trials]
    """
    key_labels, key_patterns, key_noise = jax.random.split(key, 3)

    # Uniform cluster assignment
    labels = jax.random.randint(key_labels, (n_samples,), 0, n_clusters)

    # Create distinct cluster patterns using Beta distribution
    # Low alpha/beta = more extreme (bimodal near 0 and 1)
    cluster_probs = jax.random.beta(
        key_patterns, beta_a, beta_b, shape=(n_clusters, n_features)
    )

    # Sample from Binomial with cluster-specific probabilities
    def sample_for_cluster(label: Array, key: Array) -> Array:
        probs = cluster_probs[label]
        return jax.random.binomial(key, n_trials, probs)

    sample_keys = jax.random.split(key_noise, n_samples)
    data = jax.vmap(sample_for_cluster)(labels, sample_keys)

    return data.astype(jnp.float32), labels


def compute_batch_entropy(all_probs: Array) -> Array:
    """Compute entropy of average cluster distribution across batch.

    This encourages the model to use all clusters across the dataset,
    preventing collapse to a single cluster.

    Args:
        all_probs: Shape (batch_size, n_clusters) - cluster probs for each sample

    Returns:
        Entropy of the average cluster distribution (higher = more uniform)
    """
    avg_probs = jnp.mean(all_probs, axis=0)
    probs_safe = jnp.clip(avg_probs, 1e-10, 1.0)
    return -jnp.sum(probs_safe * jnp.log(probs_safe))


def compute_cluster_purity(
    assignments: Array,
    labels: Array,
    n_clusters: int,
) -> float:
    """Compute cluster purity given true labels.

    For each predicted cluster, finds the most common true label and counts
    how many samples match. Higher is better.

    Args:
        assignments: Predicted cluster assignments
        labels: True labels
        n_clusters: Number of clusters

    Returns:
        Purity score in [0, 1]
    """
    n_correct = 0
    for k in range(n_clusters):
        mask = assignments == k
        if jnp.sum(mask) > 0:
            cluster_labels = labels[mask]
            _, counts = jnp.unique(cluster_labels, return_counts=True)
            if len(counts) > 0:
                n_correct += jnp.max(counts)
    return float(n_correct / len(labels))


def main():
    initialize_jax()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)

    # Create model
    print(
        f"Creating model: {N_OBSERVABLE} obs, {N_LATENT} latent, {N_CLUSTERS} clusters"
    )
    model = binomial_bernoulli_mixture(
        n_observable=N_OBSERVABLE,
        n_latent=N_LATENT,
        n_clusters=N_CLUSTERS,
        n_trials=N_TRIALS,
    )

    # Generate synthetic data
    key, data_key = jax.random.split(key)
    print("Generating synthetic data...")

    train_data, _ = generate_synthetic_data(
        jax.random.fold_in(data_key, 0),
        N_TRAIN,
        N_OBSERVABLE,
        N_CLUSTERS,
        N_TRIALS,
    )
    test_data, test_labels = generate_synthetic_data(
        jax.random.fold_in(data_key, 1),
        N_TEST,
        N_OBSERVABLE,
        N_CLUSTERS,
        N_TRIALS,
    )

    print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")

    # Initialize model
    key, init_key = jax.random.split(key)
    params = model.initialize_from_sample(init_key, train_data, location=0.0, shape=0.1)

    # Initialize cluster-specific parameters to break symmetry
    key, cluster_key = jax.random.split(key)
    rho, hrm_params = model.split_coords(params)
    obs_bias, int_params, lat_params = model.hrm.split_coords(hrm_params)
    mix_obs_bias, _, cat_params = model.mix_man.split_coords(lat_params)

    mix_int_dim = model.n_latent * (model.n_clusters - 1)
    new_mix_int_mat = jax.random.normal(cluster_key, (mix_int_dim,)) * 0.5

    new_lat_params = model.mix_man.join_coords(
        mix_obs_bias, new_mix_int_mat, cat_params
    )
    new_hrm_params = model.hrm.join_coords(obs_bias, int_params, new_lat_params)
    params = model.join_coords(rho, new_hrm_params)

    # Setup optimizer
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    # Loss function: negative ELBO - entropy bonus
    def loss_fn(
        params: Array, key: Array, batch: Array
    ) -> tuple[Array, tuple[Array, Array]]:
        """Compute loss = -ELBO - entropy_weight * batch_entropy."""
        # ELBO via model's built-in method
        elbo = model.mean_elbo(key, params, batch, N_MC_SAMPLES)

        # Batch entropy for cluster regularization
        def get_cluster_probs(x: Array) -> Array:
            q_params = model.approximate_posterior_at(params, x)
            return model.get_cluster_probs(q_params)

        all_probs = jax.vmap(get_cluster_probs)(batch)
        entropy = compute_batch_entropy(all_probs)

        loss = -elbo - ENTROPY_WEIGHT * entropy
        return loss, (elbo, entropy)

    # JIT-compiled training step
    @jax.jit
    def train_step(carry: Any, _: None) -> tuple[Any, Array]:
        params, opt_state, key, data = carry
        key, step_key, batch_key = jax.random.split(key, 3)

        # Sample batch
        batch_idx = jax.random.choice(batch_key, N_TRAIN, shape=(BATCH_SIZE,))
        batch = data[batch_idx]

        # Compute loss and gradients
        (_, (elbo, _)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, step_key, batch
        )

        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return (new_params, new_opt_state, key, data), elbo

    # Training loop
    print(f"\nTraining for {N_STEPS} steps with entropy weight {ENTROPY_WEIGHT}...")
    print(f"Baseline purity for {N_CLUSTERS} clusters: {100 / N_CLUSTERS:.1f}%")

    key, train_key = jax.random.split(key)
    init_carry = (params, opt_state, train_key, train_data)

    (final_params, _, _, _), elbos = jax.lax.scan(
        train_step, init_carry, None, length=N_STEPS
    )

    print(f"Final ELBO: {elbos[-1]:.4f}")

    # Evaluation
    print("\nEvaluating on test set...")

    # Get predictions
    def get_predictions(x: Array) -> tuple[Array, Array, Array]:
        """Get cluster assignment, probabilities, and reconstruction."""
        q_params = model.approximate_posterior_at(final_params, x)
        probs = model.get_cluster_probs(q_params)
        assignment = jnp.argmax(probs)
        recon = model.reconstruct(final_params, x)
        return assignment, probs, recon

    assignments, cluster_probs, reconstructions = jax.vmap(get_predictions)(test_data)

    # Metrics
    purity = compute_cluster_purity(assignments, test_labels, N_CLUSTERS)
    recon_error = float(model.normalized_reconstruction_error(final_params, test_data))

    cluster_counts = [int(jnp.sum(assignments == k)) for k in range(N_CLUSTERS)]

    print("\nResults:")
    print(f"  Cluster purity: {100 * purity:.1f}%")
    print(f"  Reconstruction error: {recon_error:.4f}")
    print(f"  Cluster distribution: {cluster_counts}")

    # Save results
    results = BinomialBernoulliResults(
        test_data=test_data.tolist(),
        test_labels=test_labels.tolist(),
        predicted_labels=assignments.tolist(),
        cluster_probs=cluster_probs.tolist(),
        reconstructions=reconstructions.tolist(),
        training_elbos=elbos.tolist(),
        cluster_purity=purity,
        reconstruction_error=recon_error,
        cluster_distribution=cluster_counts,
    )
    paths.save_analysis(results)
    print(f"\nResults saved to {paths.analysis_path}")


if __name__ == "__main__":
    main()
