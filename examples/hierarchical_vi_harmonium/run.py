"""Train Hierarchical VI with Mixture of VonMises for MNIST Clustering.

This example demonstrates hierarchical variational inference with a three-level
generative model for unsupervised digit clustering:

The model:
- Observable (X): Binomial (MNIST pixels in [0, 16])
- Latent (Z): Product of VonMises (circular latent variables)
- Latent (K): Categorical (cluster assignments)

Key features:
- Conditional independence X ⊥ K | Z via MixtureObservableEmbedding
- Learnable correction rho_Z for structured variational posterior
- REINFORCE gradients for non-reparameterizable VonMises sampling

Training:
- ELBO maximization via gradient descent
- Separate optimizers for harmonium and rho parameters

Run with: python -m examples.hierarchical_vi_harmonium.run
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from goal.models import binomial_vonmises_mixture

from ..shared import example_paths, initialize_jax
from .core import (
    HierarchicalVIConfig,
    HierarchicalVIState,
    get_cluster_assignments,
    make_elbo_metrics_fn,
    sample_from_model,
)
from .training import (
    initialize_hierarchical_vi,
    make_train_epoch_fn,
    make_train_step_fn,
)
from .types import HierarchicalVIResults

# Configuration
N_LATENT = 500  # Number of VonMises latent units
N_CLUSTERS = 10  # Number of mixture components (for 10 digits)
BATCH_SIZE = 512
LEARNING_RATE_HARM = 3e-4
LEARNING_RATE_RHO = 3e-4
N_EPOCHS = 500  # Training epochs
N_MC_SAMPLES = 5  # MC samples for ELBO estimation
KL_WEIGHT = 1.0  # Standard VAE
N_TRAIN = 40000  # Subset of MNIST for training
N_TEST = 1000  # Subset for testing/clustering evaluation
N_TRIALS = 16  # Pixel values in [0, 16]

IMG_HEIGHT = 28
IMG_WIDTH = 28
N_OBSERVABLE = IMG_HEIGHT * IMG_WIDTH


def load_mnist_sklearn() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load MNIST using scikit-learn's fetch_openml."""
    try:
        from sklearn.datasets import (
            fetch_openml,
        )

        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        return mnist.data, mnist.target  # pyright: ignore[reportAttributeAccessIssue]  # noqa: TRY300
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for this example. "
            + "Install with: pip install scikit-learn"
        ) from e


def load_mnist() -> tuple[Array, Array, Array, Array]:
    """Load MNIST and split into train/test with labels.

    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    print(f"Loading MNIST data and downsampling to [0, {N_TRIALS}]...")
    data, labels = load_mnist_sklearn()

    # Convert labels to integers
    labels = labels.astype(np.int32)

    # Downsample: [0, 255] -> [0, N_TRIALS]
    data = np.round(data * N_TRIALS / 255.0).astype(np.float32)

    # Split into train and test
    train_data = jnp.array(data[:N_TRAIN])
    train_labels = jnp.array(labels[:N_TRAIN])
    test_data = jnp.array(data[60000 : 60000 + N_TEST])
    test_labels = jnp.array(labels[60000 : 60000 + N_TEST])

    print(f"Train: {train_data.shape[0]} samples, Test: {test_data.shape[0]} samples")
    print(f"Data range: [{data.min()}, {data.max()}]")

    return train_data, train_labels, test_data, test_labels


def compute_cluster_purity(
    predicted: Array, true_labels: Array, n_clusters: int
) -> tuple[float, list[int]]:
    """Compute cluster purity score.

    For each cluster, find the most common true label and count matches.
    Purity = (sum of max matches per cluster) / total samples

    Args:
        predicted: Predicted cluster assignments
        true_labels: True digit labels
        n_clusters: Number of clusters

    Returns:
        Tuple of (purity_score, cluster_counts)
    """
    total_correct = 0
    cluster_counts = []

    for k in range(n_clusters):
        mask = predicted == k
        cluster_size = int(jnp.sum(mask))
        cluster_counts.append(cluster_size)

        if cluster_size > 0:
            cluster_labels = true_labels[mask]
            # Count occurrences of each label in this cluster
            label_counts = jnp.bincount(cluster_labels, length=10)
            most_common_count = int(jnp.max(label_counts))
            total_correct += most_common_count

    purity = total_correct / len(true_labels)
    return purity, cluster_counts


def train_hierarchical_vi(
    key: Array,
    model: Any,
    train_data: Array,
    test_data: Array,
    test_labels: Array,
    config: HierarchicalVIConfig,
) -> tuple[HierarchicalVIState, list[float], list[float], list[float], list[float]]:
    """Train hierarchical VI model.

    Args:
        key: JAX random key
        model: The hierarchical harmonium model
        train_data: Training data
        test_data: Test data for clustering evaluation
        test_labels: True labels for test data
        config: VI configuration

    Returns:
        Tuple of (final_state, elbo_history, recon_history, kl_history, rho_norm_history)
    """
    # Initialize
    key, init_key = jax.random.split(key)
    state = initialize_hierarchical_vi(
        init_key, model, train_data, shape=0.01, zero_interaction=True
    )

    # Create optimizers
    opt_harmonium = optax.adamw(learning_rate=LEARNING_RATE_HARM)
    opt_harmonium_state = opt_harmonium.init(state.harmonium_params)

    opt_rho = optax.adamw(learning_rate=LEARNING_RATE_RHO)
    opt_rho_state = opt_rho.init(state.rho_params)

    # Create JIT-compiled functions
    train_step_fn = make_train_step_fn(model, config)
    train_epoch_fn = make_train_epoch_fn(
        train_step_fn,
        opt_harmonium,
        opt_rho,
        BATCH_SIZE,
    )
    metrics_fn = make_elbo_metrics_fn(model, config)

    # Training history
    elbo_history: list[float] = []
    recon_history: list[float] = []
    kl_history: list[float] = []
    rho_norm_history: list[float] = []

    harm_params = state.harmonium_params
    rho_params = state.rho_params

    # Evaluate on a subset for monitoring
    eval_data = train_data[:500]

    for epoch in range(N_EPOCHS):
        # Train one epoch
        key, epoch_key = jax.random.split(key)
        harm_params, rho_params, opt_harmonium_state, opt_rho_state, key, _ = (
            train_epoch_fn(
                epoch_key,
                harm_params,
                rho_params,
                opt_harmonium_state,
                opt_rho_state,
                train_data,
            )
        )

        # Compute metrics on eval subset
        key, metrics_key = jax.random.split(key)
        mean_elbo, mean_recon, mean_kl = metrics_fn(
            metrics_key, harm_params, rho_params, eval_data
        )

        elbo_history.append(float(mean_elbo))
        recon_history.append(float(mean_recon))
        kl_history.append(float(mean_kl))

        rho_norm = float(jnp.linalg.norm(rho_params))
        rho_norm_history.append(rho_norm)

        # Check for NaN
        if jnp.isnan(mean_elbo):
            print(f"[Hierarchical VI] Epoch {epoch + 1}: NaN detected, stopping early")
            break

        # Compute clustering purity every 20 epochs
        if (epoch + 1) % 20 == 0:
            # Update get_assignments_batch with current params
            assignments = jax.vmap(
                lambda x: get_cluster_assignments(model, harm_params, rho_params, x)
            )(test_data)
            purity, _ = compute_cluster_purity(assignments, test_labels, N_CLUSTERS)

            print(
                f"[Hierarchical VI] Epoch {epoch + 1}/{N_EPOCHS}: "
                + f"ELBO={float(mean_elbo):.2f}, Recon={float(mean_recon):.2f}, "
                + f"KL={float(mean_kl):.2f}, Purity={purity:.3f}"
            )
        else:
            print(
                f"[Hierarchical VI] Epoch {epoch + 1}/{N_EPOCHS}: "
                + f"ELBO={float(mean_elbo):.2f}, Recon={float(mean_recon):.2f}, "
                + f"KL={float(mean_kl):.2f}"
            )

    final_state = HierarchicalVIState(harm_params, rho_params)
    return (
        final_state,
        elbo_history,
        recon_history,
        kl_history,
        rho_norm_history,
    )


def main():
    initialize_jax("gpu")
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)

    # Load data (train_labels not used in training, just for evaluation)
    train_data, _, test_data, test_labels = load_mnist()

    # Create model
    print(
        f"\nCreating Hierarchical VI model: {N_OBSERVABLE} observable, "
        + f"{N_LATENT} latent, {N_CLUSTERS} clusters"
    )
    print(f"Binomial n_trials = {N_TRIALS}")
    model = binomial_vonmises_mixture(N_OBSERVABLE, N_LATENT, N_CLUSTERS, N_TRIALS)
    print(f"Model dim: {model.dim}")
    print(f"  Observable (Binomials): {model.obs_man.dim}")
    print(f"  Interaction: {model.int_man.dim}")
    print(f"  Latent (Mixture[VonMises]): {model.pst_man.dim}")
    print(f"  - VonMises bias: {model.vonmises_man.dim}")
    print(f"  - Mixture int: {model.pst_man.int_man.dim}")
    print(f"  - Categorical: {model.pst_man.lat_man.dim}")

    # Configuration
    config = HierarchicalVIConfig(
        n_mc_samples=N_MC_SAMPLES,
        kl_weight=KL_WEIGHT,
    )

    # Train
    print(f"\n{'=' * 60}")
    print(f"Training HIERARCHICAL VI for {N_EPOCHS} epochs")
    print(f"  n_mc_samples={config.n_mc_samples}, kl_weight={config.kl_weight}")
    print(f"  lr_harm={LEARNING_RATE_HARM}, lr_rho={LEARNING_RATE_RHO}")
    print(f"{'=' * 60}")

    key, train_key = jax.random.split(key)
    state, elbo_hist, recon_hist, kl_hist, rho_hist = train_hierarchical_vi(
        train_key, model, train_data, test_data, test_labels, config
    )

    # Final clustering evaluation
    print("\n" + "=" * 60)
    print("Final Clustering Evaluation")
    print("=" * 60)

    # Get cluster assignments for test set
    assignments = jax.vmap(
        lambda x: get_cluster_assignments(
            model, state.harmonium_params, state.rho_params, x
        )
    )(test_data)
    assignments_list = [int(a) for a in assignments]

    purity, cluster_counts = compute_cluster_purity(
        assignments, test_labels, N_CLUSTERS
    )
    print(f"Cluster purity: {purity:.4f}")
    print(f"Cluster sizes: {cluster_counts}")

    # Get weight filters
    filters = model.get_filters(state.harmonium_params, (IMG_HEIGHT, IMG_WIDTH))

    # Reconstruct some samples
    n_viz = 10
    originals = test_data[:n_viz]
    recons = jax.vmap(model.reconstruct, in_axes=(None, 0))(
        state.harmonium_params, originals
    )

    # Generate samples (z_samples and k_samples not used for visualization)
    print("\nGenerating samples...")
    key, sample_key = jax.random.split(key)
    x_samples, _, _ = sample_from_model(
        sample_key, model, state.harmonium_params, n=n_viz
    )

    # Get per-cluster samples (5 samples from each cluster)
    per_cluster_samples: list[list[list[float]]] = []
    for k in range(N_CLUSTERS):
        cluster_mask = assignments == k
        cluster_indices = jnp.where(cluster_mask)[0]
        if len(cluster_indices) >= 5:
            cluster_data = test_data[cluster_indices[:5]]
        else:
            # Pad with zeros if not enough samples
            cluster_data = test_data[cluster_indices]
            if len(cluster_data) < 5:
                padding = jnp.zeros((5 - len(cluster_data), N_OBSERVABLE))
                cluster_data = jnp.concatenate([cluster_data, padding])
        per_cluster_samples.append(cluster_data.tolist())

    # Compute cluster prototypes (mean image per cluster)
    cluster_prototypes: list[list[float]] = []
    for k in range(N_CLUSTERS):
        cluster_mask = assignments == k
        if jnp.sum(cluster_mask) > 0:
            prototype = jnp.mean(test_data[cluster_mask], axis=0)
        else:
            prototype = jnp.zeros(N_OBSERVABLE)
        cluster_prototypes.append(prototype.tolist())

    # Save results
    results: HierarchicalVIResults = {
        # Training metrics
        "elbo_history": elbo_hist,
        "reconstruction_history": recon_hist,
        "kl_history": kl_hist,
        "rho_norm_history": rho_hist,
        # Clustering metrics
        "cluster_assignments": assignments_list,
        "true_labels": [int(label) for label in test_labels],
        "cluster_purity": purity,
        "cluster_counts": cluster_counts,
        # Model outputs
        "weight_filters": filters.tolist(),
        "reconstructed_images": recons.tolist(),
        "generated_samples": x_samples.tolist(),
        "cluster_prototypes": cluster_prototypes,
        "per_cluster_samples": per_cluster_samples,
        # Shared
        "original_images": originals.tolist(),
        # Config
        "config": {
            "n_mc_samples": config.n_mc_samples,
            "kl_weight": config.kl_weight,
            "learning_rate_harmonium": LEARNING_RATE_HARM,
            "learning_rate_rho": LEARNING_RATE_RHO,
            "n_clusters": N_CLUSTERS,
            "n_latent": N_LATENT,
        },
    }
    paths.save_analysis(results)
    print(f"\nResults saved to {paths.analysis_path}")


if __name__ == "__main__":
    main()
