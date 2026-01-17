"""Train RBM on binarized MNIST digits.

This example demonstrates:
1. Loading and binarizing MNIST digits
2. Training an RBM using contrastive divergence
3. Visualizing learned filters and reconstructions
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from goal.geometry import Optimizer
from goal.models import rbm

from ..shared import example_paths, initialize_jax
from .types import RBMResults

# Configuration
N_HIDDEN = 100  # Number of hidden units
BATCH_SIZE = 64
LEARNING_RATE = 0.01
N_EPOCHS = 20
CD_STEPS = 1
N_TRAIN = 5000  # Subset of MNIST for faster training

IMG_HEIGHT = 28
IMG_WIDTH = 28
N_VISIBLE = IMG_HEIGHT * IMG_WIDTH


def load_mnist_sklearn() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load MNIST using scikit-learn's fetch_openml."""
    try:
        from sklearn.datasets import fetch_openml  # pyright: ignore[reportMissingTypeStubs]

        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        return mnist.data, mnist.target  # pyright: ignore[reportAttributeAccessIssue]
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for this example. "
            + "Install with: pip install scikit-learn"
        ) from e


def load_mnist() -> Array:
    """Load and binarize MNIST digits."""
    print("Loading MNIST data...")
    data, _ = load_mnist_sklearn()

    # Take subset and normalize
    data = data[:N_TRAIN].astype(np.float32) / 255.0

    # Binarize (threshold at 0.5)
    data = (data > 0.5).astype(np.float32)

    return jnp.array(data)


def train_rbm(
    key: Array,
    model: Any,
    data: Array,
) -> tuple[Array, list[float], list[float]]:
    """Train RBM using contrastive divergence."""
    n_samples = data.shape[0]
    n_batches = n_samples // BATCH_SIZE

    # Initialize from sample data
    key, init_key = jax.random.split(key)
    params = model.initialize_from_sample(init_key, data, shape=0.01)

    optimizer = Optimizer.adamw(man=model, learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(params)

    recon_errors: list[float] = []
    free_energies: list[float] = []

    for epoch in range(N_EPOCHS):
        # Shuffle data
        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, n_samples)
        data_shuffled = data[perm]

        epoch_recon = 0.0

        for batch_idx in range(n_batches):
            batch = data_shuffled[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

            # CD update
            key, cd_key = jax.random.split(key)
            cd_grad = model.mean_contrastive_divergence_gradient(
                cd_key, params, batch, k=CD_STEPS
            )
            opt_state, params = optimizer.update(opt_state, cd_grad, params)

            # Reconstruction error for this batch
            recon_error = model.reconstruction_error(params, batch)
            epoch_recon += float(recon_error)

        avg_recon = epoch_recon / n_batches
        recon_errors.append(avg_recon)

        # Compute mean free energy on subset for monitoring
        fe = model.mean_free_energy(params, data[:1000])
        free_energies.append(float(fe))

        print(
            f"Epoch {epoch+1}/{N_EPOCHS}: "
            + f"Recon Error={avg_recon:.4f}, Free Energy={fe:.2f}"
        )

    return params, recon_errors, free_energies


def main():
    initialize_jax()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)

    # Load data
    data = load_mnist()
    print(f"Loaded {data.shape[0]} samples, shape: {data.shape}")

    # Create model
    print(f"\nCreating RBM: {N_VISIBLE} visible, {N_HIDDEN} hidden units")
    model = rbm(N_VISIBLE, N_HIDDEN)

    # Train
    print(f"\nTraining with CD-{CD_STEPS} for {N_EPOCHS} epochs...")
    key, train_key = jax.random.split(key)
    params, recon_errors, free_energies = train_rbm(train_key, model, data)

    # Get weight filters
    filters = model.get_filters(params, (IMG_HEIGHT, IMG_WIDTH))

    # Reconstruct some samples
    n_viz = 10
    originals = data[:n_viz]
    reconstructions = jax.vmap(model.reconstruct, in_axes=(None, 0))(params, originals)

    # Generate samples via Gibbs chain
    print("\nGenerating samples...")
    key, sample_key = jax.random.split(key)
    samples = model.sample(sample_key, params, n=n_viz)
    # Extract just visible part
    generated = samples[:, :N_VISIBLE]

    # Save results
    results: RBMResults = {
        "reconstruction_errors": recon_errors,
        "free_energies": free_energies,
        "weight_filters": filters.tolist(),
        "original_images": originals.tolist(),
        "reconstructed_images": reconstructions.tolist(),
        "generated_samples": generated.tolist(),
    }
    paths.save_analysis(results)
    print(f"\nResults saved to {paths.analysis_path}")


if __name__ == "__main__":
    main()
