"""Train Binomial-VonMises RBM on downsampled MNIST with Structured VI.

This example demonstrates structured variational inference where the
approximate posterior has the same form as the true conjugate posterior,
but with a learnable correction term rho.

The model:
- Observable: Binomial (MNIST pixels in [0, 16])
- Latent: VonMises (circular latent variables)

Training:
- ELBO maximization via gradient descent
- Single optimizer for full parameters (rho + harmonium)

Run with: python -m examples.structured_vi_harmonium.run
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from goal.models import (
    BinomialVonMisesVI,
    binomial_vonmises_vi,
    make_binomial_vonmises_loss_fn,
    make_binomial_vonmises_metrics_fn,
)

from ..shared import example_paths, initialize_jax
from .types import StructuredVIResults

# Configuration
N_LATENT = 50  # Number of latent VonMises units (each has 2D natural params)
BATCH_SIZE = 512
LEARNING_RATE = 3e-4
N_EPOCHS = 1000  # Training epochs
N_MC_SAMPLES = 10  # MC samples for ELBO estimation
KL_WEIGHT = 1.0  # Standard VAE (set < 1 for beta-VAE)
N_TRAIN = 5000  # Subset of MNIST for faster training
N_TRIALS = 16  # Number of trials for Binomial (pixel values in [0, 16])

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


def load_mnist() -> Array:
    """Load MNIST and downsample to counts in [0, N_TRIALS]."""
    print(f"Loading MNIST data and downsampling to [0, {N_TRIALS}]...")
    data, _ = load_mnist_sklearn()

    # Downsample: [0, 255] -> [0, N_TRIALS]
    # Scale and round to get integer counts
    data = np.round(data * N_TRIALS / 255.0).astype(np.float32)

    print(f"Data range: [{data.min()}, {data.max()}]")
    print(f"Mean pixel value: {data.mean():.3f}")
    return jnp.array(data)


def train_structured_vi(
    key: Array,
    model: BinomialVonMisesVI,
    data: Array,
    n_mc_samples: int = 10,
    kl_weight: float = 1.0,
) -> tuple[Array, list[float], list[float], list[float], list[float], list[float]]:
    """Train Binomial-VonMises model with structured variational inference.

    Args:
        key: JAX random key
        model: The variational conjugated model
        data: Training data
        n_mc_samples: Number of MC samples for ELBO estimation
        kl_weight: Weight on KL term (1.0 = standard VAE, <1.0 = beta-VAE)

    Returns:
        Tuple of (final_params, elbo_history, recon_history, kl_history,
                  rho_norm_history, conjugation_error_history)
    """
    # Initialize parameters
    key, init_key = jax.random.split(key)
    params = model.initialize_from_sample(init_key, data[:1000], shape=0.01)

    # Zero out interaction weights initially
    rho, hrm_params = model.split_coords(params)
    obs_bias, int_params, lat_params = model.hrm.split_coords(hrm_params)
    int_params = jnp.zeros_like(int_params)
    hrm_params = model.hrm.join_coords(obs_bias, int_params, lat_params)
    params = model.join_coords(rho, hrm_params)

    # Create optimizer
    opt = optax.adamw(learning_rate=LEARNING_RATE)
    opt_state = opt.init(params)

    # Create JIT-compiled functions
    loss_and_grad_fn = make_binomial_vonmises_loss_fn(model, n_mc_samples, kl_weight)
    metrics_fn = make_binomial_vonmises_metrics_fn(model, n_mc_samples, kl_weight)

    # Training history
    elbo_history: list[float] = []
    recon_history: list[float] = []
    kl_history: list[float] = []
    rho_norm_history: list[float] = []
    conj_error_history: list[float] = []

    # Evaluate on a subset for monitoring
    eval_data = data[:500]

    for epoch in range(N_EPOCHS):
        # Shuffle data
        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, data.shape[0])
        shuffled_data = data[perm]

        # Train on batches
        n_batches = data.shape[0] // BATCH_SIZE
        for batch_idx in range(n_batches):
            batch = shuffled_data[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
            key, step_key = jax.random.split(key)

            _, grad = loss_and_grad_fn(step_key, params, batch)
            updates, opt_state = opt.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

        # Compute metrics on eval subset
        key, metrics_key = jax.random.split(key)
        mean_elbo, mean_recon, mean_kl, mean_conj_err = metrics_fn(
            metrics_key, params, eval_data
        )

        elbo_history.append(float(mean_elbo))
        recon_history.append(float(mean_recon))
        kl_history.append(float(mean_kl))
        conj_error_history.append(float(mean_conj_err))

        rho = model.conjugation_params(params)  # pyright: ignore[reportArgumentType]
        rho_norm = float(jnp.linalg.norm(rho))
        rho_norm_history.append(rho_norm)

        # Check for NaN
        if jnp.isnan(mean_elbo):
            print(f"[Structured VI] Epoch {epoch + 1}: NaN detected, stopping early")
            break

        print(
            f"[Structured VI] Epoch {epoch + 1}/{N_EPOCHS}: "
            + f"ELBO={float(mean_elbo):.2f}, Recon={float(mean_recon):.2f}, "
            + f"KL={float(mean_kl):.2f}, ConjErr={float(mean_conj_err):.4f}"
        )

    return (
        params,  # pyright: ignore[reportReturnType]
        elbo_history,
        recon_history,
        kl_history,
        rho_norm_history,
        conj_error_history,
    )


def main():
    initialize_jax("gpu")
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)

    # Load data
    data = load_mnist()
    print(f"Loaded {data.shape[0]} samples, shape: {data.shape}")

    # Create model
    print(
        f"\nCreating Binomial-VonMises VI model: {N_OBSERVABLE} observable, {N_LATENT} latent units"
    )
    print(f"Binomial n_trials = {N_TRIALS}")
    model = binomial_vonmises_vi(N_OBSERVABLE, N_LATENT, N_TRIALS)
    print(
        f"Full param dim: {model.dim} = rho({model.fst_man.dim}) + hrm({model.snd_man.dim})"
    )

    # Train with structured VI
    print(f"\n{'=' * 60}")
    print(f"Training STRUCTURED VI for {N_EPOCHS} epochs")
    print(f"  n_mc_samples={N_MC_SAMPLES}, kl_weight={KL_WEIGHT}")
    print(f"{'=' * 60}")
    key, vi_key = jax.random.split(key)
    params, elbo_hist, recon_hist, kl_hist, rho_hist, conj_err_hist = train_structured_vi(
        vi_key, model, data, N_MC_SAMPLES, KL_WEIGHT
    )

    # Get weight filters
    hrm_params = model.generative_params(params)
    vi_filters = model.hrm.get_filters(hrm_params, (IMG_HEIGHT, IMG_WIDTH))

    # Reconstruct some samples
    n_viz = 10
    originals = data[:n_viz]
    vi_recons = jax.vmap(model.reconstruct, in_axes=(None, 0))(params, originals)

    # Generate samples
    print("\nGenerating samples...")
    key, sample_key = jax.random.split(key)
    vi_samples = model.sample_from_prior(sample_key, params, n=n_viz)
    vi_generated = vi_samples[:, : N_OBSERVABLE]

    # Save results
    results: StructuredVIResults = {
        # Structured VI results
        "elbo_history": elbo_hist,
        "reconstruction_history": recon_hist,
        "kl_history": kl_hist,
        "rho_norm_history": rho_hist,
        "conjugation_error_history": conj_err_hist,
        "weight_filters": vi_filters.tolist(),
        "reconstructed_images": vi_recons.tolist(),
        "generated_samples": vi_generated.tolist(),
        # Empty baseline (not run)
        "baseline_reconstruction_errors": [],
        "baseline_free_energies": [],
        "baseline_weight_filters": [],
        "baseline_reconstructed_images": [],
        "baseline_generated_samples": [],
        # Shared
        "original_images": originals.tolist(),
        # Config
        "config": {
            "n_mc_samples": N_MC_SAMPLES,
            "kl_weight": KL_WEIGHT,
            "learning_rate": LEARNING_RATE,
        },
    }
    paths.save_analysis(results)
    print(f"\nResults saved to {paths.analysis_path}")


if __name__ == "__main__":
    main()
