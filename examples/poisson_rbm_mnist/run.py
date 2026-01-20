"""Train Poisson RBM on grayscale MNIST digits.

This example demonstrates:
1. Loading MNIST as grayscale counts (0-255)
2. Training a Poisson RBM using contrastive divergence (baseline)
3. Training a Poisson RBM with approximate conjugation regularization
4. Comparing the two approaches

The Poisson RBM treats each pixel as a Poisson(rate_i) random variable,
where rate_i = exp(b_i + W_i^T z). Unlike Binomial, Poisson counts are
unbounded, so outputs are clipped to [0, 255] for visualization.

Key advantage: For Poisson distributions, the "sum of tuning curves being
constant" property that enables effective conjugation-based learning holds
exactly, unlike for Binomial distributions where it's only approximate.

The combined objective for conjugated training is:
    L = alpha_cd * D(P_X || Q_X) + alpha_conj * D(Q_rho || Q_Z)

where D(Q_rho || Q_Z) keeps the latent marginal close to an exponential family.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from goal.geometry import Optimizer
from goal.models import poisson_rbm

from ..shared import example_paths, initialize_jax
from .approximate_conjugation import (
    ApproximateConjugationConfig,
    ConjugatedHarmoniumState,
    initialize_conjugated,
    make_estimate_conjugation_error_fn,
    make_train_epoch_fn,
    make_train_step_fn,
)
from .types import PoissonRBMResults

# Configuration
N_LATENT = 10000  # Number of latent units
BATCH_SIZE = 64
LEARNING_RATE = 3e-4  # Lower learning rate for count data
N_EPOCHS = 10000
CD_STEPS = 1
N_TRAIN = 5000  # Subset of MNIST for faster training
PIXEL_SCALE = 255.0  # Maximum pixel value for normalization/clipping

IMG_HEIGHT = 28
IMG_WIDTH = 28
N_OBSERVABLE = IMG_HEIGHT * IMG_WIDTH

# Conjugation configuration
ALPHA_CD = 1.0
ALPHA_CONJ = 10.0  # Lower weight since counts have larger scale
N_CONJ_SAMPLES = 100
N_GIBBS_CONJ = 5


def load_mnist_sklearn() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load MNIST using scikit-learn's fetch_openml."""
    try:
        from sklearn.datasets import (  # pyright: ignore[reportMissingTypeStubs]
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
    """Load MNIST as grayscale counts (0-255)."""
    print("Loading MNIST data as grayscale counts...")
    data, _ = load_mnist_sklearn()

    # Take subset - data is already 0-255
    data = data[:N_TRAIN].astype(np.float32)

    print(f"Data range: [{data.min()}, {data.max()}]")
    return jnp.array(data)


def clip_for_display(x: Array) -> Array:
    """Clip Poisson counts to valid grayscale range [0, 255]."""
    return jnp.clip(x, 0, PIXEL_SCALE)


def train_baseline_rbm(
    key: Array,
    model: Any,
    data: Array,
) -> tuple[Array, list[float], list[float]]:
    """Train baseline Poisson RBM using standard contrastive divergence with scan."""
    n_samples = data.shape[0]
    n_batches = n_samples // BATCH_SIZE

    # Initialize from sample data
    key, init_key = jax.random.split(key)
    params = model.initialize_from_sample(init_key, data, shape=0.01)

    optimizer = Optimizer.adamw(man=model, learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(params)

    recon_errors: list[float] = []
    free_energies: list[float] = []

    # JIT compile the CD gradient and reconstruction error
    @jax.jit
    def cd_step(step_key: Array, p: Array, batch: Array) -> Array:
        return model.mean_contrastive_divergence_gradient(
            step_key, p, batch, k=CD_STEPS
        )

    @jax.jit
    def compute_metrics(p: Array, d: Array) -> tuple[Array, Array]:
        recon = model.normalized_reconstruction_error(p, d, scale=PIXEL_SCALE)
        fe = model.mean_free_energy(p, d[:1000])
        return recon, fe

    for epoch in range(N_EPOCHS):
        # Shuffle data
        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, n_samples)
        data_shuffled = data[perm]

        # Reshape into batches for scan
        batches = data_shuffled[: n_batches * BATCH_SIZE].reshape(
            n_batches, BATCH_SIZE, -1
        )
        key, batch_keys_key = jax.random.split(key)
        batch_keys = jax.random.split(batch_keys_key, n_batches)

        def scan_body(carry, inputs):
            p, opt_s = carry
            batch_key, batch = inputs
            grad = cd_step(batch_key, p, batch)
            new_opt_s, new_p = optimizer.update(opt_s, grad, p)
            return (new_p, new_opt_s), None

        (params, opt_state), _ = jax.lax.scan(
            scan_body, (params, opt_state), (batch_keys, batches)
        )

        # Compute metrics
        recon, fe = compute_metrics(params, data)
        recon_errors.append(float(recon))
        free_energies.append(float(fe))

        print(
            f"[Baseline] Epoch {epoch + 1}/{N_EPOCHS}: "
            + f"Recon={float(recon):.4f}, FE={float(fe):.2f}"
        )

    return params, recon_errors, free_energies


def train_conjugated_rbm(
    key: Array,
    model: Any,
    data: Array,
    config: ApproximateConjugationConfig,
) -> tuple[
    ConjugatedHarmoniumState,
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Train Poisson RBM with approximate conjugation regularization using scan."""
    # Initialize with zero interaction for exact conjugation at start
    key, init_key = jax.random.split(key)
    state = initialize_conjugated(
        init_key, model, data, shape=0.01, zero_interaction=True
    )

    # Separate optimizers for harmonium and conjugation parameters
    opt_harmonium = Optimizer.adamw(man=model, learning_rate=LEARNING_RATE)
    opt_harmonium_state = opt_harmonium.init(state.harmonium_params)

    opt_conj = Optimizer.adamw(man=model.pst_man, learning_rate=LEARNING_RATE * 0.1)
    opt_conj_state = opt_conj.init(state.conjugation_params)

    # Create JIT-compiled functions
    train_step_fn = make_train_step_fn(model, config, CD_STEPS)
    train_epoch_fn = make_train_epoch_fn(
        train_step_fn,
        opt_harmonium.update,
        opt_conj.update,
        BATCH_SIZE,
    )
    estimate_error_fn = make_estimate_conjugation_error_fn(model, n_samples=200)

    @jax.jit
    def compute_metrics(
        harm_params: Array, conj_params: Array, d: Array, err_key: Array
    ) -> tuple[Array, Array, Array]:
        recon = model.normalized_reconstruction_error(harm_params, d, scale=PIXEL_SCALE)
        fe = model.mean_free_energy(harm_params, d[:1000])
        conj_err = estimate_error_fn(err_key, harm_params, conj_params)
        return recon, fe, conj_err

    recon_errors: list[float] = []
    free_energies: list[float] = []
    conj_errors: list[float] = []
    rho_norms: list[float] = []

    harm_params = state.harmonium_params
    conj_params = state.conjugation_params

    for epoch in range(N_EPOCHS):
        # Train one epoch with scan
        key, epoch_key = jax.random.split(key)
        harm_params, conj_params, opt_harmonium_state, opt_conj_state, key = (
            train_epoch_fn(
                epoch_key,
                harm_params,
                conj_params,
                opt_harmonium_state,
                opt_conj_state,
                data,
            )
        )

        # Compute metrics
        key, err_key = jax.random.split(key)
        recon, fe, conj_err = compute_metrics(harm_params, conj_params, data, err_key)

        recon_errors.append(float(recon))
        free_energies.append(float(fe))
        conj_errors.append(float(conj_err))

        rho_norm = float(jnp.linalg.norm(conj_params))
        rho_norms.append(rho_norm)

        print(
            f"[Conjugated] Epoch {epoch + 1}/{N_EPOCHS}: "
            + f"Recon={float(recon):.4f}, FE={float(fe):.2f}, "
            + f"ConjErr={float(conj_err):.4f}, ||rho||={rho_norm:.4f}"
        )

    final_state = ConjugatedHarmoniumState(harm_params, conj_params)
    return final_state, recon_errors, free_energies, conj_errors, rho_norms


def main():
    initialize_jax("gpu")
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)

    # Load data
    data = load_mnist()
    print(f"Loaded {data.shape[0]} samples, shape: {data.shape}")

    # Create model
    print(f"\nCreating Poisson RBM: {N_OBSERVABLE} observable, {N_LATENT} latent units")
    print("Note: Poisson counts are unbounded; outputs clipped to [0, 255] for display")
    model = poisson_rbm(N_OBSERVABLE, N_LATENT)

    # Configuration for approximate conjugation
    config = ApproximateConjugationConfig(
        alpha_cd=ALPHA_CD,
        alpha_conj=ALPHA_CONJ,
        n_conj_samples=N_CONJ_SAMPLES,
        n_gibbs_conj=N_GIBBS_CONJ,
    )

    # Train baseline
    print(f"\n{'=' * 60}")
    print(f"Training BASELINE with CD-{CD_STEPS} for {N_EPOCHS} epochs")
    print(f"{'=' * 60}")
    key, baseline_key = jax.random.split(key)
    baseline_params, baseline_recon, baseline_fe = train_baseline_rbm(
        baseline_key, model, data
    )

    # Train with conjugation
    print(f"\n{'=' * 60}")
    print(f"Training CONJUGATED with CD-{CD_STEPS} for {N_EPOCHS} epochs")
    print(f"  alpha_cd={config.alpha_cd}, alpha_conj={config.alpha_conj}")
    print(f"{'=' * 60}")
    key, conj_key = jax.random.split(key)
    conj_state, conj_recon, conj_fe, conj_errors, rho_norms = train_conjugated_rbm(
        conj_key, model, data, config
    )

    # Get weight filters
    baseline_filters = model.get_filters(baseline_params, (IMG_HEIGHT, IMG_WIDTH))
    conj_filters = model.get_filters(
        conj_state.harmonium_params, (IMG_HEIGHT, IMG_WIDTH)
    )

    # Reconstruct some samples
    n_viz = 10
    originals = data[:n_viz]
    baseline_recons = jax.vmap(model.reconstruct, in_axes=(None, 0))(
        baseline_params, originals
    )
    conj_recons = jax.vmap(model.reconstruct, in_axes=(None, 0))(
        conj_state.harmonium_params, originals
    )

    # Clip reconstructions for display
    baseline_recons = clip_for_display(baseline_recons)
    conj_recons = clip_for_display(conj_recons)

    # Generate samples via Gibbs chain
    print("\nGenerating samples...")
    key, sample_key1, sample_key2 = jax.random.split(key, 3)
    baseline_samples = model.sample(sample_key1, baseline_params, n=n_viz)
    baseline_generated = clip_for_display(baseline_samples[:, :N_OBSERVABLE])

    conj_samples = model.sample(sample_key2, conj_state.harmonium_params, n=n_viz)
    conj_generated = clip_for_display(conj_samples[:, :N_OBSERVABLE])

    # Save results
    results: PoissonRBMResults = {
        # Baseline results
        "baseline_reconstruction_errors": baseline_recon,
        "baseline_free_energies": baseline_fe,
        "baseline_weight_filters": baseline_filters.tolist(),
        "baseline_reconstructed_images": baseline_recons.tolist(),
        "baseline_generated_samples": baseline_generated.tolist(),
        # Conjugated results
        "conj_reconstruction_errors": conj_recon,
        "conj_free_energies": conj_fe,
        "conj_weight_filters": conj_filters.tolist(),
        "conj_reconstructed_images": conj_recons.tolist(),
        "conj_generated_samples": conj_generated.tolist(),
        # Conjugation-specific metrics
        "conjugation_errors": conj_errors,
        "rho_norms": rho_norms,
        # Shared
        "original_images": originals.tolist(),
        # Config
        "config": {
            "alpha_cd": config.alpha_cd,
            "alpha_conj": config.alpha_conj,
            "n_conj_samples": config.n_conj_samples,
            "n_gibbs_conj": config.n_gibbs_conj,
        },
    }
    paths.save_analysis(results)
    print(f"\nResults saved to {paths.analysis_path}")


if __name__ == "__main__":
    main()
