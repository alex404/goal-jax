"""Binomial-Bernoulli Mixture MNIST with analytical rho.

This script trains the model with analytically-computed rho* instead of
learning rho as a free parameter. This prevents the optimizer from "cheating"
by flattening the likelihood to make rho appear to work well.

Key differences from run.py:
1. Only optimizes hrm_params (rho is computed analytically each step)
2. Computes rho* via least-squares regression at each training step
3. Penalizes (1-R²) to encourage linearizable ψ_X
4. Gradients flow through lstsq (JAX handles implicit differentiation)

Usage:
    python -m examples.binomial_bernoulli_mnist.run_analytical
    python -m examples.binomial_bernoulli_mnist.run_analytical --n-latent 1024 --conj-weight 1.0
"""

import argparse
import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from sklearn.datasets import fetch_openml

from goal.models import binomial_bernoulli_mixture

from ..shared import example_paths, initialize_jax
from .analytical_rho import (
    analytical_rho_with_var_penalty,
    compute_elbo_with_analytical_rho,
)
from .types import AnalyticalModelResults, BinomialBernoulliMNISTAnalyticalResults

# MNIST dimensions
IMG_SIZE = 28
N_OBSERVABLE = IMG_SIZE * IMG_SIZE  # 784

# Model architecture
N_LATENT = 4096  # Binary hidden units (like RBM)
N_CLUSTERS = 10  # For 10 digits
N_TRIALS = 16  # Binomial trials for pixel values

# Data configuration
N_TRAIN = 60000
N_TEST = 10000

# Training configuration
BATCH_SIZE = 512
N_STEPS = 50000
LEARNING_RATE = 1e-4
N_MC_SAMPLES = 10
ENTROPY_WEIGHT = 1.0  # Prevent cluster collapse
CONJ_WEIGHT = 1.0  # Weight for (1-R²) penalty

# Analytical rho computation
# IMPORTANT: n_samples must be >> n_latent to avoid overfitting in lstsq
# Default multiplier: use 5x n_latent samples for regression
CONJ_SAMPLES_MULTIPLIER = 5
RHO_UPDATE_INTERVAL = 1  # Update rho every n steps (1 = every step)
LOG_INTERVAL = 200  # How often to print progress

# Weight regularization
L2_INT_WEIGHT = 0.0  # L2 on interaction weights


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Binomial-Bernoulli Mixture on MNIST with analytical rho"
    )
    parser.add_argument(
        "--n-latent", type=int, default=N_LATENT, help="Number of binary latent units"
    )
    parser.add_argument(
        "--n-steps", type=int, default=N_STEPS, help="Number of training steps"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate"
    )
    parser.add_argument(
        "--conj-weight",
        type=float,
        default=CONJ_WEIGHT,
        help="Weight for (1-R²) penalty",
    )
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=ENTROPY_WEIGHT,
        help="Weight for batch entropy (prevents cluster collapse)",
    )
    parser.add_argument(
        "--l2-int-weight",
        type=float,
        default=L2_INT_WEIGHT,
        help="L2 regularization on interaction weights",
    )
    parser.add_argument(
        "--conj-samples-mult",
        type=int,
        default=CONJ_SAMPLES_MULTIPLIER,
        help="Multiplier for n_conj_samples = mult * n_latent (ensures overdetermined system)",
    )
    parser.add_argument(
        "--conj-mode",
        choices=["r2", "var_f", "none"],
        default="none",
        help="Conjugation penalty mode: 'r2' uses (1-R²), 'var_f' uses Var[f̃], 'none' disables penalty",
    )
    parser.add_argument(
        "--use-cosine-lr",
        action="store_true",
        help="Use cosine annealing learning rate schedule",
    )
    parser.add_argument(
        "--rho-update-interval",
        type=int,
        default=RHO_UPDATE_INTERVAL,
        help="Update rho every n steps (1 = every step, higher values reduce cost)",
    )
    return parser.parse_args()


def load_mnist_sklearn() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load MNIST using scikit-learn's fetch_openml."""
    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        return mnist.data, mnist.target  # pyright: ignore[reportAttributeAccessIssue]  # noqa: TRY300
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for this example. Install with: pip install scikit-learn"
        ) from e


def load_mnist() -> tuple[Array, Array, Array, Array]:
    """Load MNIST and split into train/test with labels."""
    print(f"Loading MNIST data and downsampling to [0, {N_TRIALS}]...")
    data, labels = load_mnist_sklearn()

    labels = labels.astype(np.int32)
    data = np.round(data * N_TRIALS / 255.0).astype(np.float32)

    train_data = jnp.array(data[:N_TRAIN])
    train_labels = jnp.array(labels[:N_TRAIN])
    test_data = jnp.array(data[60000 : 60000 + N_TEST])
    test_labels = jnp.array(labels[60000 : 60000 + N_TEST])

    print(f"Train: {train_data.shape[0]} samples, Test: {test_data.shape[0]} samples")
    return train_data, train_labels, test_data, test_labels


def compute_batch_entropy(all_probs: Array) -> Array:
    """Compute entropy of average cluster distribution across batch."""
    avg_probs = jnp.mean(all_probs, axis=0)
    probs_safe = jnp.clip(avg_probs, 1e-10, 1.0)
    return -jnp.sum(probs_safe * jnp.log(probs_safe))


def compute_cluster_purity(
    assignments: Array,
    labels: Array,
    n_clusters: int,
) -> float:
    """Compute cluster purity given true labels."""
    n_correct = 0
    for k in range(n_clusters):
        mask = assignments == k
        if jnp.sum(mask) > 0:
            cluster_labels = labels[mask]
            label_counts = jnp.bincount(cluster_labels, length=10)
            n_correct += int(jnp.max(label_counts))
    return float(n_correct / len(labels))


def compute_nmi(assignments: Array, labels: Array) -> float:
    """Compute Normalized Mutual Information between assignments and labels."""
    from sklearn.metrics import normalized_mutual_info_score

    return float(
        normalized_mutual_info_score(
            np.array(labels), np.array(assignments), average_method="arithmetic"
        )
    )


def compute_cluster_accuracy(
    assignments: Array, labels: Array, n_clusters: int, n_classes: int = 10
) -> float:
    """Compute accuracy via Hungarian algorithm for optimal cluster-to-class mapping."""
    from scipy.optimize import linear_sum_assignment

    contingency = np.zeros((n_clusters, n_classes), dtype=np.int32)
    for k in range(n_clusters):
        mask = np.array(assignments) == k
        if np.sum(mask) > 0:
            for c in range(n_classes):
                contingency[k, c] = int(np.sum(np.array(labels)[mask] == c))

    row_ind, col_ind = linear_sum_assignment(-contingency)
    return float(contingency[row_ind, col_ind].sum() / len(labels))


def train_model_analytical(
    key: Array,
    model: Any,
    train_data: Array,
    test_data: Array,
    test_labels: Array,
    conj_weight: float = CONJ_WEIGHT,
    conj_mode: str = "none",
    entropy_weight: float = ENTROPY_WEIGHT,
    n_steps: int = N_STEPS,
    learning_rate: float = LEARNING_RATE,
    l2_int_weight: float = L2_INT_WEIGHT,
    n_conj_samples: int | None = None,
    conj_samples_mult: int = CONJ_SAMPLES_MULTIPLIER,
    use_cosine_lr: bool = False,
    rho_update_interval: int = RHO_UPDATE_INTERVAL,
) -> tuple[Array, AnalyticalModelResults]:
    """Train model with analytically-computed rho.

    Args:
        key: Random key
        model: The BinomialBernoulliMixture model
        train_data: Training data
        test_data: Test data
        test_labels: Test labels
        conj_weight: Weight for conjugation penalty
        conj_mode: Penalty mode - 'r2' uses (1-R²), 'var_f' uses Var[f̃], 'none' disables
        entropy_weight: Weight for batch entropy (prevents cluster collapse)
        n_steps: Number of training steps
        learning_rate: Learning rate
        l2_int_weight: L2 regularization on interaction weights
        n_conj_samples: Number of samples for analytical rho computation (overrides multiplier)
        conj_samples_mult: Multiplier to compute n_conj_samples = mult * n_latent
        use_cosine_lr: Use cosine annealing learning rate schedule
        rho_update_interval: Update rho every n steps (1 = every step)

    Returns:
        Tuple of (final_params, results_dict)
    """
    # Compute n_conj_samples from multiplier if not explicitly provided
    if n_conj_samples is None:
        n_conj_samples = conj_samples_mult * model.n_latent

    print(f"\n{'=' * 60}")
    print(f"Training with ANALYTICAL rho (weight={conj_weight})")
    print(f"  n_conj_samples = {n_conj_samples} ({n_conj_samples / model.n_latent:.1f}x n_latent)")
    print(f"{'=' * 60}")

    # Initialize model and extract hrm_params only
    key, init_key = jax.random.split(key)
    full_params = model.initialize_from_sample(
        init_key, train_data, location=0.0, shape=0.1
    )

    # Extract hrm_params (discard initial rho)
    _, hrm_params = model.split_coords(full_params)

    # Break symmetry in cluster-specific parameters
    key, cluster_key = jax.random.split(key)
    obs_bias, int_params, lat_params = model.hrm.split_coords(hrm_params)
    mix_obs_bias, _, cat_params = model.mix_man.split_coords(lat_params)

    mix_int_dim = model.n_latent * (model.n_clusters - 1)
    new_mix_int_mat = jax.random.normal(cluster_key, (mix_int_dim,)) * 0.5

    new_lat_params = model.mix_man.join_coords(
        mix_obs_bias, new_mix_int_mat, cat_params
    )
    hrm_params = model.hrm.join_coords(obs_bias, int_params, new_lat_params)

    # Optimizer only for hrm_params (not rho!)
    if use_cosine_lr:
        lr_schedule = optax.cosine_decay_schedule(
            init_value=learning_rate, decay_steps=n_steps, alpha=0.01
        )
        optimizer = optax.adam(lr_schedule)
    else:
        optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(hrm_params)

    def loss_fn(
        hrm_params: Array, key: Array, batch: Array
    ) -> tuple[Array, tuple[Array, Array, Array, Array, Array]]:
        """Loss function with analytical rho computation.

        Loss = -ELBO - entropy + conj_weight * penalty + l2_reg

        Where penalty is (1-R²) or Var[f̃] depending on conj_mode.
        Gradients flow through lstsq via JAX's implicit differentiation.
        """
        key, elbo_key, conj_key = jax.random.split(key, 3)

        # Compute analytical rho* and both metrics
        var_f, rho_star, r_squared, std_f = analytical_rho_with_var_penalty(
            conj_key, model, hrm_params, n_conj_samples
        )

        # Compute ELBO with analytical rho
        elbo = compute_elbo_with_analytical_rho(
            elbo_key, model, hrm_params, rho_star, batch, N_MC_SAMPLES
        )

        # Build full params for entropy computation
        params = model.join_coords(rho_star, hrm_params)

        def get_cluster_probs(x: Array) -> Array:
            q_params = model.approximate_posterior_at(params, x)
            return model.get_cluster_probs(q_params)

        all_probs = jax.vmap(get_cluster_probs)(batch)
        entropy = compute_batch_entropy(all_probs)

        # L2 regularization on interaction weights
        _, int_params_inner, _ = model.hrm.split_coords(hrm_params)
        l2_reg = l2_int_weight * jnp.sum(int_params_inner**2)

        # Conjugation penalty based on mode
        if conj_mode == "r2":
            r_squared_clamped = jnp.clip(r_squared, -1.0, 1.0)
            conj_penalty = 1.0 - r_squared_clamped
        elif conj_mode == "var_f":
            conj_penalty = var_f
        else:  # "none"
            conj_penalty = 0.0

        # Total loss
        loss = -elbo - entropy_weight * entropy + conj_weight * conj_penalty + l2_reg

        return loss, (elbo, r_squared, var_f, std_f, rho_star)

    @jax.jit
    def train_step(
        carry: Any, _: None
    ) -> tuple[Any, tuple[Array, Array, Array, Array, Array]]:
        hrm_params, opt_state, key, data = carry
        key, step_key, batch_key = jax.random.split(key, 3)

        batch_idx = jax.random.choice(batch_key, N_TRAIN, shape=(BATCH_SIZE,))
        batch = data[batch_idx]

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            hrm_params, step_key, batch
        )

        updates, new_opt_state = optimizer.update(grads, opt_state, hrm_params)
        new_hrm_params = optax.apply_updates(hrm_params, updates)

        elbo, r_squared, var_f, std_f, rho_star = metrics
        rho_norm = jnp.linalg.norm(rho_star)

        return (new_hrm_params, new_opt_state, key, data), (
            elbo,
            r_squared,
            var_f,
            std_f,
            rho_norm,
        )

    # Training loop
    elbos_list: list[float] = []
    r2s_list: list[float] = []
    var_f_list: list[float] = []
    std_f_list: list[float] = []
    rho_norms_list: list[float] = []

    key, train_key = jax.random.split(key)
    current_hrm_params = hrm_params
    current_opt_state = opt_state

    print(f"Training for {n_steps} steps...")

    for step in range(n_steps):
        train_key, step_key = jax.random.split(train_key)
        (current_hrm_params, current_opt_state, train_key, _), metrics = train_step(
            (current_hrm_params, current_opt_state, step_key, train_data), None
        )

        elbo, r_squared, var_f, std_f, rho_norm = metrics

        elbos_list.append(float(elbo))
        r2s_list.append(float(r_squared))
        var_f_list.append(float(var_f))
        std_f_list.append(float(std_f))
        rho_norms_list.append(float(rho_norm))

        if step % LOG_INTERVAL == 0:
            print(
                f"  Step {step}: ELBO={elbo:.2f}, R²={r_squared:.4f}, "
                f"Std[f̃]={std_f:.4f}, ||rho*||={rho_norm:.4f}"
            )

    # Final rho computation for evaluation
    key, final_conj_key = jax.random.split(key)
    _, final_rho, _, _ = analytical_rho_with_var_penalty(
        final_conj_key, model, current_hrm_params, n_conj_samples
    )
    final_params = model.join_coords(final_rho, current_hrm_params)

    print(f"Final ELBO: {elbos_list[-1]:.4f}")
    print(f"Final R²: {r2s_list[-1]:.4f}")
    print(f"Final Std[f̃]: {std_f_list[-1]:.4f}")

    # Evaluation
    print("Evaluating on test set...")

    def get_assignment(x: Array) -> Array:
        q_params = model.approximate_posterior_at(final_params, x)
        probs = model.get_cluster_probs(q_params)
        return jnp.argmax(probs)

    assignments = jax.vmap(get_assignment)(test_data)

    purity = compute_cluster_purity(assignments, test_labels, N_CLUSTERS)
    nmi = compute_nmi(assignments, test_labels)
    cluster_accuracy = compute_cluster_accuracy(assignments, test_labels, N_CLUSTERS)
    recon_error = float(model.normalized_reconstruction_error(final_params, test_data))
    cluster_counts = [int(jnp.sum(assignments == k)) for k in range(N_CLUSTERS)]

    print(f"  Purity: {100 * purity:.1f}%")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Accuracy: {100 * cluster_accuracy:.1f}%")
    print(f"  Reconstruction error: {recon_error:.4f}")
    print(f"  Clusters: {cluster_counts}")

    # Generate samples
    key, sample_key = jax.random.split(key)
    x_samples, _, _ = model.sample_from_model(sample_key, final_params, n=20)

    # Cluster prototypes
    cluster_prototypes: list[list[float]] = []
    for k in range(N_CLUSTERS):
        cluster_mask = assignments == k
        if jnp.sum(cluster_mask) > 0:
            prototype = jnp.mean(test_data[cluster_mask], axis=0)
        else:
            prototype = jnp.zeros(N_OBSERVABLE)
        cluster_prototypes.append(prototype.tolist())

    results: AnalyticalModelResults = {
        "training_elbos": elbos_list,
        "r_squared_history": r2s_list,
        "var_f_history": var_f_list,
        "std_f_history": std_f_list,
        "rho_norms": rho_norms_list,
        "cluster_purity": purity,
        "nmi": nmi,
        "cluster_accuracy": cluster_accuracy,
        "reconstruction_error": recon_error,
        "cluster_counts": cluster_counts,
        "cluster_assignments": assignments.tolist(),
        "generated_samples": x_samples.tolist(),
        "cluster_prototypes": cluster_prototypes,
    }

    return final_params, results


def main():
    args = parse_args()

    # Print configuration
    print("=" * 60)
    print("ANALYTICAL RHO CONFIGURATION")
    print("=" * 60)
    print(f"  N latent: {args.n_latent}")
    print(f"  N steps: {args.n_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Conjugation weight: {args.conj_weight}")
    print(f"  Conjugation mode: {args.conj_mode}")
    print(f"  Entropy weight: {args.entropy_weight}")
    print(f"  L2 int weight: {args.l2_int_weight}")
    print(f"  Conj samples multiplier: {args.conj_samples_mult}x n_latent")
    print(f"  Use cosine LR: {args.use_cosine_lr}")
    print("=" * 60)

    initialize_jax("gpu")
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)

    # Load data
    train_data, _, test_data, test_labels = load_mnist()

    # Create model
    n_latent = args.n_latent
    print(
        f"\nCreating model: {N_OBSERVABLE} obs, {n_latent} latent, {N_CLUSTERS} clusters"
    )
    model = binomial_bernoulli_mixture(
        n_observable=N_OBSERVABLE,
        n_latent=n_latent,
        n_clusters=N_CLUSTERS,
        n_trials=N_TRIALS,
    )
    print(f"Model dim: {model.dim}")
    print(f"  Rho params: {model.rho_man.dim}")
    print(f"  Harmonium params: {model.hrm.dim}")

    # Train
    key, train_key = jax.random.split(key)
    final_params, results = train_model_analytical(
        train_key,
        model,
        train_data,
        test_data,
        test_labels,
        conj_weight=args.conj_weight,
        conj_mode=args.conj_mode,
        entropy_weight=args.entropy_weight,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        l2_int_weight=args.l2_int_weight,
        conj_samples_mult=args.conj_samples_mult,
        use_cosine_lr=args.use_cosine_lr,
    )

    # Get reconstructions
    def get_recon(x: Array) -> Array:
        return model.reconstruct(final_params, x)

    reconstructions = jax.vmap(get_recon)(test_data[:20])

    # Save results to a custom file (analytical_results.json)
    full_results = BinomialBernoulliMNISTAnalyticalResults(
        analytical_rho=results,
        true_labels=test_labels.tolist(),
        original_images=test_data[:20].tolist(),
        reconstructed_images=reconstructions.tolist(),
    )
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    analytical_results_path = paths.results_dir / "analytical_results.json"
    with open(analytical_results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to {analytical_results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<22} {'Value':>14}")
    print("-" * 40)
    print(f"{'Cluster purity':<22} {100 * results['cluster_purity']:>13.1f}%")
    print(f"{'Accuracy':<22} {100 * results['cluster_accuracy']:>13.1f}%")
    print(f"{'NMI':<22} {results['nmi']:>14.4f}")
    print(f"{'Reconstruction error':<22} {results['reconstruction_error']:>14.4f}")
    print(f"{'Final R²':<22} {results['r_squared_history'][-1]:>14.4f}")
    print(f"{'Final Std[f̃]':<22} {results['std_f_history'][-1]:>14.4f}")
    print(f"{'Final Var[f̃]':<22} {results['var_f_history'][-1]:>14.4f}")
    print(f"{'Final ||rho*||':<22} {results['rho_norms'][-1]:>14.4f}")


if __name__ == "__main__":
    main()
