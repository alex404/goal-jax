"""Binomial-Bernoulli Mixture MNIST clustering example.

This example demonstrates clustering MNIST digits with a hierarchical model:
- Observable: Binomials (pixel count data)
- Latent: Bernoullis (binary hidden units)
- Mixture: Categorical over 10 clusters

Compares three training modes:
1. Learnable rho (free conjugation parameters)
2. Fixed rho=0 (tied weights, like RBM)
3. Conjugate-regularized (learnable rho with Var[f̃] penalty)

Usage:
    python -m examples.binomial_bernoulli_mnist.run
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from sklearn.datasets import fetch_openml

from goal.models import binomial_bernoulli_mixture

from ..shared import example_paths, initialize_jax
from .types import BinomialBernoulliMNISTResults, ModelResults

# MNIST dimensions
IMG_SIZE = 28
N_OBSERVABLE = IMG_SIZE * IMG_SIZE  # 784

# Model architecture
N_LATENT = 4096  # Binary hidden units (like RBM)
N_CLUSTERS = 10  # For 10 digits
N_TRIALS = 16  # Binomial trials for pixel values

# Data configuration
N_TRAIN = 60000  # Subset for faster training
N_TEST = 10000

# Training configuration
BATCH_SIZE = 512
N_STEPS = 50000
LEARNING_RATE = 1e-4
N_MC_SAMPLES = 10
ENTROPY_WEIGHT = 1.0  # Prevent cluster collapse
CONJ_REG_WEIGHT = 1.0  # Weight for conjugation regularization

# Conjugation error estimation
N_CONJ_SAMPLES = 100  # Samples for conjugation error estimation
LOG_INTERVAL = 200  # How often to print progress


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
    from sklearn.metrics import (
        normalized_mutual_info_score,
    )

    return float(
        normalized_mutual_info_score(
            np.array(labels), np.array(assignments), average_method="arithmetic"
        )
    )


def compute_conjugation_metrics_jit(
    model: Any, params: Array, key: Array, n_samples: int
) -> tuple[Array, Array, Array]:
    """JIT-compatible conjugation metrics computation.

    Samples from the CURRENT prior and computes Var[f̃], Std[f̃], and R².

    f_tilde(y) = rho · s_Y(y) - psi_X(theta_X + Theta_XY · s_Y(y))

    R² = 1 - Var[f̃] / Var[psi_X] measures how well rho·s approximates psi_X.

    Returns:
        Tuple of (variance, std, r_squared)
    """
    rho, hrm_params = model.split_coords(params)

    # Sample from CURRENT prior (not stale samples!)
    _, _, prior_mixture_params = model.hrm.split_coords(hrm_params)
    y_samples, _ = model.sample_from_posterior(key, prior_mixture_params, n_samples)

    def reduced_learning_signal(y: Array) -> Array:
        s_y = model.bas_lat_man.sufficient_statistic(y)
        term1 = jnp.dot(rho, s_y)
        lkl_params = model.likelihood_at(params, y)
        term2 = model.obs_man.log_partition_function(lkl_params)
        return term1 - term2

    def psi_x_at_y(y: Array) -> Array:
        lkl_params = model.likelihood_at(params, y)
        return model.obs_man.log_partition_function(lkl_params)

    f_vals = jax.vmap(reduced_learning_signal)(y_samples)
    psi_vals = jax.vmap(psi_x_at_y)(y_samples)

    var_f = jnp.var(f_vals)
    std_f = jnp.sqrt(var_f)
    var_psi = jnp.var(psi_vals)

    # R² = 1 - Var[residual] / Var[target]
    r_squared = jnp.where(var_psi > 1e-10, 1.0 - var_f / var_psi, 1.0)

    return var_f, std_f, r_squared


def train_model(
    key: Array,
    model: Any,
    train_data: Array,
    test_data: Array,
    test_labels: Array,
    fix_rho: bool = False,
    conj_reg_weight: float = 0.0,
) -> tuple[Array, ModelResults]:
    """Train a model and return final params and results.

    Args:
        key: Random key
        model: The BinomialBernoulliMixture model
        train_data: Training data
        test_data: Test data
        test_labels: Test labels
        fix_rho: If True, keep rho=0 throughout training (tied weights)
        conj_reg_weight: Weight for conjugation error regularization (0 = no reg)

    Returns:
        Tuple of (final_params, results_dict)
    """
    if fix_rho:
        mode_str = "FIXED rho=0"
    elif conj_reg_weight > 0:
        mode_str = f"CONJUGATE REGULARIZED (weight={conj_reg_weight})"
    else:
        mode_str = "LEARNABLE rho"

    print(f"\n{'=' * 60}")
    print(f"Training with {mode_str}")
    print(f"{'=' * 60}")

    # Initialize model
    key, init_key = jax.random.split(key)
    params = model.initialize_from_sample(init_key, train_data, location=0.0, shape=0.1)

    # Break symmetry in cluster-specific parameters
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

    zero_rho = jnp.zeros_like(rho)

    if fix_rho:
        optimizer = optax.adam(LEARNING_RATE)
        opt_state = optimizer.init(new_hrm_params)
    else:
        optimizer = optax.adam(LEARNING_RATE)
        opt_state = optimizer.init(params)

    # Loss functions
    def loss_fn_full(
        params: Array, key: Array, batch: Array
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        """Loss for learnable rho with optional conjugation regularization."""
        key, elbo_key, conj_key = jax.random.split(key, 3)
        elbo = model.mean_elbo(elbo_key, params, batch, N_MC_SAMPLES)

        def get_cluster_probs(x: Array) -> Array:
            q_params = model.approximate_posterior_at(params, x)
            return model.get_cluster_probs(q_params)

        all_probs = jax.vmap(get_cluster_probs)(batch)
        entropy = compute_batch_entropy(all_probs)

        # Compute all conjugation metrics
        conj_var, conj_std, conj_r2 = compute_conjugation_metrics_jit(
            model, params, conj_key, N_CONJ_SAMPLES
        )

        loss = -elbo - ENTROPY_WEIGHT * entropy + conj_reg_weight * conj_var

        return loss, (elbo, conj_var, conj_std, conj_r2)

    def loss_fn_fixed(
        hrm_params: Array, key: Array, batch: Array
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        """Loss for fixed rho=0."""
        key, elbo_key, conj_key = jax.random.split(key, 3)
        params = model.join_coords(zero_rho, hrm_params)
        elbo = model.mean_elbo(elbo_key, params, batch, N_MC_SAMPLES)

        def get_cluster_probs(x: Array) -> Array:
            q_params = model.approximate_posterior_at(params, x)
            return model.get_cluster_probs(q_params)

        all_probs = jax.vmap(get_cluster_probs)(batch)
        entropy = compute_batch_entropy(all_probs)

        # Compute all conjugation metrics for monitoring
        conj_var, conj_std, conj_r2 = compute_conjugation_metrics_jit(
            model, params, conj_key, N_CONJ_SAMPLES
        )

        loss = -elbo - ENTROPY_WEIGHT * entropy
        return loss, (elbo, conj_var, conj_std, conj_r2)

    # Training steps
    if fix_rho:

        @jax.jit
        def train_step(
            carry: Any, _: None
        ) -> tuple[Any, tuple[Array, Array, Array, Array]]:
            hrm_params, opt_state, key, data = carry
            key, step_key, batch_key = jax.random.split(key, 3)

            batch_idx = jax.random.choice(batch_key, N_TRAIN, shape=(BATCH_SIZE,))
            batch = data[batch_idx]

            (_, metrics), grads = jax.value_and_grad(loss_fn_fixed, has_aux=True)(
                hrm_params, step_key, batch
            )

            updates, new_opt_state = optimizer.update(grads, opt_state, hrm_params)
            new_hrm_params = optax.apply_updates(hrm_params, updates)

            return (new_hrm_params, new_opt_state, key, data), metrics

    else:

        @jax.jit
        def train_step(
            carry: Any, _: None
        ) -> tuple[Any, tuple[Array, Array, Array, Array]]:
            params, opt_state, key, data = carry
            key, step_key, batch_key = jax.random.split(key, 3)

            batch_idx = jax.random.choice(batch_key, N_TRAIN, shape=(BATCH_SIZE,))
            batch = data[batch_idx]

            (_, metrics), grads = jax.value_and_grad(loss_fn_full, has_aux=True)(
                params, step_key, batch
            )

            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            return (new_params, new_opt_state, key, data), metrics

    # Training loop
    elbos_list: list[float] = []
    conj_vars_list: list[float] = []
    conj_stds_list: list[float] = []
    conj_r2s_list: list[float] = []
    rho_norms_list: list[float] = []

    key, train_key = jax.random.split(key)

    current_params = params
    current_hrm_params = new_hrm_params
    current_opt_state = opt_state

    print(f"Training for {N_STEPS} steps...")

    for step in range(N_STEPS):
        if fix_rho:
            train_key, step_key = jax.random.split(train_key)
            (current_hrm_params, current_opt_state, train_key, _), metrics = train_step(
                (current_hrm_params, current_opt_state, step_key, train_data), None
            )
            current_params = model.join_coords(zero_rho, current_hrm_params)
        else:
            train_key, step_key = jax.random.split(train_key)
            (current_params, current_opt_state, train_key, _), metrics = train_step(
                (current_params, current_opt_state, step_key, train_data), None
            )

        elbo, conj_var, conj_std, conj_r2 = metrics

        # Store metrics every step
        elbos_list.append(float(elbo))
        conj_vars_list.append(float(conj_var))
        conj_stds_list.append(float(conj_std))
        conj_r2s_list.append(float(conj_r2))

        rho_current, _ = model.split_coords(current_params)
        rho_norm = float(jnp.linalg.norm(rho_current))
        rho_norms_list.append(rho_norm)

        # Print progress periodically
        if step % LOG_INTERVAL == 0:
            print(
                f"  Step {step}: ELBO={elbo:.2f}, Std[f̃]={conj_std:.2f}, R²={conj_r2:.4f}, ||rho||={rho_norm:.4f}"
            )

    final_params = current_params
    print(f"Final ELBO: {elbos_list[-1]:.4f}")

    # Evaluation
    print("Evaluating on test set...")

    def get_assignment(x: Array) -> Array:
        q_params = model.approximate_posterior_at(final_params, x)
        probs = model.get_cluster_probs(q_params)
        return jnp.argmax(probs)

    assignments = jax.vmap(get_assignment)(test_data)

    purity = compute_cluster_purity(assignments, test_labels, N_CLUSTERS)
    nmi = compute_nmi(assignments, test_labels)
    recon_error = float(model.normalized_reconstruction_error(final_params, test_data))
    cluster_counts = [int(jnp.sum(assignments == k)) for k in range(N_CLUSTERS)]

    print(f"  Purity: {100 * purity:.1f}%")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Reconstruction error: {recon_error:.4f}")
    print(f"  Clusters: {cluster_counts}")
    print(f"  Final R²: {conj_r2s_list[-1]:.4f}")

    # Generate samples from this model
    key, sample_key = jax.random.split(key)
    x_samples, _, _ = model.sample_from_model(sample_key, final_params, n=20)

    # Compute cluster prototypes
    cluster_prototypes: list[list[float]] = []
    for k in range(N_CLUSTERS):
        cluster_mask = assignments == k
        if jnp.sum(cluster_mask) > 0:
            prototype = jnp.mean(test_data[cluster_mask], axis=0)
        else:
            prototype = jnp.zeros(N_OBSERVABLE)
        cluster_prototypes.append(prototype.tolist())

    results: ModelResults = {
        "training_elbos": elbos_list,
        "conjugation_errors": conj_vars_list,
        "conjugation_stds": conj_stds_list,
        "conjugation_r2s": conj_r2s_list,
        "rho_norms": rho_norms_list,
        "cluster_purity": purity,
        "nmi": nmi,
        "reconstruction_error": recon_error,
        "cluster_counts": cluster_counts,
        "cluster_assignments": assignments.tolist(),
        "generated_samples": x_samples.tolist(),
        "cluster_prototypes": cluster_prototypes,
    }

    return final_params, results


def main():
    initialize_jax("gpu")
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)

    # Load data
    train_data, _, test_data, test_labels = load_mnist()

    # Create model
    print(
        f"\nCreating model: {N_OBSERVABLE} obs, {N_LATENT} latent, {N_CLUSTERS} clusters"
    )
    model = binomial_bernoulli_mixture(
        n_observable=N_OBSERVABLE,
        n_latent=N_LATENT,
        n_clusters=N_CLUSTERS,
        n_trials=N_TRIALS,
    )
    print(f"Model dim: {model.dim}")
    print(f"  Rho params: {model.rho_man.dim}")
    print(f"  Harmonium params: {model.hrm.dim}")

    # Train all three models
    key, key1, key2, key3 = jax.random.split(key, 4)

    final_params_learnable, results_learnable = train_model(
        key1,
        model,
        train_data,
        test_data,
        test_labels,
        fix_rho=False,
        conj_reg_weight=0,
    )

    _, results_fixed = train_model(
        key2, model, train_data, test_data, test_labels, fix_rho=True, conj_reg_weight=0
    )

    final_params_conj_reg, results_conj_reg = train_model(
        key3,
        model,
        train_data,
        test_data,
        test_labels,
        fix_rho=False,
        conj_reg_weight=CONJ_REG_WEIGHT,
    )

    # Get reconstructions from best model (by NMI)
    best_nmi = max(
        results_learnable["nmi"], results_fixed["nmi"], results_conj_reg["nmi"]
    )
    if results_conj_reg["nmi"] == best_nmi:
        best_params = final_params_conj_reg
        best_name = "conjugate-regularized"
    elif results_fixed["nmi"] == best_nmi:
        best_params = model.join_coords(
            jnp.zeros(model.rho_man.dim),
            model.generative_params(final_params_learnable),
        )
        best_name = "fixed rho=0"
    else:
        best_params = final_params_learnable
        best_name = "learnable rho"

    print(f"\nBest model by NMI: {best_name}")

    # Get reconstructions from best model
    def get_recon(x: Array) -> Array:
        return model.reconstruct(best_params, x)

    reconstructions = jax.vmap(get_recon)(test_data[:20])

    # Save results
    results = BinomialBernoulliMNISTResults(
        learnable_rho=results_learnable,
        fixed_rho=results_fixed,
        conjugate_regularized=results_conj_reg,
        true_labels=test_labels.tolist(),
        original_images=test_data[:20].tolist(),
        reconstructed_images=reconstructions.tolist(),
    )
    paths.save_analysis(results)
    print(f"\nResults saved to {paths.analysis_path}")

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<22} {'Learnable':>14} {'Fixed rho=0':>14} {'Conj. Reg.':>14}")
    print("-" * 80)

    purity_l = 100 * results_learnable["cluster_purity"]
    purity_f = 100 * results_fixed["cluster_purity"]
    purity_c = 100 * results_conj_reg["cluster_purity"]
    print(
        f"{'Cluster purity':<22} {purity_l:>13.1f}% {purity_f:>13.1f}% {purity_c:>13.1f}%"
    )

    nmi_l = results_learnable["nmi"]
    nmi_f = results_fixed["nmi"]
    nmi_c = results_conj_reg["nmi"]
    print(f"{'NMI':<22} {nmi_l:>14.4f} {nmi_f:>14.4f} {nmi_c:>14.4f}")

    recon_l = results_learnable["reconstruction_error"]
    recon_f = results_fixed["reconstruction_error"]
    recon_c = results_conj_reg["reconstruction_error"]
    print(
        f"{'Reconstruction error':<22} {recon_l:>14.4f} {recon_f:>14.4f} {recon_c:>14.4f}"
    )

    var_l = results_learnable["conjugation_errors"][-1]
    var_f = results_fixed["conjugation_errors"][-1]
    var_c = results_conj_reg["conjugation_errors"][-1]
    print(f"{'Final Var[f̃]':<22} {var_l:>14.2f} {var_f:>14.2f} {var_c:>14.2f}")

    std_l = results_learnable["conjugation_stds"][-1]
    std_f = results_fixed["conjugation_stds"][-1]
    std_c = results_conj_reg["conjugation_stds"][-1]
    print(f"{'Final Std[f̃]':<22} {std_l:>14.2f} {std_f:>14.2f} {std_c:>14.2f}")

    r2_l = results_learnable["conjugation_r2s"][-1]
    r2_f = results_fixed["conjugation_r2s"][-1]
    r2_c = results_conj_reg["conjugation_r2s"][-1]
    print(f"{'Final R²':<22} {r2_l:>14.4f} {r2_f:>14.4f} {r2_c:>14.4f}")

    rho_l = results_learnable["rho_norms"][-1]
    rho_f = results_fixed["rho_norms"][-1]
    rho_c = results_conj_reg["rho_norms"][-1]
    print(f"{'Final ||rho||':<22} {rho_l:>14.4f} {rho_f:>14.4f} {rho_c:>14.4f}")


if __name__ == "__main__":
    main()
