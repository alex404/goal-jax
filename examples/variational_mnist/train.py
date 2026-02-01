"""Training script for Variational MNIST clustering.

Compares training modes:
1. free: Learnable rho (free conjugation parameters)
2. fixed: Fixed rho=0 (tied weights, like RBM)
3. regularized: Learnable rho with conjugation penalty (Var[f_tilde])
4. analytical: Analytical rho via least squares each step
5. em: EM-style gradient computation for ELBO optimization

Supports two observable types:
- binomial: Discretized pixels in [0, n_trials] (default)
- poisson: Count data with unbounded support

Usage:
    python -m examples.variational_mnist.train
    python -m examples.variational_mnist.train --mode free --n-steps 10000
    python -m examples.variational_mnist.train --mode em --n-em-iterations 50
    python -m examples.variational_mnist.train --observable poisson --mode free
"""

import argparse
import json
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from sklearn.datasets import fetch_openml

from ..shared import example_paths, initialize_jax
from .model import (
    ANALYTICAL_RHO_SAMPLES_MULTIPLIER,
    N_OBSERVABLE,
    MixtureModel,
    compute_analytical_rho,
    compute_em_elbo_gradient,
    compute_positive_phase_stats,
    create_model,
)

# Data configuration
N_TRAIN = 60000
N_TEST = 10000

# Default hyperparameters
DEFAULT_N_LATENT = 1024
DEFAULT_N_CLUSTERS = 10
DEFAULT_N_TRIALS = 16
DEFAULT_N_STEPS = 50000
DEFAULT_BATCH_SIZE = 512
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_N_MC_SAMPLES = 10
DEFAULT_ENTROPY_WEIGHT = 1.0  # Prevents cluster collapse
DEFAULT_CONJ_WEIGHT = 1.0  # Weight for conjugation penalty

# EM-specific hyperparameters
DEFAULT_N_EM_ITERATIONS = 50
DEFAULT_N_M_STEPS = 1000
DEFAULT_N_NEGATIVE_SAMPLES = 512

# Logging
LOG_INTERVAL = 200
N_CONJ_SAMPLES = 10000


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Variational MNIST model")
    parser.add_argument(
        "--mode",
        choices=["all", "free", "fixed", "regularized", "analytical", "em"],
        default="all",
        help="Training mode: 'all' trains free/fixed/regularized, or train a specific one",
    )
    parser.add_argument(
        "--n-latent",
        type=int,
        default=DEFAULT_N_LATENT,
        help="Number of binary latent units",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help="Number of mixture components",
    )
    parser.add_argument(
        "--n-steps", type=int, default=DEFAULT_N_STEPS, help="Number of training steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=DEFAULT_ENTROPY_WEIGHT,
        help="Weight for prior entropy (prevents cluster collapse in generative model)",
    )
    parser.add_argument(
        "--posterior-entropy-weight",
        type=float,
        default=DEFAULT_ENTROPY_WEIGHT,
        help="Weight for posterior batch entropy (prevents cluster collapse in assignments)",
    )
    parser.add_argument(
        "--conj-weight",
        type=float,
        default=DEFAULT_CONJ_WEIGHT,
        help="Weight for conjugation penalty (regularized mode only)",
    )
    parser.add_argument(
        "--conj-mode",
        choices=["variance", "r2"],
        default="variance",
        help="Conjugation penalty: 'variance' uses Var[f_tilde], 'r2' uses (1-R^2)",
    )
    parser.add_argument(
        "--observable",
        choices=["binomial", "poisson"],
        default="binomial",
        help="Observable distribution type: 'binomial' (bounded) or 'poisson' (unbounded)",
    )
    parser.add_argument(
        "--latent",
        choices=["bernoulli", "vonmises"],
        default="bernoulli",
        help="Latent distribution type: 'bernoulli' (binary) or 'vonmises' (circular continuous)",
    )
    parser.add_argument(
        "--analytical-samples-mult",
        type=int,
        default=ANALYTICAL_RHO_SAMPLES_MULTIPLIER,
        help=f"For analytical mode: n_samples = mult * lat_dim (default {ANALYTICAL_RHO_SAMPLES_MULTIPLIER}x for overdetermined lstsq)",
    )
    # EM-specific arguments
    parser.add_argument(
        "--n-em-iterations",
        type=int,
        default=DEFAULT_N_EM_ITERATIONS,
        help="Number of EM iterations (outer loop) for EM mode",
    )
    parser.add_argument(
        "--n-m-steps",
        type=int,
        default=DEFAULT_N_M_STEPS,
        help="Number of M-step gradient iterations per E-step in EM mode",
    )
    parser.add_argument(
        "--n-negative-samples",
        type=int,
        default=DEFAULT_N_NEGATIVE_SAMPLES,
        help="Number of samples for negative phase in EM mode",
    )
    return parser.parse_args()


def load_mnist_sklearn() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load MNIST using scikit-learn's fetch_openml."""
    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required. Install with: pip install scikit-learn"
        ) from e
    else:
        return mnist.data, mnist.target  # pyright: ignore[reportAttributeAccessIssue]


def load_mnist(
    n_trials: int = DEFAULT_N_TRIALS,
    observable_type: Literal["binomial", "poisson"] = "binomial",
) -> tuple[Array, Array, Array, Array]:
    """Load MNIST and split into train/test with labels.

    Args:
        n_trials: Discretization level (both binomial and poisson scale to [0, n_trials])
        observable_type: Type of observable distribution

    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    print(f"Loading MNIST data ({observable_type}, scaled to [0, {n_trials}])...")
    data, labels = load_mnist_sklearn()

    labels = labels.astype(np.int32)
    # Scale to [0, n_trials] for both types to prevent Poisson rates from exploding
    data = np.round(data * n_trials / 255.0).astype(np.float32)

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


def compute_prior_entropy(model: MixtureModel, hrm_params: Array) -> Array:
    """Compute entropy of the prior's marginal cluster distribution.

    For a mixture p(y,k), the marginal over k is:
        p(k) ∝ exp(θ_K[k] + ψ_Y(θ_Y[k]))

    This directly penalizes cluster collapse in the prior, not just the posterior.
    """
    # Extract prior mixture params
    _, _, prior_mixture_params = model.hrm.split_coords(hrm_params)

    # Use the model's method which correctly computes the marginal
    probs = model.get_cluster_probs(prior_mixture_params)
    probs_safe = jnp.clip(probs, 1e-10, 1.0)

    return -jnp.sum(probs_safe * jnp.log(probs_safe))


def compute_conjugation_metrics_jit(
    model: Any, params: Array, key: Array, n_samples: int
) -> tuple[Array, Array, Array]:
    """JIT-compatible conjugation metrics computation.

    Returns:
        Tuple of (variance, std, r_squared)
    """
    rho, hrm_params = model.split_coords(params)

    # Sample from current prior
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

    variance_threshold = 1e-6
    raw_r2 = 1.0 - var_f / jnp.maximum(var_psi, variance_threshold)
    r_squared = jnp.where(
        var_psi > variance_threshold,
        jnp.clip(raw_r2, -10.0, 1.0),
        jnp.nan,
    )

    return var_f, std_f, r_squared


def train_model(  # noqa: C901
    key: Array,
    model: MixtureModel,
    train_data: Array,
    test_data: Array,
    test_labels: Array,
    mode: str,
    n_steps: int,
    learning_rate: float,
    entropy_weight: float,
    conj_weight: float,
    conj_mode: str,
    observable_type: Literal["binomial", "poisson"] = "binomial",
    n_trials: int = DEFAULT_N_TRIALS,
    analytical_samples_mult: int = ANALYTICAL_RHO_SAMPLES_MULTIPLIER,
    posterior_entropy_weight: float = DEFAULT_ENTROPY_WEIGHT,
) -> tuple[Array, dict[str, Any]]:
    """Train a model with specified mode.

    Args:
        key: Random key
        model: The BinomialBernoulliMixture model
        train_data: Training data
        test_data: Test data
        test_labels: Test labels
        mode: Training mode ('free', 'fixed', or 'regularized')
        n_steps: Number of training steps
        learning_rate: Learning rate
        entropy_weight: Weight for batch entropy
        conj_weight: Weight for conjugation penalty (regularized mode)
        conj_mode: Conjugation penalty mode ('variance' or 'r2')
        observable_type: Type of observable distribution
        n_trials: Binomial discretization level
        analytical_samples_mult: For analytical mode, n_samples = mult * lat_dim

    Returns:
        Tuple of (final_params, results_dict)
    """
    use_analytical_rho = mode == "analytical"
    fix_rho = mode == "fixed"
    use_conj_penalty = mode == "regularized"

    n_analytical_samples = analytical_samples_mult * model.bas_lat_man.dim
    mode_str = {
        "free": "LEARNABLE rho",
        "fixed": "FIXED rho=0",
        "regularized": f"REGULARIZED (weight={conj_weight}, mode={conj_mode})",
        "analytical": f"ANALYTICAL rho ({n_analytical_samples} samples, {analytical_samples_mult}x lat_dim={model.bas_lat_man.dim})",
    }[mode]

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

    mix_int_dim = model.bas_lat_man.dim * (model.n_clusters - 1)
    new_mix_int_mat = jax.random.normal(cluster_key, (mix_int_dim,)) * 0.5

    new_lat_params = model.mix_man.join_coords(
        mix_obs_bias, new_mix_int_mat, cat_params
    )
    new_hrm_params = model.hrm.join_coords(obs_bias, int_params, new_lat_params)
    params = model.join_coords(rho, new_hrm_params)

    zero_rho = jnp.zeros_like(rho)

    # Optimizer setup
    # In analytical mode, optimizer only updates hrm_params (rho is computed via lstsq)
    optimizer = optax.adam(learning_rate)
    if use_analytical_rho or fix_rho:
        opt_state = optimizer.init(new_hrm_params)
    else:
        opt_state = optimizer.init(params)

    # Loss function for learnable rho (free or regularized)
    def loss_fn_full(
        params: Array, key: Array, batch: Array
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        key, elbo_key, conj_key = jax.random.split(key, 3)
        elbo = model.mean_elbo(elbo_key, params, batch, DEFAULT_N_MC_SAMPLES)

        _, hrm_params_from_full = model.split_coords(params)
        prior_entropy = compute_prior_entropy(model, hrm_params_from_full)

        # Posterior entropy
        def get_cluster_probs(x: Array) -> Array:
            q_params = model.approximate_posterior_at(params, x)
            return model.get_cluster_probs(q_params)

        all_probs = jax.vmap(get_cluster_probs)(batch)
        posterior_entropy = compute_batch_entropy(all_probs)

        conj_var, conj_std, conj_r2 = compute_conjugation_metrics_jit(
            model, params, conj_key, N_CONJ_SAMPLES
        )

        # Conjugation penalty (only applied if regularized mode)
        if use_conj_penalty:
            if conj_mode == "r2":
                conj_penalty = 1.0 - jnp.clip(conj_r2, -1.0, 1.0)
            else:
                conj_penalty = conj_var
            conj_loss = conj_weight * conj_penalty
        else:
            conj_loss = 0.0

        loss = (
            -elbo
            - entropy_weight * prior_entropy
            - posterior_entropy_weight * posterior_entropy
            + conj_loss
        )
        return loss, (elbo, conj_var, conj_std, conj_r2)

    # Loss function for fixed rho=0
    def loss_fn_fixed(
        hrm_params: Array, key: Array, batch: Array
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        key, elbo_key, conj_key = jax.random.split(key, 3)
        params_with_zero_rho = model.join_coords(zero_rho, hrm_params)
        elbo = model.mean_elbo(
            elbo_key, params_with_zero_rho, batch, DEFAULT_N_MC_SAMPLES
        )

        prior_entropy = compute_prior_entropy(model, hrm_params)

        # Posterior entropy
        def get_cluster_probs(x: Array) -> Array:
            q_params = model.approximate_posterior_at(params_with_zero_rho, x)
            return model.get_cluster_probs(q_params)

        all_probs = jax.vmap(get_cluster_probs)(batch)
        posterior_entropy = compute_batch_entropy(all_probs)

        conj_var, conj_std, conj_r2 = compute_conjugation_metrics_jit(
            model, params_with_zero_rho, conj_key, N_CONJ_SAMPLES
        )

        loss = (
            -elbo
            - entropy_weight * prior_entropy
            - posterior_entropy_weight * posterior_entropy
        )
        return loss, (elbo, conj_var, conj_std, conj_r2)

    # Loss function for analytical rho (computed via lstsq each step)
    # NOTE: Gradients flow through lstsq via JAX implicit differentiation
    # The optimizer only updates hrm_params; rho is always optimal given hrm_params

    def loss_fn_analytical(
        hrm_params: Array, key: Array, batch: Array
    ) -> tuple[Array, tuple[Array, Array, Array, Array, Array]]:
        key, rho_key, elbo_key = jax.random.split(key, 3)

        # Compute analytical rho via lstsq (differentiable in JAX)
        rho_star, r_squared, _ = compute_analytical_rho(
            model, hrm_params, rho_key, n_samples=n_analytical_samples
        )

        # Build full params with analytical rho and compute ELBO
        params_with_rho = model.join_coords(rho_star, hrm_params)
        elbo = model.mean_elbo(elbo_key, params_with_rho, batch, DEFAULT_N_MC_SAMPLES)

        # Prior entropy to prevent cluster collapse in generative model
        prior_entropy = compute_prior_entropy(model, hrm_params)

        # Posterior entropy to prevent cluster collapse in assignments
        def get_cluster_probs(x: Array) -> Array:
            q_params = model.approximate_posterior_at(params_with_rho, x)
            return model.get_cluster_probs(q_params)

        all_probs = jax.vmap(get_cluster_probs)(batch)
        posterior_entropy = compute_batch_entropy(all_probs)

        loss = (
            -elbo
            - entropy_weight * prior_entropy
            - posterior_entropy_weight * posterior_entropy
        )
        # Return rho_star as part of aux so we can track it
        # Use r_squared for both conj_var and conj_r2 slots (var not meaningful here)
        return loss, (elbo, r_squared, r_squared, r_squared, rho_star)

    # Training step functions
    if use_analytical_rho:

        @jax.jit
        def train_step(
            carry: Any, _: None
        ) -> tuple[Any, tuple[Array, Array, Array, Array, Array]]:
            hrm_params, opt_state, step_key, data = carry
            step_key, next_key, batch_key = jax.random.split(step_key, 3)

            batch_idx = jax.random.choice(
                batch_key, N_TRAIN, shape=(DEFAULT_BATCH_SIZE,)
            )
            batch = data[batch_idx]

            (_, metrics), grads = jax.value_and_grad(loss_fn_analytical, has_aux=True)(
                hrm_params, next_key, batch
            )

            updates, new_opt_state = optimizer.update(grads, opt_state, hrm_params)
            new_hrm_params = optax.apply_updates(hrm_params, updates)

            return (new_hrm_params, new_opt_state, step_key, data), metrics

    elif fix_rho:

        @jax.jit
        def train_step(
            carry: Any, _: None
        ) -> tuple[Any, tuple[Array, Array, Array, Array]]:
            hrm_params, opt_state, step_key, data = carry
            step_key, next_key, batch_key = jax.random.split(step_key, 3)

            batch_idx = jax.random.choice(
                batch_key, N_TRAIN, shape=(DEFAULT_BATCH_SIZE,)
            )
            batch = data[batch_idx]

            (_, metrics), grads = jax.value_and_grad(loss_fn_fixed, has_aux=True)(
                hrm_params, next_key, batch
            )

            updates, new_opt_state = optimizer.update(grads, opt_state, hrm_params)
            new_hrm_params = optax.apply_updates(hrm_params, updates)

            return (new_hrm_params, new_opt_state, step_key, data), metrics
    else:

        @jax.jit
        def train_step(
            carry: Any, _: None
        ) -> tuple[Any, tuple[Array, Array, Array, Array]]:
            params, opt_state, step_key, data = carry
            step_key, next_key, batch_key = jax.random.split(step_key, 3)

            batch_idx = jax.random.choice(
                batch_key, N_TRAIN, shape=(DEFAULT_BATCH_SIZE,)
            )
            batch = data[batch_idx]

            (_, metrics), grads = jax.value_and_grad(loss_fn_full, has_aux=True)(
                params, next_key, batch
            )

            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            return (new_params, new_opt_state, step_key, data), metrics

    # Training loop
    elbos: list[float] = []
    conj_vars: list[float] = []
    conj_stds: list[float] = []
    conj_r2s: list[float] = []
    rho_norms: list[float] = []

    key, train_key = jax.random.split(key)
    current_params = params
    current_hrm_params = new_hrm_params
    current_opt_state = opt_state

    print(f"Training for {n_steps} steps...")

    # Pre-compile assignment function for logging
    from .evaluate import compute_nmi

    @jax.jit
    def get_assignments_batch(params: Array, xs: Array) -> Array:
        def get_assignment(x: Array) -> Array:
            q_params = model.approximate_posterior_at(params, x)
            probs = model.get_cluster_probs(q_params)
            return jnp.argmax(probs)

        return jax.vmap(get_assignment)(xs)

    for step in range(n_steps):
        if use_analytical_rho:
            (current_hrm_params, current_opt_state, train_key, _), metrics = train_step(
                (current_hrm_params, current_opt_state, train_key, train_data), None
            )
            # Analytical mode returns 5 values: (elbo, r2, r2, r2, rho_star)
            elbo, _, _, conj_r2, rho_star = metrics
            current_params = model.join_coords(rho_star, current_hrm_params)
            conj_var = 0.0  # Not computed in analytical mode
            conj_std = 0.0  # Not computed in analytical mode
        elif fix_rho:
            (current_hrm_params, current_opt_state, train_key, _), metrics = train_step(
                (current_hrm_params, current_opt_state, train_key, train_data), None
            )
            elbo, conj_var, conj_std, conj_r2 = metrics
            current_params = model.join_coords(zero_rho, current_hrm_params)
        else:
            (current_params, current_opt_state, train_key, _), metrics = train_step(
                (current_params, current_opt_state, train_key, train_data), None
            )
            elbo, conj_var, conj_std, conj_r2 = metrics

        elbos.append(float(elbo))
        conj_vars.append(float(conj_var))
        conj_stds.append(float(conj_std))
        conj_r2s.append(float(conj_r2))

        rho_current, hrm_current = model.split_coords(current_params)
        rho_norm = float(jnp.linalg.norm(rho_current))
        rho_norms.append(rho_norm)

        if step % LOG_INTERVAL == 0:
            assignments = get_assignments_batch(current_params, test_data)
            nmi = compute_nmi(assignments, test_labels)
            prior_ent = float(compute_prior_entropy(model, hrm_current))
            max_ent = float(jnp.log(model.n_clusters))

            print(
                f"  Step {step}: ELBO={elbo:.2f}, NMI={nmi:.4f}, R^2={conj_r2:.4f}, ||rho||={rho_norm:.2f}, H_prior={prior_ent:.2f}/{max_ent:.2f}"
            )

    final_params = current_params
    print(f"Final ELBO: {elbos[-1]:.4f}")

    # Evaluation
    print("Evaluating on test set...")

    def get_assignment(x: Array) -> Array:
        q_params = model.approximate_posterior_at(final_params, x)
        probs = model.get_cluster_probs(q_params)
        return jnp.argmax(probs)

    assignments = jax.vmap(get_assignment)(test_data)

    # Import evaluation metrics
    from .evaluate import (
        compute_cluster_accuracy,
        compute_cluster_purity,
        compute_nmi,
    )

    purity = compute_cluster_purity(assignments, test_labels, model.n_clusters)
    nmi = compute_nmi(assignments, test_labels)
    accuracy = compute_cluster_accuracy(assignments, test_labels, model.n_clusters)

    # Compute reconstruction error with appropriate normalization
    if observable_type == "poisson":
        from goal.models.graphical.hierarchical import (
            PoissonBernoulliMixture,
        )

        assert isinstance(model, PoissonBernoulliMixture)
        recon_error = float(
            model.normalized_reconstruction_error(
                final_params, test_data, scale=n_trials
            )
        )
    else:
        recon_error = float(
            model.normalized_reconstruction_error(final_params, test_data)
        )

    cluster_counts = [int(jnp.sum(assignments == k)) for k in range(model.n_clusters)]

    print(f"  Purity: {100 * purity:.1f}%")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Accuracy: {100 * accuracy:.1f}%")
    print(f"  Reconstruction error: {recon_error:.4f}")
    print(f"  Clusters: {cluster_counts}")
    print(f"  Final R^2: {conj_r2s[-1]:.4f}")

    # Generate samples (stochastic)
    key, sample_key = jax.random.split(key)
    x_samples, _, _ = model.sample_from_model(sample_key, final_params, n=20)

    # Clip samples for visualization (especially important for Poisson)
    x_samples = jnp.clip(x_samples, 0, n_trials)

    # Also generate mean-based samples (expected values, not sampled)
    # This shows what the prior "means" rather than random samples
    key, mean_key = jax.random.split(key)
    _, _, prior_mixture_params = model.hrm.split_coords(
        model.generative_params(final_params)
    )
    y_mean_samples, _ = model.sample_from_posterior(mean_key, prior_mixture_params, 20)

    def get_mean_image(y: Array) -> Array:
        lkl_params = model.likelihood_at(final_params, y)
        return model.obs_man.to_mean(lkl_params)

    x_mean_samples = jax.vmap(get_mean_image)(y_mean_samples)

    # Clip mean samples for visualization (especially important for Poisson)
    x_mean_samples = jnp.clip(x_mean_samples, 0, n_trials)

    # Cluster prototypes
    cluster_prototypes: list[list[float]] = []
    for k in range(model.n_clusters):
        mask = assignments == k
        if jnp.sum(mask) > 0:
            prototype = jnp.mean(test_data[mask], axis=0)
        else:
            prototype = jnp.zeros(N_OBSERVABLE)
        cluster_prototypes.append(prototype.tolist())

    results = {
        "mode": mode,
        "training_elbos": elbos,
        "generated_samples_mean": x_mean_samples.tolist(),
        "conjugation_vars": conj_vars,
        "conjugation_stds": conj_stds,
        "conjugation_r2s": conj_r2s,
        "rho_norms": rho_norms,
        "cluster_purity": purity,
        "nmi": nmi,
        "cluster_accuracy": accuracy,
        "reconstruction_error": recon_error,
        "cluster_counts": cluster_counts,
        "cluster_assignments": assignments.tolist(),
        "generated_samples": x_samples.tolist(),
        "cluster_prototypes": cluster_prototypes,
    }

    return final_params, results


def train_model_em(  # noqa: C901
    key: Array,
    model: MixtureModel,
    train_data: Array,
    test_data: Array,
    test_labels: Array,
    n_em_iterations: int,
    n_m_steps: int,
    n_negative_samples: int,
    learning_rate: float,
    entropy_weight: float,
    observable_type: Literal["binomial", "poisson"] = "binomial",
    n_trials: int = DEFAULT_N_TRIALS,
    analytical_samples_mult: int = ANALYTICAL_RHO_SAMPLES_MULTIPLIER,
) -> tuple[Array, dict[str, Any]]:
    """Train a model using EM-style ELBO optimization.

    EM-style training separates inference and parameter updates:
    - E-step (once per EM iteration): Fit rho analytically, compute positive phase stats
    - M-step (many gradient steps): Update generative params using positive - negative gradient

    This should be more stable than standard autodiff ELBO because:
    - Separates inference (E-step) from parameter updates (M-step)
    - Explicit gradient formula avoids pathological autodiff paths
    - Rho is recomputed each iteration to be optimal for current generative params

    Args:
        key: Random key
        model: The mixture model
        train_data: Training data
        test_data: Test data
        test_labels: Test labels
        n_em_iterations: Number of outer EM iterations
        n_m_steps: Number of M-step gradient iterations per E-step
        n_negative_samples: Number of samples for negative phase estimation
        learning_rate: Learning rate for M-step optimizer
        entropy_weight: Weight for prior entropy regularization
        observable_type: Type of observable distribution
        n_trials: Binomial discretization level
        analytical_samples_mult: Multiplier for analytical rho samples

    Returns:
        Tuple of (final_params, results_dict)
    """
    n_analytical_samples = analytical_samples_mult * model.bas_lat_man.dim

    print(f"\n{'=' * 60}")
    print(
        f"Training with EM mode: {n_em_iterations} EM iters, {n_m_steps} M-steps each"
    )
    print(f"  Negative samples: {n_negative_samples}")
    print(f"  Analytical rho samples: {n_analytical_samples}")
    print(f"{'=' * 60}")

    # Initialize model
    key, init_key = jax.random.split(key)
    params = model.initialize_from_sample(init_key, train_data, location=0.0, shape=0.1)

    # Break symmetry in cluster-specific parameters
    key, cluster_key = jax.random.split(key)
    _, hrm_params = model.split_coords(params)
    obs_bias, int_params, lat_params = model.hrm.split_coords(hrm_params)
    mix_obs_bias, _, cat_params = model.mix_man.split_coords(lat_params)

    mix_int_dim = model.bas_lat_man.dim * (model.n_clusters - 1)
    new_mix_int_mat = jax.random.normal(cluster_key, (mix_int_dim,)) * 0.5

    new_lat_params = model.mix_man.join_coords(
        mix_obs_bias, new_mix_int_mat, cat_params
    )
    hrm_params = model.hrm.join_coords(obs_bias, int_params, new_lat_params)

    # Optimizer for hrm_params only (rho is computed analytically)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(hrm_params)

    # Pre-compile functions for evaluation
    from .evaluate import compute_nmi

    @jax.jit
    def get_assignments_batch(params: Array, xs: Array) -> Array:
        def get_assignment(x: Array) -> Array:
            q_params = model.approximate_posterior_at(params, x)
            probs = model.get_cluster_probs(q_params)
            return jnp.argmax(probs)

        return jax.vmap(get_assignment)(xs)

    # Create M-step iteration function (called many times per E-step)
    def make_m_step_fn(
        positive_stats: Array,
    ) -> Any:
        """Create jitted M-step function with fixed positive_stats."""

        def entropy_loss(hrm_p: Array) -> Array:
            """Compute negative prior entropy (we minimize this)."""
            _, _, prior_mixture_params = model.hrm.split_coords(hrm_p)
            probs = model.get_cluster_probs(prior_mixture_params)
            probs_safe = jnp.clip(probs, 1e-10, 1.0)
            return jnp.sum(probs_safe * jnp.log(probs_safe))  # Negative entropy

        @jax.jit
        def m_step_iteration(
            carry: tuple[Any, Any, Any], _: None
        ) -> tuple[tuple[Any, Any, Any], None]:
            hrm_p, opt_st, step_key = carry
            step_key, grad_key = jax.random.split(step_key)

            # Compute ELBO gradient: positive - negative
            elbo_grad = compute_em_elbo_gradient(
                model, hrm_p, positive_stats, grad_key, n_negative_samples
            )

            # Compute entropy regularization gradient
            entropy_grad = jax.grad(entropy_loss)(hrm_p)

            # Total gradient: -elbo_grad (we minimize) + entropy_weight * entropy_grad
            # Note: elbo_grad points in direction of increasing ELBO
            # We want to maximize ELBO, so we use -(-elbo_grad) = elbo_grad for updates
            # But optax minimizes, so we negate: total_grad = -elbo_grad
            total_grad = -elbo_grad + entropy_weight * entropy_grad

            # Apply update
            updates, new_opt_st = optimizer.update(total_grad, opt_st, hrm_p)
            new_hrm_p = optax.apply_updates(hrm_p, updates)

            return (new_hrm_p, new_opt_st, step_key), None

        return m_step_iteration

    # Tracking
    elbos: list[float] = []
    r2s: list[float] = []
    rho_norms: list[float] = []

    print(f"Training for {n_em_iterations} EM iterations...")

    for em_iter in range(n_em_iterations):
        # === E-step ===
        # 1. Sample batch for this iteration
        key, batch_key, rho_key, elbo_key = jax.random.split(key, 4)
        batch_idx = jax.random.choice(batch_key, N_TRAIN, shape=(DEFAULT_BATCH_SIZE,))
        batch = train_data[batch_idx]

        # 2. Fit rho analytically
        rho_star, _, _ = compute_analytical_rho(
            model, hrm_params, rho_key, n_samples=n_analytical_samples
        )
        params_with_rho = model.join_coords(rho_star, hrm_params)

        # 3. Compute positive phase stats (held fixed for M-step)
        positive_stats = compute_positive_phase_stats(model, params_with_rho, batch)

        # === M-step ===
        # Run many gradient steps with fixed positive_stats
        m_step_fn = make_m_step_fn(positive_stats)

        key, m_key = jax.random.split(key)
        carry = (hrm_params, opt_state, m_key)

        # Use lax.scan for efficient iteration
        (hrm_params, opt_state, _), _ = jax.lax.scan(
            m_step_fn, carry, None, length=n_m_steps
        )

        # === Logging ===
        # Recompute rho with updated hrm_params for evaluation
        key, final_rho_key = jax.random.split(key)
        final_rho, final_r2, _ = compute_analytical_rho(
            model, hrm_params, final_rho_key, n_samples=n_analytical_samples
        )
        final_params = model.join_coords(final_rho, hrm_params)

        # Compute ELBO for monitoring
        elbo = model.mean_elbo(elbo_key, final_params, batch, DEFAULT_N_MC_SAMPLES)

        elbos.append(float(elbo))
        r2s.append(float(final_r2))
        rho_norms.append(float(jnp.linalg.norm(final_rho)))

        # Log every iteration (EM iterations are expensive)
        assignments = get_assignments_batch(final_params, test_data)
        nmi = compute_nmi(assignments, test_labels)
        _, _, prior_mix_params = model.hrm.split_coords(hrm_params)
        prior_probs = model.get_cluster_probs(prior_mix_params)
        prior_ent = float(
            -jnp.sum(prior_probs * jnp.log(jnp.clip(prior_probs, 1e-10, 1.0)))
        )
        max_ent = float(jnp.log(model.n_clusters))

        print(
            f"  EM iter {em_iter}: ELBO={elbo:.2f}, NMI={nmi:.4f}, R²={final_r2:.4f}, ||rho||={rho_norms[-1]:.2f}, H_prior={prior_ent:.2f}/{max_ent:.2f}"
        )

    # Final params
    key, final_rho_key = jax.random.split(key)
    final_rho, _, _ = compute_analytical_rho(
        model, hrm_params, final_rho_key, n_samples=n_analytical_samples
    )
    final_params = model.join_coords(final_rho, hrm_params)

    print(f"Final ELBO: {elbos[-1]:.4f}")

    # Evaluation
    print("Evaluating on test set...")

    def get_assignment(x: Array) -> Array:
        q_params = model.approximate_posterior_at(final_params, x)
        probs = model.get_cluster_probs(q_params)
        return jnp.argmax(probs)

    assignments = jax.vmap(get_assignment)(test_data)

    from .evaluate import (
        compute_cluster_accuracy,
        compute_cluster_purity,
        compute_nmi,
    )

    purity = compute_cluster_purity(assignments, test_labels, model.n_clusters)
    nmi = compute_nmi(assignments, test_labels)
    accuracy = compute_cluster_accuracy(assignments, test_labels, model.n_clusters)

    # Compute reconstruction error
    if observable_type == "poisson":
        from goal.models.graphical.hierarchical import (
            PoissonBernoulliMixture,
        )

        assert isinstance(model, PoissonBernoulliMixture)
        recon_error = float(
            model.normalized_reconstruction_error(
                final_params, test_data, scale=n_trials
            )
        )
    else:
        recon_error = float(
            model.normalized_reconstruction_error(final_params, test_data)
        )

    cluster_counts = [int(jnp.sum(assignments == k)) for k in range(model.n_clusters)]

    print(f"  Purity: {100 * purity:.1f}%")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Accuracy: {100 * accuracy:.1f}%")
    print(f"  Reconstruction error: {recon_error:.4f}")
    print(f"  Clusters: {cluster_counts}")
    print(f"  Final R²: {r2s[-1]:.4f}")

    # Generate samples
    key, sample_key = jax.random.split(key)
    x_samples, _, _ = model.sample_from_model(sample_key, final_params, n=20)
    x_samples = jnp.clip(x_samples, 0, n_trials)

    # Mean-based samples
    key, mean_key = jax.random.split(key)
    _, _, prior_mixture_params = model.hrm.split_coords(
        model.generative_params(final_params)
    )
    y_mean_samples, _ = model.sample_from_posterior(mean_key, prior_mixture_params, 20)

    def get_mean_image(y: Array) -> Array:
        lkl_params = model.likelihood_at(final_params, y)
        return model.obs_man.to_mean(lkl_params)

    x_mean_samples = jax.vmap(get_mean_image)(y_mean_samples)
    x_mean_samples = jnp.clip(x_mean_samples, 0, n_trials)

    # Cluster prototypes
    cluster_prototypes: list[list[float]] = []
    for k in range(model.n_clusters):
        mask = assignments == k
        if jnp.sum(mask) > 0:
            prototype = jnp.mean(test_data[mask], axis=0)
        else:
            prototype = jnp.zeros(N_OBSERVABLE)
        cluster_prototypes.append(prototype.tolist())

    results = {
        "mode": "em",
        "training_elbos": elbos,
        "generated_samples_mean": x_mean_samples.tolist(),
        "conjugation_vars": [0.0] * len(elbos),  # Not computed in EM mode
        "conjugation_stds": [0.0] * len(elbos),
        "conjugation_r2s": r2s,
        "rho_norms": rho_norms,
        "cluster_purity": purity,
        "nmi": nmi,
        "cluster_accuracy": accuracy,
        "reconstruction_error": recon_error,
        "cluster_counts": cluster_counts,
        "cluster_assignments": assignments.tolist(),
        "generated_samples": x_samples.tolist(),
        "cluster_prototypes": cluster_prototypes,
    }

    return final_params, results


def main():
    args = parse_args()

    print("=" * 60)
    print("VARIATIONAL MNIST CONFIGURATION")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Observable type: {args.observable}")
    print(f"  Latent type: {args.latent}")
    print(f"  N latent: {args.n_latent}")
    print(f"  N clusters: {args.n_clusters}")
    print(f"  N steps: {args.n_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Prior entropy weight: {args.entropy_weight}")
    print(f"  Posterior entropy weight: {args.posterior_entropy_weight}")
    print(f"  Conj weight: {args.conj_weight}")
    print(f"  Conj mode: {args.conj_mode}")
    if args.mode == "analytical":
        print(f"  Analytical samples mult: {args.analytical_samples_mult}")
    if args.mode == "em":
        print(f"  N EM iterations: {args.n_em_iterations}")
        print(f"  N M-steps per E-step: {args.n_m_steps}")
        print(f"  N negative samples: {args.n_negative_samples}")
        print(f"  Analytical samples mult: {args.analytical_samples_mult}")
    print("=" * 60)

    initialize_jax("gpu")
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)

    # Load data
    train_data, _, test_data, test_labels = load_mnist(
        DEFAULT_N_TRIALS, observable_type=args.observable
    )

    # Create model
    print(
        f"\nCreating {args.observable} model: {N_OBSERVABLE} obs, {args.n_latent} latent, {args.n_clusters} clusters"
    )
    model = create_model(
        n_latent=args.n_latent,
        n_clusters=args.n_clusters,
        observable_type=args.observable,
        latent_type=args.latent,
        n_trials=DEFAULT_N_TRIALS,
    )
    print(f"Model dim: {model.dim}")
    print(f"  Rho params: {model.rho_man.dim}")
    print(f"  Harmonium params: {model.hrm.dim}")

    # Train models
    all_results: dict[str, Any] = {}
    all_params: dict[str, Array] = {}

    if args.mode == "em":
        # EM mode uses a separate training function
        key, train_key = jax.random.split(key)
        params, results = train_model_em(
            train_key,
            model,
            train_data,
            test_data,
            test_labels,
            n_em_iterations=args.n_em_iterations,
            n_m_steps=args.n_m_steps,
            n_negative_samples=args.n_negative_samples,
            learning_rate=args.learning_rate,
            entropy_weight=args.entropy_weight,
            observable_type=args.observable,
            n_trials=DEFAULT_N_TRIALS,
            analytical_samples_mult=args.analytical_samples_mult,
        )
        all_results["em"] = results
        all_params["em"] = params
        modes = ["em"]
    else:
        # Standard training modes
        train_kwargs: dict[str, Any] = {
            "n_steps": args.n_steps,
            "learning_rate": args.learning_rate,
            "entropy_weight": args.entropy_weight,
            "conj_weight": args.conj_weight,
            "conj_mode": args.conj_mode,
            "observable_type": args.observable,
            "n_trials": DEFAULT_N_TRIALS,
            "analytical_samples_mult": args.analytical_samples_mult,
            "posterior_entropy_weight": args.posterior_entropy_weight,
        }

        # Determine which modes to train
        modes = ["free", "fixed", "regularized"] if args.mode == "all" else [args.mode]

        for mode in modes:
            key, train_key = jax.random.split(key)
            params, results = train_model(
                train_key,
                model,
                train_data,
                test_data,
                test_labels,
                mode=mode,
                **train_kwargs,
            )
            all_results[mode] = results
            all_params[mode] = params

    # Get reconstructions from best model by NMI
    best_mode = max(all_results.keys(), key=lambda m: all_results[m]["nmi"])
    best_params = all_params[best_mode]
    print(f"\nBest model by NMI: {best_mode}")

    def get_recon(x: Array) -> Array:
        return model.reconstruct(best_params, x)

    reconstructions = jax.vmap(get_recon)(test_data[:20])

    # Clip reconstructions for visualization
    reconstructions = jnp.clip(reconstructions, 0, DEFAULT_N_TRIALS)

    # Save results
    output = {
        "models": all_results,
        "true_labels": test_labels.tolist(),
        "original_images": test_data[:20].tolist(),
        "reconstructed_images": reconstructions.tolist(),
        "best_model": best_mode,
        "config": {
            "observable_type": args.observable,
            "latent_type": args.latent,
            "n_latent": args.n_latent,
            "n_clusters": args.n_clusters,
            "n_trials": DEFAULT_N_TRIALS,
            "n_steps": args.n_steps,
            "learning_rate": args.learning_rate,
            "prior_entropy_weight": args.entropy_weight,
            "posterior_entropy_weight": args.posterior_entropy_weight,
            "conj_weight": args.conj_weight,
            "conj_mode": args.conj_mode,
            "analytical_samples_mult": args.analytical_samples_mult,
            "n_em_iterations": args.n_em_iterations,
            "n_m_steps": args.n_m_steps,
            "n_negative_samples": args.n_negative_samples,
        },
    }

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.analysis_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {paths.analysis_path}")

    # Save parameters for later diagnosis
    params_path = paths.results_dir / "params.npz"
    params_arrays: dict[str, np.ndarray[Any, Any]] = {
        mode: np.array(params) for mode, params in all_params.items()
    }
    np.savez(str(params_path), **params_arrays)  # pyright: ignore[reportArgumentType]
    print(f"Parameters saved to {params_path}")

    # Print summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)

    if len(modes) > 1:
        header = f"{'Metric':<22}"
        for mode in modes:
            header += f" {mode:>14}"
        print(header)
        print("-" * 80)

        metrics = [
            ("Cluster purity", "cluster_purity", lambda x: f"{100 * x:>13.1f}%"),
            ("Accuracy", "cluster_accuracy", lambda x: f"{100 * x:>13.1f}%"),
            ("NMI", "nmi", lambda x: f"{x:>14.4f}"),
            ("Reconstruction error", "reconstruction_error", lambda x: f"{x:>14.4f}"),
            ("Final Var[f_tilde]", "conjugation_vars", lambda x: f"{x[-1]:>14.2f}"),
            ("Final Std[f_tilde]", "conjugation_stds", lambda x: f"{x[-1]:>14.2f}"),
            ("Final R^2", "conjugation_r2s", lambda x: f"{x[-1]:>14.4f}"),
            ("Final ||rho||", "rho_norms", lambda x: f"{x[-1]:>14.4f}"),
        ]

        for name, key_name, fmt in metrics:
            row = f"{name:<22}"
            for mode in modes:
                val = all_results[mode][key_name]
                row += fmt(val)
            print(row)
    else:
        mode = modes[0]
        r = all_results[mode]
        print(f"{'Metric':<22} {'Value':>14}")
        print("-" * 40)
        print(f"{'Cluster purity':<22} {100 * r['cluster_purity']:>13.1f}%")
        print(f"{'Accuracy':<22} {100 * r['cluster_accuracy']:>13.1f}%")
        print(f"{'NMI':<22} {r['nmi']:>14.4f}")
        print(f"{'Reconstruction error':<22} {r['reconstruction_error']:>14.4f}")
        print(f"{'Final Var[f_tilde]':<22} {r['conjugation_vars'][-1]:>14.2f}")
        print(f"{'Final Std[f_tilde]':<22} {r['conjugation_stds'][-1]:>14.2f}")
        print(f"{'Final R^2':<22} {r['conjugation_r2s'][-1]:>14.4f}")
        print(f"{'Final ||rho||':<22} {r['rho_norms'][-1]:>14.4f}")


if __name__ == "__main__":
    main()
