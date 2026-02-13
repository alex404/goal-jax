"""Training script for Variational MNIST clustering.

Training modes:
1. gradient: Gradient descent on all parameters including rho.
2. analytical: Analytical rho via least squares each step.

Both modes support --conj-weight to regularize the generative model
parameters with the conjugation variance (Var[f_tilde]).
Set --conj-weight 0 for no regularization (default), or positive for regularized.

Supports two observable types:
- binomial: Discretized pixels in [0, n_trials] (default)
- poisson: Count data with unbounded support

Usage:
    python -m examples.variational_mnist.train
    python -m examples.variational_mnist.train --mode gradient --n-steps 10000
    python -m examples.variational_mnist.train --mode gradient --conj-weight 0.1
    python -m examples.variational_mnist.train --mode analytical --conj-weight 0.1
    python -m examples.variational_mnist.train --observable poisson
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false

import argparse
import json
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from sklearn.datasets import fetch_openml

from ..shared import example_paths
from .model import (
    ANALYTICAL_RHO_SAMPLES_MULTIPLIER,
    N_OBSERVABLE,
    MixtureModel,
    create_model,
    normalized_reconstruction_error,
    reconstruct,
)

# Data configuration
N_TRAIN = 60000
N_TEST = 10000

# Default hyperparameters
DEFAULT_N_LATENT = 1024
DEFAULT_N_CLUSTERS = 10
DEFAULT_N_TRIALS = 16
DEFAULT_N_STEPS = 5000
DEFAULT_BATCH_SIZE = 512
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_N_MC_SAMPLES = 10
DEFAULT_ENTROPY_WEIGHT = 1.0  # Prevents cluster collapse
DEFAULT_CONJ_WEIGHT = 0.0  # Weight for conjugation penalty (0 = free rho)

# Logging
LOG_INTERVAL = 200
N_CONJ_SAMPLES = 10000  # For diagnostic metrics (logged every LOG_INTERVAL)
N_CONJ_PENALTY_SAMPLES = 1024  # For conjugation penalty in loss (every step)

# Schedule defaults
DEFAULT_KL_WARMUP_STEPS = 1000
DEFAULT_LR_WARMUP_STEPS = 500


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Variational MNIST model")
    parser.add_argument(
        "--mode",
        choices=["gradient", "analytical"],
        default="gradient",
        help="Training mode: 'gradient' learns rho via GD, 'analytical' computes rho via lstsq",
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
        help="Weight for conjugation penalty (gradient mode; 0 = free rho)",
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
        "--interaction",
        choices=["hierarchical", "full"],
        default="full",
        help="Interaction mode: 'hierarchical' (X↔(Y,K)) or 'full' (X↔Y↔K three-block)",
    )
    parser.add_argument(
        "--analytical-samples-mult",
        type=int,
        default=ANALYTICAL_RHO_SAMPLES_MULTIPLIER,
        help=f"For analytical mode: n_samples = mult * lat_dim (default {ANALYTICAL_RHO_SAMPLES_MULTIPLIER}x for overdetermined lstsq)",
    )
    parser.add_argument(
        "--kl-warmup-steps",
        type=int,
        default=DEFAULT_KL_WARMUP_STEPS,
        help="Steps to linearly ramp KL weight (beta) from 0 to 1",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=DEFAULT_LR_WARMUP_STEPS,
        help="Steps for learning rate warmup before cosine decay",
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


def compute_prior_entropy(model: MixtureModel, params: Array) -> Array:
    """Compute entropy of the prior's marginal cluster distribution."""
    return model.prior_entropy(params)


def compute_conjugation_metrics_jit(
    model: Any, params: Array, key: Array, n_samples: int
) -> tuple[Array, Array, Array]:
    """JIT-compatible conjugation metrics computation.

    Returns:
        Tuple of (variance, std, r_squared)
    """
    return model.conjugation_metrics(key, params, n_samples)


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
    kl_warmup_steps: int = DEFAULT_KL_WARMUP_STEPS,
    lr_warmup_steps: int = DEFAULT_LR_WARMUP_STEPS,
) -> tuple[Array, dict[str, Any]]:
    """Train a model with specified mode.

    Args:
        key: Random key
        model: The mixture model
        train_data: Training data
        test_data: Test data
        test_labels: Test labels
        mode: Training mode ('gradient' or 'analytical')
        n_steps: Number of training steps
        learning_rate: Learning rate
        entropy_weight: Weight for batch entropy
        conj_weight: Weight for conjugation penalty (gradient mode; 0 = free rho)
        conj_mode: Conjugation penalty mode ('variance' or 'r2')
        observable_type: Type of observable distribution
        n_trials: Binomial discretization level
        analytical_samples_mult: For analytical mode, n_samples = mult * lat_dim

    Returns:
        Tuple of (final_params, results_dict)
    """
    use_analytical_rho = mode == "analytical"
    use_conj_penalty = conj_weight > 0

    n_analytical_samples = analytical_samples_mult * model.bas_lat_man.dim
    conj_str = (
        f", conj penalty weight={conj_weight} mode={conj_mode}"
        if use_conj_penalty
        else ""
    )
    if use_analytical_rho:
        mode_str = f"ANALYTICAL rho ({n_analytical_samples} samples, {analytical_samples_mult}x lat_dim={model.bas_lat_man.dim}{conj_str})"
    else:
        mode_str = f"GRADIENT DESCENT{conj_str}"

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

    mix_int_dim = model.bas_lat_man.dim * (model.n_categories - 1)
    k1, k2, k3 = jax.random.split(cluster_key, 3)
    new_mix_int_mat = jax.random.normal(k1, (mix_int_dim,)) * 0.5

    new_lat_params = model.mix_man.join_coords(
        mix_obs_bias, new_mix_int_mat, cat_params
    )

    # For full model, also break symmetry in the xyk and xk interaction blocks
    if hasattr(model.hrm, 'xyk_man'):
        block_int = model.hrm.int_man
        xy_params, xyk_params, xk_params = block_int.coord_blocks(int_params)
        obs_dim = model.obs_man.dim
        lat_dim = model.bas_lat_man.dim
        # Use same scale as xy but divided across components
        xyk_scale = 0.1 / jnp.sqrt(obs_dim * lat_dim)
        xk_scale = 0.1 / jnp.sqrt(obs_dim)
        new_xyk = jax.random.normal(k2, xyk_params.shape) * xyk_scale
        new_xk = jax.random.normal(k3, xk_params.shape) * xk_scale
        int_params = jnp.concatenate([xy_params, new_xyk, new_xk])

    new_hrm_params = model.hrm.join_coords(obs_bias, int_params, new_lat_params)
    params = model.join_coords(rho, new_hrm_params)

    # Optimizer setup with linear LR warmup then constant
    # In analytical mode, optimizer only updates hrm_params (rho is computed via lstsq)
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=learning_rate,
                transition_steps=lr_warmup_steps,
            ),
            optax.constant_schedule(learning_rate),
        ],
        boundaries=[lr_warmup_steps],
    )
    optimizer = optax.adam(lr_schedule)
    if use_analytical_rho:
        opt_state = optimizer.init(new_hrm_params)
    else:
        opt_state = optimizer.init(params)

    # Loss function for gradient mode (learnable rho, optional regularization)
    def loss_fn_gradient(
        params: Array, key: Array, batch: Array, beta: float
    ) -> tuple[Array, tuple[Array, Array]]:
        key, elbo_key, conj_key = jax.random.split(key, 3)
        elbo = model.mean_elbo(
            elbo_key, params, batch, DEFAULT_N_MC_SAMPLES, kl_weight=beta
        )

        prior_entropy = model.prior_entropy(params)

        # Posterior entropy
        def get_cluster_probs(x: Array) -> Array:
            q_params = model.approximate_posterior_at(params, x)
            return model.get_cluster_probs(q_params)

        all_probs = jax.vmap(get_cluster_probs)(batch)
        posterior_entropy = compute_batch_entropy(all_probs)

        # Conjugation penalty shares beta with KL — both are about prior structure
        # Var[f̃] measures how well the linear conjugation approximates the true
        # posterior, which is the same geometric quantity the KL gap depends on
        if use_conj_penalty:
            conj_var = model.conjugation_error(conj_key, params, N_CONJ_PENALTY_SAMPLES)
            conj_loss = beta * conj_weight * conj_var
        else:
            conj_var = jnp.array(0.0)
            conj_loss = 0.0

        # Entropy regularization at full strength from step 0 (prevents collapse)
        # Only KL (inside ELBO) and conjugation penalty use beta warmup
        loss = (
            -elbo
            - entropy_weight * prior_entropy
            - posterior_entropy_weight * posterior_entropy
            + conj_loss
        )
        return loss, (elbo, conj_var)

    # Loss function for analytical rho (computed via lstsq each step)
    # NOTE: Gradients flow through lstsq via JAX implicit differentiation
    # The optimizer only updates hrm_params; rho is always optimal given hrm_params

    def loss_fn_analytical(
        hrm_params: Array, key: Array, batch: Array, beta: float
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        key, rho_key, elbo_key, conj_key = jax.random.split(key, 4)

        # Compute analytical rho via regress_conjugation_parameters
        zero_rho = jnp.zeros(model.rho_man.dim)
        dummy_params = model.join_coords(zero_rho, hrm_params)
        rho_star, _, _ = model.regress_conjugation_parameters(
            rho_key, dummy_params, n_analytical_samples
        )

        # Build full params with analytical rho and compute ELBO
        params_with_rho = model.join_coords(rho_star, hrm_params)
        elbo = model.mean_elbo(
            elbo_key, params_with_rho, batch, DEFAULT_N_MC_SAMPLES, kl_weight=beta
        )

        # Prior entropy to prevent cluster collapse in generative model
        prior_entropy = model.prior_entropy(params_with_rho)

        # Posterior entropy to prevent cluster collapse in assignments
        def get_cluster_probs(x: Array) -> Array:
            q_params = model.approximate_posterior_at(params_with_rho, x)
            return model.get_cluster_probs(q_params)

        all_probs = jax.vmap(get_cluster_probs)(batch)
        posterior_entropy = compute_batch_entropy(all_probs)

        # Conjugation penalty shares beta with KL
        if use_conj_penalty:
            conj_var = model.conjugation_error(conj_key, params_with_rho, N_CONJ_PENALTY_SAMPLES)
            conj_loss = beta * conj_weight * conj_var
        else:
            conj_var = jnp.array(0.0)
            conj_loss = 0.0

        # Entropy regularization at full strength from step 0 (prevents collapse)
        loss = (
            -elbo
            - entropy_weight * prior_entropy
            - posterior_entropy_weight * posterior_entropy
            + conj_loss
        )
        return loss, (elbo, conj_var, rho_star)

    # Training step functions
    # beta is passed as a traced Array to avoid JIT recompilation per step
    if use_analytical_rho:

        @jax.jit
        def train_step(
            carry: Any, _: None, beta: Array = jnp.array(1.0)
        ) -> tuple[Any, tuple[Array, Array, Array]]:
            hrm_params, opt_state, step_key, data = carry
            step_key, next_key, batch_key = jax.random.split(step_key, 3)

            batch_idx = jax.random.choice(
                batch_key, N_TRAIN, shape=(DEFAULT_BATCH_SIZE,)
            )
            batch = data[batch_idx]

            (_, metrics), grads = jax.value_and_grad(loss_fn_analytical, has_aux=True)(
                hrm_params, next_key, batch, beta
            )

            updates, new_opt_state = optimizer.update(grads, opt_state, hrm_params)
            new_hrm_params = optax.apply_updates(hrm_params, updates)

            return (new_hrm_params, new_opt_state, step_key, data), metrics

    else:

        @jax.jit
        def train_step(
            carry: Any, _: None, beta: Array = jnp.array(1.0)
        ) -> tuple[Any, tuple[Array, Array]]:
            params, opt_state, step_key, data = carry
            step_key, next_key, batch_key = jax.random.split(step_key, 3)

            batch_idx = jax.random.choice(
                batch_key, N_TRAIN, shape=(DEFAULT_BATCH_SIZE,)
            )
            batch = data[batch_idx]

            (_, metrics), grads = jax.value_and_grad(loss_fn_gradient, has_aux=True)(
                params, next_key, batch, beta
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

    # JIT-compile diagnostic metrics (called only at LOG_INTERVAL)
    @jax.jit
    def compute_diagnostics(params: Array, key: Array) -> tuple[Array, Array, Array]:
        return model.conjugation_metrics(key, params, N_CONJ_SAMPLES)

    for step in range(n_steps):
        # Beta warmup: ramps KL weight and conj penalty together from 0 to 1.
        # Both are about prior structure — KL penalizes q deviating from the prior,
        # Var[f̃] penalizes the conjugation gap that makes q ≠ p(z|x).
        beta = jnp.array(min(1.0, step / max(kl_warmup_steps, 1)))

        if use_analytical_rho:
            (current_hrm_params, current_opt_state, train_key, _), metrics = train_step(
                (current_hrm_params, current_opt_state, train_key, train_data),
                None,
                beta=beta,
            )
            elbo, conj_var, rho_star = metrics
            current_params = model.join_coords(rho_star, current_hrm_params)
        else:
            (current_params, current_opt_state, train_key, _), metrics = train_step(
                (current_params, current_opt_state, train_key, train_data),
                None,
                beta=beta,
            )
            elbo, conj_var = metrics

        elbos.append(float(elbo))
        conj_vars.append(float(conj_var))

        rho_current, _ = model.split_coords(current_params)
        rho_norm = float(jnp.linalg.norm(rho_current))
        rho_norms.append(rho_norm)

        if step % LOG_INTERVAL == 0:
            # Compute full diagnostic metrics (expensive, only at log intervals)
            train_key, diag_key = jax.random.split(train_key)
            _, conj_std, conj_r2 = compute_diagnostics(current_params, diag_key)
            conj_stds.append(float(conj_std))
            conj_r2s.append(float(conj_r2))

            assignments = get_assignments_batch(current_params, test_data)
            nmi = compute_nmi(assignments, test_labels)
            prior_ent = float(model.prior_entropy(current_params))
            max_ent = float(jnp.log(model.n_categories))

            print(
                f"  Step {step}: ELBO={elbo:.2f}, NMI={nmi:.4f}, R^2={conj_r2:.4f}, beta={float(beta):.3f}, ||rho||={rho_norm:.2f}, H_prior={prior_ent:.2f}/{max_ent:.2f}"
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

    purity = compute_cluster_purity(assignments, test_labels, model.n_categories)
    nmi = compute_nmi(assignments, test_labels)
    accuracy = compute_cluster_accuracy(assignments, test_labels, model.n_categories)

    # Compute reconstruction error with appropriate normalization
    if observable_type == "poisson":
        recon_error = float(
            normalized_reconstruction_error(
                model, final_params, test_data, scale=n_trials
            )
        )
    else:
        recon_error = float(
            normalized_reconstruction_error(model, final_params, test_data)
        )

    cluster_counts = [int(jnp.sum(assignments == k)) for k in range(model.n_categories)]

    print(f"  Purity: {100 * purity:.1f}%")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Accuracy: {100 * accuracy:.1f}%")
    print(f"  Reconstruction error: {recon_error:.4f}")
    print(f"  Clusters: {cluster_counts}")
    if conj_r2s:
        print(f"  Final R^2: {conj_r2s[-1]:.4f}")

    # Generate samples (stochastic)
    key, sample_key = jax.random.split(key)
    xz_samples = model.sample(sample_key, final_params, n=20)
    x_samples = xz_samples[:, :model.obs_man.dim]

    # Clip samples for visualization (especially important for Poisson)
    x_samples = jnp.clip(x_samples, 0, n_trials)

    # Also generate mean-based samples (expected values, not sampled)
    key, mean_key = jax.random.split(key)
    _, _, prior_mixture_params = model.hrm.split_coords(
        model.generative_params(final_params)
    )
    z_mean_samples = model.mix_man.sample(mean_key, prior_mixture_params, 20)

    def get_mean_image(z: Array) -> Array:
        lkl_params = model.likelihood_at(final_params, z)
        return model.obs_man.to_mean(lkl_params)

    x_mean_samples = jax.vmap(get_mean_image)(z_mean_samples)

    # Clip mean samples for visualization (especially important for Poisson)
    x_mean_samples = jnp.clip(x_mean_samples, 0, n_trials)

    # Cluster prototypes
    cluster_prototypes: list[list[float]] = []
    for k in range(model.n_categories):
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


def main():
    args = parse_args()

    print("=" * 60)
    print("VARIATIONAL MNIST CONFIGURATION")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Observable type: {args.observable}")
    print(f"  N latent: {args.n_latent}")
    print(f"  N clusters: {args.n_clusters}")
    print(f"  N steps: {args.n_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Prior entropy weight: {args.entropy_weight}")
    print(f"  Posterior entropy weight: {args.posterior_entropy_weight}")
    print(f"  Conj weight: {args.conj_weight}")
    print(f"  Conj mode: {args.conj_mode}")
    print(f"  Interaction: {args.interaction}")
    print(f"  KL warmup steps: {args.kl_warmup_steps}")
    print(f"  LR warmup steps: {args.lr_warmup_steps}")
    if args.mode == "analytical":
        print(f"  Analytical samples mult: {args.analytical_samples_mult}")
    print("=" * 60)

    jax.config.update("jax_platform_name", "gpu")
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
        n_trials=DEFAULT_N_TRIALS,
        interaction=args.interaction,
    )
    print(f"Model dim: {model.dim}")
    print(f"  Rho params: {model.rho_man.dim}")
    print(f"  Harmonium params: {model.hrm.dim}")

    # Train model
    key, train_key = jax.random.split(key)
    params, results = train_model(
        train_key,
        model,
        train_data,
        test_data,
        test_labels,
        mode=args.mode,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
        conj_weight=args.conj_weight,
        conj_mode=args.conj_mode,
        observable_type=args.observable,
        n_trials=DEFAULT_N_TRIALS,
        analytical_samples_mult=args.analytical_samples_mult,
        posterior_entropy_weight=args.posterior_entropy_weight,
        kl_warmup_steps=args.kl_warmup_steps,
        lr_warmup_steps=args.lr_warmup_steps,
    )

    # Reconstructions
    def get_recon(x: Array) -> Array:
        return reconstruct(model, params, x)

    reconstructions = jax.vmap(get_recon)(test_data[:20])
    reconstructions = jnp.clip(reconstructions, 0, DEFAULT_N_TRIALS)

    # Save results
    output = {
        "models": {args.mode: results},
        "true_labels": test_labels.tolist(),
        "original_images": test_data[:20].tolist(),
        "reconstructed_images": reconstructions.tolist(),
        "best_model": args.mode,
        "config": {
            "observable_type": args.observable,
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
            "interaction": args.interaction,
            "kl_warmup_steps": args.kl_warmup_steps,
            "lr_warmup_steps": args.lr_warmup_steps,
        },
    }

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.analysis_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {paths.analysis_path}")

    # Save parameters for later diagnosis
    params_path = paths.results_dir / "params.npz"
    np.savez(str(params_path), **{args.mode: np.array(params)})  # pyright: ignore[reportArgumentType]
    print(f"Parameters saved to {params_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<22} {'Value':>14}")
    print("-" * 40)
    print(f"{'Cluster purity':<22} {100 * results['cluster_purity']:>13.1f}%")
    print(f"{'Accuracy':<22} {100 * results['cluster_accuracy']:>13.1f}%")
    print(f"{'NMI':<22} {results['nmi']:>14.4f}")
    print(f"{'Reconstruction error':<22} {results['reconstruction_error']:>14.4f}")
    print(f"{'Final Var[f_tilde]':<22} {results['conjugation_vars'][-1]:>14.2f}")
    if results['conjugation_stds']:
        print(f"{'Final Std[f_tilde]':<22} {results['conjugation_stds'][-1]:>14.2f}")
    if results['conjugation_r2s']:
        print(f"{'Final R^2':<22} {results['conjugation_r2s'][-1]:>14.4f}")
    print(f"{'Final ||rho||':<22} {results['rho_norms'][-1]:>14.4f}")


if __name__ == "__main__":
    main()
