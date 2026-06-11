"""Training script for Variational MNIST clustering.

Training modes:
1. gradient: Gradient descent on all parameters including rho.
2. analytical: Analytical rho via least squares each step.

Both modes support --conj-weight to regularize the generative model
parameters with the conjugation variance (Var[r]).
Set --conj-weight 0 for no regularization (default), or positive for regularized.

Supports three observable types:
- binomial: Discretized pixels in [0, n_trials] (default)
- poisson: Count data with unbounded support
- normal: DiagonalNormal continuous observable (pixels rescaled to [0,1])

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
import time
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from sklearn.datasets import fetch_openml

from goal.geometry.exponential_family.variational import (
    conjugation_metrics,
    regress_conjugation_parameters,
)

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
DEFAULT_CONJ_WARMUP_STEPS = 2000


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
        help="Conjugation penalty: 'variance' uses Var[r], 'r2' uses (1-R^2)",
    )
    parser.add_argument(
        "--observable",
        choices=["binomial", "poisson", "normal"],
        default="binomial",
        help=(
            "Observable distribution type: 'binomial' (bounded counts), "
            "'poisson' (unbounded counts), or 'normal' (DiagonalNormal "
            "continuous observable; pixels rescaled to [0,1] floats; "
            "--n-trials is ignored)"
        ),
    )
    parser.add_argument(
        "--interaction",
        choices=["hierarchical", "full"],
        default="full",
        help="Interaction mode: 'hierarchical' (X<->(Y,K)) or 'full' (X<->Y<->K three-block)",
    )
    parser.add_argument(
        "--normal-precision-floor",
        type=float,
        default=1e-3,
        help=(
            "Per-coordinate floor on the natural-parameter precision of the "
            "DiagonalNormal observable. Caps per-pixel variance at "
            "1/<floor> and prevents the log_partition NaN that the affine "
            "likelihood b+L(s(z)) can trigger when an interaction-driven "
            "precision crosses zero for some posterior-sampled z. Only used "
            "with --observable normal."
        ),
    )
    parser.add_argument(
        "--latent",
        choices=["bernoullis", "chordal_boltzmann"],
        default="bernoullis",
        help="Base latent kind: 'bernoullis' (independent units) or 'chordal_boltzmann' (chordal pairwise couplings, exact JT inference)",
    )
    parser.add_argument(
        "--latent-graph",
        choices=["chain"],
        default="chain",
        help="Sparsity pattern for chordal_boltzmann latent (only 'chain' is wired up)",
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
    parser.add_argument(
        "--conj-warmup-steps",
        type=int,
        default=DEFAULT_CONJ_WARMUP_STEPS,
        help="Steps to linearly ramp the conjugation regularizer from 0 to conj_weight",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        default="adam",
        help=(
            "Outer optimizer. ``adam`` is the historical default. ``adamw`` "
            "adds decoupled weight decay (set ``--weight-decay`` to control "
            "the strength) which acts as L2 regularization on the natural "
            "parameters."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help=(
            "Decoupled weight-decay coefficient for AdamW. Ignored when "
            "``--optimizer adam``. Typical values 1e-5 to 1e-3."
        ),
    )
    return parser.parse_args()


def load_mnist_sklearn() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load MNIST using scikit-learn's fetch_openml."""
    # Store data in project .data/ directory to avoid cluttering home directory
    data_home = str(Path(__file__).parents[2] / ".data")
    try:
        mnist = fetch_openml(
            "mnist_784", version=1, as_frame=False, parser="auto", data_home=data_home
        )
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required. Install with: pip install scikit-learn"
        ) from e
    else:
        return mnist.data, mnist.target


def load_mnist(
    n_trials: int = DEFAULT_N_TRIALS,
    observable_type: Literal["binomial", "poisson", "normal"] = "binomial",
) -> tuple[Array, Array, Array, Array]:
    """Load MNIST and split into train/test with labels.

    Args:
        n_trials: Discretization level for the count-style observables
            (``binomial``, ``poisson``); ignored for ``normal``.
        observable_type: Type of observable distribution.

    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    if observable_type == "normal":
        print("Loading MNIST data (normal, rescaled to [0, 1] floats)...")
    else:
        print(f"Loading MNIST data ({observable_type}, scaled to [0, {n_trials}])...")
    data, labels = load_mnist_sklearn()

    labels = labels.astype(np.int32)
    if observable_type == "normal":
        data = (data / 255.0).astype(np.float32)
    else:
        # Scale to [0, n_trials] for count-style observables.
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
    return conjugation_metrics(model, key, params, n_samples)


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
    observable_type: Literal["binomial", "poisson", "normal"] = "binomial",
    n_trials: int = DEFAULT_N_TRIALS,
    analytical_samples_mult: int = ANALYTICAL_RHO_SAMPLES_MULTIPLIER,
    posterior_entropy_weight: float = DEFAULT_ENTROPY_WEIGHT,
    kl_warmup_steps: int = DEFAULT_KL_WARMUP_STEPS,
    lr_warmup_steps: int = DEFAULT_LR_WARMUP_STEPS,
    conj_warmup_steps: int = DEFAULT_CONJ_WARMUP_STEPS,
    optimizer_name: Literal["adam", "adamw"] = "adam",
    weight_decay: float = 0.0,
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
    lat_params, lkl_params, rho = model.split_coords(params)
    obs_bias, int_params = model.gen_hrm.lkl_fun_man.split_coords(lkl_params)
    mix_obs_bias, _, cat_params = model.mix_man.split_coords(lat_params)

    mix_int_dim = model.bas_lat_man.dim * (model.n_categories - 1)
    k1, k2, k3 = jax.random.split(cluster_key, 3)
    new_mix_int_mat = jax.random.normal(k1, (mix_int_dim,)) * 0.5

    new_lat_params = model.mix_man.join_coords(
        mix_obs_bias, new_mix_int_mat, cat_params
    )

    # For full model, also break symmetry in the xyk and xk interaction blocks
    if hasattr(model.gen_hrm, 'xyk_man'):
        block_int = model.gen_hrm.int_man
        xy_params, xyk_params, xk_params = block_int.coord_blocks(int_params)
        obs_dim = model.obs_man.dim
        lat_dim = model.bas_lat_man.dim
        # Use same scale as xy but divided across components
        xyk_scale = 0.1 / jnp.sqrt(obs_dim * lat_dim)
        xk_scale = 0.1 / jnp.sqrt(obs_dim)
        new_xyk = jax.random.normal(k2, xyk_params.shape) * xyk_scale
        new_xk = jax.random.normal(k3, xk_params.shape) * xk_scale
        int_params = jnp.concatenate([xy_params, new_xyk, new_xk])

    new_lkl_params = model.gen_hrm.lkl_fun_man.join_coords(obs_bias, int_params)
    params = model.join_coords(new_lat_params, new_lkl_params, rho)

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
    if optimizer_name == "adamw":
        optimizer = optax.adamw(lr_schedule, weight_decay=weight_decay)
    else:
        optimizer = optax.adam(lr_schedule)
    prior_dim = model.prr_man.dim
    init_prior, init_lkl, _ = model.split_coords(params)
    gen_params = jnp.concatenate([init_prior, init_lkl])
    if use_analytical_rho:
        opt_state = optimizer.init(gen_params)
    else:
        opt_state = optimizer.init(params)

    # Loss function for gradient mode (learnable rho, optional regularization)
    def loss_fn_gradient(
        params: Array, key: Array, batch: Array, beta: float, conj_beta: float
    ) -> tuple[Array, tuple[Array, Array]]:
        key, elbo_key, conj_key = jax.random.split(key, 3)
        # beta-warmup applied externally as L_beta = L_1 + (1-beta)*KL
        elbo_1 = model.mean_elbo(elbo_key, params, batch, DEFAULT_N_MC_SAMPLES)
        mean_kl = jnp.mean(
            jax.vmap(lambda x: model.elbo_divergence(params, x))(batch)
        )
        elbo = elbo_1 + (1.0 - beta) * mean_kl

        prior_entropy = model.prior_entropy(params)

        # Posterior entropy
        def get_cluster_probs(x: Array) -> Array:
            q_params = model.approximate_posterior_at(params, x)
            return model.get_cluster_probs(q_params)

        all_probs = jax.vmap(get_cluster_probs)(batch)
        posterior_entropy = compute_batch_entropy(all_probs)

        # Conjugation penalty: independent ramp via conj_beta so we can train
        # at standard ELBO (beta=1) while gradually introducing the conjugation
        # regularizer over its own (typically slower) schedule.
        if use_conj_penalty:
            conj_var, _, _ = conjugation_metrics(
                model, conj_key, params, N_CONJ_PENALTY_SAMPLES
            )
            conj_loss = conj_beta * conj_weight * conj_var
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
        # Report the true (beta=1) ELBO; the beta-scaled value lives only in the loss.
        return loss, (elbo_1, conj_var)

    # Loss function for analytical rho (computed via lstsq each step)
    # NOTE: Gradients flow through lstsq via JAX implicit differentiation
    # The optimizer only updates hrm_params; rho is always optimal given hrm_params

    def loss_fn_analytical(
        gen_params: Array, key: Array, batch: Array, beta: float, conj_beta: float
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        key, rho_key, elbo_key = jax.random.split(key, 3)

        prior_p = gen_params[:prior_dim]
        lkl_p = gen_params[prior_dim:]

        # Compute analytical rho via regress_conjugation_parameters
        zero_rho = jnp.zeros(model.cnj_man.dim)
        dummy_params = model.join_coords(prior_p, lkl_p, zero_rho)
        rho_star, _, _, regression_var = regress_conjugation_parameters(
            model, rho_key, dummy_params, n_analytical_samples
        )

        # Build full params with analytical rho and compute ELBO
        params_with_rho = model.join_coords(prior_p, lkl_p, rho_star)
        # beta-warmup applied externally as L_beta = L_1 + (1-beta)*KL
        elbo_1 = model.mean_elbo(
            elbo_key, params_with_rho, batch, DEFAULT_N_MC_SAMPLES
        )
        mean_kl = jnp.mean(
            jax.vmap(lambda x: model.elbo_divergence(params_with_rho, x))(batch)
        )
        elbo = elbo_1 + (1.0 - beta) * mean_kl

        # Prior entropy to prevent cluster collapse in generative model
        prior_entropy = model.prior_entropy(params_with_rho)

        # Posterior entropy to prevent cluster collapse in assignments
        def get_cluster_probs(x: Array) -> Array:
            q_params = model.approximate_posterior_at(params_with_rho, x)
            return model.get_cluster_probs(q_params)

        all_probs = jax.vmap(get_cluster_probs)(batch)
        posterior_entropy = compute_batch_entropy(all_probs)

        # Conjugation penalty on independent ramp via conj_beta — use residual
        # variance from the regression rather than sampling the prior again.
        if use_conj_penalty:
            conj_var = regression_var
            conj_loss = conj_beta * conj_weight * conj_var
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
        # Report the true (beta=1) ELBO; the beta-scaled value lives only in the loss.
        return loss, (elbo_1, conj_var, rho_star)

    # Training step functions
    # beta is passed as a traced Array to avoid JIT recompilation per step
    if use_analytical_rho:

        @jax.jit
        def train_step(
            carry: Any,
            _: None,
            beta: Array | None = None,
            conj_beta: Array | None = None,
        ) -> tuple[Any, tuple[Array, Array, Array]]:
            if beta is None:
                beta = jnp.array(1.0)
            if conj_beta is None:
                conj_beta = jnp.array(1.0)
            gen_params, opt_state, step_key, data = carry
            step_key, next_key, batch_key = jax.random.split(step_key, 3)

            batch_idx = jax.random.choice(
                batch_key, N_TRAIN, shape=(DEFAULT_BATCH_SIZE,)
            )
            batch = data[batch_idx]

            (_, metrics), grads = jax.value_and_grad(loss_fn_analytical, has_aux=True)(
                gen_params, next_key, batch, beta, conj_beta
            )

            updates, new_opt_state = optimizer.update(grads, opt_state, gen_params)
            new_gen_params = optax.apply_updates(gen_params, updates)

            return (new_gen_params, new_opt_state, step_key, data), metrics

    else:

        @jax.jit
        def train_step(
            carry: Any,
            _: None,
            beta: Array | None = None,
            conj_beta: Array | None = None,
        ) -> tuple[Any, tuple[Array, Array]]:
            if beta is None:
                beta = jnp.array(1.0)
            if conj_beta is None:
                conj_beta = jnp.array(1.0)
            params, opt_state, step_key, data = carry
            step_key, next_key, batch_key = jax.random.split(step_key, 3)

            batch_idx = jax.random.choice(
                batch_key, N_TRAIN, shape=(DEFAULT_BATCH_SIZE,)
            )
            batch = data[batch_idx]

            (_, metrics), grads = jax.value_and_grad(loss_fn_gradient, has_aux=True)(
                params, next_key, batch, beta, conj_beta
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
    # Per-step wallclock in seconds; step 0 includes JIT compile cost,
    # subsequent steps measure the warm step cost.
    step_times: list[float] = []

    key, train_key = jax.random.split(key)
    current_params = params
    current_gen_params = gen_params
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

        # Chunked like normalized_reconstruction_error: a full-test-set vmap
        # materializes the latent log-partition workspace per lane and OOMs
        # with the ChainBoltzmann associative-scan kernel at n_latent=1024.
        return jax.lax.map(get_assignment, xs, batch_size=256)

    # JIT-compile diagnostic metrics (called only at LOG_INTERVAL)
    @jax.jit
    def compute_diagnostics(params: Array, key: Array) -> tuple[Array, Array, Array]:
        return conjugation_metrics(model, key, params, N_CONJ_SAMPLES)

    for step in range(n_steps):
        # Independent warmup schedules. `beta` linearly ramps the KL penalty
        # (L_beta = L_1 + (1-beta)*KL externally); `conj_beta` linearly ramps
        # the conjugation regularizer. Decoupled because they target different
        # structures: KL controls posterior collapse, conj controls residual
        # variance.
        beta = jnp.array(min(1.0, step / max(kl_warmup_steps, 1)))
        conj_beta = jnp.array(min(1.0, step / max(conj_warmup_steps, 1)))

        step_start = time.perf_counter()
        if use_analytical_rho:
            (current_gen_params, current_opt_state, train_key, _), metrics = train_step(
                (current_gen_params, current_opt_state, train_key, train_data),
                None,
                beta=beta,
                conj_beta=conj_beta,
            )
            elbo, conj_var, rho_star = metrics
            jax.block_until_ready(elbo)
            current_params = model.join_coords(
                current_gen_params[:prior_dim],
                current_gen_params[prior_dim:],
                rho_star,
            )
        else:
            (current_params, current_opt_state, train_key, _), metrics = train_step(
                (current_params, current_opt_state, train_key, train_data),
                None,
                beta=beta,
                conj_beta=conj_beta,
            )
            elbo, conj_var = metrics
            jax.block_until_ready(elbo)
        step_times.append(time.perf_counter() - step_start)

        elbos.append(float(elbo))
        conj_vars.append(float(conj_var))

        rho_current = model.conjugation_parameters(current_params)
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

    # Chunked for the same reason as get_assignments_batch above.
    assignments = jax.lax.map(get_assignment, test_data, batch_size=256)

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

    cluster_counts = jnp.bincount(assignments, length=model.n_categories).tolist()

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

    # Clip samples for visualization; range depends on observable type.
    clip_max = 1.0 if observable_type == "normal" else float(n_trials)
    x_samples = jnp.clip(x_samples, 0, clip_max)

    # Also generate mean-based samples (expected values, not sampled)
    key, mean_key = jax.random.split(key)
    prior_mixture_params = model.prior_params(final_params)
    z_mean_samples = model.mix_man.sample(mean_key, prior_mixture_params, 20)

    def get_mean_image(z: Array) -> Array:
        lkl_params = model.likelihood_at(final_params, z)
        return model.obs_man.to_mean(lkl_params)

    x_mean_samples = jax.vmap(get_mean_image)(z_mean_samples)

    # Clip mean samples for visualization; matches the observable range.
    x_mean_samples = jnp.clip(x_mean_samples, 0, clip_max)

    # Cluster prototypes
    cluster_prototypes: list[list[float]] = []
    for k in range(model.n_categories):
        mask = assignments == k
        if jnp.sum(mask) > 0:
            prototype = jnp.mean(test_data[mask], axis=0)
        else:
            prototype = jnp.zeros(N_OBSERVABLE)
        cluster_prototypes.append(prototype.tolist())

    # Split step wallclock series: step 0 is JIT compile + first step;
    # the rest are warm steps used for the per-step throughput metric.
    jit_compile_seconds = step_times[0] if step_times else 0.0
    warm_step_times = step_times[1:] if len(step_times) > 1 else []
    mean_warm_step_seconds = (
        float(sum(warm_step_times) / len(warm_step_times)) if warm_step_times else 0.0
    )

    results = {
        "mode": mode,
        "observable_type": observable_type,
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
        "step_times_seconds": step_times,
        "jit_compile_seconds": jit_compile_seconds,
        "mean_warm_step_seconds": mean_warm_step_seconds,
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
    print(f"  Latent: {args.latent}")
    if args.latent == "chordal_boltzmann":
        print(f"  Latent graph: {args.latent_graph}")
    print(f"  KL warmup steps: {args.kl_warmup_steps}")
    print(f"  LR warmup steps: {args.lr_warmup_steps}")
    print(f"  Conj warmup steps: {args.conj_warmup_steps}")
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
        latent=args.latent,
        latent_graph=args.latent_graph,
        normal_precision_floor=args.normal_precision_floor,
    )
    print(f"Model dim: {model.dim}")
    print(f"  Rho params: {model.cnj_man.dim}")
    print(f"  Harmonium params: {model.gen_hrm.dim}")

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
        conj_warmup_steps=args.conj_warmup_steps,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
    )

    # Reconstructions
    def get_recon(x: Array) -> Array:
        return reconstruct(model, params, x)

    reconstructions = jax.vmap(get_recon)(test_data[:20])
    recon_clip_max = 1.0 if args.observable == "normal" else float(DEFAULT_N_TRIALS)
    reconstructions = jnp.clip(reconstructions, 0, recon_clip_max)

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
            "latent": args.latent,
            "latent_graph": args.latent_graph,
            "kl_warmup_steps": args.kl_warmup_steps,
            "lr_warmup_steps": args.lr_warmup_steps,
            "conj_warmup_steps": args.conj_warmup_steps,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "normal_precision_floor": args.normal_precision_floor,
        },
    }

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.analysis_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {paths.analysis_path}")

    # Save parameters for later diagnosis
    params_path = paths.results_dir / "params.npz"
    np.savez(str(params_path), **{args.mode: np.array(params)})
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
    print(f"{'Final Var[r]':<22} {results['conjugation_vars'][-1]:>14.2f}")
    if results['conjugation_stds']:
        print(f"{'Final Std[r]':<22} {results['conjugation_stds'][-1]:>14.2f}")
    if results['conjugation_r2s']:
        print(f"{'Final R^2':<22} {results['conjugation_r2s'][-1]:>14.4f}")
    print(f"{'Final ||rho||':<22} {results['rho_norms'][-1]:>14.4f}")
    print(
        f"{'JIT compile (s)':<22} {results['jit_compile_seconds']:>14.4f}"
    )
    print(
        f"{'Mean warm step (s)':<22} {results['mean_warm_step_seconds']:>14.4f}"
    )


if __name__ == "__main__":
    main()
