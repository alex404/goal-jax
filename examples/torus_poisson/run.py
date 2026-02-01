"""Training script for Poisson-VonMises variational harmonium.

Demonstrates variational inference for recovering toroidal latent structure
from Poisson spike counts. Generates synthetic data from a known ground-truth
model and trains to recover the parameters.

Supports three training modes:
1. free: Learnable rho via gradient descent
2. regularized: Learnable rho with conjugation variance penalty
3. analytical: Least-squares rho computation each step

Usage:
    python -m examples.torus_poisson.run
"""

import json
from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.models import (
    PoissonVonMisesHarmonium,
    VariationalPoissonVonMises,
    poisson_vonmises_harmonium,
    variational_poisson_vonmises,
)

from ..shared import example_paths, jax_cli
from .types import GroundTruth, GTConjugationMetrics, ModeResults, TuningParams


def von_mises_inverse_cdf(u: Array, kappa: float, mu: float = 0.0) -> Array:
    """Compute inverse CDF of von Mises distribution via numerical root finding.

    For von Mises with concentration kappa and mean mu:
        p(θ) = exp(κ·cos(θ - μ)) / (2π·I₀(κ))

    Args:
        u: Quantiles in [0, 1], shape (n,)
        kappa: Concentration parameter (0 = uniform)
        mu: Mean direction

    Returns:
        Angles θ such that F(θ) = u, shape (n,)
    """
    import numpy as np
    import scipy.optimize as opt
    import scipy.special as sp

    if kappa < 1e-6:
        # Uniform distribution
        return jnp.array(u * 2 * np.pi)

    # Normalization constant
    norm = 2 * np.pi * sp.i0(kappa)

    def cdf(theta: float) -> float:
        """Compute CDF via numerical integration."""
        from scipy.integrate import quad

        def integrand(t: float) -> float:
            return np.exp(kappa * np.cos(t - mu))

        result, _ = quad(integrand, 0, theta)
        return result / norm

    # Invert CDF for each quantile
    u_np = np.asarray(u)
    theta_out = np.zeros_like(u_np)

    for i, ui in enumerate(u_np.flat):
        # Find theta such that CDF(theta) = ui
        def objective(theta: float) -> float:
            return cdf(theta) - ui

        # Use bracketed root finding
        result = opt.brentq(objective, 0, 2 * np.pi)
        theta_out.flat[i] = result

    return jnp.array(theta_out)


# Default hyperparameters
N_NEURONS = 64  # Square number for grid
N_LATENT = 2  # T^2 (2-torus)
N_SAMPLES = 10000  # Large sample for stable training
N_STEPS = 5000
BATCH_SIZE = 256
LEARNING_RATE = 5e-4  # Lower LR for stability
N_MC_SAMPLES = 10
SEED = 42

# Ground truth parameters
BASELINE_RATE = 2.0
GAIN = 1.5
PRIOR_CONCENTRATION = 0.1
DISTORTION = 0.0  # Zero = perfect convolutional structure (conjugated)
COVERAGE = 1.0  # Fraction of torus covered (1.0=full, 0.5=half -> non-zero rho)
# Density modulation: von Mises concentration of neuron preferred locations
DENSITY_KAPPA1 = 0.0  # 0 = uniform, >0 = concentrated around density_mu1
DENSITY_KAPPA2 = 0.0  # 0 = uniform, >0 = concentrated around density_mu2
DENSITY_MU1 = 0.0  # Center of concentration in theta1
DENSITY_MU2 = 0.0  # Center of concentration in theta2

# Training mode parameters
CONJ_WEIGHT = 1.0  # Weight for var[RLS] penalty in regularized mode
ANALYTICAL_RHO_SAMPLES_MULTIPLIER = 50  # More samples for stable analytical rho

# Logging
LOG_INTERVAL = 200
N_CONJ_SAMPLES = 10000  # Large sample for stable var[RLS] evaluation


def create_ground_truth_model(
    n_neurons: int,
    n_latent: int,
    key: Array,
    baseline_rate: float = BASELINE_RATE,
    gain: float = GAIN,
    prior_concentration: float = PRIOR_CONCENTRATION,  # pyright: ignore[reportUnusedParameter]
    distortion: float = DISTORTION,
    coverage: float = COVERAGE,
    density_kappa1: float = DENSITY_KAPPA1,
    density_kappa2: float = DENSITY_KAPPA2,
    density_mu1: float = DENSITY_MU1,
    density_mu2: float = DENSITY_MU2,
) -> tuple[PoissonVonMisesHarmonium, Array, TuningParams]:
    """Create ground truth population code on n-torus.

    For convolutional structure (uniform locations covering full torus, same gains):
    - rho is approximately 0 (exact symmetry)

    To get non-zero rho with good conjugation (high R²):
    - Set density_kappa1 > 0 to concentrate neurons around density_mu1 in theta1
    - Set density_kappa2 > 0 to concentrate neurons around density_mu2 in theta2
    - For best R² (near 1), use single-dimension modulation (kappa1 > 0, kappa2 = 0)

    Alternative (but causes systematic bias):
    - Set coverage < 1.0 to cover only part of the torus (e.g., 0.5 = half)
    - Or set distortion > 0 to add noise to locations/gains

    Args:
        n_neurons: Number of observable neurons
        n_latent: Number of latent dimensions (torus dimension)
        key: JAX random key
        baseline_rate: Baseline firing rate
        gain: Tuning curve gain
        prior_concentration: Prior von Mises concentration
        distortion: Distortion level (0=convolutional, >0=asymmetric)
        coverage: Fraction of torus covered in theta1 (1.0=full, 0.5=half)
        density_kappa1: Von Mises concentration for theta1 density (0=uniform)
        density_kappa2: Von Mises concentration for theta2 density (0=uniform)
        density_mu1: Center of density concentration in theta1
        density_mu2: Center of density concentration in theta2

    Returns:
        Tuple of (model, params, tuning_params)
    """
    model = poisson_vonmises_harmonium(n_neurons, n_latent)

    # For 2D torus: arrange neurons on a grid (possibly non-uniform)
    if n_latent == 2:
        n_per_dim = int(jnp.sqrt(n_neurons))
        if n_per_dim * n_per_dim != n_neurons:
            # Not a perfect square, use ceil and take first n_neurons
            n_per_dim = int(jnp.ceil(jnp.sqrt(n_neurons)))

        # Create grid of preferred (theta_1, theta_2) locations
        # Use inverse CDF for von Mises density modulation
        quantiles_1d = jnp.linspace(0, 1, n_per_dim, endpoint=False) + 0.5 / n_per_dim

        # Apply von Mises modulation independently per dimension
        if density_kappa1 > 0:
            angles_1d_1 = von_mises_inverse_cdf(quantiles_1d, density_kappa1, density_mu1)
        else:
            max_angle_1 = 2 * jnp.pi * coverage
            angles_1d_1 = jnp.linspace(0, max_angle_1, n_per_dim, endpoint=False)

        if density_kappa2 > 0:
            angles_1d_2 = von_mises_inverse_cdf(quantiles_1d, density_kappa2, density_mu2)
        else:
            angles_1d_2 = jnp.linspace(0, 2 * jnp.pi, n_per_dim, endpoint=False)

        grid_1, grid_2 = jnp.meshgrid(angles_1d_1, angles_1d_2)
        preferred_1 = grid_1.ravel()[:n_neurons]
        preferred_2 = grid_2.ravel()[:n_neurons]
    else:
        # For higher-dimensional torus, use random locations
        key, loc_key = jax.random.split(key)
        locations = jax.random.uniform(loc_key, (n_neurons, n_latent)) * 2 * jnp.pi
        locations = locations.at[:, 0].set(locations[:, 0] * coverage)  # Apply coverage
        preferred_1 = locations[:, 0]
        preferred_2 = locations[:, 1] if n_latent > 1 else jnp.zeros(n_neurons)

    # Add distortion if requested
    gains = jnp.ones(n_neurons) * gain
    if distortion > 0:
        key, dist_key, gain_key = jax.random.split(key, 3)
        noise = jax.random.normal(dist_key, (n_neurons, 2)) * distortion
        preferred_1 = preferred_1 + noise[:, 0]
        preferred_2 = preferred_2 + noise[:, 1]
        gains = gains * jnp.exp(
            jax.random.normal(gain_key, (n_neurons,)) * distortion * 0.5
        )

    # Wrap angles to [0, 2*pi)
    preferred_1 = jnp.mod(preferred_1, 2 * jnp.pi)
    preferred_2 = jnp.mod(preferred_2, 2 * jnp.pi)

    # Build interaction matrix: W[i, :] = [g*cos(p1), g*sin(p1), g*cos(p2), g*sin(p2)]
    int_matrix = jnp.stack(
        [
            gains * jnp.cos(preferred_1),
            gains * jnp.sin(preferred_1),
            gains * jnp.cos(preferred_2),
            gains * jnp.sin(preferred_2),
        ],
        axis=1,
    )  # (n_neurons, 4)

    # Baselines (log of baseline firing rate)
    baselines = jnp.ones(n_neurons) * jnp.log(baseline_rate)

    # Prior (low concentration = nearly uniform on torus)
    # VonMises params: [kappa*cos(mu), kappa*sin(mu)] for each dimension
    prior_params = jnp.zeros(2 * n_latent)  # Zero natural params = uniform-ish

    # Join into harmonium params
    params = model.join_coords(baselines, int_matrix.ravel(), prior_params)

    # Store tuning parameters for visualization
    tuning: TuningParams = {
        "preferred_theta1": preferred_1.tolist(),
        "preferred_theta2": preferred_2.tolist(),
        "gains": gains.tolist(),
        "baselines": baselines.tolist(),
    }

    return model, params, tuning


def extract_tuning_params(
    model: VariationalPoissonVonMises, params: Array
) -> TuningParams:
    """Extract tuning curve parameters from learned model.

    For each neuron, the weight row W[i, :] = [w1, w2, w3, w4] represents:
    - Dimension 1: gain_1 * [cos(pref_1), sin(pref_1)]
    - Dimension 2: gain_2 * [cos(pref_2), sin(pref_2)]

    We extract:
    - Preferred angles: arctan2(sin_coeff, cos_coeff)
    - Gains: sqrt(cos_coeff^2 + sin_coeff^2)
    """
    _, hrm_params = model.split_coords(params)
    obs_bias, int_params, _ = model.hrm.split_coords(hrm_params)
    int_matrix = int_params.reshape(model.n_neurons, 2 * model.n_latent)

    # Extract preferred angles and gains for each torus dimension
    # W[i, :] = [g1*cos(p1), g1*sin(p1), g2*cos(p2), g2*sin(p2)]
    cos_1, sin_1 = int_matrix[:, 0], int_matrix[:, 1]
    cos_2, sin_2 = int_matrix[:, 2], int_matrix[:, 3]

    preferred_1 = jnp.arctan2(sin_1, cos_1)
    preferred_2 = jnp.arctan2(sin_2, cos_2)

    # Wrap to [0, 2*pi)
    preferred_1 = jnp.mod(preferred_1, 2 * jnp.pi)
    preferred_2 = jnp.mod(preferred_2, 2 * jnp.pi)

    # Compute gains as norm of (cos, sin) pairs
    gain_1 = jnp.sqrt(cos_1**2 + sin_1**2)
    gain_2 = jnp.sqrt(cos_2**2 + sin_2**2)
    # Use average gain for simplicity
    gains = (gain_1 + gain_2) / 2

    return {
        "preferred_theta1": preferred_1.tolist(),
        "preferred_theta2": preferred_2.tolist(),
        "gains": gains.tolist(),
        "baselines": obs_bias.tolist(),
    }


def compute_analytical_rho(
    model: VariationalPoissonVonMises,
    hrm_params: Array,
    key: Array,
    n_samples: int,
) -> tuple[Array, Array, Array]:
    """Compute optimal rho via least squares regression.

    Solves: rho* = argmin sum((chi + rho*s(z) - psi_X(z))^2)

    Args:
        model: Variational model
        hrm_params: Harmonium parameters (without rho)
        key: JAX random key
        n_samples: Number of samples for regression

    Returns:
        Tuple of (rho_star, r_squared, chi)
    """
    lat_dim = model.pst_man.dim  # 2*n_latent

    # Sample from prior - use stop_gradient since VonMises sampling
    # uses rejection sampling which cannot be differentiated through
    _, _, prior_params = model.hrm.split_coords(hrm_params)
    z_samples = jax.lax.stop_gradient(
        model.pst_man.sample(key, prior_params, n_samples)
    )

    # Get sufficient statistics: shape (n_samples, lat_dim)
    s_z_samples = jax.vmap(model.pst_man.sufficient_statistic)(z_samples)

    # Compute psi_X at each sample
    obs_bias, int_params, _ = model.hrm.split_coords(hrm_params)
    int_matrix = int_params.reshape(model.n_neurons, lat_dim)

    def compute_psi(z: Array) -> Array:
        s_z = model.pst_man.sufficient_statistic(z)
        lkl_nat = obs_bias + int_matrix @ s_z
        return model.obs_man.log_partition_function(lkl_nat)

    psi_values = jax.vmap(compute_psi)(z_samples)

    # Least squares: psi approx chi + rho*s_Z
    design = jnp.concatenate([jnp.ones((n_samples, 1)), s_z_samples], axis=1)
    coeffs = jnp.linalg.lstsq(design, psi_values, rcond=None)[0]
    chi, rho_star = coeffs[0], coeffs[1:]

    # R^2 for diagnostics
    predicted = design @ coeffs
    ss_res = jnp.sum((psi_values - predicted) ** 2)
    ss_tot = jnp.sum((psi_values - jnp.mean(psi_values)) ** 2)
    r_squared = 1.0 - ss_res / jnp.maximum(ss_tot, 1e-6)

    return rho_star, jnp.clip(r_squared, -10.0, 1.0), chi


def compute_conjugation_metrics_jit(
    model: VariationalPoissonVonMises, params: Array, key: Array, n_samples: int
) -> tuple[Array, Array, Array]:
    """JIT-compatible conjugation metrics computation.

    Uses stop_gradient on samples since VonMises uses rejection sampling
    which cannot be differentiated through.

    Returns:
        Tuple of (variance, std, r_squared)
    """
    rho, hrm_params = model.split_coords(params)

    # Sample from current prior - use stop_gradient since VonMises sampling
    # uses rejection sampling which cannot be differentiated through
    _, _, prior_params = model.hrm.split_coords(hrm_params)
    z_samples = jax.lax.stop_gradient(
        model.pst_man.sample(key, prior_params, n_samples)
    )

    def reduced_learning_signal(z: Array) -> Array:
        s_z = model.pst_man.sufficient_statistic(z)
        term1 = jnp.dot(rho, s_z)
        lkl_params = model.likelihood_at(params, z)
        term2 = model.obs_man.log_partition_function(lkl_params)
        return term1 - term2

    def psi_x_at_z(z: Array) -> Array:
        lkl_params = model.likelihood_at(params, z)
        return model.obs_man.log_partition_function(lkl_params)

    f_vals = jax.vmap(reduced_learning_signal)(z_samples)
    psi_vals = jax.vmap(psi_x_at_z)(z_samples)

    var_f = jnp.var(f_vals)
    std_f = jnp.sqrt(var_f)
    var_psi = jnp.var(psi_vals)

    variance_threshold = 1e-6
    raw_r2 = 1.0 - var_f / jnp.maximum(var_psi, variance_threshold)
    r_squared = jnp.where(
        var_psi > variance_threshold, jnp.clip(raw_r2, -10.0, 1.0), jnp.nan
    )

    return var_f, std_f, r_squared


def compute_ground_truth_conjugation(
    gt_model: PoissonVonMisesHarmonium,
    gt_params: Array,
    key: Array,
    n_samples: int = 10000,
) -> GTConjugationMetrics:
    """Compute conjugation metrics for the ground truth model.

    This computes the optimal rho via least squares and then evaluates
    R² and Var[RLS] with that optimal rho.

    Args:
        gt_model: Ground truth harmonium model
        gt_params: Ground truth parameters
        key: JAX random key
        n_samples: Number of samples for Monte Carlo estimation

    Returns:
        GTConjugationMetrics with optimal rho and associated metrics
    """
    lat_dim = gt_model.pst_man.dim  # 2*n_latent

    # Sample from prior
    _, _, prior_params = gt_model.split_coords(gt_params)
    z_samples = gt_model.pst_man.sample(key, prior_params, n_samples)

    # Get sufficient statistics: shape (n_samples, lat_dim)
    s_z_samples = jax.vmap(gt_model.pst_man.sufficient_statistic)(z_samples)

    # Compute psi_X at each sample
    obs_bias, int_params, _ = gt_model.split_coords(gt_params)
    int_matrix = int_params.reshape(gt_model.n_neurons, lat_dim)

    def compute_psi(z: Array) -> Array:
        s_z = gt_model.pst_man.sufficient_statistic(z)
        lkl_nat = obs_bias + int_matrix @ s_z
        return gt_model.obs_man.log_partition_function(lkl_nat)

    psi_values = jax.vmap(compute_psi)(z_samples)

    # Least squares: psi approx chi + rho*s_Z
    design = jnp.concatenate([jnp.ones((n_samples, 1)), s_z_samples], axis=1)
    coeffs = jnp.linalg.lstsq(design, psi_values, rcond=None)[0]
    _, optimal_rho = coeffs[0], coeffs[1:]

    # Compute R² and var[RLS] with optimal rho
    predicted = design @ coeffs
    residuals = psi_values - predicted  # This is the reduced learning signal (shifted)

    var_rls = float(jnp.var(residuals))
    var_psi = float(jnp.var(psi_values))

    # R² = 1 - Var[RLS] / Var[psi]
    if var_psi > 1e-6:
        r_squared = 1.0 - var_rls / var_psi
    else:
        r_squared = float("nan")

    optimal_rho_norm = float(jnp.linalg.norm(optimal_rho))

    return {
        "optimal_rho": optimal_rho.tolist(),
        "optimal_rho_norm": optimal_rho_norm,
        "r_squared": float(r_squared),
        "var_rls": var_rls,
        "var_psi": var_psi,
    }


def train_model(
    key: Array,
    model: VariationalPoissonVonMises,
    train_data: Array,
    mode: str,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    n_mc_samples: int,
    conj_weight: float = CONJ_WEIGHT,
    analytical_samples_mult: int = ANALYTICAL_RHO_SAMPLES_MULTIPLIER,
) -> tuple[Array, ModeResults]:
    """Train a model with specified mode.

    Args:
        key: Random key
        model: The VariationalPoissonVonMises model
        train_data: Training spike counts
        mode: Training mode ('free', 'regularized', or 'analytical')
        n_steps: Number of training steps
        batch_size: Batch size
        learning_rate: Learning rate
        n_mc_samples: Number of MC samples for ELBO
        conj_weight: Weight for conjugation penalty (regularized mode)
        analytical_samples_mult: Multiplier for analytical rho samples

    Returns:
        Tuple of (final_params, mode_results)
    """
    use_analytical_rho = mode == "analytical"
    use_conj_penalty = mode == "regularized"

    n_analytical_samples = analytical_samples_mult * model.pst_man.dim
    n_samples = train_data.shape[0]

    mode_str = {
        "free": "LEARNABLE rho",
        "regularized": f"REGULARIZED (weight={conj_weight})",
        "analytical": f"ANALYTICAL rho ({n_analytical_samples} samples)",
    }[mode]

    print(f"\n{'=' * 60}")
    print(f"Training with {mode_str}")
    print(f"{'=' * 60}")

    # Initialize model
    key, init_key = jax.random.split(key)
    params = model.initialize_from_sample(
        init_key, train_data, location=0.0, shape=0.1
    )

    rho, hrm_params = model.split_coords(params)
    _ = jnp.zeros_like(rho)

    # Optimizer setup
    optimizer = optax.adam(learning_rate)
    if use_analytical_rho:
        opt_state = optimizer.init(hrm_params)
    else:
        opt_state = optimizer.init(params)

    # Loss function for learnable rho (free or regularized)
    def loss_fn_full(
        params: Array, key: Array, batch: Array
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        key, elbo_key, conj_key = jax.random.split(key, 3)
        elbo = model.mean_elbo(elbo_key, params, batch, n_mc_samples)

        conj_var, conj_std, conj_r2 = compute_conjugation_metrics_jit(
            model, params, conj_key, N_CONJ_SAMPLES
        )

        if use_conj_penalty:
            conj_loss = conj_weight * conj_var
        else:
            conj_loss = 0.0

        loss = -elbo + conj_loss
        return loss, (elbo, conj_var, conj_std, conj_r2)

    # Loss function for analytical rho
    def loss_fn_analytical(
        hrm_params: Array, key: Array, batch: Array
    ) -> tuple[Array, tuple[Array, Array, Array, Array, Array]]:
        key, rho_key, elbo_key = jax.random.split(key, 3)

        # Compute analytical rho via lstsq
        rho_star, r_squared, _ = compute_analytical_rho(
            model, hrm_params, rho_key, n_samples=n_analytical_samples
        )

        # Build full params with analytical rho and compute ELBO
        params_with_rho = model.join_coords(rho_star, hrm_params)
        elbo = model.mean_elbo(elbo_key, params_with_rho, batch, n_mc_samples)

        loss = -elbo
        return loss, (elbo, r_squared, r_squared, r_squared, rho_star)

    # Training step functions
    if use_analytical_rho:

        @jax.jit
        def train_step(
            carry: Any, _: None
        ) -> tuple[Any, tuple[Array, Array, Array, Array, Array]]:
            hrm_params, opt_state, step_key, data = carry
            step_key, next_key, batch_key = jax.random.split(step_key, 3)

            batch_idx = jax.random.choice(batch_key, n_samples, shape=(batch_size,))
            batch = data[batch_idx]

            (_, metrics), grads = jax.value_and_grad(loss_fn_analytical, has_aux=True)(
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

            batch_idx = jax.random.choice(batch_key, n_samples, shape=(batch_size,))
            batch = data[batch_idx]

            (_, metrics), grads = jax.value_and_grad(loss_fn_full, has_aux=True)(
                params, next_key, batch
            )

            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            return (new_params, new_opt_state, step_key, data), metrics

    # Training loop
    elbos: list[float] = []
    reconstruction_errors: list[float] = []
    r_squared: list[float] = []
    var_rls_history: list[float] = []  # Variance of reduced learning signal
    rho_norms: list[float] = []
    conjugation_errors: list[float] = []

    key, train_key = jax.random.split(key)
    current_params = params
    current_hrm_params = hrm_params
    current_opt_state = opt_state

    print(f"Training for {n_steps} steps...")

    for step in range(n_steps):
        if use_analytical_rho:
            (current_hrm_params, current_opt_state, train_key, _), metrics = train_step(
                (current_hrm_params, current_opt_state, train_key, train_data), None
            )
            elbo, _, _, _, rho_star = metrics
            current_params = model.join_coords(rho_star, current_hrm_params)
            conj_var = 0.0
        else:
            (current_params, current_opt_state, train_key, _), metrics = train_step(
                (current_params, current_opt_state, train_key, train_data), None
            )
            elbo, conj_var, _, _ = metrics

        elbos.append(float(elbo))
        conjugation_errors.append(float(conj_var))

        rho_current, _ = model.split_coords(current_params)
        rho_norm = float(jnp.linalg.norm(rho_current))
        rho_norms.append(rho_norm)

        if step % LOG_INTERVAL == 0:
            # Compute reconstruction error and conjugation metrics
            recon_error = float(model.reconstruction_error(current_params, train_data[:500]))
            reconstruction_errors.append(recon_error)

            # Compute var[RLS] explicitly
            key, conj_key = jax.random.split(key)
            var_f, _, r2_val = compute_conjugation_metrics_jit(
                model, current_params, conj_key, N_CONJ_SAMPLES
            )
            r_squared.append(float(r2_val))
            var_rls_history.append(float(var_f))

            print(
                f"  Step {step}: ELBO={elbo:.2f}, R^2={r2_val:.4f}, var[RLS]={var_f:.4f}, ||rho||={rho_norm:.2f}"
            )

    # Final evaluation
    print("\nFinal evaluation...")
    key, eval_key = jax.random.split(key)

    final_recon_error = float(model.reconstruction_error(current_params, train_data))
    final_var_rls, _, final_r2 = compute_conjugation_metrics_jit(
        model, current_params, eval_key, N_CONJ_SAMPLES
    )
    final_rho, _ = model.split_coords(current_params)
    final_rho_norm = float(jnp.linalg.norm(final_rho))

    print(f"  Final ELBO: {elbos[-1]:.4f}")
    print(f"  Final R^2: {float(final_r2):.4f}")
    print(f"  Final var[RLS]: {float(final_var_rls):.4f}")
    print(f"  Final ||rho||: {final_rho_norm:.4f}")
    print(f"  Reconstruction error: {final_recon_error:.4f}")

    # Extract learned parameters
    learned_weights = model.get_weight_matrix(current_params)
    _, hrm_p = model.split_coords(current_params)
    learned_baselines, _, _ = model.hrm.split_coords(hrm_p)
    learned_tuning = extract_tuning_params(model, current_params)

    results: ModeResults = {
        "history": {
            "elbos": elbos,
            "reconstruction_errors": reconstruction_errors,
            "r_squared": r_squared,
            "var_rls": var_rls_history,
            "rho_norms": rho_norms,
            "conjugation_errors": conjugation_errors,
        },
        "final_rho_norm": final_rho_norm,
        "final_r_squared": float(final_r2),
        "final_var_rls": float(final_var_rls),
        "final_elbo": elbos[-1],
        "final_reconstruction_error": final_recon_error,
        "learned_weight_matrix": learned_weights.tolist(),
        "learned_baselines": learned_baselines.tolist(),
        "learned_tuning": learned_tuning,
    }

    return current_params, results


def main():
    """Main entry point."""
    jax_cli()
    paths = example_paths(__file__)

    print("=" * 60)
    print("POISSON-VONMISES VARIATIONAL HARMONIUM")
    print("=" * 60)
    print(f"  n_neurons: {N_NEURONS}")
    print(f"  n_latent: {N_LATENT}")
    print(f"  n_samples: {N_SAMPLES}")
    print(f"  n_steps: {N_STEPS}")
    print(f"  distortion: {DISTORTION}")
    print(f"  coverage: {COVERAGE}")
    print(f"  baseline_rate: {BASELINE_RATE}")
    print(f"  gain: {GAIN}")
    print("=" * 60)

    key = jax.random.PRNGKey(SEED)

    # Create ground truth model
    print("\nCreating ground truth model...")
    key, gt_key = jax.random.split(key)
    gt_model, gt_params, gt_tuning = create_ground_truth_model(
        n_neurons=N_NEURONS,
        n_latent=N_LATENT,
        key=gt_key,
        baseline_rate=BASELINE_RATE,
        gain=GAIN,
        prior_concentration=PRIOR_CONCENTRATION,
        distortion=DISTORTION,
        coverage=COVERAGE,
        density_kappa1=DENSITY_KAPPA1,
        density_kappa2=DENSITY_KAPPA2,
        density_mu1=DENSITY_MU1,
        density_mu2=DENSITY_MU2,
    )

    # Generate synthetic data from ground truth
    print("Generating synthetic data from ground truth model...")
    key, sample_key = jax.random.split(key)

    # Sample latent angles from prior
    _, _, gt_prior_params = gt_model.split_coords(gt_params)
    z_samples = gt_model.pst_man.sample(sample_key, gt_prior_params, N_SAMPLES)

    # Sample spike counts given latent angles
    key, spike_key = jax.random.split(key)
    spike_keys = jax.random.split(spike_key, N_SAMPLES)

    def sample_spikes(subkey: Array, z: Array) -> Array:
        rates = gt_model.observable_rates(gt_params, z)
        return jax.random.poisson(subkey, rates).astype(jnp.float32)

    spike_counts = jax.vmap(sample_spikes)(spike_keys, z_samples)

    print(f"  Spike counts shape: {spike_counts.shape}")
    print(f"  Mean spike count: {jnp.mean(spike_counts):.2f}")
    print(f"  Max spike count: {jnp.max(spike_counts):.0f}")

    # Extract ground truth info
    gt_weights = gt_model.get_weight_matrix(gt_params)
    gt_baselines, _, gt_prior = gt_model.split_coords(gt_params)

    # Compute ground truth conjugation metrics
    print("\nComputing ground truth conjugation metrics...")
    key, gt_conj_key = jax.random.split(key)
    gt_conjugation = compute_ground_truth_conjugation(
        gt_model, gt_params, gt_conj_key, n_samples=N_CONJ_SAMPLES * 2
    )
    print(f"  GT R^2 (with optimal rho): {gt_conjugation['r_squared']:.4f}")
    print(f"  GT var[RLS]: {gt_conjugation['var_rls']:.4f}")
    print(f"  GT var[psi_X]: {gt_conjugation['var_psi']:.4f}")
    print(f"  GT ||rho_optimal||: {gt_conjugation['optimal_rho_norm']:.4f}")

    ground_truth: GroundTruth = {
        "weight_matrix": gt_weights.tolist(),
        "baselines": gt_baselines.tolist(),
        "prior_params": gt_prior.tolist(),
        "tuning": gt_tuning,
        "conjugation": gt_conjugation,
    }

    # Create variational model for training
    model = variational_poisson_vonmises(N_NEURONS, N_LATENT)
    print("\nVariational model created:")
    print(f"  Total params: {model.dim}")
    print(f"  Rho params: {model.rho_man.dim}")
    print(f"  Harmonium params: {model.hrm.dim}")

    # Train all modes
    modes = ["free", "regularized", "analytical"]

    # Train models
    all_results: dict[str, ModeResults] = {}
    all_params: dict[str, Array] = {}

    for mode in modes:
        key, train_key = jax.random.split(key)
        params, results = train_model(
            key=train_key,
            model=model,
            train_data=spike_counts,
            mode=mode,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            n_mc_samples=N_MC_SAMPLES,
            conj_weight=CONJ_WEIGHT,
            analytical_samples_mult=ANALYTICAL_RHO_SAMPLES_MULTIPLIER,
        )
        all_results[mode] = results
        all_params[mode] = params

    # Find best model by R^2
    best_mode = max(all_results.keys(), key=lambda m: all_results[m]["final_r_squared"])
    print(f"\nBest model by R^2: {best_mode}")

    # Save results
    output: dict[str, Any] = {
        "models": all_results,
        "ground_truth": ground_truth,
        "best_model": best_mode,
        "config": {
            "n_neurons": N_NEURONS,
            "n_latent": N_LATENT,
            "n_samples": N_SAMPLES,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "n_mc_samples": N_MC_SAMPLES,
            "seed": SEED,
            "distortion": DISTORTION,
            "coverage": COVERAGE,
            "baseline_rate": BASELINE_RATE,
            "gain": GAIN,
            "prior_concentration": PRIOR_CONCENTRATION,
            "conj_weight": CONJ_WEIGHT,
            "analytical_samples_mult": ANALYTICAL_RHO_SAMPLES_MULTIPLIER,
            "density_kappa1": DENSITY_KAPPA1,
            "density_kappa2": DENSITY_KAPPA2,
            "density_mu1": DENSITY_MU1,
            "density_mu2": DENSITY_MU2,
        },
    }

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.analysis_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {paths.analysis_path}")

    # Print summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)

    header = f"{'Metric':<25}"
    for mode in modes:
        header += f" {mode:>14}"
    print(header)
    print("-" * 80)

    metrics = [
        ("Final ELBO", "final_elbo", lambda x: f"{x:>14.2f}"),
        ("Final R^2", "final_r_squared", lambda x: f"{x:>14.4f}"),
        ("Final var[RLS]", "final_var_rls", lambda x: f"{x:>14.4f}"),
        ("Final ||rho||", "final_rho_norm", lambda x: f"{x:>14.4f}"),
        ("Reconstruction error", "final_reconstruction_error", lambda x: f"{x:>14.4f}"),
    ]

    for name, key_name, fmt in metrics:
        row = f"{name:<25}"
        for mode in modes:
            val = all_results[mode][key_name]  # type: ignore
            row += fmt(val)
        print(row)

    # Print GT reference values
    print("-" * 80)
    print(f"{'GT R^2 (optimal rho)':<25} {gt_conjugation['r_squared']:>14.4f}")
    print(f"{'GT var[RLS] (optimal rho)':<25} {gt_conjugation['var_rls']:>14.4f}")


if __name__ == "__main__":
    main()
