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

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.models import (
    PoissonVonMisesHarmonium,
    VonMisesPopulationCode,
)

from ..shared import example_paths, jax_cli
from .types import GroundTruth, GTConjugationMetrics, ModeResults, TuningParams


def von_mises_inverse_cdf(u: Array, kappa: float, mu: float = 0.0) -> Array:
    """Compute inverse CDF of von Mises distribution via numerical root finding.

    For von Mises with concentration kappa and mean mu:
        p(\\theta) = exp(\\kappa cos(\\theta - \\mu)) / (2\\pi I_0(\\kappa))

    Args:
        u: Quantiles in [0, 1], shape (n,)
        kappa: Concentration parameter (0 = uniform)
        mu: Mean direction

    Returns:
        Angles \\theta such that F(\\theta) = u, shape (n,)
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


def create_ground_truth_model(
    n_neurons: int,
    n_latent: int,
    key: Array,
    baseline_rate: float,
    gain: float,
    prior_concentration: float,  # pyright: ignore[reportUnusedParameter]
    distortion: float,
    coverage: float,
    density_kappa1: float,
    density_kappa2: float,
    density_mu1: float,
    density_mu2: float,
) -> tuple[PoissonVonMisesHarmonium, Array, TuningParams]:
    """Create ground truth population code on n-torus."""
    model = PoissonVonMisesHarmonium(n_neurons, n_latent)

    # For 2D torus: arrange neurons on a grid (possibly non-uniform)
    if n_latent == 2:
        n_per_dim = int(jnp.sqrt(n_neurons))
        if n_per_dim * n_per_dim != n_neurons:
            n_per_dim = int(jnp.ceil(jnp.sqrt(n_neurons)))

        quantiles_1d = jnp.linspace(0, 1, n_per_dim, endpoint=False) + 0.5 / n_per_dim

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
        key, loc_key = jax.random.split(key)
        locations = jax.random.uniform(loc_key, (n_neurons, n_latent)) * 2 * jnp.pi
        locations = locations.at[:, 0].set(locations[:, 0] * coverage)
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
    )

    baselines = jnp.ones(n_neurons) * jnp.log(baseline_rate)
    prior_params = jnp.zeros(2 * n_latent)
    params = model.join_coords(baselines, int_matrix.ravel(), prior_params)

    tuning: TuningParams = {
        "preferred_theta1": preferred_1.tolist(),
        "preferred_theta2": preferred_2.tolist(),
        "gains": gains.tolist(),
        "baselines": baselines.tolist(),
    }

    return model, params, tuning


def extract_tuning_params(
    model: VonMisesPopulationCode, params: Array
) -> TuningParams:
    """Extract tuning curve parameters from learned model."""
    _, hrm_params = model.split_coords(params)
    obs_bias, int_params, _ = model.hrm.split_coords(hrm_params)
    int_matrix = int_params.reshape(model.n_neurons, 2 * model.n_latent)

    cos_1, sin_1 = int_matrix[:, 0], int_matrix[:, 1]
    cos_2, sin_2 = int_matrix[:, 2], int_matrix[:, 3]

    preferred_1 = jnp.mod(jnp.arctan2(sin_1, cos_1), 2 * jnp.pi)
    preferred_2 = jnp.mod(jnp.arctan2(sin_2, cos_2), 2 * jnp.pi)

    gain_1 = jnp.sqrt(cos_1**2 + sin_1**2)
    gain_2 = jnp.sqrt(cos_2**2 + sin_2**2)
    gains = (gain_1 + gain_2) / 2

    return {
        "preferred_theta1": preferred_1.tolist(),
        "preferred_theta2": preferred_2.tolist(),
        "gains": gains.tolist(),
        "baselines": obs_bias.tolist(),
    }


def compute_gt_conjugation(
    gt_model: PoissonVonMisesHarmonium,
    gt_params: Array,
    key: Array,
    n_samples: int = 10000,
) -> GTConjugationMetrics:
    """Compute conjugation metrics for the ground truth model using library methods."""
    # Wrap GT harmonium in variational model to use library methods
    var_model = VonMisesPopulationCode(hrm=gt_model)
    zero_rho = jnp.zeros(var_model.rho_man.dim)
    var_params = var_model.join_coords(zero_rho, gt_params)

    # Compute optimal rho via regression
    key, reg_key = jax.random.split(key)
    rho_star, r_squared, _, _ = var_model.regress_conjugation_parameters(
        reg_key, var_params, n_samples
    )

    # Var[\psi_X] = Var[RLS] when rho=0 (since RLS = 0 \cdot s - \psi = -\psi)
    key, m0_key = jax.random.split(key)
    var_psi, _, _ = var_model.conjugation_metrics(m0_key, var_params, n_samples)

    # Var[RLS] with optimal rho
    optimal_params = var_model.join_coords(rho_star, gt_params)
    key, m1_key = jax.random.split(key)
    var_rls, _, _ = var_model.conjugation_metrics(m1_key, optimal_params, n_samples)

    return {
        "optimal_rho": rho_star.tolist(),
        "optimal_rho_norm": float(jnp.linalg.norm(rho_star)),
        "r_squared": float(r_squared),
        "var_rls": float(var_rls),
        "var_psi": float(var_psi),
    }


def train_model(
    key: Array,
    model: VonMisesPopulationCode,
    train_data: Array,
    mode: str,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    n_mc_samples: int,
    init_params: Array,
    conj_weight: float,
    analytical_samples_mult: int,
    n_conj_samples: int,
    log_interval: int,
) -> tuple[Array, ModeResults]:
    """Train a model with specified mode."""
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

    params = init_params
    _, hrm_params = model.split_coords(params)
    zero_rho = jnp.zeros(model.rho_man.dim)

    # Optimizer setup
    if use_analytical_rho:
        # Gradient clipping stabilizes the implicit gradient through lstsq,
        # which can amplify noise early in training.
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
        opt_state = optimizer.init(hrm_params)
    else:
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

    # --- Loss functions (close over train_data) ---

    def loss_fn_free(
        params: Array, key: Array, batch: Array
    ) -> tuple[Array, Array]:
        elbo = model.mean_elbo(key, params, batch, n_mc_samples)
        return -elbo, elbo

    def loss_fn_regularized(
        params: Array, key: Array, batch: Array
    ) -> tuple[Array, tuple[Array, Array]]:
        elbo_key, conj_key = jax.random.split(key)
        elbo = model.mean_elbo(elbo_key, params, batch, n_mc_samples)

        # Conjugation penalty with stop_gradient on samples (VonMises is
        # non-reparameterizable, so we can't differentiate through sampling)
        p_params = model.prior_params(params)
        z_sg = jax.lax.stop_gradient(
            model.pst_man.sample(conj_key, p_params, n_conj_samples)
        )
        f_vals = jax.vmap(lambda z: model.reduced_learning_signal(params, z))(z_sg)
        conj_var = jnp.var(f_vals)

        loss = -elbo + conj_weight * conj_var
        return loss, (elbo, conj_var)

    def loss_fn_analytical(
        hrm_params: Array, key: Array, batch: Array
    ) -> tuple[Array, tuple[Array, Array]]:
        rho_key, elbo_key, conj_key = jax.random.split(key, 3)

        # Compute analytical rho via library regression
        dummy_params = model.join_coords(zero_rho, hrm_params)
        rho_star, _, _, _ = model.regress_conjugation_parameters(
            rho_key, dummy_params, n_analytical_samples
        )

        # Let implicit gradient flow through lstsq for proper rho coupling
        params_with_rho = model.join_coords(rho_star, hrm_params)
        elbo = model.mean_elbo(elbo_key, params_with_rho, batch, n_mc_samples)

        # Conjugation penalty: explicitly discourage nonlinear \psi_X
        p_params = model.prior_params(params_with_rho)
        z_sg = jax.lax.stop_gradient(
            model.pst_man.sample(conj_key, p_params, n_conj_samples)
        )
        f_vals = jax.vmap(lambda z: model.reduced_learning_signal(params_with_rho, z))(
            z_sg
        )
        conj_var = jnp.var(f_vals)

        loss = -elbo + conj_weight * conj_var
        return loss, (elbo, rho_star)

    # --- Training step functions (close over train_data, no @jax.jit - scan handles compilation) ---

    def train_step_analytical(
        carry: Any, _: None
    ) -> tuple[Any, tuple[Array, Array]]:
        hrm_params, opt_state, step_key = carry
        step_key, next_key, batch_key = jax.random.split(step_key, 3)
        batch = train_data[jax.random.choice(batch_key, n_samples, shape=(batch_size,))]

        (_, (elbo, rho_star)), grads = jax.value_and_grad(
            loss_fn_analytical, has_aux=True
        )(hrm_params, next_key, batch)

        updates, new_opt_state = optimizer.update(grads, opt_state, hrm_params)
        new_hrm_params = optax.apply_updates(hrm_params, updates)

        return (new_hrm_params, new_opt_state, step_key), (
            elbo,
            rho_star,
        )

    def train_step_regularized(
        carry: Any, _: None
    ) -> tuple[Any, tuple[Array, Array]]:
        params, opt_state, step_key = carry
        step_key, next_key, batch_key = jax.random.split(step_key, 3)
        batch = train_data[jax.random.choice(batch_key, n_samples, shape=(batch_size,))]

        (_, (elbo, conj_var)), grads = jax.value_and_grad(
            loss_fn_regularized, has_aux=True
        )(params, next_key, batch)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return (new_params, new_opt_state, step_key), (elbo, conj_var)

    def train_step_free(
        carry: Any, _: None
    ) -> tuple[Any, Array]:
        params, opt_state, step_key = carry
        step_key, next_key, batch_key = jax.random.split(step_key, 3)
        batch = train_data[jax.random.choice(batch_key, n_samples, shape=(batch_size,))]

        (_, elbo), grads = jax.value_and_grad(loss_fn_free, has_aux=True)(
            params, next_key, batch
        )

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return (new_params, new_opt_state, step_key), elbo

    # --- Chunked training loop: inner scan for speed, outer loop for periodic metrics ---
    all_elbos: list[Array] = []
    reconstruction_errors: list[float] = []
    r_squared: list[float] = []
    var_rls_history: list[float] = []
    rho_norms: list[float] = []
    all_conj_errors: list[Array] = []

    key, train_key = jax.random.split(key)
    current_params = params
    current_hrm_params = hrm_params
    current_opt_state = opt_state

    n_chunks = n_steps // log_interval
    remainder = n_steps % log_interval

    print(f"Training for {n_steps} steps ({n_chunks} chunks of {log_interval})...")

    for chunk in range(n_chunks):
        if use_analytical_rho:
            carry = (current_hrm_params, current_opt_state, train_key)
            carry, (elbos_chunk, rho_stars_chunk) = jax.lax.scan(
                train_step_analytical, carry, None, length=log_interval
            )
            current_hrm_params, current_opt_state, train_key = carry
            current_params = model.join_coords(rho_stars_chunk[-1], current_hrm_params)
            all_elbos.append(elbos_chunk)
            all_conj_errors.append(jnp.zeros(log_interval))
        elif use_conj_penalty:
            carry = (current_params, current_opt_state, train_key)
            carry, (elbos_chunk, conj_vars_chunk) = jax.lax.scan(
                train_step_regularized, carry, None, length=log_interval
            )
            current_params, current_opt_state, train_key = carry
            all_elbos.append(elbos_chunk)
            all_conj_errors.append(conj_vars_chunk)
        else:
            carry = (current_params, current_opt_state, train_key)
            carry, elbos_chunk = jax.lax.scan(
                train_step_free, carry, None, length=log_interval
            )
            current_params, current_opt_state, train_key = carry
            all_elbos.append(elbos_chunk)
            all_conj_errors.append(jnp.zeros(log_interval))

        # Collect per-chunk rho norm
        rho_current, _ = model.split_coords(current_params)
        rho_norm = float(jnp.linalg.norm(rho_current))
        rho_norms.append(rho_norm)

        # Periodic metrics (expensive, computed once per chunk)
        recon_error = float(
            model.reconstruction_error(current_params, train_data[:500])
        )
        reconstruction_errors.append(recon_error)

        key, conj_key = jax.random.split(key)
        var_f, _, r2_val = model.conjugation_metrics(
            conj_key, current_params, n_conj_samples
        )
        r_squared.append(float(r2_val))
        var_rls_history.append(float(var_f))

        step_num = (chunk + 1) * log_interval
        last_elbo = float(elbos_chunk[-1])
        print(
            f"  Step {step_num}: ELBO={last_elbo:.2f}, R^2={r2_val:.4f}, var[RLS]={var_f:.4f}, ||rho||={rho_norm:.2f}"
        )

    # Handle remainder steps
    if remainder > 0:
        if use_analytical_rho:
            carry = (current_hrm_params, current_opt_state, train_key)
            carry, (elbos_chunk, rho_stars_chunk) = jax.lax.scan(
                train_step_analytical, carry, None, length=remainder
            )
            current_hrm_params, current_opt_state, train_key = carry
            current_params = model.join_coords(rho_stars_chunk[-1], current_hrm_params)
            all_elbos.append(elbos_chunk)
            all_conj_errors.append(jnp.zeros(remainder))
        elif use_conj_penalty:
            carry = (current_params, current_opt_state, train_key)
            carry, (elbos_chunk, conj_vars_chunk) = jax.lax.scan(
                train_step_regularized, carry, None, length=remainder
            )
            current_params, current_opt_state, train_key = carry
            all_elbos.append(elbos_chunk)
            all_conj_errors.append(conj_vars_chunk)
        else:
            carry = (current_params, current_opt_state, train_key)
            carry, elbos_chunk = jax.lax.scan(
                train_step_free, carry, None, length=remainder
            )
            current_params, current_opt_state, train_key = carry
            all_elbos.append(elbos_chunk)
            all_conj_errors.append(jnp.zeros(remainder))

    # Flatten collected arrays
    elbos = jnp.concatenate(all_elbos).tolist()
    conjugation_errors = jnp.concatenate(all_conj_errors).tolist()

    # Final evaluation
    print("\nFinal evaluation...")

    # For analytical mode, recompute rho with many samples for stable evaluation
    if use_analytical_rho:
        key, rho_eval_key = jax.random.split(key)
        zero_rho = jnp.zeros(model.rho_man.dim)
        eval_params = model.join_coords(zero_rho, current_hrm_params)
        rho_final, _, _, _ = model.regress_conjugation_parameters(
            rho_eval_key, eval_params, n_conj_samples * 2
        )
        current_params = model.join_coords(rho_final, current_hrm_params)

    key, eval_key = jax.random.split(key)

    final_recon_error = float(model.reconstruction_error(current_params, train_data))
    final_var_rls, _, final_r2 = model.conjugation_metrics(
        eval_key, current_params, n_conj_samples
    )
    final_rho, _ = model.split_coords(current_params)
    final_rho_norm = float(jnp.linalg.norm(final_rho))

    print(f"  Final ELBO: {elbos[-1]:.4f}")
    print(f"  Final R^2: {float(final_r2):.4f}")
    print(f"  Final var[RLS]: {float(final_var_rls):.4f}")
    print(f"  Final ||rho||: {final_rho_norm:.4f}")
    print(f"  Reconstruction error: {final_recon_error:.4f}")

    # Extract learned parameters
    _, hrm_p = model.split_coords(current_params)
    learned_baselines, learned_int_params, _ = model.hrm.split_coords(hrm_p)
    learned_weights = learned_int_params.reshape(model.n_neurons, 2 * model.n_latent)
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

    # Hyperparameters
    n_neurons = 64  # Square number for grid
    n_latent = 2  # T^2 (2-torus)
    n_samples = 10000  # Large sample for stable training
    n_steps = 4000
    batch_size = 256
    learning_rate = 5e-4  # Lower LR for stability
    n_mc_samples = 10
    seed = 42

    # Ground truth parameters
    baseline_rate = 2.0
    gain = 1.5
    prior_concentration = 0.1
    distortion = 0.0  # Zero = perfect convolutional structure (conjugated)
    coverage = 1.0  # Fraction of torus covered (1.0=full, 0.5=half -> non-zero rho)
    # Density modulation: von Mises concentration of neuron preferred locations
    density_kappa1 = 0.0  # 0 = uniform, >0 = concentrated around density_mu1
    density_kappa2 = 0.0  # 0 = uniform, >0 = concentrated around density_mu2
    density_mu1 = 0.0  # Center of concentration in theta1
    density_mu2 = 0.0  # Center of concentration in theta2

    # Training mode parameters
    conj_weight = 1.0  # Weight for var[RLS] penalty in regularized mode
    analytical_rho_samples_multiplier = 500  # Enough samples for stable lstsq regression

    # Logging
    log_interval = 200
    n_conj_samples = 10000  # Large sample for stable var[RLS] evaluation

    print("=" * 60)
    print("POISSON-VONMISES VARIATIONAL HARMONIUM")
    print("=" * 60)
    print(f"  n_neurons: {n_neurons}")
    print(f"  n_latent: {n_latent}")
    print(f"  n_samples: {n_samples}")
    print(f"  n_steps: {n_steps}")
    print(f"  distortion: {distortion}")
    print(f"  coverage: {coverage}")
    print(f"  baseline_rate: {baseline_rate}")
    print(f"  gain: {gain}")
    print("=" * 60)

    key = jax.random.PRNGKey(seed)

    # Create ground truth model
    print("\nCreating ground truth model...")
    key, gt_key = jax.random.split(key)
    gt_model, gt_params, gt_tuning = create_ground_truth_model(
        n_neurons=n_neurons,
        n_latent=n_latent,
        key=gt_key,
        baseline_rate=baseline_rate,
        gain=gain,
        prior_concentration=prior_concentration,
        distortion=distortion,
        coverage=coverage,
        density_kappa1=density_kappa1,
        density_kappa2=density_kappa2,
        density_mu1=density_mu1,
        density_mu2=density_mu2,
    )

    # Generate synthetic data from ground truth
    print("Generating synthetic data from ground truth model...")
    key, sample_key = jax.random.split(key)
    _, _, gt_prior_params = gt_model.split_coords(gt_params)
    z_samples = gt_model.pst_man.sample(sample_key, gt_prior_params, n_samples)

    key, spike_key = jax.random.split(key)
    spike_keys = jax.random.split(spike_key, n_samples)

    def sample_spikes(subkey: Array, z: Array) -> Array:
        lkl_params = gt_model.likelihood_at(gt_params, z)
        rates = gt_model.obs_man.to_mean(lkl_params)
        return jax.random.poisson(subkey, rates).astype(jnp.float32)

    spike_counts = jax.vmap(sample_spikes)(spike_keys, z_samples)

    print(f"  Spike counts shape: {spike_counts.shape}")
    print(f"  Mean spike count: {jnp.mean(spike_counts):.2f}")
    print(f"  Max spike count: {jnp.max(spike_counts):.0f}")

    # Compute ground truth conjugation metrics using library methods
    print("\nComputing ground truth conjugation metrics...")
    key, gt_conj_key = jax.random.split(key)
    gt_conjugation = compute_gt_conjugation(
        gt_model, gt_params, gt_conj_key, n_samples=n_conj_samples * 2
    )
    print(f"  GT R^2 (with optimal rho): {gt_conjugation['r_squared']:.4f}")
    print(f"  GT var[RLS]: {gt_conjugation['var_rls']:.4f}")
    print(f"  GT var[psi_X]: {gt_conjugation['var_psi']:.4f}")
    print(f"  GT ||rho_optimal||: {gt_conjugation['optimal_rho_norm']:.4f}")

    # Extract ground truth info
    gt_baselines, gt_int_params, gt_prior = gt_model.split_coords(gt_params)
    gt_weights = gt_int_params.reshape(gt_model.n_neurons, 2 * gt_model.n_latent)

    ground_truth: GroundTruth = {
        "weight_matrix": gt_weights.tolist(),
        "baselines": gt_baselines.tolist(),
        "prior_params": gt_prior.tolist(),
        "tuning": gt_tuning,
        "conjugation": gt_conjugation,
    }

    # Create variational model for training
    model = VonMisesPopulationCode(hrm=PoissonVonMisesHarmonium(n_neurons, n_latent))
    print("\nVariational model created:")
    print(f"  Total params: {model.dim}")
    print(f"  Rho params: {model.rho_man.dim}")
    print(f"  Harmonium params: {model.hrm.dim}")

    # Train all modes
    modes = ["free", "regularized", "analytical"]
    all_results: dict[str, ModeResults] = {}
    all_params: dict[str, Array] = {}

    # Pre-compute shared initialization so all modes start from the same point
    key, init_key = jax.random.split(key)
    shared_init_params = model.initialize_from_sample(
        init_key, spike_counts, location=0.0, shape=0.1
    )

    for mode_idx, mode in enumerate(modes):
        train_key = jax.random.PRNGKey(seed + mode_idx + 1)
        params, results = train_model(
            key=train_key,
            model=model,
            train_data=spike_counts,
            mode=mode,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_mc_samples=n_mc_samples,
            init_params=shared_init_params,
            conj_weight=conj_weight,
            analytical_samples_mult=analytical_rho_samples_multiplier,
            n_conj_samples=n_conj_samples,
            log_interval=log_interval,
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
            "n_neurons": n_neurons,
            "n_latent": n_latent,
            "n_samples": n_samples,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "n_mc_samples": n_mc_samples,
            "seed": seed,
            "distortion": distortion,
            "coverage": coverage,
            "baseline_rate": baseline_rate,
            "gain": gain,
            "prior_concentration": prior_concentration,
            "conj_weight": conj_weight,
            "analytical_samples_mult": analytical_rho_samples_multiplier,
            "density_kappa1": density_kappa1,
            "density_kappa2": density_kappa2,
            "density_mu1": density_mu1,
            "density_mu2": density_mu2,
        },
    }

    paths.save_analysis(output)
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
