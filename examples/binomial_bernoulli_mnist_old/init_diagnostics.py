"""Diagnostic script for conjugation metrics at initialization.

This script examines how Var[f̃], Std[f̃], and R² behave at initialization
as we vary the number of latent neurons. No training is performed.

The key quantities:
- ψ_X(y) = log partition function of observable at latent y
- f̃ = χ + ρ · s_Y(y) - ψ_X(y)  (residual after linear approximation)
- We fit: ψ_X ≈ χ + ρ · s_Y  via least squares
- Var[f̃] = residual variance (amortization gap)
- Std[f̃] = sqrt(Var[f̃])
- R² = 1 - Var[f̃] / Var[ψ_X]

Math check:
- If ψ_X is perfectly linear in s_Y, then R² = 1, Var[f̃] = 0
- If ψ_X is unrelated to s_Y, then R² ≈ 0, Var[f̃] ≈ Var[ψ_X]

Usage:
    python -m examples.binomial_bernoulli_mnist.init_diagnostics
"""

import jax
import jax.numpy as jnp
from jax import Array

from goal.models import binomial_bernoulli_mixture

# Configuration
N_OBSERVABLE = 784  # MNIST dimensions
N_CLUSTERS = 10
N_TRIALS = 16
N_SEEDS = 5  # Multiple seeds for robustness
INIT_SHAPE = 0.1  # Default initialization scale


def compute_conjugation_metrics(
    model,
    hrm_params: Array,
    y_samples: Array,
) -> dict[str, float]:
    """Compute conjugation metrics given samples.

    Returns dict with:
    - r2: R² goodness of fit
    - var_f: Var[f̃] (residual variance)
    - std_f: Std[f̃] = sqrt(Var[f̃])
    - var_psi: Var[ψ_X] (total variance of log partition)
    - std_psi: Std[ψ_X]
    - chi: intercept from regression
    - rho_norm: ||ρ||₂
    """
    # Extract harmonium components
    obs_bias, int_params, _ = model.hrm.split_coords(hrm_params)
    int_matrix = int_params.reshape(model.n_observable, model.n_latent)

    # Compute sufficient statistics for all y samples
    s_y_samples = jax.vmap(model.bas_lat_man.sufficient_statistic)(y_samples)

    # Compute ψ_X at each sample point
    def compute_psi_at_y(y: Array) -> Array:
        s_y = model.bas_lat_man.sufficient_statistic(y)
        lkl_nat = obs_bias + int_matrix @ s_y
        return model.obs_man.log_partition_function(lkl_nat)

    psi_values = jax.vmap(compute_psi_at_y)(y_samples)

    # Build design matrix [1, s_Y(y)]
    n_samples = y_samples.shape[0]
    ones = jnp.ones((n_samples, 1))
    design = jnp.concatenate([ones, s_y_samples], axis=1)

    # Solve least squares: ψ ≈ χ + ρ · s_Y
    coeffs = jnp.linalg.lstsq(design, psi_values, rcond=None)[0]
    chi = coeffs[0]
    rho = coeffs[1:]

    # Compute predictions and residuals
    predicted = design @ coeffs
    residuals = psi_values - predicted  # This is f̃

    # Compute statistics
    ss_res = jnp.sum(residuals ** 2)
    ss_tot = jnp.sum((psi_values - jnp.mean(psi_values)) ** 2)

    var_f = jnp.var(residuals)  # Var[f̃]
    std_f = jnp.sqrt(var_f)
    var_psi = jnp.var(psi_values)  # Var[ψ_X]
    std_psi = jnp.sqrt(var_psi)

    # R² with numerical stability
    r2 = 1.0 - ss_res / jnp.maximum(ss_tot, 1e-10)
    r2 = jnp.clip(r2, -10.0, 1.0)

    return {
        "r2": float(r2),
        "var_f": float(var_f),
        "std_f": float(std_f),
        "var_psi": float(var_psi),
        "std_psi": float(std_psi),
        "chi": float(chi),
        "rho_norm": float(jnp.linalg.norm(rho)),
        "mean_psi": float(jnp.mean(psi_values)),
    }


def sample_from_prior(key: Array, model, hrm_params: Array, n_samples: int) -> Array:
    """Sample y from the prior distribution."""
    _, _, prior_mixture_params = model.hrm.split_coords(hrm_params)
    y_samples, _ = model.sample_from_posterior(key, prior_mixture_params, n_samples)
    return y_samples


def run_diagnostics(
    n_latent_values: list[int],
    n_samples_mult: int = 10,
    init_shape: float = INIT_SHAPE,
    seed: int = 42,
):
    """Run diagnostics for various latent population sizes."""

    print("=" * 80)
    print("CONJUGATION METRICS AT INITIALIZATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  n_observable = {N_OBSERVABLE}")
    print(f"  n_clusters = {N_CLUSTERS}")
    print(f"  n_trials = {N_TRIALS}")
    print(f"  init_shape = {init_shape}")
    print(f"  n_samples = {n_samples_mult} × n_latent")
    print(f"  n_seeds = {N_SEEDS}")
    print()

    # Header - focusing on Std[f̃] and Var[f̃]
    print(f"{'n_lat':>6} | {'Var[f̃]':>12} | {'Std[f̃]':>10} | {'Var[ψ]':>12} | {'Std[ψ]':>10} | {'||ρ||':>10} | {'χ':>12}")
    print("-" * 85)

    results = []

    for n_latent in n_latent_values:
        # Number of samples must be >> n_latent for overdetermined system
        n_samples = n_samples_mult * n_latent

        # Aggregate over multiple seeds
        var_f_list, std_f_list = [], []
        var_psi_list, std_psi_list = [], []
        rho_norm_list, chi_list = [], []

        for seed_offset in range(N_SEEDS):
            key = jax.random.PRNGKey(seed + seed_offset * 1000)
            key_init, key_sample = jax.random.split(key)

            # Create model
            model = binomial_bernoulli_mixture(
                n_observable=N_OBSERVABLE,
                n_latent=n_latent,
                n_clusters=N_CLUSTERS,
                n_trials=N_TRIALS,
            )

            # Initialize parameters
            params = model.initialize(key_init, shape=init_shape)
            _, hrm_params = model.split_coords(params)

            # Sample from prior
            y_samples = sample_from_prior(key_sample, model, hrm_params, n_samples)

            # Compute metrics
            metrics = compute_conjugation_metrics(model, hrm_params, y_samples)

            var_f_list.append(metrics["var_f"])
            std_f_list.append(metrics["std_f"])
            var_psi_list.append(metrics["var_psi"])
            std_psi_list.append(metrics["std_psi"])
            rho_norm_list.append(metrics["rho_norm"])
            chi_list.append(metrics["chi"])

        # Compute means
        var_f_mean = sum(var_f_list) / len(var_f_list)
        std_f_mean = sum(std_f_list) / len(std_f_list)
        var_psi_mean = sum(var_psi_list) / len(var_psi_list)
        std_psi_mean = sum(std_psi_list) / len(std_psi_list)
        rho_norm_mean = sum(rho_norm_list) / len(rho_norm_list)
        chi_mean = sum(chi_list) / len(chi_list)

        print(f"{n_latent:>6} | {var_f_mean:>12.4f} | {std_f_mean:>10.4f} | {var_psi_mean:>12.4f} | {std_psi_mean:>10.4f} | {rho_norm_mean:>10.4f} | {chi_mean:>12.2f}")

        results.append({
            "n_latent": n_latent,
            "n_samples": n_samples,
            "var_f": var_f_mean,
            "std_f": std_f_mean,
            "var_psi": var_psi_mean,
            "std_psi": std_psi_mean,
            "rho_norm": rho_norm_mean,
            "chi": chi_mean,
        })

    print("-" * 85)

    return results


def detailed_analysis_single(n_latent: int, seed: int = 42):
    """Detailed analysis for a single n_latent value."""

    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS: n_latent = {n_latent}")
    print(f"{'='*80}")

    key = jax.random.PRNGKey(seed)
    key_init, key_sample = jax.random.split(key)

    # Create model
    model = binomial_bernoulli_mixture(
        n_observable=N_OBSERVABLE,
        n_latent=n_latent,
        n_clusters=N_CLUSTERS,
        n_trials=N_TRIALS,
    )

    # Initialize parameters
    params = model.initialize(key_init, shape=INIT_SHAPE)
    _, hrm_params = model.split_coords(params)

    # Extract components for analysis
    obs_bias, int_params, prior_params = model.hrm.split_coords(hrm_params)
    int_matrix = int_params.reshape(model.n_observable, model.n_latent)

    print(f"\nParameter shapes:")
    print(f"  obs_bias: {obs_bias.shape} (dim={obs_bias.shape[0]})")
    print(f"  int_matrix: {int_matrix.shape}")
    print(f"  prior_params: {prior_params.shape}")

    print(f"\nParameter statistics:")
    print(f"  ||obs_bias||₂ = {jnp.linalg.norm(obs_bias):.4f}")
    print(f"  ||int_matrix||_F = {jnp.linalg.norm(int_matrix):.4f}")
    print(f"  ||int_matrix||_F / sqrt(n_obs × n_lat) = {jnp.linalg.norm(int_matrix) / jnp.sqrt(N_OBSERVABLE * n_latent):.4f}")
    print(f"  mean(int_matrix) = {jnp.mean(int_matrix):.6f}")
    print(f"  std(int_matrix) = {jnp.std(int_matrix):.6f}")

    # Compute ψ_X properties
    n_samples = 10 * n_latent
    y_samples = sample_from_prior(key_sample, model, hrm_params, n_samples)

    def compute_psi_at_y(y: Array) -> Array:
        s_y = model.bas_lat_man.sufficient_statistic(y)
        lkl_nat = obs_bias + int_matrix @ s_y
        return model.obs_man.log_partition_function(lkl_nat)

    psi_values = jax.vmap(compute_psi_at_y)(y_samples)

    print(f"\nψ_X statistics over {n_samples} samples:")
    print(f"  mean(ψ_X) = {jnp.mean(psi_values):.4f}")
    print(f"  std(ψ_X) = {jnp.std(psi_values):.4f}")
    print(f"  min(ψ_X) = {jnp.min(psi_values):.4f}")
    print(f"  max(ψ_X) = {jnp.max(psi_values):.4f}")

    # Compute natural parameters at samples
    def compute_lkl_nat_at_y(y: Array) -> Array:
        s_y = model.bas_lat_man.sufficient_statistic(y)
        return obs_bias + int_matrix @ s_y

    lkl_nats = jax.vmap(compute_lkl_nat_at_y)(y_samples)

    print(f"\nLikelihood natural params θ_X(y) statistics:")
    print(f"  mean(θ_X) = {jnp.mean(lkl_nats):.4f}")
    print(f"  std(θ_X) = {jnp.std(lkl_nats):.4f}")
    print(f"  min(θ_X) = {jnp.min(lkl_nats):.4f}")
    print(f"  max(θ_X) = {jnp.max(lkl_nats):.4f}")

    # Recall: For Binomial(n_trials, p), ψ(θ) = n_trials * log(1 + exp(θ))
    # This is n_trials × softplus, which is nearly linear for large |θ|
    print(f"\nNote: ψ_X(θ) = {N_TRIALS} × log(1 + exp(θ)) [Binomial log-partition]")
    print(f"  Nearly linear when |θ| >> 1")

    # Compute metrics
    metrics = compute_conjugation_metrics(model, hrm_params, y_samples)

    print(f"\nConjugation metrics (f̃ = χ + ρ·s_Y - ψ_X):")
    print(f"  Var[f̃] = {metrics['var_f']:.6f}")
    print(f"  Std[f̃] = {metrics['std_f']:.6f}")
    print(f"  Var[ψ_X] = {metrics['var_psi']:.6f}")
    print(f"  Std[ψ_X] = {metrics['std_psi']:.6f}")
    print(f"  χ (intercept) = {metrics['chi']:.4f}")
    print(f"  ||ρ*||₂ = {metrics['rho_norm']:.4f}")

    # Relative residual
    print(f"\nRelative residual:")
    ratio = metrics['std_f'] / metrics['std_psi'] if metrics['std_psi'] > 0 else 0
    print(f"  Std[f̃] / Std[ψ_X] = {ratio:.6f}  (fraction of variance unexplained)")


def run_varying_init_scale(n_latent: int = 64, seed: int = 42):
    """Examine how conjugation metrics depend on initialization scale."""

    print(f"\n{'='*80}")
    print(f"EFFECT OF INITIALIZATION SCALE (n_latent = {n_latent})")
    print(f"{'='*80}")

    init_shapes = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\n{'shape':>8} | {'Var[f̃]':>12} | {'Std[f̃]':>10} | {'Var[ψ]':>12} | {'Std[ψ]':>10} | {'||W||_F':>10}")
    print("-" * 75)

    for init_shape in init_shapes:
        key = jax.random.PRNGKey(seed)
        key_init, key_sample = jax.random.split(key)

        model = binomial_bernoulli_mixture(
            n_observable=N_OBSERVABLE,
            n_latent=n_latent,
            n_clusters=N_CLUSTERS,
            n_trials=N_TRIALS,
        )

        params = model.initialize(key_init, shape=init_shape)
        _, hrm_params = model.split_coords(params)

        _, int_params, _ = model.hrm.split_coords(hrm_params)
        int_matrix = int_params.reshape(model.n_observable, model.n_latent)

        n_samples = 10 * n_latent
        y_samples = sample_from_prior(key_sample, model, hrm_params, n_samples)

        metrics = compute_conjugation_metrics(model, hrm_params, y_samples)
        w_norm = float(jnp.linalg.norm(int_matrix))

        print(f"{init_shape:>8.2f} | {metrics['var_f']:>12.4f} | {metrics['std_f']:>10.4f} | {metrics['var_psi']:>12.4f} | {metrics['std_psi']:>10.4f} | {w_norm:>10.4f}")


def analyze_softplus_linearity():
    """Analyze why Binomial-Bernoulli is inherently conjugate."""

    print(f"\n{'='*80}")
    print("WHY BINOMIAL-BERNOULLI IS INHERENTLY CONJUGATE")
    print(f"{'='*80}")

    print("""
The Binomial log-partition function is:
    ψ(θ) = n × log(1 + exp(θ)) = n × softplus(θ)

The softplus function:
    softplus(θ) = log(1 + exp(θ))

is famously "almost linear":
    - For θ << 0: softplus(θ) ≈ 0
    - For θ >> 0: softplus(θ) ≈ θ
    - Near θ = 0: softplus(θ) ≈ log(2) + θ/2 + O(θ²)

Let's see how linear it is at different scales:
""")

    # Analyze softplus curvature
    thetas = jnp.linspace(-5, 5, 1000)
    psi_single = 16 * jnp.log(1 + jnp.exp(thetas))  # n_trials = 16

    # Fit linear model to softplus
    design = jnp.stack([jnp.ones_like(thetas), thetas], axis=1)
    coeffs = jnp.linalg.lstsq(design, psi_single, rcond=None)[0]
    psi_linear = coeffs[0] + coeffs[1] * thetas
    residual = psi_single - psi_linear

    print(f"  Linear fit to ψ(θ) = 16 × softplus(θ) over θ ∈ [-5, 5]:")
    print(f"    ψ ≈ {coeffs[0]:.4f} + {coeffs[1]:.4f} × θ")
    print(f"    max|residual| = {float(jnp.max(jnp.abs(residual))):.4f}")
    print(f"    R² = {float(1 - jnp.var(residual) / jnp.var(psi_single)):.6f}")

    # Key insight: sum over many units
    print(f"""
KEY INSIGHT: The total log-partition is a SUM over {N_OBSERVABLE} observables.

For each observable i: ψᵢ(θᵢ) = n × softplus(θᵢ)

Total: Ψ(θ) = Σᵢ ψᵢ(θᵢ)

Since θᵢ = bias_i + Σⱼ wᵢⱼ × yⱼ, each θᵢ varies with y in a linear way.

The non-linearity (curvature of softplus) only matters when θᵢ ≈ 0.
But θ values are distributed across a range, so:
    - Many θᵢ << 0: contribute ≈ 0 (dead pixels)
    - Many θᵢ >> 0: contribute ≈ n × θᵢ (active pixels, linear)
    - Some θᵢ ≈ 0: contribute curvature

The curvature from the few θᵢ ≈ 0 terms is drowned out by the linear terms.
""")

    # Simulate this effect
    print("Simulation: Sum of softplus over many units")
    print("-" * 50)

    def test_regime(biases, weights, n_lat, label):
        """Test a specific parameter regime."""
        n_obs = len(biases)
        n_trials = 16

        key = jax.random.PRNGKey(42)
        n_samples = 10 * n_lat
        y_samples = jax.random.bernoulli(key, 0.5, (n_samples, n_lat)).astype(jnp.float32)

        def total_psi(y):
            theta = biases + weights @ y
            return n_trials * jnp.sum(jnp.log(1 + jnp.exp(theta)))

        psi_values = jax.vmap(total_psi)(y_samples)

        design = jnp.concatenate([jnp.ones((n_samples, 1)), y_samples], axis=1)
        coeffs = jnp.linalg.lstsq(design, psi_values, rcond=None)[0]
        predicted = design @ coeffs
        residuals = psi_values - predicted

        r2 = float(1 - jnp.var(residuals) / jnp.var(psi_values))
        std_f = float(jnp.std(residuals))
        std_psi = float(jnp.std(psi_values))

        # Distribution of theta values
        theta_at_sample = biases + weights @ y_samples[0]
        pct_curved = int(jnp.sum(jnp.abs(theta_at_sample) < 1)) / n_obs * 100

        print(f"\n  {label}:")
        print(f"    n_observable={n_obs}, n_latent={n_lat}")
        print(f"    R² = {r2:.6f}, Std[f̃] = {std_f:.4f}, Std[ψ] = {std_psi:.4f}")
        print(f"    θ distribution: mean={float(jnp.mean(theta_at_sample)):.2f}, std={float(jnp.std(theta_at_sample)):.2f}")
        print(f"    {pct_curved:.0f}% of θ values in curved region (|θ| < 1)")
        return r2

    n_obs = 784
    n_lat = 64
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Case 1: Large biases (original) - most θ in linear region
    biases1 = jax.random.normal(key1, (n_obs,)) * 2
    weights1 = jax.random.normal(key2, (n_obs, n_lat)) * 0.1
    test_regime(biases1, weights1, n_lat, "Large biases (std=2), small weights (std=0.1)")

    # Case 2: Zero biases, small weights - all θ near 0
    biases2 = jnp.zeros(n_obs)
    weights2 = jax.random.normal(key2, (n_obs, n_lat)) * 0.1
    test_regime(biases2, weights2, n_lat, "Zero biases, small weights (std=0.1)")

    # Case 3: Zero biases, tiny weights - θ ≈ 0 for all
    biases3 = jnp.zeros(n_obs)
    weights3 = jax.random.normal(key2, (n_obs, n_lat)) * 0.01
    test_regime(biases3, weights3, n_lat, "Zero biases, tiny weights (std=0.01)")

    # Case 4: Uniform biases at θ=0 (all in curved region), larger weights
    biases4 = jnp.zeros(n_obs)
    weights4 = jax.random.normal(key2, (n_obs, n_lat)) * 0.5
    test_regime(biases4, weights4, n_lat, "Zero biases, larger weights (std=0.5)")

    # Case 5: Single observable (extreme test)
    print("\n  --- Extreme test: n_observable = 1 ---")
    biases5 = jnp.zeros(1)
    weights5 = jax.random.normal(key2, (1, n_lat)) * 0.5
    test_regime(biases5, weights5, n_lat, "Single observable, zero bias")

    # Case 6: Few observables - this is where linearization fails!
    print("\n  --- Effect of n_observable (this is where linearization breaks) ---")
    print(f"\n  {'n_obs':>8} | {'R²':>8} | {'Std[f̃]':>10} | {'Std[ψ]':>10}")
    print("  " + "-" * 45)
    for n_obs_small in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 784]:
        biases_s = jnp.zeros(n_obs_small)
        key_w = jax.random.PRNGKey(123)
        weights_s = jax.random.normal(key_w, (n_obs_small, n_lat)) * 0.5
        key_y = jax.random.PRNGKey(456)
        n_samples = 10 * n_lat
        y_samples = jax.random.bernoulli(key_y, 0.5, (n_samples, n_lat)).astype(jnp.float32)

        def total_psi(y):
            theta = biases_s + weights_s @ y
            return 16 * jnp.sum(jnp.log(1 + jnp.exp(theta)))

        psi_values = jax.vmap(total_psi)(y_samples)
        design = jnp.concatenate([jnp.ones((n_samples, 1)), y_samples], axis=1)
        coeffs = jnp.linalg.lstsq(design, psi_values, rcond=None)[0]
        predicted = design @ coeffs
        residuals = psi_values - predicted
        r2 = float(1 - jnp.var(residuals) / jnp.var(psi_values))
        std_f = float(jnp.std(residuals))
        std_psi = float(jnp.std(psi_values))
        print(f"  {n_obs_small:>8} | {r2:>8.4f} | {std_f:>10.2f} | {std_psi:>10.2f}")

    # Single softplus analysis - force θ to be in curved region
    print("\n  --- Single softplus: θ forced into curved region [0, 1] ---")
    n_samples = 10000
    key_y = jax.random.PRNGKey(789)

    # Weight chosen so that θ = 0.5 * mean(y) stays around 0.25
    # y ~ Bernoulli(0.5), so E[θ] ≈ 0.25
    weight_single = 0.5
    y_1d = jax.random.bernoulli(key_y, 0.5, (n_samples,)).astype(jnp.float32)

    # θ = 0 + weight * y, where y ∈ {0, 1}
    theta_vals = weight_single * y_1d  # θ ∈ {0, 0.5}
    psi_vals = 16 * jnp.log(1 + jnp.exp(theta_vals))

    # Fit linear: ψ ≈ a + b*y
    design_1d = jnp.stack([jnp.ones_like(y_1d), y_1d], axis=1)
    coeffs_1d = jnp.linalg.lstsq(design_1d, psi_vals, rcond=None)[0]
    pred_1d = design_1d @ coeffs_1d
    res_1d = psi_vals - pred_1d

    r2_1d = float(1 - jnp.var(res_1d) / jnp.var(psi_vals))

    print(f"  θ ∈ {{0, 0.5}} (fully in curved region)")
    print(f"  ψ(0) = {float(16 * jnp.log(2)):.4f}")
    print(f"  ψ(0.5) = {float(16 * jnp.log(1 + jnp.exp(0.5))):.4f}")
    print(f"  Linear fit: ψ ≈ {float(coeffs_1d[0]):.4f} + {float(coeffs_1d[1]):.4f} × y")
    print(f"  R² = {r2_1d:.6f}")
    print(f"  Std[residual] = {float(jnp.std(res_1d)):.6f}")

    # Ground truth analysis
    psi_0 = 16 * jnp.log(2)  # ψ(0)
    psi_half = 16 * jnp.log(1 + jnp.exp(0.5))  # ψ(0.5)
    # If perfectly linear: ψ(θ) = a + b*θ
    # Then ψ(0) = a, ψ(0.5) = a + 0.5*b
    # So b = 2*(ψ(0.5) - ψ(0)) = 2*(ψ(0.5) - psi_0)
    b_linear = 2 * (psi_half - psi_0)
    print(f"\n  If ψ were perfectly linear in θ:")
    print(f"    slope b = 2*(ψ(0.5) - ψ(0)) = {float(b_linear):.4f}")
    print(f"    But actual slope of softplus at θ=0.25 is: 16 × σ(0.25) = {float(16 / (1 + jnp.exp(-0.25))):.4f}")
    print(f"    Discrepancy comes from curvature of softplus")

    print("""
  KEY INSIGHT: For a SINGLE observable with θ in [0, 1]:
  - R² ≈ 1.0 because the linear fit absorbs most variance
  - But the residual is small ONLY because the range [0, 0.5] is small
  - The RELATIVE curvature (Std[f̃]/Std[ψ]) reveals the non-linearity

  The true test is: what happens when θ has LARGE variance while staying near 0?
""")

    # More revealing test: multiple y dimensions, θ sums up but stays curved
    print("  --- n_lat=64 latents, n_obs=1, with θ centered at 0 ---")
    n_lat = 64
    # Weights scaled so that θ = sum_j w_j * y_j has std ≈ 1 when y ~ Bernoulli(0.5)
    # Var[θ] = sum_j w_j² * Var[y_j] = sum_j w_j² * 0.25
    # Want std(θ) ≈ 1, so sum_j w_j² ≈ 4, so w_j ≈ sqrt(4/64) ≈ 0.25
    key_w = jax.random.PRNGKey(999)
    weights_centered = jax.random.normal(key_w, (1, n_lat)) * 0.25

    n_samples = 640
    y_samples = jax.random.bernoulli(key_y, 0.5, (n_samples, n_lat)).astype(jnp.float32)

    theta_centered = (weights_centered @ y_samples.T).flatten()  # Shape (n_samples,)
    psi_centered = 16 * jnp.log(1 + jnp.exp(theta_centered))

    # Fit linear in y
    design_full = jnp.concatenate([jnp.ones((n_samples, 1)), y_samples], axis=1)
    coeffs_full = jnp.linalg.lstsq(design_full, psi_centered, rcond=None)[0]
    pred_full = design_full @ coeffs_full
    res_full = psi_centered - pred_full

    _ = float(1 - jnp.var(res_full) / jnp.var(psi_centered))

    print(f"  θ distribution: mean={float(jnp.mean(theta_centered)):.2f}, std={float(jnp.std(theta_centered)):.2f}")
    print(f"  {int(jnp.sum(jnp.abs(theta_centered) < 1)) / n_samples * 100:.0f}% of θ in curved region")
    var_f_full = float(jnp.var(res_full))
    std_f_full = float(jnp.std(res_full))
    var_psi_full = float(jnp.var(psi_centered))
    std_psi_full = float(jnp.std(psi_centered))

    print(f"  Var[f̃] = {var_f_full:.4f}")
    print(f"  Std[f̃] = {std_f_full:.4f}")
    print(f"  Var[ψ] = {var_psi_full:.4f}")
    print(f"  Std[ψ] = {std_psi_full:.4f}")
    print(f"  Ratio Std[f̃]/Std[ψ] = {std_f_full / std_psi_full:.4f} ({std_f_full / std_psi_full * 100:.1f}% unexplained)")

    # Final summary
    print("""
================================================================================
SUMMARY: WHEN DOES BINOMIAL-BERNOULLI CONJUGATION FAIL?
================================================================================

CONJUGATION WORKS (Std[f̃]/Std[ψ] < 0.05) WHEN:
  1. Weights are small → θ barely varies → ψ is "trivially linear"
  2. n_observable is large (e.g., 784 for MNIST) → law of large numbers
     averages out the curvature of individual softplus functions

CONJUGATION FAILS (Std[f̃]/Std[ψ] > 0.2) WHEN:
  1. n_observable is small (e.g., 1-16 observables)
  2. θ values are centered near 0 (curved region of softplus)
  3. θ has significant variance (std > 1)

FOR MNIST-SIZED MODELS (n_obs = 784):
  The architecture is INHERENTLY nearly-conjugate due to the law of large
  numbers. The curvature from individual softplus(θ_i) terms averages out.
  This explains why training works well without explicit conjugation penalties.

IMPLICATION:
  If you want to study conjugation/non-conjugation regimes, you need either:
  - A model with fewer observables
  - A different likelihood function with stronger curvature
""")


if __name__ == "__main__":
    # Initialize JAX
    print("Initializing JAX...")
    jax.config.update("jax_platform_name", "cpu")  # Use CPU for reproducibility

    # Test with various latent population sizes, including n_latent=1
    n_latent_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    # Run main diagnostics
    results = run_diagnostics(n_latent_values)

    # Detailed analysis for specific cases
    detailed_analysis_single(n_latent=1)
    detailed_analysis_single(n_latent=16)
    detailed_analysis_single(n_latent=256)

    # Effect of initialization scale
    run_varying_init_scale(n_latent=64)
    run_varying_init_scale(n_latent=256)

    # Why is this model inherently conjugate?
    analyze_softplus_linearity()
