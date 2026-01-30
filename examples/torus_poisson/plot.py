"""Plotting for Poisson-VonMises variational harmonium example.

Creates a 4-panel abstract-ready figure:
1. TL: ELBO training curves
2. TR: Conjugation quality (R² or var[RLS])
3. BL: Tuning curve location recovery
4. BR: Conjugate Bayesian inference (evidence accumulation)

Usage:
    python -m examples.torus_poisson.plot
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from ..shared import apply_style, colors, example_paths, model_colors


def circular_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute minimum angular distance, handling wrap-around."""
    diff = np.abs(a - b)
    return np.minimum(diff, 2 * np.pi - diff)


def find_optimal_matching(
    gt_theta1: NDArray[np.float64],
    gt_theta2: NDArray[np.float64],
    learned_theta1: NDArray[np.float64],
    learned_theta2: NDArray[np.float64],
    n_offsets: int = 16,
) -> tuple[NDArray[np.intp], float, float, bool]:
    """Find optimal neuron matching accounting for torus symmetries.

    Accounts for:
    - Rotational symmetry in each dimension (θ → θ + offset)
    - Dimension permutation symmetry (θ₁ ↔ θ₂)

    Uses Hungarian algorithm for optimal assignment.

    Args:
        gt_theta1, gt_theta2: Ground truth preferred angles
        learned_theta1, learned_theta2: Learned preferred angles
        n_offsets: Number of offsets to search in each dimension

    Returns:
        Tuple of (matching_indices, best_offset1, best_offset2, swapped)
        matching_indices[i] = j means GT neuron i matches learned neuron j
    """
    n_neurons = len(gt_theta1)
    best_cost = float('inf')
    best_matching = np.arange(n_neurons)
    best_offset1, best_offset2 = 0.0, 0.0
    best_swapped = False

    offsets = np.linspace(0, 2 * np.pi, n_offsets, endpoint=False)

    # Try both dimension orderings
    for swapped in [False, True]:
        if swapped:
            l1, l2 = learned_theta2, learned_theta1
        else:
            l1, l2 = learned_theta1, learned_theta2

        for offset1 in offsets:
            for offset2 in offsets:
                # Apply offsets
                shifted1 = np.mod(l1 + offset1, 2 * np.pi)
                shifted2 = np.mod(l2 + offset2, 2 * np.pi)

                # Compute cost matrix using circular distance
                cost_matrix = np.zeros((n_neurons, n_neurons))
                for i in range(n_neurons):
                    d1 = circular_distance(gt_theta1[i], shifted1)
                    d2 = circular_distance(gt_theta2[i], shifted2)
                    cost_matrix[i, :] = d1 + d2

                # Hungarian algorithm
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                total_cost = cost_matrix[row_ind, col_ind].sum()

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_matching = col_ind
                    best_offset1 = offset1
                    best_offset2 = offset2
                    best_swapped = swapped

    return best_matching, best_offset1, best_offset2, best_swapped


def apply_alignment(
    theta1: NDArray[np.float64],
    theta2: NDArray[np.float64],
    offset1: float,
    offset2: float,
    swapped: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply alignment transformation to learned angles."""
    if swapped:
        t1, t2 = theta2, theta1
    else:
        t1, t2 = theta1, theta2
    aligned1 = np.mod(t1 + offset1, 2 * np.pi)
    aligned2 = np.mod(t2 + offset2, 2 * np.pi)
    return aligned1, aligned2


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = paths.load_analysis()

    # Extract data
    ground_truth = results["ground_truth"]
    models = results["models"]
    config = results["config"]
    best_mode = results["best_model"]

    n_neurons = config["n_neurons"]
    coverage = config.get("coverage", 1.0)

    # GT conjugation info
    gt_conjugation = ground_truth.get("conjugation", {})
    gt_r2 = gt_conjugation.get("r_squared", None)
    gt_var_rls = gt_conjugation.get("var_rls", None)

    # Get list of modes
    modes = list(models.keys())

    # Pi tick setup
    pi_ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    pi_labels = ["0", "π/2", "π", "3π/2", "2π"]

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # =========================================================================
    # Panel A (TL): ELBO Training Curves
    # =========================================================================
    ax1 = axes[0, 0]
    for i, mode in enumerate(modes):
        history = models[mode]["history"]
        ax1.plot(history["elbos"], color=model_colors[i], linewidth=1.2, label=mode)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("ELBO")
    ax1.set_title("A. Model Fit (ELBO)", fontweight='bold', loc='left')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Panel B (TR): Conjugation Quality (var[RLS])
    # =========================================================================
    ax2 = axes[0, 1]
    for i, mode in enumerate(modes):
        history = models[mode]["history"]
        var_rls_data = history.get("var_rls", [])
        if len(var_rls_data) > 0:
            n_total = len(history["elbos"])
            log_interval = n_total // len(var_rls_data) if len(var_rls_data) > 0 else 1
            steps = [j * log_interval for j in range(len(var_rls_data))]
            ax2.plot(steps, var_rls_data, color=model_colors[i], linewidth=1.5, label=mode)

    # GT reference line
    if gt_var_rls is not None:
        ax2.axhline(y=gt_var_rls, color=colors["ground_truth"], linestyle="--",
                   linewidth=2, label=f"GT optimal")
    ax2.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Var[RLS]")
    ax2.set_title("B. Conjugation Quality", fontweight='bold', loc='left')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Panel C (BL): Tuning Curve Locations on 2D Manifold
    # =========================================================================
    ax3 = axes[1, 0]

    # Extract GT tuning
    gt_theta1 = np.array(ground_truth["tuning"]["preferred_theta1"])
    gt_theta2 = np.array(ground_truth["tuning"]["preferred_theta2"])

    # Best mode tuning
    best_results = models[best_mode]
    learned_theta1 = np.array(best_results["learned_tuning"]["preferred_theta1"])
    learned_theta2 = np.array(best_results["learned_tuning"]["preferred_theta2"])

    # Find optimal matching and alignment
    matching, offset1, offset2, swapped = find_optimal_matching(
        gt_theta1, gt_theta2, learned_theta1, learned_theta2
    )
    aligned1, aligned2 = apply_alignment(learned_theta1, learned_theta2, offset1, offset2, swapped)
    matched_theta1 = aligned1[matching]
    matched_theta2 = aligned2[matching]

    # Compute errors for annotation
    err1 = circular_distance(gt_theta1, matched_theta1)
    err2 = circular_distance(gt_theta2, matched_theta2)
    mean_err = np.mean(err1 + err2) / 2

    # Plot GT locations
    ax3.scatter(gt_theta1, gt_theta2, c=colors["ground_truth"], s=50, alpha=0.8,
                marker='o', label='Ground Truth', edgecolors='white', linewidths=0.5)

    # Plot learned locations (aligned)
    ax3.scatter(matched_theta1, matched_theta2, c=model_colors[0], s=50, alpha=0.8,
                marker='^', label=f'Learned ({best_mode})', edgecolors='white', linewidths=0.5)

    # Draw lines connecting GT to learned for each neuron
    for i in range(len(gt_theta1)):
        ax3.plot([gt_theta1[i], matched_theta1[i]], [gt_theta2[i], matched_theta2[i]],
                 'k-', alpha=0.2, linewidth=0.5)

    ax3.set_xlabel("θ₁")
    ax3.set_ylabel("θ₂")
    ax3.set_xlim(0, 2 * np.pi)
    ax3.set_ylim(0, 2 * np.pi)
    ax3.set_aspect("equal")
    ax3.set_title("C. Preferred Locations on T²", fontweight='bold', loc='left')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(pi_ticks)
    ax3.set_xticklabels(pi_labels)
    ax3.set_yticks(pi_ticks)
    ax3.set_yticklabels(pi_labels)

    # Error annotation
    ax3.text(0.05, 0.05, f"Mean error: {np.degrees(mean_err):.1f}°/dim",
             transform=ax3.transAxes, fontsize=9, verticalalignment="bottom",
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # =========================================================================
    # Panel D (BR): Evidence Accumulation via Conjugate Inference
    # =========================================================================
    ax4 = axes[1, 1]

    # Run evidence accumulation analysis
    evidence_results = compute_evidence_accumulation(results)

    if evidence_results is not None:
        n_obs_list = evidence_results["n_observations"]
        mean_log_probs = evidence_results["mean_log_probs"]
        std_log_probs = evidence_results["std_log_probs"]

        # Plot log probability vs number of observations
        ax4.errorbar(n_obs_list, mean_log_probs,
                    yerr=std_log_probs,
                    marker='o', capsize=3, color=model_colors[0], linewidth=1.5,
                    markersize=8)

        ax4.set_xlabel("Number of Observations")
        ax4.set_ylabel("Log p(z* | x)")
        ax4.set_title("D. Evidence Accumulation", fontweight='bold', loc='left')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(n_obs_list)

        # Annotation showing improvement
        improvement = mean_log_probs[-1] - mean_log_probs[0]
        ax4.text(0.05, 0.95, f"Δ = +{improvement:.1f} nats",
                transform=ax4.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, "Evidence accumulation\ndata not available",
                ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        ax4.set_title("D. Evidence Accumulation", fontweight='bold', loc='left')

    plt.tight_layout()

    # Summary annotation
    best_elbo = models[best_mode]["final_elbo"]
    best_var_rls = models[best_mode].get("final_var_rls", 0.0)
    summary = f"Best: {best_mode} | ELBO: {best_elbo:.0f} | Var[RLS]: {best_var_rls:.1f}"
    if gt_var_rls is not None:
        summary += f" | GT: {gt_var_rls:.1f}"
    summary += f" | n={n_neurons} | cov={coverage}"

    fig.text(0.5, 0.01, summary, ha='center', fontsize=8, family='monospace')

    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


def compute_evidence_accumulation(results: dict) -> dict | None:
    """Compute evidence accumulation showing posterior concentrates with more observations.

    Uses the analytical model's learned parameters. For conjugate inference,
    combining n observations means:
        posterior_params = prior + W^T @ (x1 + ... + xn) - n * rho

    The rho correction scales with n because each observation contributes a
    ψ_X(z) term that rho approximates.

    Returns dict with n_observations, mean_log_probs, std_log_probs
    """
    try:
        import jax
        import jax.numpy as jnp
        from scipy.special import i0 as bessel_i0
        from goal.models import variational_poisson_vonmises

        config = results["config"]
        models = results["models"]

        n_neurons = config["n_neurons"]
        n_latent = config["n_latent"]

        # Use analytical model specifically
        if "analytical" not in models:
            print("Analytical model not found")
            return None

        var_model = variational_poisson_vonmises(n_neurons, n_latent)

        # Get analytical model's learned parameters
        learned_weights = np.array(models["analytical"]["learned_weight_matrix"])
        learned_baselines = np.array(models["analytical"]["learned_baselines"])

        # Reconstruct harmonium params
        int_params = jnp.array(learned_weights.ravel())
        obs_bias = jnp.array(learned_baselines)
        prior_params = jnp.zeros(2 * n_latent)
        hrm_params = var_model.hrm.join_coords(obs_bias, int_params, prior_params)

        # Compute analytical rho
        key = jax.random.PRNGKey(99999)
        key, rho_key = jax.random.split(key)

        _, _, prior_p = var_model.hrm.split_coords(hrm_params)
        z_samples_rho = var_model.pst_man.sample(rho_key, prior_p, 1000)
        s_z = jax.vmap(var_model.pst_man.sufficient_statistic)(z_samples_rho)

        int_matrix = int_params.reshape(n_neurons, 2 * n_latent)

        def compute_psi(z):
            s = var_model.pst_man.sufficient_statistic(z)
            lkl_nat = obs_bias + int_matrix @ s
            return var_model.obs_man.log_partition_function(lkl_nat)

        psi_vals = jax.vmap(compute_psi)(z_samples_rho)
        design = jnp.concatenate([jnp.ones((1000, 1)), s_z], axis=1)
        coeffs = jnp.linalg.lstsq(design, psi_vals, rcond=None)[0]
        rho = coeffs[1:]

        # Stable log I_0(κ) computation
        def log_bessel_i0(kappa: float) -> float:
            """Numerically stable log I_0(κ)."""
            if kappa < 500:
                return float(np.log(bessel_i0(kappa)))
            else:
                # Asymptotic: I_0(κ) ≈ exp(κ) / sqrt(2πκ)
                return kappa - 0.5 * np.log(2 * np.pi * kappa)

        def von_mises_log_prob(z, theta) -> float:
            """Log probability of z under VonMises product with natural params theta."""
            total = 0.0
            for dim in range(n_latent):
                theta_cos = float(theta[2 * dim])
                theta_sin = float(theta[2 * dim + 1])
                kappa = np.sqrt(theta_cos**2 + theta_sin**2)

                # θ · s(z) = κ * cos(z - μ)
                s_z_cos = float(jnp.cos(z[dim]))
                s_z_sin = float(jnp.sin(z[dim]))
                dot_product = theta_cos * s_z_cos + theta_sin * s_z_sin

                total += dot_product - log_bessel_i0(kappa)
            return total

        def von_mises_max_log_prob(theta) -> float:
            """Maximum log probability (at the mode) for VonMises with natural params theta."""
            total = 0.0
            for dim in range(n_latent):
                theta_cos = float(theta[2 * dim])
                theta_sin = float(theta[2 * dim + 1])
                kappa = np.sqrt(theta_cos**2 + theta_sin**2)
                # At mode, cos(z - μ) = 1, so dot product = κ
                total += kappa - log_bessel_i0(kappa)
            return total

        # Evidence accumulation test
        key = jax.random.PRNGKey(12345)
        n_trials = 100
        n_obs_list = [1, 2, 4, 8]

        all_log_probs = {n: [] for n in n_obs_list}

        for trial in range(n_trials):
            key, z_key, spike_key = jax.random.split(key, 3)

            # Sample true latent z from learned prior
            z_true = var_model.pst_man.sample(z_key, prior_p, 1)[0]

            # Generate observations from learned model at z_true
            max_obs = max(n_obs_list)
            spike_keys = jax.random.split(spike_key, max_obs)
            observations = []
            for i in range(max_obs):
                rates = var_model.hrm.observable_rates(hrm_params, z_true)
                x = jax.random.poisson(spike_keys[i], rates).astype(jnp.float32)
                observations.append(x)

            for n_obs in n_obs_list:
                x_combined = sum(observations[:n_obs])

                # Conjugate posterior: prior + W^T @ sum(x) - n * rho
                post_params_raw = var_model.hrm.posterior_at(hrm_params, x_combined)
                post_params = post_params_raw - n_obs * rho

                # Log probability of true z under posterior
                # Normalize by max log prob to show how well centered the posterior is
                log_prob = von_mises_log_prob(z_true, post_params)
                log_prob_max = von_mises_max_log_prob(post_params)
                # This gives log p(z*)/p(mode) which is ≤ 0, closer to 0 is better
                normalized_log_prob = log_prob - log_prob_max
                all_log_probs[n_obs].append(normalized_log_prob)

        mean_log_probs = np.array([np.mean(all_log_probs[n]) for n in n_obs_list])
        std_log_probs = np.array([np.std(all_log_probs[n]) / np.sqrt(n_trials) for n in n_obs_list])

        return {
            "n_observations": n_obs_list,
            "mean_log_probs": mean_log_probs,
            "std_log_probs": std_log_probs,
        }

    except Exception as e:
        print(f"Evidence accumulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
