"""Diagnostic script to understand why generated samples are blank.

This investigates the gap between reconstruction (posterior-based) and
generation (prior-based) in the trained model.

Supports both Binomial and Poisson observable types.
"""

import json
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..shared import example_paths, initialize_jax
from .model import MixtureModel, create_model
from .train import DEFAULT_N_TRIALS, load_mnist


def diagnose_trained_model(mode: str = "free"):
    """Diagnose a trained model by loading its parameters."""
    initialize_jax("gpu")
    paths = example_paths(__file__)

    # Load results and config
    try:
        with open(paths.analysis_path) as f:
            results = json.load(f)
    except FileNotFoundError:
        print("No results found. Run train.py first.")
        return

    # Load parameters
    params_path = paths.results_dir / "params.npz"
    try:
        params_data = np.load(params_path)
    except FileNotFoundError:
        print(f"No parameters found at {params_path}. Re-run train.py to save params.")
        return

    if mode not in params_data:
        print(f"Mode '{mode}' not found. Available: {list(params_data.keys())}")
        return

    params = jnp.array(params_data[mode])

    # Get config
    config = results.get("config", {})
    n_latent = config.get("n_latent", 1024)
    n_clusters = config.get("n_clusters", 10)
    n_trials = config.get("n_trials", DEFAULT_N_TRIALS)
    observable_type: Literal["binomial", "poisson"] = config.get(
        "observable_type", "binomial"
    )
    latent_type: Literal["bernoulli", "vonmises"] = config.get(
        "latent_type", "bernoulli"
    )

    # Load test data
    _, _, test_data, _ = load_mnist(n_trials, observable_type=observable_type)

    # Create model
    model: MixtureModel = create_model(
        n_latent=n_latent,
        n_clusters=n_clusters,
        observable_type=observable_type,
        latent_type=latent_type,
        n_trials=n_trials,
    )

    key = jax.random.PRNGKey(42)

    print(f"\n{'=' * 60}")
    print(f"DIAGNOSING TRAINED MODEL: {mode} ({observable_type})")
    print(f"{'=' * 60}")

    # Split parameters
    rho, hrm_params = model.split_coords(params)
    obs_bias, int_params, lat_params = model.hrm.split_coords(hrm_params)

    print("\n--- TRAINED PARAMETERS ---")
    print(
        f"Rho: norm={float(jnp.linalg.norm(rho)):.4f}, min={float(rho.min()):.4f}, max={float(rho.max()):.4f}"
    )
    print(
        f"Observable bias: min={float(obs_bias.min()):.4f}, max={float(obs_bias.max()):.4f}, mean={float(obs_bias.mean()):.4f}"
    )

    lat_dim = model.bas_lat_man.dim
    int_matrix = int_params.reshape(model.n_observable, lat_dim)
    print(
        f"Interaction matrix: shape={int_matrix.shape}, norm={float(jnp.linalg.norm(int_matrix)):.4f}, mean={float(int_matrix.mean()):.6f}, std={float(int_matrix.std()):.6f}"
    )

    # Sample from prior
    print("\n--- PRIOR SAMPLES ---")
    key, sample_key = jax.random.split(key)
    _, _, prior_mixture_params = model.hrm.split_coords(hrm_params)
    y_samples, k_samples = model.sample_from_posterior(
        sample_key, prior_mixture_params, 100
    )

    if latent_type == "vonmises":
        print(
            f"z samples: mean={float(y_samples.mean()):.4f}, std={float(y_samples.std()):.4f}, range=[{float(y_samples.min()):.4f}, {float(y_samples.max()):.4f}]"
        )
    else:
        print(
            f"y samples: mean={float(y_samples.mean()):.4f}, fraction 1s={float((y_samples > 0.5).mean()):.4f}"
        )
    print(f"k samples: distribution={[int((k_samples == k).sum()) for k in range(10)]}")

    # What are the likelihood parameters for prior samples?
    def get_lkl_params(y):
        return model.likelihood_at(params, y)

    lkl_params_prior = jax.vmap(get_lkl_params)(y_samples)
    print("\nLikelihood natural params (prior samples):")
    print(
        f"  min={float(lkl_params_prior.min()):.4f}, max={float(lkl_params_prior.max()):.4f}, mean={float(lkl_params_prior.mean()):.4f}"
    )

    # Convert to means (expected pixel values)
    lkl_means_prior = jax.vmap(model.obs_man.to_mean)(lkl_params_prior)
    print("Expected pixel values (prior samples):")
    print(
        f"  min={float(lkl_means_prior.min()):.4f}, max={float(lkl_means_prior.max()):.4f}, mean={float(lkl_means_prior.mean()):.4f}"
    )

    # Now compare to posterior samples
    print("\n--- POSTERIOR SAMPLES (from test data) ---")
    key, _post_key = jax.random.split(key)

    # Sample from posterior for several test images
    n_test = 10
    all_y_post = []
    all_lkl_means_post = []

    for i in range(n_test):
        x_sample = test_data[i]
        q_params = model.approximate_posterior_at(params, x_sample)
        key, sub_key = jax.random.split(key)
        y_post, _ = model.sample_from_posterior(sub_key, q_params, 10)
        all_y_post.append(y_post)

        lkl_params_post = jax.vmap(get_lkl_params)(y_post)
        lkl_means_post = jax.vmap(model.obs_man.to_mean)(lkl_params_post)
        all_lkl_means_post.append(lkl_means_post)

    all_y_post = jnp.concatenate(all_y_post)
    all_lkl_means_post = jnp.concatenate(all_lkl_means_post)

    if latent_type == "vonmises":
        print(
            f"z samples: mean={float(all_y_post.mean()):.4f}, std={float(all_y_post.std()):.4f}"
        )
    else:
        print(
            f"y samples: mean={float(all_y_post.mean()):.4f}, fraction 1s={float((all_y_post > 0.5).mean()):.4f}"
        )
    print("Expected pixel values (posterior samples):")
    print(
        f"  min={float(all_lkl_means_post.min()):.4f}, max={float(all_lkl_means_post.max()):.4f}, mean={float(all_lkl_means_post.mean()):.4f}"
    )

    # Key comparison
    print("\n--- KEY COMPARISON ---")
    if latent_type == "vonmises":
        print(f"Prior z std:     {float(y_samples.std()):.4f}")
        print(f"Posterior z std: {float(all_y_post.std()):.4f}")
    else:
        print(f"Prior y activation rate:     {float((y_samples > 0.5).mean()) * 100:.1f}%")
        print(f"Posterior y activation rate: {float((all_y_post > 0.5).mean()) * 100:.1f}%")
    print(f"Prior expected pixel mean:     {float(lkl_means_prior.mean()):.4f}")
    print(f"Posterior expected pixel mean: {float(all_lkl_means_post.mean()):.4f}")
    print(f"Test data actual mean:         {float(test_data.mean()):.4f}")

    # Check the prior component parameters
    print("\n--- PRIOR MIXTURE COMPONENTS ---")
    mix_obs_bias, _mix_int_params, _cat_params = model.mix_man.split_coords(lat_params)
    print(
        f"Mixture observable bias ({latent_type} prior): min={float(mix_obs_bias.min()):.4f}, max={float(mix_obs_bias.max()):.4f}, mean={float(mix_obs_bias.mean()):.4f}"
    )

    # Get component-wise parameters
    components, _ = model.mix_man.split_natural_mixture(prior_mixture_params)
    comp_params_2d = model.mix_man.cmp_man.to_2d(components)

    if latent_type == "vonmises":
        print("\nComponent Von Mises mean parameters:")
        for k in range(min(10, model.n_clusters)):
            comp_mean = model.bas_lat_man.to_mean(comp_params_2d[k])
            mean_norm = float(jnp.linalg.norm(comp_mean))
            print(f"  Component {k}: mean norm = {mean_norm:.4f}")
    else:
        print("\nComponent Bernoulli activation rates (sigmoid of natural params):")
        for k in range(min(10, model.n_clusters)):
            comp_mean = model.bas_lat_man.to_mean(comp_params_2d[k])
            activation_rate = float(comp_mean.mean())
            print(f"  Component {k}: mean activation = {activation_rate:.4f}")

    # Diagnosis summary
    print("\n--- DIAGNOSIS ---")
    prior_pixel_mean = float(lkl_means_prior.mean())
    post_pixel_mean = float(all_lkl_means_post.mean())
    float(test_data.mean())

    # Generation gap: ratio of prior to posterior pixel means
    # If this is near 1.0, prior and posterior produce similar images
    # If this is near 0, prior produces blank images while posterior doesn't
    generation_gap = prior_pixel_mean / max(post_pixel_mean, 1e-6)
    print(f"Generation gap (prior/posterior pixel mean): {generation_gap:.4f}")
    print("  - 1.0 = prior matches aggregate posterior well")
    print("  - 0.0 = prior produces blank, posterior produces meaningful images")

    if generation_gap < 0.1:
        print("\nPROBLEM: Severe prior-posterior mismatch (gap < 0.1)")
        print("Prior samples produce nearly blank images while posterior works fine.")
        print("\nPossible causes:")
        print("  1. Prior p(y|k) not learning to match aggregate posterior")
        print("  2. Observable bias too negative, requiring specific y patterns")
        print("\nPossible fixes:")
        print("  1. KL annealing (start with low KL weight, increase over training)")
        print("  2. Learned/flexible prior instead of fixed mixture")
        print("  3. Better initialization of prior parameters")
    elif generation_gap < 0.5:
        print("\nWARNING: Moderate prior-posterior mismatch (gap < 0.5)")
    else:
        print("\nOK: Prior samples produce reasonable images")

    # Check if the issue is observable bias
    mean_obs_bias = float(obs_bias.mean())
    if mean_obs_bias < -5:
        print(f"\nWARNING: Observable bias is very negative (mean={mean_obs_bias:.2f})")
        print("This requires strong interaction term to produce non-zero pixels")

    # ========== LATENT CODE COLLAPSE DIAGNOSTICS ==========
    print("\n--- LATENT CODE COLLAPSE DIAGNOSTICS ---")

    # 1. Analyze interaction weight distribution per latent dimension
    # Each column of int_matrix corresponds to one latent neuron
    weight_norms_per_latent = jnp.linalg.norm(int_matrix, axis=0)  # (n_latent,)
    _ = jnp.abs(int_matrix).mean(axis=0)  # (n_latent,)

    print(f"\nInteraction weight norms per latent dim ({model.n_observable} obs -> {lat_dim} latent dims):")
    print(f"  min={float(weight_norms_per_latent.min()):.4f}, max={float(weight_norms_per_latent.max()):.4f}, mean={float(weight_norms_per_latent.mean()):.4f}, std={float(weight_norms_per_latent.std()):.4f}")

    # Count "dead" latents (very small weight norm)
    dead_threshold = 0.1 * float(weight_norms_per_latent.mean())
    n_dead_latents = int(jnp.sum(weight_norms_per_latent < dead_threshold))
    print(f"  'Dead' latent dims (norm < 10% of mean): {n_dead_latents}/{lat_dim} ({100*n_dead_latents/lat_dim:.1f}%)")

    # 2. Analyze posterior activation variance across test data
    # Sample posteriors for many test images and look at activation patterns
    print("\nPosterior latent activation analysis (100 test images):")
    key, _ = jax.random.split(key)
    n_analysis = 100

    all_y_activations = []
    for i in range(n_analysis):
        x_sample = test_data[i]
        q_params = model.approximate_posterior_at(params, x_sample)

        # Get mean activation for each component, weighted by component probs
        probs = model.get_cluster_probs(q_params)
        components, _ = model.mix_man.split_natural_mixture(q_params)
        comp_params_2d = model.mix_man.cmp_man.to_2d(components)

        # Weighted average of Bernoulli means across components
        comp_means = jax.vmap(model.bas_lat_man.to_mean)(comp_params_2d)
        weighted_mean = jnp.einsum("k,ki->i", probs, comp_means)
        all_y_activations.append(weighted_mean)

    all_y_activations = jnp.stack(all_y_activations)  # (100, n_latent)

    # Variance of each latent dimension across samples
    latent_variances = jnp.var(all_y_activations, axis=0)  # (n_latent,)
    latent_means = jnp.mean(all_y_activations, axis=0)  # (n_latent,)

    print(f"  Latent activation means: min={float(latent_means.min()):.6f}, max={float(latent_means.max()):.6f}, mean={float(latent_means.mean()):.6f}")
    print(f"  Latent activation variances: min={float(latent_variances.min()):.8f}, max={float(latent_variances.max()):.6f}, mean={float(latent_variances.mean()):.6f}")

    # Count latents with near-zero variance (not responding to input)
    var_threshold = 1e-6
    n_latent_dims = all_y_activations.shape[1]
    n_constant_latents = int(jnp.sum(latent_variances < var_threshold))
    print(f"  Constant latent dims (var < {var_threshold}): {n_constant_latents}/{n_latent_dims} ({100*n_constant_latents/n_latent_dims:.1f}%)")

    if latent_type != "vonmises":
        # Count latents with very low mean (almost never active) — Bernoulli only
        mean_threshold = 0.01
        n_inactive_latents = int(jnp.sum(latent_means < mean_threshold))
        print(f"  Inactive latents (mean < {mean_threshold}): {n_inactive_latents}/{model.n_latent} ({100*n_inactive_latents/model.n_latent:.1f}%)")

    # 3. Effective dimensionality via participation ratio
    # PR = (sum of variances)^2 / sum of (variances^2)
    # This gives ~1 if only one latent varies, ~n if all vary equally
    total_var = float(jnp.sum(latent_variances))
    sum_var_sq = float(jnp.sum(latent_variances**2))
    if sum_var_sq > 1e-12:
        participation_ratio = (total_var**2) / sum_var_sq
    else:
        participation_ratio = 0.0

    print(f"\n  Effective dimensionality (participation ratio): {participation_ratio:.1f}/{n_latent_dims}")
    print(f"  Utilization: {100*participation_ratio/n_latent_dims:.1f}%")

    # 4. Histogram of activation rates / variance distribution
    if latent_type == "vonmises":
        print(f"\nLatent variance distribution:")
        var_bins = [0, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, float("inf")]
        for i in range(len(var_bins) - 1):
            count = int(jnp.sum((latent_variances >= var_bins[i]) &
                               (latent_variances < var_bins[i+1])))
            print(f"  [{var_bins[i]:.1e}, {var_bins[i+1]:.1e}): {count} latent dims")
    else:
        activation_rate_per_latent = latent_means  # For Bernoulli, mean = P(y=1)
        print(f"\nActivation rate distribution:")
        bins = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        for i in range(len(bins) - 1):
            count = int(jnp.sum((activation_rate_per_latent >= bins[i]) &
                               (activation_rate_per_latent < bins[i+1])))
            print(f"  [{bins[i]:.3f}, {bins[i+1]:.3f}): {count} latents")

    # Summary
    print("\n--- LATENT COLLAPSE SUMMARY ---")
    if participation_ratio < n_latent_dims * 0.1:
        print(f"SEVERE: Only {participation_ratio:.0f} effective latent dimensions ({100*participation_ratio/n_latent_dims:.1f}% of {n_latent_dims})")
        print("This indicates significant latent code collapse.")
    elif participation_ratio < n_latent_dims * 0.3:
        print(f"MODERATE: {participation_ratio:.0f} effective latent dimensions ({100*participation_ratio/n_latent_dims:.1f}% of {n_latent_dims})")
    else:
        print(f"OK: {participation_ratio:.0f} effective latent dimensions ({100*participation_ratio/n_latent_dims:.1f}% of {n_latent_dims})")


def main():
    """Run diagnosis for all trained modes."""
    initialize_jax("gpu")
    paths = example_paths(__file__)

    # Load parameters
    params_path = paths.results_dir / "params.npz"
    try:
        params_data = np.load(params_path)
        modes = list(params_data.keys())
    except FileNotFoundError:
        print(f"No parameters found at {params_path}")
        print("Run train.py first to train and save models.")
        return

    for mode in modes:
        diagnose_trained_model(mode)


if __name__ == "__main__":
    main()
