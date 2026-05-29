"""Diagnostic script to understand why generated samples are blank.

This investigates the gap between reconstruction (posterior-based) and
generation (prior-based) in the trained model.

Supports Binomial, Poisson, and Normal (DiagonalNormal) observable types.
When the trained latent is ``ChordalBoltzmann``, also writes a coupling-
pattern visualisation alongside the standard diagnostic output.
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false

import json
from typing import Literal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from ..shared import example_paths, model_color
from .model import MixtureModel, create_model
from .train import DEFAULT_N_TRIALS, load_mnist


def diagnose_trained_model(mode: str = "gradient"):  # noqa: C901
    """Diagnose a trained model by loading its parameters."""
    jax.config.update("jax_platform_name", "gpu")
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
    observable_type: Literal["binomial", "poisson", "normal"] = config.get(
        "observable_type", "binomial"
    )
    interaction: Literal["hierarchical", "full"] = config.get("interaction", "full")
    latent: Literal["bernoullis", "chordal_boltzmann"] = config.get(
        "latent", "bernoullis"
    )
    latent_graph: Literal["chain"] = config.get("latent_graph", "chain")

    # Load test data
    _, _, test_data, _ = load_mnist(n_trials, observable_type=observable_type)

    # Create model
    model: MixtureModel = create_model(
        n_latent=n_latent,
        n_clusters=n_clusters,
        observable_type=observable_type,
        n_trials=n_trials,
        interaction=interaction,
        latent=latent,
        latent_graph=latent_graph,
    )

    key = jax.random.PRNGKey(42)

    print(f"\n{'=' * 60}")
    print(f"DIAGNOSING TRAINED MODEL: {mode} ({observable_type})")
    print(f"{'=' * 60}")

    # Split parameters
    lat_params, lkl_params, rho = model.split_coords(params)
    obs_bias, int_params = model.gen_hrm.lkl_fun_man.split_coords(lkl_params)

    print("\n--- TRAINED PARAMETERS ---")
    print(
        f"Rho: norm={float(jnp.linalg.norm(rho)):.4f}, min={float(rho.min()):.4f}, max={float(rho.max()):.4f}"
    )
    print(
        f"Observable bias: min={float(obs_bias.min()):.4f}, max={float(obs_bias.max()):.4f}, mean={float(obs_bias.mean()):.4f}"
    )

    lat_dim = model.bas_lat_man.dim  # param dim (= n + n_chordal_edges for ChordalBoltzmann)
    lat_data_dim = model.bas_lat_man.data_dim  # sample dim (= n binary units)
    # The naive ``(obs_dim, lat_dim)`` reshape assumes the entire ``int_params``
    # vector is a single ``xy`` block. In ``full`` interaction mode it actually
    # contains three concatenated blocks (xy, xyk, xk) and the sizes will not
    # match. Skip the reshape-based diagnostic in that case rather than aborting
    # the whole script — the per-component prior analysis below still runs.
    int_matrix = None
    expected_size = int(model.obs_man.dim) * int(lat_dim)
    if int_params.size == expected_size:
        int_matrix = int_params.reshape(model.obs_man.dim, lat_dim)
        print(
            f"Interaction matrix: shape={int_matrix.shape}, norm={float(jnp.linalg.norm(int_matrix)):.4f}, mean={float(int_matrix.mean()):.6f}, std={float(int_matrix.std()):.6f}"
        )
    else:
        print(
            f"Interaction params size {int_params.size} != obs_dim*lat_dim"
            + f" ({expected_size}); skipping single-block interaction-matrix"
            + " diagnostic (full-interaction model packs xy/xyk/xk blocks)."
        )
    if int_matrix is not None and lat_dim != lat_data_dim:
        # ChordalBoltzmann: first lat_data_dim columns are unit interactions
        # x_i, remainder are pair interactions x_i * x_j over chordal edges.
        unit_block = int_matrix[:, :lat_data_dim]
        pair_block = int_matrix[:, lat_data_dim:]
        print(
            f"  unit cols: shape={unit_block.shape}, norm={float(jnp.linalg.norm(unit_block)):.4f}"
        )
        print(
            f"  pair cols: shape={pair_block.shape}, norm={float(jnp.linalg.norm(pair_block)):.4f}"
        )

    # ChordalBoltzmann-only: visualise learned off-diagonal couplings.
    # ``lat_dim > lat_data_dim`` is the structural signature: the extra entries
    # are the coupling weights on the chordal edges (``n_latent - 1`` for a chain).
    # Hoisted before prior sampling so the visualisation still produces output
    # even if downstream analyses fail for some model configurations.
    if lat_dim != lat_data_dim:
        n_edges = lat_dim - lat_data_dim
        print(
            f"\n--- CHORDAL COUPLINGS ({n_edges} edges,"
            + f" graph={latent_graph}) ---"
        )
        early_components, _ = model.mix_man.split_natural_mixture(lat_params)
        early_comp_params_2d = model.mix_man.cmp_man.to_2d(early_components)
        couplings_per_component = []
        for k in range(model.n_categories):
            comp_natural = early_comp_params_2d[k]
            couplings = np.asarray(comp_natural[lat_data_dim:])
            couplings_per_component.append(couplings)
        couplings_arr = np.stack(couplings_per_component)
        print(
            "  Coupling magnitudes per component:"
            + f" mean |c|={float(np.mean(np.abs(couplings_arr))):.4f},"
            + f" max |c|={float(np.max(np.abs(couplings_arr))):.4f}"
        )

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        edge_idx = np.arange(n_edges)
        for k in range(model.n_categories):
            axes[0].plot(
                edge_idx,
                couplings_arr[k],
                color=model_color(k),
                alpha=0.7,
                label=f"comp {k}",
            )
        axes[0].axhline(0.0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_ylabel("Coupling weight")
        axes[0].set_title(
            f"Learned chordal couplings per component ({latent_graph} graph)"
        )
        axes[0].legend(ncol=5, fontsize=7)
        agg = np.mean(np.abs(couplings_arr), axis=0)
        axes[1].plot(edge_idx, agg, color=model_color(0))
        axes[1].set_xlabel("Edge index (along chain)")
        axes[1].set_ylabel("Mean |coupling|")
        axes[1].set_title("Mean coupling magnitude across components")
        coupling_plot_path = paths.results_dir / f"couplings_{mode}.png"
        fig.tight_layout()
        fig.savefig(coupling_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Coupling visualisation saved to {coupling_plot_path}")

    # Sample from prior
    print("\n--- PRIOR SAMPLES ---")
    key, sample_key = jax.random.split(key)
    z_samples_prior = model.mix_man.sample(sample_key, lat_params, 100)
    # Extract y and k components (k is a scalar category index). y has length
    # ``lat_data_dim`` (number of binary units); ChordalBoltzmann's larger
    # ``lat_dim`` (n + n_chordal_edges) is the parameter dimension, not the
    # sample dimension.
    y_samples = z_samples_prior[:, :lat_data_dim]
    k_samples = z_samples_prior[:, lat_data_dim].astype(jnp.int32)

    print(
        f"y samples: mean={float(y_samples.mean()):.4f}, fraction 1s={float((y_samples > 0.5).mean()):.4f}"
    )
    print(f"k samples: distribution={[int((k_samples == k).sum()) for k in range(10)]}")

    # What are the likelihood parameters for prior samples?
    # ``likelihood_at`` takes the posterior-shaped latent ``z``. In hierarchical
    # mode that is just ``y``; in full mode it is the joint ``(y, k)`` row that
    # ``mix_man.sample`` already produces.
    def get_lkl_params(z: Array) -> Array:
        return model.likelihood_at(params, z)

    lkl_params_prior = jax.vmap(get_lkl_params)(z_samples_prior)
    print("\nLikelihood natural params (prior samples):")
    print(
        f"  min={float(lkl_params_prior.min()):.4f}, max={float(lkl_params_prior.max()):.4f}, mean={float(lkl_params_prior.mean()):.4f}"
    )

    # Convert to means (expected pixel values). For Normal observables the
    # mean vector packs both first and second moments; slice to data_dim so
    # the printed stats live in the pixel space.
    lkl_means_prior_full = jax.vmap(model.obs_man.to_mean)(lkl_params_prior)
    lkl_means_prior = lkl_means_prior_full[..., : model.obs_man.data_dim]
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
        z_post = model.mix_man.sample(sub_key, q_params, 10)
        y_post = z_post[:, :lat_data_dim]
        all_y_post.append(y_post)

        lkl_params_post = jax.vmap(get_lkl_params)(z_post)
        lkl_means_post_full = jax.vmap(model.obs_man.to_mean)(lkl_params_post)
        # Slice to data_dim for Normal observables; identity for count obs.
        lkl_means_post = lkl_means_post_full[..., : model.obs_man.data_dim]
        all_lkl_means_post.append(lkl_means_post)

    all_y_post = jnp.concatenate(all_y_post)
    all_lkl_means_post = jnp.concatenate(all_lkl_means_post)

    print(
        f"y samples: mean={float(all_y_post.mean()):.4f}, fraction 1s={float((all_y_post > 0.5).mean()):.4f}"
    )
    print("Expected pixel values (posterior samples):")
    print(
        f"  min={float(all_lkl_means_post.min()):.4f}, max={float(all_lkl_means_post.max()):.4f}, mean={float(all_lkl_means_post.mean()):.4f}"
    )

    # Key comparison
    print("\n--- KEY COMPARISON ---")
    print(
        f"Prior y activation rate:     {float((y_samples > 0.5).mean()) * 100:.1f}%"
    )
    print(
        f"Posterior y activation rate: {float((all_y_post > 0.5).mean()) * 100:.1f}%"
    )
    print(f"Prior expected pixel mean:     {float(lkl_means_prior.mean()):.4f}")
    print(f"Posterior expected pixel mean: {float(all_lkl_means_post.mean()):.4f}")
    print(f"Test data actual mean:         {float(test_data.mean()):.4f}")

    # Check the prior component parameters
    print("\n--- PRIOR MIXTURE COMPONENTS ---")
    mix_obs_bias, _mix_int_params, _cat_params = model.mix_man.split_coords(lat_params)
    print(
        f"Mixture observable bias (bernoulli prior): min={float(mix_obs_bias.min()):.4f}, max={float(mix_obs_bias.max()):.4f}, mean={float(mix_obs_bias.mean()):.4f}"
    )

    # Get component-wise parameters
    components, _ = model.mix_man.split_natural_mixture(lat_params)
    comp_params_2d = model.mix_man.cmp_man.to_2d(components)

    print("\nComponent Bernoulli activation rates (sigmoid of natural params):")
    for k in range(min(10, model.n_categories)):
        comp_mean = model.bas_lat_man.to_mean(comp_params_2d[k])
        activation_rate = float(comp_mean.mean())
        print(f"  Component {k}: mean activation = {activation_rate:.4f}")

    # Diagnosis summary
    print("\n--- DIAGNOSIS ---")
    prior_pixel_mean = float(lkl_means_prior.mean())
    post_pixel_mean = float(all_lkl_means_post.mean())
    float(test_data.mean())

    # Generation gap: ratio of prior to posterior pixel means
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
    # Only runs in hierarchical mode where ``int_matrix`` is a single block.
    if int_matrix is not None:
        weight_norms_per_latent = jnp.linalg.norm(int_matrix, axis=0)  # (n_latent,)
        _ = jnp.abs(int_matrix).mean(axis=0)  # (n_latent,)

        print(
            f"\nInteraction weight norms per latent dim ({model.obs_man.dim} obs -> {lat_dim} latent dims):"
        )
        print(
            f"  min={float(weight_norms_per_latent.min()):.4f}, max={float(weight_norms_per_latent.max()):.4f}, mean={float(weight_norms_per_latent.mean()):.4f}, std={float(weight_norms_per_latent.std()):.4f}"
        )

        # Count "dead" latents (very small weight norm)
        dead_threshold = 0.1 * float(weight_norms_per_latent.mean())
        n_dead_latents = int(jnp.sum(weight_norms_per_latent < dead_threshold))
        print(
            f"  'Dead' latent dims (norm < 10% of mean): {n_dead_latents}/{lat_dim} ({100 * n_dead_latents / lat_dim:.1f}%)"
        )

    # 2. Analyze posterior activation variance across test data
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

    print(
        f"  Latent activation means: min={float(latent_means.min()):.6f}, max={float(latent_means.max()):.6f}, mean={float(latent_means.mean()):.6f}"
    )
    print(
        f"  Latent activation variances: min={float(latent_variances.min()):.8f}, max={float(latent_variances.max()):.6f}, mean={float(latent_variances.mean()):.6f}"
    )

    # Count latents with near-zero variance (not responding to input)
    var_threshold = 1e-6
    n_latent_dims = all_y_activations.shape[1]
    n_constant_latents = int(jnp.sum(latent_variances < var_threshold))
    print(
        f"  Constant latent dims (var < {var_threshold}): {n_constant_latents}/{n_latent_dims} ({100 * n_constant_latents / n_latent_dims:.1f}%)"
    )

    # Count latents with very low mean (almost never active) -- Bernoulli only
    mean_threshold = 0.01
    n_inactive_latents = int(jnp.sum(latent_means < mean_threshold))
    print(
        f"  Inactive latents (mean < {mean_threshold}): {n_inactive_latents}/{lat_dim} ({100 * n_inactive_latents / lat_dim:.1f}%)"
    )

    # 3. Effective dimensionality via participation ratio
    total_var = float(jnp.sum(latent_variances))
    sum_var_sq = float(jnp.sum(latent_variances**2))
    participation_ratio = (total_var**2) / sum_var_sq if sum_var_sq > 1e-12 else 0.0

    print(
        f"\n  Effective dimensionality (participation ratio): {participation_ratio:.1f}/{n_latent_dims}"
    )
    print(f"  Utilization: {100 * participation_ratio / n_latent_dims:.1f}%")

    # 4. Histogram of activation rates
    activation_rate_per_latent = latent_means  # For Bernoulli, mean = P(y=1)
    print("\nActivation rate distribution:")
    bins = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    for i in range(len(bins) - 1):
        count = int(
            jnp.sum(
                (activation_rate_per_latent >= bins[i])
                & (activation_rate_per_latent < bins[i + 1])
            )
        )
        print(f"  [{bins[i]:.3f}, {bins[i + 1]:.3f}): {count} latents")

    # Summary
    print("\n--- LATENT COLLAPSE SUMMARY ---")
    if participation_ratio < n_latent_dims * 0.1:
        print(
            f"SEVERE: Only {participation_ratio:.0f} effective latent dimensions ({100 * participation_ratio / n_latent_dims:.1f}% of {n_latent_dims})"
        )
        print("This indicates significant latent code collapse.")
    elif participation_ratio < n_latent_dims * 0.3:
        print(
            f"MODERATE: {participation_ratio:.0f} effective latent dimensions ({100 * participation_ratio / n_latent_dims:.1f}% of {n_latent_dims})"
        )
    else:
        print(
            f"OK: {participation_ratio:.0f} effective latent dimensions ({100 * participation_ratio / n_latent_dims:.1f}% of {n_latent_dims})"
        )


def main():
    """Run diagnosis for all trained modes."""
    jax.config.update("jax_platform_name", "gpu")
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
