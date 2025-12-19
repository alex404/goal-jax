"""Mixture of Factor Analyzers (MFA) example using MixtureOfConjugated."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import Optimizer, OptState
from goal.models import FactorAnalysis, Normal
from goal.models.graphical.mixture import MixtureOfConjugated

from ..shared import example_paths, initialize_jax
from .types import MFAResults

### Ground Truth Sample Generation ###


def create_ground_truth_samples(
    key: Array,
    sample_size: int,
) -> tuple[
    Array,
    Array,
    Array,
    FactorAnalysis,
    FactorAnalysis,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
]:
    """Create samples from a known MFA by manual mixing.

    Strategy: Define two FA models, sample component assignments,
    then sample from corresponding FA.

    Returns
    -------
    samples : Array
        Observable samples (sample_size, 3)
    component_indicators : Array
        Which component generated each sample (sample_size,)
    ground_truth_mfa_params : Array
        The true MFA parameters used to generate the data
    fa1, fa2 : FactorAnalysis
        The two FA component models
    fa1_params, fa2_params : Array
        Parameters for each FA component
    mixing_probs : Array
        Mixture weights [w1, w2]
    loadings_1, means_1, diags_1 : Array
        Component 1 FA structure
    loadings_2, means_2, diags_2 : Array
        Component 2 FA structure
    """
    obs_dim, lat_dim = 3, 2

    # Component 1: Factors oriented along different directions
    loadings_1 = jnp.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ]
    )
    means_1 = jnp.array([2.0, 1.0, 2.0])
    diags_1 = jnp.array([0.3, 0.3, 0.3])

    # Component 2: Different loading orientation and mean
    loadings_2 = jnp.array(
        [
            [-1.0, 0.0],
            [0.5, -0.5],
            [0.0, -1.0],
        ]
    )
    means_2 = jnp.array([-2.0, -1.0, -2.0])
    diags_2 = jnp.array([0.3, 0.3, 0.3])

    # Mixing weights (60% component 1, 40% component 2)
    mixing_probs = jnp.array([0.6, 0.4])

    # Create FA models
    fa1 = FactorAnalysis(obs_dim, lat_dim)
    fa2 = FactorAnalysis(obs_dim, lat_dim)

    fa1_params = fa1.from_loadings(loadings_1, means_1, diags_1)
    fa2_params = fa2.from_loadings(loadings_2, means_2, diags_2)

    # Construct ground truth MFA parameters
    # This requires constructing the mixture from the two FA parameter sets
    from goal.models.graphical.mixture import MixtureOfConjugated

    mfa_gt = MixtureOfConjugated(n_categories=2, hrm=fa1)

    # Extract FA parameters
    obs1, int1, lat1 = fa1.split_coords(fa1_params)
    obs2, int2, lat2 = fa2.split_coords(fa2_params)

    # For MFA, we need to construct:
    # - obs_params: Use component 1's observable params as base
    # - int_params: (xy block, xyk block, xk block)
    # - lat_params: CompleteMixture structure

    # Observable params: use component 1 as base
    mfa_obs = obs1

    # Interaction params:
    # - xy block: use component 1's interaction
    # - xyk block: difference in interactions (obs2 - obs1)
    # - xk block: observable bias from mixing (difference in observable params)
    xy_block = int1
    xyk_block = (int2 - int1).ravel()  # Flatten to 1D for single difference component
    xk_block = (obs2 - obs1).ravel()  # Flatten to 1D

    mfa_int = jnp.concatenate([xy_block, xyk_block, xk_block])

    # Latent params: CompleteMixture(observable=lat1, interaction=lat2-lat1, categorical=log_odds)
    # For mixing_probs = [0.6, 0.4], log odds = log(0.4/0.6)
    log_odds = jnp.log(mixing_probs[1] / mixing_probs[0])
    mfa_lat = mfa_gt.lat_man.join_coords(
        lat1,  # Base latent from component 1
        (lat2 - lat1).ravel(),  # Difference in latent params
        jnp.array([log_odds]),  # Categorical log odds
    )

    ground_truth_mfa_params = mfa_gt.join_coords(mfa_obs, mfa_int, mfa_lat)

    # Sample component assignments
    key_assign, key_sample = jax.random.split(key)
    component_indicators = jax.random.categorical(
        key_assign,
        jnp.log(mixing_probs),
        shape=(sample_size,),
    )

    # Sample from each component
    keys_sample = jax.random.split(key_sample, sample_size)

    def sample_one(key_i: Array, component: Array) -> Array:
        """Sample one observation from the appropriate component."""
        # Sample from component 0 or 1
        sample_0 = fa1.observable_sample(key_i, fa1_params, 1)[0]
        sample_1 = fa2.observable_sample(key_i, fa2_params, 1)[0]
        return jnp.where(component == 0, sample_0, sample_1)

    samples = jax.vmap(sample_one)(keys_sample, component_indicators)

    return (
        samples,
        component_indicators,
        ground_truth_mfa_params,
        fa1,
        fa2,
        fa1_params,
        fa2_params,
        mixing_probs,
        loadings_1,
        means_1,
        diags_1,
        loadings_2,
        means_2,
        diags_2,
    )


### Model Fitting ###


def fit_mfa(
    key: Array,
    mfa: MixtureOfConjugated[Normal, Normal],
    n_steps: int,
    sample: Array,
    learning_rate: float = 1e-3,
) -> tuple[Array, Array, Array]:
    """Train MFA using gradient descent.

    Returns
    -------
    lls : Array
        Training log likelihoods (n_steps,)
    init_params : Array
        Initial parameters
    final_params : Array
        Final parameters after training
    """
    # Initialize from sample
    init_params = mfa.initialize_from_sample(key, sample)

    # Create optimizer
    optimizer: Optimizer[MixtureOfConjugated[Normal, Normal]] = Optimizer.adamw(
        man=mfa, learning_rate=learning_rate
    )
    opt_state = optimizer.init(init_params)

    def cross_entropy_loss(params: Array) -> Array:
        """Negative log likelihood loss."""
        return -mfa.average_log_observable_density(params, sample)

    def grad_step(
        opt_state_and_params: tuple[OptState, Array], _: Any
    ) -> tuple[tuple[OptState, Array], Array]:
        """Single gradient descent step."""
        opt_state, params = opt_state_and_params
        loss_val, grads = jax.value_and_grad(cross_entropy_loss)(params)
        opt_state, params = optimizer.update(opt_state, grads, params)
        return (opt_state, params), -loss_val  # Return log likelihood

    (_, final_params), lls = jax.lax.scan(
        grad_step, (opt_state, init_params), None, length=n_steps
    )

    return lls.ravel(), init_params, final_params


# JIT compile for performance
fit_mfa = jax.jit(fit_mfa, static_argnames=["mfa", "n_steps"])


### Density Computation ###


def compute_marginal_density_2d(
    loadings_1: Array,
    means_1: Array,
    diags_1: Array,
    loadings_2: Array,
    means_2: Array,
    diags_2: Array,
    mixing_weights: Array,
    x_range: Array,
    dims: tuple[int, int],
) -> Array:
    """Compute true 2D marginal density for a mixture of two FAs.

    Computes p(x_i, x_j) = Σ_k w_k N(x | μ_k^{ij}, Σ_k^{ij,ij})
    by marginalizing out the third dimension analytically.

    For FA: Cov = Λ Λ^T + Ψ where Λ is loadings and Ψ is diagonal noise.

    Parameters
    ----------
    loadings_1, loadings_2 : Array
        Loading matrices for each component (obs_dim, lat_dim)
    means_1, means_2 : Array
        Mean vectors for each component (obs_dim,)
    diags_1, diags_2 : Array
        Diagonal noise variances for each component (obs_dim,)
    mixing_weights : Array
        Mixture weights [w1, w2]
    x_range : Array
        1D array of coordinates (n_points,)
    dims : tuple[int, int]
        Which two dimensions to plot (e.g., (0, 1) for x1 vs x2)

    Returns
    -------
    densities : Array
        2D grid of marginal densities (n_points, n_points)
    """
    # Create 2D grid
    n_points = len(x_range)
    grid_2d = jnp.stack(jnp.meshgrid(x_range, x_range), axis=-1).reshape(-1, 2)

    def marginal_density_at_point(point_2d: Array) -> Array:
        """Compute marginal density at a single 2D point."""
        densities = []

        for loadings, means, diags in [
            (loadings_1, means_1, diags_1),
            (loadings_2, means_2, diags_2),
        ]:
            # FA observable distribution: N(μ, Λ Λ^T + Ψ)
            mu = means  # (obs_dim,)
            cov = jnp.dot(loadings, loadings.T) + jnp.diag(diags)  # (obs_dim, obs_dim)

            # Extract marginal for selected dimensions
            dims_array = jnp.array(dims)
            mu_marginal = mu[dims_array]  # (2,)
            cov_marginal = cov[dims_array][:, dims_array]  # (2, 2)

            # Evaluate 2D Gaussian density
            diff = point_2d - mu_marginal
            inv_cov = jnp.linalg.inv(cov_marginal)
            log_det = jnp.linalg.slogdet(cov_marginal)[1]
            log_dens = (
                -0.5 * jnp.dot(diff, jnp.dot(inv_cov, diff))
                - 0.5 * log_det
                - jnp.log(2 * jnp.pi)
            )
            densities.append(jnp.exp(log_dens))

        # Mixture density
        return mixing_weights[0] * densities[0] + mixing_weights[1] * densities[1]

    # Compute density at each grid point
    densities = jax.vmap(marginal_density_at_point)(grid_2d)

    return densities.reshape(n_points, n_points)


def compute_marginals_from_components(
    loadings_1: Array,
    means_1: Array,
    diags_1: Array,
    loadings_2: Array,
    means_2: Array,
    diags_2: Array,
    mixing_weights: Array,
    x_range: Array,
) -> tuple[Array, Array, Array]:
    """Compute marginals from known FA components."""
    dens_x1x2 = compute_marginal_density_2d(
        loadings_1,
        means_1,
        diags_1,
        loadings_2,
        means_2,
        diags_2,
        mixing_weights,
        x_range,
        dims=(0, 1),
    )
    dens_x1x3 = compute_marginal_density_2d(
        loadings_1,
        means_1,
        diags_1,
        loadings_2,
        means_2,
        diags_2,
        mixing_weights,
        x_range,
        dims=(0, 2),
    )
    dens_x2x3 = compute_marginal_density_2d(
        loadings_1,
        means_1,
        diags_1,
        loadings_2,
        means_2,
        diags_2,
        mixing_weights,
        x_range,
        dims=(1, 2),
    )
    return dens_x1x2, dens_x1x3, dens_x2x3


def compute_marginals_from_mfa(
    mfa: MixtureOfConjugated[Normal, Normal],
    mfa_params: Array,
    x_range: Array,
) -> tuple[Array, Array, Array]:
    """Compute 2D marginal densities by numerically integrating MFA density.

    This works for any MFA parameters without needing to extract component structure.

    Returns
    -------
    dens_x1x2, dens_x1x3, dens_x2x3 : Arrays
        2D density grids for each pair of dimensions
    """
    n_points = len(x_range)

    def compute_marginal_2d(dims: tuple[int, int], marg_dim: int) -> Array:
        """Compute marginal by integrating over marg_dim."""
        grid_2d = jnp.stack(jnp.meshgrid(x_range, x_range), axis=-1).reshape(-1, 2)

        # Integration range for marginalized dimension
        marg_range = jnp.linspace(-10, 10, 50)
        dmarg = marg_range[1] - marg_range[0]

        def marginal_at_point(point_2d: Array) -> Array:
            """Integrate p(x1, x2, x3) over x_marg."""

            def density_at_marg(x_marg: Array) -> Array:
                # Construct full 3D point
                point_3d = jnp.zeros(3)
                point_3d = point_3d.at[dims[0]].set(point_2d[0])
                point_3d = point_3d.at[dims[1]].set(point_2d[1])
                point_3d = point_3d.at[marg_dim].set(x_marg)
                return mfa.observable_density(mfa_params, point_3d)

            # Numerical integration
            densities = jax.vmap(density_at_marg)(marg_range)
            return jnp.sum(densities) * dmarg

        marginals = jax.vmap(marginal_at_point)(grid_2d)
        return marginals.reshape(n_points, n_points)

    # X1-X2 (marginalize over X3)
    dens_x1x2 = compute_marginal_2d(dims=(0, 1), marg_dim=2)

    # X1-X3 (marginalize over X2)
    dens_x1x3 = compute_marginal_2d(dims=(0, 2), marg_dim=1)

    # X2-X3 (marginalize over X1)
    dens_x2x3 = compute_marginal_2d(dims=(1, 2), marg_dim=0)

    return dens_x1x2, dens_x1x3, dens_x2x3


### Responsibility Computation ###


def compute_posterior_assignments(
    mfa: MixtureOfConjugated[Normal, Normal],
    params: Array,
    samples: Array,
) -> Array:
    """Compute posterior assignment probabilities p(z|x) for each sample.

    Parameters
    ----------
    mfa : MixtureOfConjugated
        The MFA model
    params : Array
        Model parameters (natural coordinates)
    samples : Array
        Observations (n_samples, obs_dim)

    Returns
    -------
    assignments : Array
        Posterior assignment probabilities (n_samples, n_components)
    """
    return jax.vmap(mfa.posterior_assignments, in_axes=(None, 0))(params, samples)


### Main Analysis ###


def compute_mfa_results(
    key: Array,
    sample_size: int = 1000,
    n_steps: int = 20000,
    n_points: int = 50,
) -> MFAResults:
    """Run complete MFA analysis.

    Parameters
    ----------
    key : Array
        Random key
    sample_size : int
        Number of samples to generate
    n_steps : int
        Number of gradient descent iterations
    n_points : int
        Grid resolution for density plots

    Returns
    -------
    MFAResults
        Complete analysis results
    """
    # Create MFA model (direct instantiation)
    base_fa = FactorAnalysis(obs_dim=3, lat_dim=2)
    mfa = MixtureOfConjugated(n_categories=2, hrm=base_fa)

    # Generate ground truth samples
    key_sample, key_train = jax.random.split(key)
    (
        samples,
        gt_components,
        gt_mfa_params,
        _fa1,
        _fa2,
        _fa1_params,
        _fa2_params,
        mixing_probs,
        loadings_1,
        means_1,
        diags_1,
        loadings_2,
        means_2,
        diags_2,
    ) = create_ground_truth_samples(key_sample, sample_size)

    # Fit model with a small learning rate to avoid numerical instability
    lls, init_params, final_params = fit_mfa(
        key_train, mfa, n_steps, samples, learning_rate=1e-2
    )

    # Create coordinate range for density plots
    x_range = jnp.linspace(-5.0, 5.0, n_points)

    # Compute marginal densities for ground truth using true FA components (analytical)
    gt_x1x2, gt_x1x3, gt_x2x3 = compute_marginals_from_components(
        loadings_1,
        means_1,
        diags_1,
        loadings_2,
        means_2,
        diags_2,
        mixing_probs,
        x_range,
    )

    # Compute marginal densities for fitted models by numerically integrating MFA density
    print("Computing initial marginals...")
    init_x1x2, init_x1x3, init_x2x3 = compute_marginals_from_mfa(
        mfa, init_params, x_range
    )

    print("Computing final marginals...")
    final_x1x2, final_x1x3, final_x2x3 = compute_marginals_from_mfa(
        mfa, final_params, x_range
    )

    # Compute posterior assignments
    gt_assignments = jnp.eye(2)[gt_components]  # One-hot encode ground truth
    final_assignments = compute_posterior_assignments(mfa, final_params, samples)

    # Compute ground truth log likelihood (using true ground truth params)
    gt_ll = mfa.average_log_observable_density(gt_mfa_params, samples)

    return MFAResults(
        observations=samples.tolist(),
        ground_truth_components=gt_components.tolist(),
        plot_range=x_range.tolist(),
        log_likelihoods=lls.tolist(),
        ground_truth_ll=float(gt_ll),
        ground_truth_density_x1x2=gt_x1x2.tolist(),
        ground_truth_density_x1x3=gt_x1x3.tolist(),
        ground_truth_density_x2x3=gt_x2x3.tolist(),
        initial_density_x1x2=init_x1x2.tolist(),
        initial_density_x1x3=init_x1x3.tolist(),
        initial_density_x2x3=init_x2x3.tolist(),
        final_density_x1x2=final_x1x2.tolist(),
        final_density_x1x3=final_x1x3.tolist(),
        final_density_x2x3=final_x2x3.tolist(),
        ground_truth_assignments=gt_assignments.tolist(),
        final_assignments=final_assignments.tolist(),
    )


def main():
    """Main entry point."""
    initialize_jax()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(0)

    results = compute_mfa_results(key)
    paths.save_analysis(results)
    print(f"Analysis complete. Results saved to {paths.results_dir}")
    print(f"Initial log likelihood: {results['log_likelihoods'][0]:.3f}")
    print(f"Final log likelihood: {results['log_likelihoods'][-1]:.3f}")
    print(
        f"Improvement: {results['log_likelihoods'][-1] - results['log_likelihoods'][0]:.3f}"
    )


if __name__ == "__main__":
    main()
