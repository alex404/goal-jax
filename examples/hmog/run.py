"""Hierarchical Mixture of Gaussians (HMoG) example.

Fits HMoGs with isotropic and diagonal observable noise by EM, and the diagonal HMoG again as a
variational model of two levels: the linear Gaussian model $p(x \\mid y)$ gets a constant
conjugation function $\\rho_Y$, and the mixture $p(y \\mid k)$ a perceptron $\\rho_K$ of its $y$
bias, both learned with the other parameters by Adam on the ELBO minus a residual variance penalty
on $\\rho$ alone. Both levels are conjugated, so the learned $\\rho_Y$ and the recognition model
$q(k \\mid x)$ that $\\rho_K$ defines can be compared with the exact ones at the same likelihood.

Every fit starts from a diagonal Gaussian mixture fit to the sample in $x$: the loading points
along the difference of its component means. From the default initialization, EM sits on the
plateau of one effective component for half of the seeds.
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.geometry import (
    BiasConjugationFunction,
    ConstantConjugationFunction,
    Diagonal,
    DifferentiableDeepModel,
    MultilayerPerceptron,
    PositiveDefinite,
    Scale,
)
from goal.geometry.exponential_family.graphical import GraphicalLevel
from goal.geometry.exponential_family.variational import VariationalConjugated
from goal.models import AnalyticHMoG, AnalyticMixture, analytic_hmog, diagonal_normal

from ..shared import example_paths, jax_cli
from .types import HMoGResults, VariationalHistory

# Model configuration
width = 0.1
height = 8.0
sep = 3.0


def create_ground_truth() -> tuple[AnalyticHMoG[Diagonal], Array]:
    """Create ground truth HMoG model."""
    hmog = analytic_hmog(obs_dim=2, obs_rep=Diagonal(), lat_dim=1, n_components=2)

    cm = hmog.pst_man.prr_man
    cat_natural = jnp.array([0.5])
    cat_mean = cm.to_mean(cat_natural)

    um = hmog.pst_man
    lm = um.obs_man
    y0_means = lm.join_mean_covariance(jnp.array([-sep / 2]), jnp.array([1.0]))
    y1_means = lm.join_mean_covariance(jnp.array([sep / 2]), jnp.array([1.0]))
    components = jnp.concatenate([y0_means, y1_means])
    mix_mean = um.join_mean_mixture(components, cat_mean)
    mix_natural = um.to_natural(mix_mean)

    om = hmog.lwr_hrm.obs_man
    obs_mean = om.join_mean_covariance(
        jnp.array([0.0, 0.0]), jnp.array([width, height])
    )
    obs_natural = om.to_natural(obs_mean)
    int_mat0 = jnp.array([1.0, 0.0])
    obs_prs = om.split_location_precision(obs_natural)[1]
    (xy_map,) = hmog.crs_maps
    int_mat = xy_map.from_matrix(om.cov_man.to_matrix(obs_prs) @ int_mat0)

    lkl_params = hmog.lkl_fun_man.join_coords(obs_natural, int_mat)
    return hmog, hmog.join_conjugated(lkl_params, mix_natural)


def initialize_from_mixture[Rep: PositiveDefinite](
    key: Array, hmog: AnalyticHMoG[Rep], sample: Array
) -> Array:
    """HMoG natural parameters from a two-component diagonal Gaussian mixture fit to the sample by EM.

    With component means $\\mu_0, \\mu_1$, the observable mean is $(\\mu_0 + \\mu_1) / 2$, the
    loading $(\\mu_1 - \\mu_0) / 2$, the observable covariance the average component covariance,
    and $y \\mid k$ is placed at $\\mp 1$ with variance $0.1$, with the mixture's weights.
    """
    nrm = diagonal_normal(hmog.obs_man.data_dim)
    gmm = AnalyticMixture(nrm, 2)

    def em_step(params: Array, _: Any) -> tuple[Array, None]:
        return gmm.expectation_maximization(params, sample), None

    gmm_params, _ = jax.lax.scan(
        em_step, gmm.initialize_from_sample(key, sample, shape=0.1), None, length=500
    )
    comps, weights = gmm.split_natural_mixture(gmm_params)
    mus, covs = jax.vmap(lambda c: nrm.split_mean_covariance(nrm.to_mean(c)))(
        gmm.cmp_man.to_2d(comps)
    )

    obs_man = hmog.lwr_hrm.obs_man
    cov = obs_man.cov_man.from_matrix(jnp.diag(jnp.mean(covs, axis=0)))
    obs_params = obs_man.to_natural(
        obs_man.join_mean_covariance((mus[0] + mus[1]) / 2, cov)
    )
    obs_prs = obs_man.split_location_precision(obs_params)[1]
    (xy_map,) = hmog.crs_maps
    int_params = xy_map.from_matrix(
        obs_man.cov_man.to_matrix(obs_prs) @ ((mus[1] - mus[0]) / 2)
    )

    mix = hmog.pst_man
    lat_man = mix.obs_man
    lat_comps = jnp.concatenate(
        [
            lat_man.join_mean_covariance(jnp.array([-1.0]), jnp.array([0.1])),
            lat_man.join_mean_covariance(jnp.array([1.0]), jnp.array([0.1])),
        ]
    )
    mix_params = mix.to_natural(
        mix.join_mean_mixture(lat_comps, mix.pst_man.to_mean(weights))
    )
    lkl_params = hmog.lkl_fun_man.join_coords(obs_params, int_params)
    return hmog.join_conjugated(lkl_params, mix_params)


def fit_hmog[Rep: PositiveDefinite](
    key: Array, hmog: AnalyticHMoG[Rep], sample: Array, n_steps: int
) -> tuple[Array, Array, Array]:
    """Fit HMoG via EM."""
    init_params = initialize_from_mixture(key, hmog, sample)

    def em_step(params: Array, _: Any) -> tuple[Array, Array]:
        ll = hmog.average_log_observable_density(params, sample)
        return hmog.expectation_maximization(params, sample), ll

    final_params, lls = jax.lax.scan(em_step, init_params, None, length=n_steps)
    return lls.ravel(), init_params, final_params


fit_hmog = jax.jit(fit_hmog, static_argnames=["hmog", "n_steps"])


def fit_variational(
    key: Array,
    hmog: AnalyticHMoG[Diagonal],
    sample: Array,
    n_steps: int,
    chunk: int,
) -> VariationalHistory:
    """Fit the HMoG as two variational levels by Adam on the ELBO minus a residual variance penalty on $\\rho$.

    The penalty is the variance of each level's conjugation residual, at the prior and, for the
    mixture, at the recognition models of a minibatch. The harmonium parameters and the latent
    samples carry no gradient through it, so it fits $\\rho$ to the current harmonium by least
    squares, which fixes the constant of $\\rho_K$ that the ELBO leaves free, and does not pull
    the harmonium toward conjugation.
    """
    lwr, mix = hmog.lwr_hrm, hmog.pst_man
    cat = mix.pst_man
    rho_k_map = MultilayerPerceptron(cat, mix.obs_man, (16,), jax.nn.tanh)
    upper = VariationalConjugated(
        GraphicalLevel(((mix, (0,)),), cat),
        (BiasConjugationFunction(rho_k_map, mix.lkl_fun_man),),
        DifferentiableDeepModel(cat),
    )
    model = VariationalConjugated(
        GraphicalLevel(((lwr, (0,)),), upper.gph_hrm),
        (ConstantConjugationFunction(lwr.pst_man, lwr.lkl_fun_man),),
        upper,
    )
    penalty_weight = 0.1

    def residual_variance(
        level: VariationalConjugated[Any], params: Array, key: Array, n: int
    ) -> Array:
        """Variance of the residual of the top level of ``params``, at samples of its fixed prior."""
        zs = level.dep_man.sample(
            key, jax.lax.stop_gradient(level.conjugated_prior_params(params)), n
        )
        return jnp.var(jax.vmap(lambda z: level.conjugation_residual(params, z))(zs))

    def rho_penalty(key: Array, params: Array) -> Array:
        k_y, k_k, k_idx, k_q = jax.random.split(key, 4)
        hrm_params, cnj_params = model.split_coords(params)
        _, rho_k_params = model.snd_man.split_coords(cnj_params)
        fixed_hrm = jax.lax.stop_gradient(hrm_params)
        fixed = jax.lax.stop_gradient(params)

        def at_upper(up_params: Array) -> Array:
            up_hrm, _ = upper.split_coords(jax.lax.stop_gradient(up_params))
            return upper.join_coords(up_hrm, rho_k_params)

        var_y = residual_variance(
            model, model.join_coords(fixed_hrm, cnj_params), k_y, 64
        )
        var_k = residual_variance(
            upper, at_upper(model.conjugated_prior_params(fixed)), k_k, 64
        )
        idx = jax.random.choice(k_idx, sample.shape[0], (64,), replace=False)
        var_kq = jnp.mean(
            jax.vmap(
                lambda k, x: residual_variance(
                    upper, at_upper(model.recognition_at(fixed, x)), k, 16
                )
            )(jax.random.split(k_q, 64), sample[idx])
        )
        return var_y + var_k + var_kq

    def loss(key: Array, params: Array) -> Array:
        k_elbo, k_pen = jax.random.split(key)
        elbo = model.mean_elbo(k_elbo, params, sample, 16)
        return -elbo + penalty_weight * rho_penalty(k_pen, params)

    optimizer = optax.adam(1e-2)

    def step(carry: tuple[Array, Any, Array], _: None) -> tuple[Any, None]:
        params, opt_state, key = carry
        key, k_loss = jax.random.split(key)
        grads = jax.grad(lambda p: loss(k_loss, p))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state, key), None

    def k_logits(params: Array, x: Array) -> tuple[Array, Array]:
        """Logit of $k = 1$ under the recognition model and under the exact posterior of the harmonium."""
        rec_logit = upper.conjugated_prior_params(model.recognition_at(params, x))
        hrm_params, _ = model.split_coords(params)
        _, exact_logit = mix.split_conjugated(hmog.posterior_at(hrm_params, x))
        return rec_logit[0], exact_logit[0]

    @jax.jit
    def measure(params: Array) -> tuple[Array, Array, Array, Array]:
        hrm_params, cnj_params = model.split_coords(params)
        rho_y, _ = model.snd_man.split_coords(cnj_params)
        obs_params, int_params, _ = hmog.split_coords(hrm_params)
        exact_rho_y = lwr.conjugation_parameters(
            hmog.lkl_fun_man.join_coords(obs_params, int_params)
        )
        elbo = model.mean_elbo(jax.random.PRNGKey(0), params, sample, 256)
        ll = hmog.average_log_observable_density(hrm_params, sample)
        return elbo, ll, rho_y, exact_rho_y

    k_init, k_rho, k_train = jax.random.split(key, 3)
    rho_y = jnp.zeros(lwr.pst_man.dim)
    rho_k = rho_k_map.glorot_initialize(k_rho)
    params = model.join_coords(
        initialize_from_mixture(k_init, hmog, sample),
        model.snd_man.join_coords(rho_y, rho_k),
    )
    run_chunk = jax.jit(lambda carry: jax.lax.scan(step, carry, None, chunk)[0])
    carry = (params, optimizer.init(params), k_train)
    history = VariationalHistory(
        steps=[],
        elbo=[],
        log_likelihood=[],
        rho=[],
        exact_rho=[],
        recognition_logits=[],
        exact_logits=[],
    )
    for i in range(n_steps // chunk + 1):
        if i > 0:
            carry = run_chunk(carry)
        elbo, ll, rho, exact_rho = measure(carry[0])
        history["steps"].append(i * chunk)
        history["elbo"].append(float(elbo))
        history["log_likelihood"].append(float(ll))
        history["rho"].append(rho.tolist())
        history["exact_rho"].append(exact_rho.tolist())
    rec_logits, exact_logits = jax.jit(jax.vmap(k_logits, in_axes=(None, 0)))(
        carry[0], sample
    )
    history["recognition_logits"] = rec_logits.tolist()
    history["exact_logits"] = exact_logits.tolist()
    return history


def main():
    jax_cli()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(0)
    key_sample, key_test, key_train = jax.random.split(key, 3)

    # Ground truth
    gt_hmog, gt_params = create_ground_truth()
    sample = gt_hmog.observable_sample(key_sample, gt_params, 400)
    test_sample = gt_hmog.observable_sample(key_test, gt_params, 4000)

    # Models
    iso_hmog = analytic_hmog(2, Scale(), 1, 2)
    dia_hmog = analytic_hmog(2, Diagonal(), 1, 2)

    # Training
    iso_lls, iso_init, iso_final = fit_hmog(key_train, iso_hmog, sample, 5000)
    dia_lls, dia_init, dia_final = fit_hmog(key_train, dia_hmog, sample, 5000)
    variational = fit_variational(key_train, dia_hmog, sample, 5000, 100)

    # Grid for densities
    x = jnp.linspace(-10, 10, 100)
    y = jnp.linspace(-10, 10, 100)
    grid_points = jnp.stack([g.ravel() for g in jnp.meshgrid(x, y)], axis=1)
    lat_y = jnp.linspace(-6, 6, 100)

    def obs_density[Rep: PositiveDefinite](
        model: AnalyticHMoG[Rep], params: Array
    ) -> Array:
        return jax.vmap(model.observable_density, in_axes=(None, 0))(
            params, grid_points
        ).reshape(100, 100)

    def mix_density[Rep: PositiveDefinite](
        model: AnalyticHMoG[Rep], params: Array
    ) -> Array:
        mix_params = model.split_conjugated(params)[1]
        return jax.vmap(model.pst_man.observable_density, in_axes=(None, 0))(
            mix_params, lat_y[:, None]
        )

    results = HMoGResults(
        plot_range_x1=x.tolist(),
        plot_range_x2=y.tolist(),
        observations=sample.tolist(),
        log_likelihoods={"Isotropic": iso_lls.tolist(), "Diagonal": dia_lls.tolist()},
        test_log_likelihoods={
            "Ground Truth": float(
                gt_hmog.average_log_observable_density(gt_params, test_sample)
            ),
            "Isotropic": float(
                iso_hmog.average_log_observable_density(iso_final, test_sample)
            ),
            "Diagonal": float(
                dia_hmog.average_log_observable_density(dia_final, test_sample)
            ),
        },
        variational=variational,
        observable_densities={
            "Ground Truth": obs_density(gt_hmog, gt_params).tolist(),
            "Isotropic Initial": obs_density(iso_hmog, iso_init).tolist(),
            "Isotropic Final": obs_density(iso_hmog, iso_final).tolist(),
            "Diagonal Initial": obs_density(dia_hmog, dia_init).tolist(),
            "Diagonal Final": obs_density(dia_hmog, dia_final).tolist(),
        },
        plot_range_y=lat_y.tolist(),
        mixture_densities={
            "Ground Truth": mix_density(gt_hmog, gt_params).tolist(),
            "Isotropic Initial": mix_density(iso_hmog, iso_init).tolist(),
            "Isotropic Final": mix_density(iso_hmog, iso_final).tolist(),
            "Diagonal Initial": mix_density(dia_hmog, dia_init).tolist(),
            "Diagonal Final": mix_density(dia_hmog, dia_final).tolist(),
        },
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
