"""Tests for geometry/exponential_family/variational.py, on shipped models.

A single level: ground-truth verification of the variational estimators and their
stop_gradient policy, on a ``PoissonPopulationCode`` with a single VonMises latent, a
and a ``BoltzmannPopulationCode`` with a one-dimensional Normal latent, both fit as a
``VariationalPopulationCode``.
Every latent expectation is then an integral over a grid: over [0, 2pi) for
the VonMises latent, where the periodic trapezoid rule is exact to near machine precision
for smooth integrands, and over a wide interval for the Normal latent. ``jax.grad`` of the quadrature expressions therefore yields exact
gradients --- including the dependence of the sampling measure on the parameters ---
against which the autodiff gradients of the Monte-Carlo estimators are compared over many
independent keys (a z-test on each parameter coordinate).

This pins down the stop_gradient sites: a missing sg would double-count the
direct gradient, a spurious sg would drop the score correction, and either
error shifts the MC gradient mean away from the exact gradient by far more
than its standard error.

Nested levels: the canonical circuit (models/variational/population_codes.py) in both variants,
its ELBO decomposition at depths 1 and 2, and at depth 1 its ELBO against enumeration of the neurons
and quadrature, and its recognition model normalizes at random conjugation functions.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import JunctionTree, PositiveDefinite
from goal.geometry.exponential_family.variational import (
    DifferentiableVariationalConjugated,
    regress_conjugation_parameters,
)
from goal.models import (
    BoltzmannPopulationCode,
    CanonicalCircuit,
    ChordalBoltzmann,
    Normal,
    PoissonPopulationCode,
    VonMisesProduct,
    canonical_circuit,
    variational_population_code,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

N_KEYS = 128
N_MC = 128
Z_THRESHOLD = 5.0


@dataclass(frozen=True)
class Case:
    name: str
    model: DifferentiableVariationalConjugated[Any, Any]
    x: Array
    z_grid: Array
    """Quadrature points of the latent, on an equally spaced grid."""
    dz: float
    """Volume of a grid cell."""


N_GRID = 2048
CASES = [
    Case(
        "von_mises",
        variational_population_code(PoissonPopulationCode(VonMisesProduct(1), 4)),
        jnp.array([2.0, 0.0, 1.0, 3.0]),
        (jnp.arange(N_GRID) / N_GRID * 2 * jnp.pi).reshape(-1, 1),
        2 * jnp.pi / N_GRID,
    ),
    Case(
        "boltzmann",
        variational_population_code(
            BoltzmannPopulationCode(
                Normal(1, PositiveDefinite()),
                ChordalBoltzmann.from_edges(4, [(0, 1), (1, 2), (2, 3)]),
            )
        ),
        jnp.array([1.0, 0.0, 1.0, 1.0]),
        jnp.linspace(-30.0, 30.0, N_GRID).reshape(-1, 1),
        60.0 / (N_GRID - 1),
    ),
]
CASE_IDS = [case.name for case in CASES]


def _params(case: Case) -> Array:
    """Generic (non-conjugate, nonzero-rho) parameters."""
    model = case.model
    k_init, k_rho = jax.random.split(jax.random.PRNGKey(0))
    hrm_p, _ = model.split_coords(model.initialize(k_init, shape=0.5))
    rho = 0.3 * jax.random.normal(k_rho, (model.cnj_fun_man.dim,))
    return model.join_coords(hrm_p, rho)


# --- Quadrature ground truths (exact values and exact autodiff gradients) ---


def _recognition_weights(case: Case, params: Array) -> tuple[Array, Array]:
    model = case.model
    q_params = model.recognition_at(params, case.x)
    log_q = jax.vmap(lambda z: model.pst_man.log_density(q_params, z))(case.z_grid)
    return jnp.exp(log_q), log_q


def _prior_log_density(case: Case, params: Array) -> Array:
    model = case.model
    prior_p = model.conjugated_prior_params(params)
    return jax.vmap(lambda z: model.pst_man.log_density(prior_p, z))(case.z_grid)


def _exact_elbo(case: Case, params: Array) -> Array:
    model = case.model
    w, log_q = _recognition_weights(case, params)
    lkl_nat = jax.vmap(lambda z: model.likelihood_at(params, z))(case.z_grid)
    xs = jnp.tile(case.x, (len(case.z_grid), 1))
    log_pxz = jax.vmap(model.obs_man.log_density)(lkl_nat, xs)
    log_pz = _prior_log_density(case, params)
    return case.dz * jnp.sum(w * (log_pxz + log_pz - log_q))


def _exact_mean_r(case: Case, params: Array) -> Array:
    w, _ = _recognition_weights(case, params)
    r = jax.vmap(lambda z: case.model.elbo_residual(params, case.x, z))(case.z_grid)
    return case.dz * jnp.sum(w * r)


def _exact_kl(case: Case, params: Array) -> Array:
    w, log_q = _recognition_weights(case, params)
    return case.dz * jnp.sum(w * (log_q - _prior_log_density(case, params)))


def _exact_var_q_r(case: Case, params: Array) -> Array:
    w, _ = _recognition_weights(case, params)
    r = jax.vmap(lambda z: case.model.elbo_residual(params, case.x, z))(case.z_grid)
    mean_r = case.dz * jnp.sum(w * r)
    return case.dz * jnp.sum(w * (r - mean_r) ** 2)


def _exact_var_p_r(case: Case, params: Array) -> Array:
    w = jnp.exp(_prior_log_density(case, params))
    r = jax.vmap(lambda z: case.model.conjugation_residual(params, z))(case.z_grid)
    mean_r = case.dz * jnp.sum(w * r)
    return case.dz * jnp.sum(w * (r - mean_r) ** 2)


# --- MC estimator statistics over independent keys ---


def _mc_value_and_grad(fn, params: Array) -> tuple[Array, Array, Array, Array]:
    """Mean value/gradient of a stochastic estimator with standard errors."""
    keys = jax.random.split(jax.random.PRNGKey(42), N_KEYS)
    vals, grads = jax.vmap(jax.value_and_grad(fn, argnums=1))(
        keys, jnp.tile(params, (N_KEYS, 1))
    )
    val_se = jnp.std(vals) / jnp.sqrt(N_KEYS)
    grad_se = jnp.std(grads, axis=0) / jnp.sqrt(N_KEYS)
    return jnp.mean(vals), val_se, jnp.mean(grads, axis=0), grad_se


def _assert_matches(
    exact_val: Array,
    exact_grad: Array,
    mc_val: Array,
    val_se: Array,
    mc_grad: Array,
    grad_se: Array,
) -> None:
    assert jnp.abs(mc_val - exact_val) < Z_THRESHOLD * val_se + 1e-9
    z_scores = jnp.abs(mc_grad - exact_grad) / jnp.maximum(grad_se, 1e-9)
    assert jnp.max(z_scores) < Z_THRESHOLD


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_quadrature_is_exact(case: Case):
    """The grid carries the whole mass of the recognition model and the prior."""
    params = _params(case)
    w, _ = _recognition_weights(case, params)
    prior_mass = case.dz * jnp.sum(jnp.exp(_prior_log_density(case, params)))
    assert jnp.allclose(case.dz * jnp.sum(w), 1.0, atol=1e-10)
    assert jnp.allclose(prior_mass, 1.0, atol=1e-10)


class TestStandardFormElbo:
    """elbo_at: value and full gradient against exact quadrature."""

    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    @pytest.mark.parametrize("n_mc", [1, 2, N_MC])
    def test_value_and_gradient_match_quadrature(self, case: Case, n_mc: int):
        """Also with one or two samples, where the leave-one-out baseline degenerates or is rescaled."""
        model, params = case.model, _params(case)
        exact_val = _exact_elbo(case, params)
        exact_grad = jax.grad(lambda p: _exact_elbo(case, p))(params)
        mc_val, val_se, mc_grad, grad_se = _mc_value_and_grad(
            lambda k, p: model.elbo_at(k, p, case.x, n_mc), params
        )
        # The gradient must be right in every block: a wrong sg on the samples
        # or on r inside the score term would shift the rho/prior blocks.
        _assert_matches(exact_val, exact_grad, mc_val, val_se, mc_grad, grad_se)

    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_standard_form_decomposition(self, case: Case):
        """Quadrature identity L(x) = c(x) + E_q[r] --- validates the c/r split."""
        params = _params(case)
        c_x = case.model.conjugation_baseline(params, case.x)
        lhs = _exact_elbo(case, params)
        rhs = c_x + _exact_mean_r(case, params)
        assert jnp.allclose(lhs, rhs, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_value_excludes_score_term(self, case: Case):
        """The ``- sg(score)`` site: the returned value is exactly c(x) + mean(r).

        Replicates the internal sampling with the same key, so any leakage of
        the score correction into the value would show up exactly.
        """
        model, params = case.model, _params(case)
        key = jax.random.PRNGKey(7)
        surrogate = model.elbo_at(key, params, case.x, N_MC)

        z_samples = model.sample_recognition(key, params, case.x, N_MC)
        r_vals = jax.vmap(lambda z: model.elbo_residual(params, case.x, z))(z_samples)
        clean = model.conjugation_baseline(params, case.x) + jnp.mean(r_vals)
        assert jnp.allclose(surrogate, clean, rtol=1e-12, atol=1e-12)


class TestElboDivergence:
    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_kl_matches_quadrature(self, case: Case):
        params = _params(case)
        closed_form = case.model.elbo_divergence(params, case.x)
        assert jnp.allclose(closed_form, _exact_kl(case, params), rtol=1e-8, atol=1e-10)


class TestRecognitionResidualVariance:
    """Var_q[r] with score correction: the sampling measure shares parameters
    with r, so the exact gradient has both a direct and a score term."""

    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_value_and_gradient_match_quadrature(self, case: Case):
        model, params = case.model, _params(case)
        exact_val = _exact_var_q_r(case, params)
        exact_grad = jax.grad(lambda p: _exact_var_q_r(case, p))(params)
        mc_val, val_se, mc_grad, grad_se = _mc_value_and_grad(
            lambda k, p: model.recognition_residual_variances_at(k, p, case.x, N_MC)[0],
            params,
        )
        # Dropping the score correction here would bias the prior/rho blocks
        # (the measure q depends on theta_Z, rho, and Theta).
        _assert_matches(exact_val, exact_grad, mc_val, val_se, mc_grad, grad_se)


class TestPriorResidualVariance:
    """Var_p[r] and its full gradient, incl. the score term through theta_Z."""

    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_value_and_gradient_match_quadrature(self, case: Case):
        model, params = case.model, _params(case)
        exact_val = _exact_var_p_r(case, params)
        exact_grad = jax.grad(lambda p: _exact_var_p_r(case, p))(params)
        mc_val, val_se, mc_grad, grad_se = _mc_value_and_grad(
            lambda k, p: model.conjugation_residual_variances(k, p, N_MC)[0], params
        )
        _assert_matches(exact_val, exact_grad, mc_val, val_se, mc_grad, grad_se)


class TestRegression:
    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_fit_lowers_the_residual_variance(self, case: Case):
        """``regress_conjugation_parameters`` against $\\rho = 0$, at the same prior, by quadrature."""
        model, params = case.model, _params(case)
        hrm_params, _ = model.split_coords(params)
        obs_p, int_p, _ = model.gph_hrm.split_coords(hrm_params)
        prior = model.conjugated_prior_params(params)
        rho_zero = model.cnj_fun_man.zeros()

        def at_prior(rho: Array) -> Array:
            hrm = model.gph_hrm.join_coords(obs_p, int_p, prior - rho)
            return model.join_coords(hrm, rho)

        rho_fit, _, _, _ = regress_conjugation_parameters(
            model, jax.random.PRNGKey(3), at_prior(rho_zero), 4000
        )
        assert _exact_var_p_r(case, at_prior(rho_fit)) < 0.5 * _exact_var_p_r(
            case, at_prior(rho_zero)
        )


# --- Nested levels: the canonical circuit ---

VARIANTS = ["partially_exact", "approximate"]
DEPTHS = [1, 2]
X_CIRCUIT = jnp.array([0.4])


def _circuit(variant: str, depth: int) -> CanonicalCircuit:
    """Three neurons per layer on a chain, one-dimensional latents and observable."""
    return canonical_circuit(
        1,
        PositiveDefinite(),
        (JunctionTree.from_edges(3, ((0, 1), (1, 2))),) * depth,
        (1,) * depth,
        variant,  # pyright: ignore[reportArgumentType]
        (4,),
    )


def _standard_normal_params(nrm: Any) -> Array:
    """Natural parameters of the standard normal."""
    return nrm.join_location_precision(
        jnp.zeros(nrm.data_dim), nrm.cov_man.from_matrix(jnp.eye(nrm.data_dim))
    )


def _circuit_params(circuit: CanonicalCircuit, key: Array) -> Array:
    """Random natural parameters: standard normal Gaussians, random neurons, tuning curves with positive precision, and random conjugation functions."""
    keys = iter(jax.random.split(key, 8 * len(circuit.lvl_mans)))

    def draw(n: int) -> Array:
        return 0.5 * jax.random.normal(next(keys), (n,))

    obs_int_paramss: list[tuple[Array, Array]] = []
    cnj_fun_paramss: list[Array] = []
    levels = circuit.lvl_mans
    for gb_level, pop_level in zip(levels[::2], levels[1::2], strict=True):
        (gb_hrm,), (gb_fun,) = gb_level.gph_hrm.obs_hrms, gb_level.cnj_funs
        (pop_hrm,), (pop_fun,) = pop_level.gph_hrm.obs_hrms, pop_level.cnj_funs
        obs_int_paramss.append(
            (_standard_normal_params(gb_hrm.obs_man), draw(gb_hrm.int_man.dim))
        )
        lat = pop_hrm.pst_man
        shape = (pop_hrm.obs_man.data_dim, lat.data_dim)
        rows = jax.vmap(lat.join_location_precision)(
            draw(shape[0] * shape[1]).reshape(shape),
            jnp.abs(draw(shape[0] * shape[1])).reshape(shape) + 0.5,
        )
        (nz_map,) = pop_hrm.crs_maps
        obs_int_paramss.append((draw(pop_hrm.obs_man.dim), nz_map.from_matrix(rows)))
        cnj_fun_paramss.append(draw(gb_fun.dim))
        cnj_fun_paramss.append(draw(pop_fun.dim))
    hrm_params = _standard_normal_params(levels[-1].pst_man)
    for level, (obs_params, int_params) in zip(
        reversed(levels), reversed(obs_int_paramss), strict=True
    ):
        hrm_params = level.gph_hrm.join_coords(obs_params, int_params, hrm_params)
    return circuit.join_coords(hrm_params, jnp.concatenate(cnj_fun_paramss))


def _with_conjugation_noise(
    circuit: CanonicalCircuit, params: Array, key: Array, scale: float
) -> Array:
    """Circuit parameters with Gaussian noise of s.d. ``scale`` on every conjugation function's parameters."""
    hrm_params, cnj_fun_params = circuit.split_coords(params)
    noise = scale * jax.random.normal(key, cnj_fun_params.shape)
    return circuit.join_coords(hrm_params, cnj_fun_params + noise)


def _exact_circuit_elbo(circuit: CanonicalCircuit, params: Array) -> Array:
    """The ELBO of a one-layer circuit: the three neurons enumerated, the latent by quadrature.

    Its gradient includes the dependence of the recognition model on the parameters at both
    levels, the target of the score terms of the nested estimator.
    """
    states = ((jnp.arange(8)[:, None] >> jnp.arange(3)) & 1).astype(float)
    zs = jnp.linspace(-50.0, 50.0, 4001)[:, None]
    ys = jnp.concatenate(
        [jnp.repeat(states, len(zs), axis=0), jnp.tile(zs, (len(states), 1))], axis=1
    )
    log_q = jax.vmap(lambda y: circuit.recognition_log_density(params, X_CIRCUIT, y))(
        ys
    )
    log_p = jax.vmap(
        lambda y: circuit.log_density(params, jnp.concatenate([X_CIRCUIT, y]))
    )(ys)
    return (zs[1, 0] - zs[0, 0]) * jnp.sum(jnp.exp(log_q) * (log_p - log_q))


class TestCanonicalCircuit:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_elbo_matches_quadrature(self, variant: str):
        """Value and gradient away from conjugacy, where every level has a score term."""
        circuit = _circuit(variant, 1)
        params = _circuit_params(circuit, jax.random.PRNGKey(6))
        exact_val = _exact_circuit_elbo(circuit, params)
        exact_grad = jax.grad(lambda p: _exact_circuit_elbo(circuit, p))(params)
        mc_val, val_se, mc_grad, grad_se = _mc_value_and_grad(
            lambda k, p: circuit.elbo_at(k, p, X_CIRCUIT, 16), params
        )
        _assert_matches(exact_val, exact_grad, mc_val, val_se, mc_grad, grad_se)

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("depth", DEPTHS)
    def test_decomposition(self, variant: str, depth: int):
        """$\\log \\tilde p(x, z) - \\log q(z \\mid x) = c(x) + $ the ELBO residual, pointwise."""
        circuit = _circuit(variant, depth)
        params = _circuit_params(circuit, jax.random.PRNGKey(3))
        zs = circuit.sample_recognition(jax.random.PRNGKey(4), params, X_CIRCUIT, 8)

        def gap(z: Array) -> Array:
            log_joint = circuit.log_density(params, jnp.concatenate([X_CIRCUIT, z]))
            log_q = circuit.recognition_log_density(params, X_CIRCUIT, z)
            return log_joint - log_q - circuit.elbo_residual(params, X_CIRCUIT, z)

        gaps = jax.vmap(gap)(zs)
        baseline = circuit.conjugation_baseline(params, X_CIRCUIT)
        assert jnp.allclose(gaps, baseline, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_recognition_model_normalizes(self, variant: str):
        """The recognition model is a distribution at random conjugation functions: the conjugation function of the population code adds precision to $z$.

        The grid of $z$ spans the samples of $q$ by ten standard deviations on each side, as random
        conjugation functions can place $q$ far from the origin.
        """
        circuit = _circuit(variant, 1)
        params0 = _circuit_params(circuit, jax.random.PRNGKey(12))
        states = ((jnp.arange(8)[:, None] >> jnp.arange(3)) & 1).astype(float)
        for i in range(3):
            params = _with_conjugation_noise(
                circuit, params0, jax.random.PRNGKey(13 + i), 1.0
            )
            samples = circuit.sample_recognition(
                jax.random.PRNGKey(16 + i), params, X_CIRCUIT, 1000
            )[:, -1]
            spread = 10.0 * jnp.std(samples)
            zs = jnp.linspace(
                jnp.min(samples) - spread, jnp.max(samples) + spread, 20001
            )[:, None]
            ys = jnp.concatenate(
                [jnp.repeat(states, len(zs), axis=0), jnp.tile(zs, (len(states), 1))],
                axis=1,
            )
            log_q = jax.vmap(
                lambda y: circuit.recognition_log_density(params, X_CIRCUIT, y)
            )(ys)
            total = (zs[1, 0] - zs[0, 0]) * jnp.sum(jnp.exp(log_q))
            assert jnp.allclose(total, 1.0, atol=1e-6)
