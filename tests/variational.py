"""Tests for geometry/exponential_family/variational.py, on three shipped models.

A single level: ground-truth verification of the variational estimators and their
stop_gradient policy, on a ``PoissonPopulationCode`` with a single VonMises latent, a
``BoltzmannPopulationCode`` with a one-dimensional Normal latent, and a
``PoissonPopulationCode`` with two subpopulations, each tuned to its own VonMises latent.
Every latent expectation is then an integral over a grid: over [0, 2pi) or its square for
the VonMises latents, where the periodic trapezoid rule is exact to near machine precision
for smooth integrands, and over a wide interval for the Normal latent. ``jax.grad`` of the quadrature expressions therefore yields exact
gradients --- including the dependence of the sampling measure on the parameters ---
against which the autodiff gradients of the Monte-Carlo estimators are compared over many
independent keys (a z-test on each parameter coordinate).

This pins down the stop_gradient sites: a missing sg would double-count the
direct gradient, a spurious sg would drop the score correction, and either
error shifts the MC gradient mean away from the exact gradient by far more
than its standard error.

Nested levels: the canonical circuit (models/graphical/canonical_circuit.py) at depths 1 and 2,
in both variants, against its exact log-likelihood by enumeration of the neurons, and at
depth 1 against its exact ELBO by enumeration and quadrature.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import DifferentiableTuple, PositiveDefinite
from goal.geometry.exponential_family.variational import (
    DifferentiableVariationalConjugated,
    regress_conjugation_parameters,
)
from goal.models import (
    CanonicalCircuit,
    PoissonPopulationCode,
    PoissonPopulationHarmonium,
    VonMisesProduct,
    canonical_circuit,
    chordal_boltzmann_population_code,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

N_KEYS = 128
N_MC = 128
Z_THRESHOLD = 5.0


@dataclass(frozen=True)
class Case:
    name: str
    model: DifferentiableVariationalConjugated[Any, Any, Any]
    x: Array
    z_grid: Array
    """Quadrature points of the latent, on an equally spaced grid."""
    dz: float
    """Volume of a grid cell."""


N_GRID = 2048
N_GRID_2D = 128
_ANGLES = jnp.arange(N_GRID_2D) / N_GRID_2D * 2 * jnp.pi
CASES = [
    Case(
        "von_mises",
        PoissonPopulationCode(
            PoissonPopulationHarmonium((4,), VonMisesProduct(1), ((0, 0),))
        ),
        jnp.array([2.0, 0.0, 1.0, 3.0]),
        (jnp.arange(N_GRID) / N_GRID * 2 * jnp.pi).reshape(-1, 1),
        2 * jnp.pi / N_GRID,
    ),
    Case(
        "boltzmann",
        chordal_boltzmann_population_code(4, [(0, 1), (1, 2), (2, 3)], 1),
        jnp.array([1.0, 0.0, 1.0, 1.0]),
        jnp.linspace(-30.0, 30.0, N_GRID).reshape(-1, 1),
        60.0 / (N_GRID - 1),
    ),
    Case(
        "poisson_subpopulations",
        PoissonPopulationCode(
            PoissonPopulationHarmonium(
                (3, 2),
                DifferentiableTuple((VonMisesProduct(1), VonMisesProduct(1))),
                ((0, 0), (1, 1)),
            )
        ),
        jnp.array([2.0, 0.0, 1.0, 3.0, 1.0]),
        jnp.stack(jnp.meshgrid(_ANGLES, _ANGLES, indexing="ij"), axis=-1).reshape(
            -1, 2
        ),
        (2 * jnp.pi / N_GRID_2D) ** 2,
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
    return jax.vmap(lambda z: model.prr_man.log_density(prior_p, z))(case.z_grid)


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
        obs_p, int_p, _ = model.gen_hrm.split_coords(hrm_params)
        prior = model.conjugated_prior_params(params)
        rho_zero = model.cnj_fun_man.zeros()

        def at_prior(rho: Array) -> Array:
            hrm = model.gen_hrm.join_coords(obs_p, int_p, prior - rho)
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
        (3,) * depth,
        (((0, 1), (1, 2)),) * depth,
        (1,) * depth,
        variant,  # pyright: ignore[reportArgumentType]
        (4,),
    )


def _circuit_params(circuit: CanonicalCircuit, key: Array, exact: bool) -> Array:
    """Random generative coordinates, with a standard normal observable and positive tuning precisions.

    With ``exact``, every population code is decoupled from its latent and every learned
    conjugation function is zero, and in the approximate variant every readout is decoupled
    from its neurons as well: then every level is exactly conjugate.
    """
    man = circuit.generative_man
    blocks = list(man.split_coords(0.5 * jax.random.normal(key, (man.dim,))))
    obs_man = circuit.rdt_hrm.obs_man
    blocks[0] = obs_man.join_location_precision(
        jnp.zeros(obs_man.data_dim),
        obs_man.cov_man.from_matrix(jnp.eye(obs_man.data_dim)),
    )
    for k in range(len(circuit.layers)):
        int_idx, loc_idx, prc_idx = 1 + 4 * k, 3 + 4 * k, 4 + 4 * k
        blocks[prc_idx] = jnp.abs(blocks[prc_idx]) + 0.5
        if exact:
            blocks[loc_idx] = jnp.zeros_like(blocks[loc_idx])
            blocks[prc_idx] = jnp.zeros_like(blocks[prc_idx])
            if circuit.cnj_map is not None:
                blocks[int_idx] = jnp.zeros_like(blocks[int_idx])
    if exact:
        blocks[-1] = jnp.zeros_like(blocks[-1])
    return circuit.tie(man.join_coords(*blocks))


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
    @pytest.mark.parametrize("depth", DEPTHS)
    def test_elbo_is_the_log_density_when_exact(self, variant: str, depth: int):
        """Every residual vanishes, so the estimator is exact, with any samples."""
        circuit = _circuit(variant, depth)
        params = _circuit_params(circuit, jax.random.PRNGKey(0), exact=True)
        elbo = circuit.elbo_at(jax.random.PRNGKey(1), params, X_CIRCUIT, 4)
        log_p = circuit.exact_log_observable_density(params, X_CIRCUIT)
        assert jnp.allclose(elbo, log_p, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("depth", DEPTHS)
    def test_elbo_gradient_is_the_log_density_gradient_when_exact(
        self, variant: str, depth: int
    ):
        """The divergence of the recognition model is zero, its minimum, so its gradient vanishes."""
        circuit = _circuit(variant, depth)
        params = _circuit_params(circuit, jax.random.PRNGKey(2), exact=True)
        exact_grad = jax.grad(circuit.exact_log_observable_density)(params, X_CIRCUIT)
        _, _, mc_grad, grad_se = _mc_value_and_grad(
            lambda k, p: circuit.elbo_at(k, p, X_CIRCUIT, 16), params
        )
        z_scores = jnp.abs(mc_grad - exact_grad) / jnp.maximum(grad_se, 1e-9)
        assert jnp.max(z_scores) < Z_THRESHOLD

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_elbo_matches_quadrature(self, variant: str):
        """Value and gradient away from conjugacy, where every level has a score term."""
        circuit = _circuit(variant, 1)
        params = _circuit_params(circuit, jax.random.PRNGKey(6), exact=False)
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
        params = _circuit_params(circuit, jax.random.PRNGKey(3), exact=False)
        zs = circuit.sample_recognition(jax.random.PRNGKey(4), params, X_CIRCUIT, 8)

        def gap(z: Array) -> Array:
            log_joint = circuit.log_density(params, jnp.concatenate([X_CIRCUIT, z]))
            log_q = circuit.recognition_log_density(params, X_CIRCUIT, z)
            return log_joint - log_q - circuit.elbo_residual(params, X_CIRCUIT, z)

        gaps = jax.vmap(gap)(zs)
        baseline = circuit.conjugation_baseline(params, X_CIRCUIT)
        assert jnp.allclose(gaps, baseline, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("depth", DEPTHS)
    def test_exact_density_normalizes(self, variant: str, depth: int):
        circuit = _circuit(variant, depth)
        params = _circuit_params(circuit, jax.random.PRNGKey(5), exact=False)
        xs = jnp.linspace(-12.0, 12.0, 4001)[:, None]
        log_ps = jax.vmap(lambda x: circuit.exact_log_observable_density(params, x))(xs)
        assert jnp.allclose(
            jnp.sum(jnp.exp(log_ps)) * (xs[1, 0] - xs[0, 0]), 1.0, atol=1e-6
        )
