"""Tests for geometry/exponential_family/variational.py and models/graphical/variational.py.

Ground-truth verification of the variational estimators and their
stop_gradient policy. A ``VonMisesPopulationCode`` with a single VonMises
latent makes every latent expectation a 1D integral over [0, 2pi), where the
periodic trapezoid rule is exact to near machine precision for smooth
integrands. ``jax.grad`` of the quadrature expressions therefore yields exact
gradients --- including the dependence of the sampling measure on the
parameters --- against which the autodiff gradients of the Monte-Carlo
estimators are compared over many independent keys (a z-test on each
parameter coordinate).

This pins down the stop_gradient sites: a missing sg would double-count the
direct gradient, a spurious sg would drop the score correction, and either
error shifts the MC gradient mean away from the exact gradient by far more
than its standard error.

The final class covers the graphical specialization:
``VariationalHierarchicalMixture`` stores a BaseLatent-shaped conjugation
correction while its Posterior/Prior is the full mixture, so the inherited
machinery only composes if ``conjugation_parameters`` zero-pads correctly
into mixture shape. Those tests pin the pad structure, the exact vanishing
of the residual at exact conjugation, and agreement between
``regress_conjugation_parameters`` and the prior conjugation loss.
"""

from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    EmbeddedMap,
    Harmonium,
    IdentityEmbedding,
    ObservableEmbedding,
    Rectangular,
)
from goal.geometry.exponential_family.variational import (
    regress_conjugation_parameters,
)
from goal.models import (
    Bernoullis,
    Binomials,
    CompleteMixture,
    PoissonVonMisesHarmonium,
    VariationalHierarchicalMixture,
    VonMisesPopulationCode,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

N_NEURONS = 4
N_GRID = 2048
N_KEYS = 128
N_MC = 128
Z_THRESHOLD = 5.0

X_OBS = jnp.array([2.0, 0.0, 1.0, 3.0])
Z_GRID = (jnp.arange(N_GRID) / N_GRID * 2 * jnp.pi).reshape(-1, 1)
DZ = 2 * jnp.pi / N_GRID


def _setup() -> tuple[VonMisesPopulationCode, Array]:
    """Model with generic (non-conjugate, nonzero-rho) parameters."""
    model = VonMisesPopulationCode(_gen_hrm=PoissonVonMisesHarmonium(N_NEURONS, 1))
    k_init, k_rho = jax.random.split(jax.random.PRNGKey(0))
    prior_p, lkl_p, _ = model.split_coords(model.initialize(k_init, shape=0.5))
    rho = 0.3 * jax.random.normal(k_rho, (model.cnj_man.dim,))
    return model, model.join_coords(prior_p, lkl_p, rho)


# --- Quadrature ground truths (exact values and exact autodiff gradients) ---


def _recognition_weights(
    model: VonMisesPopulationCode, params: Array
) -> tuple[Array, Array]:
    q_params = model.approximate_posterior_at(params, X_OBS)
    log_q = jax.vmap(lambda z: model.pst_man.log_density(q_params, z))(Z_GRID)
    return jnp.exp(log_q), log_q


def _exact_elbo(model: VonMisesPopulationCode, params: Array) -> Array:
    w, log_q = _recognition_weights(model, params)
    lkl_nat = jax.vmap(lambda z: model.likelihood_at(params, z))(Z_GRID)
    log_pxz = jax.vmap(model.obs_man.log_density)(
        lkl_nat, jnp.tile(X_OBS, (N_GRID, 1))
    )
    prior_p = model.prior_params(params)
    log_pz = jax.vmap(lambda z: model.prr_man.log_density(prior_p, z))(Z_GRID)
    return DZ * jnp.sum(w * (log_pxz + log_pz - log_q))


def _exact_mean_r(model: VonMisesPopulationCode, params: Array) -> Array:
    w, _ = _recognition_weights(model, params)
    r = jax.vmap(lambda z: model.conjugation_residual(params, z, X_OBS))(Z_GRID)
    return DZ * jnp.sum(w * r)


def _exact_kl(model: VonMisesPopulationCode, params: Array) -> Array:
    w, log_q = _recognition_weights(model, params)
    prior_p = model.prior_params(params)
    log_pz = jax.vmap(lambda z: model.prr_man.log_density(prior_p, z))(Z_GRID)
    return DZ * jnp.sum(w * (log_q - log_pz))


def _exact_var_q_r(model: VonMisesPopulationCode, params: Array) -> Array:
    w, _ = _recognition_weights(model, params)
    r = jax.vmap(lambda z: model.conjugation_residual(params, z, X_OBS))(Z_GRID)
    mean_r = DZ * jnp.sum(w * r)
    return DZ * jnp.sum(w * (r - mean_r) ** 2)


def _exact_var_p_r(
    model: VonMisesPopulationCode, params: Array, freeze_measure: bool
) -> Array:
    prior_p = model.prior_params(params)
    log_p = jax.vmap(lambda z: model.prr_man.log_density(prior_p, z))(Z_GRID)
    w = jnp.exp(log_p)
    if freeze_measure:
        w = jax.lax.stop_gradient(w)
    r = jax.vmap(lambda z: model.conjugation_residual(params, z))(Z_GRID)
    mean_r = DZ * jnp.sum(w * r)
    return DZ * jnp.sum(w * (r - mean_r) ** 2)


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


class TestStandardFormElbo:
    """elbo_at: value and full gradient against exact quadrature."""

    def test_value_and_gradient_match_quadrature(self):
        model, params = _setup()
        exact_val = _exact_elbo(model, params)
        exact_grad = jax.grad(lambda p: _exact_elbo(model, p))(params)
        mc_val, val_se, mc_grad, grad_se = _mc_value_and_grad(
            lambda k, p: model.elbo_at(k, p, X_OBS, N_MC), params
        )
        # The gradient must be right in every block: a wrong sg on the samples
        # or on r inside the score term would shift the rho/prior blocks.
        _assert_matches(exact_val, exact_grad, mc_val, val_se, mc_grad, grad_se)

    def test_standard_form_decomposition(self):
        """Quadrature identity L(x) = c(x) + E_q[r] --- validates the c/r split."""
        model, params = _setup()
        c_x = model.conjugation_baseline(params, X_OBS)
        lhs = _exact_elbo(model, params)
        rhs = c_x + _exact_mean_r(model, params)
        assert jnp.allclose(lhs, rhs, rtol=1e-10, atol=1e-10)

    def test_value_excludes_score_term(self):
        """The ``- sg(score)`` site: the returned value is exactly c(x) + mean(r).

        Replicates the internal sampling with the same key, so any leakage of
        the score correction into the value would show up exactly.
        """
        model, params = _setup()
        key = jax.random.PRNGKey(7)
        surrogate = model.elbo_at(key, params, X_OBS, N_MC)

        q_params = model.approximate_posterior_at(params, X_OBS)
        z_samples = model.pst_man.sample(key, q_params, N_MC)
        r_vals = jax.vmap(
            lambda z: model.conjugation_residual(params, z, X_OBS)
        )(z_samples)
        clean = model.conjugation_baseline(params, X_OBS) + jnp.mean(r_vals)
        assert jnp.allclose(surrogate, clean, rtol=1e-12, atol=1e-12)


class TestElboDivergence:
    def test_kl_matches_quadrature(self):
        model, params = _setup()
        closed_form = model.elbo_divergence(params, X_OBS)
        quadrature = _exact_kl(model, params)
        assert jnp.allclose(closed_form, quadrature, rtol=1e-8, atol=1e-10)


class TestRecognitionConjugationLoss:
    """Var_q[r] with score correction: the sampling measure shares parameters
    with r, so the exact gradient has both a direct and a score term."""

    def test_value_and_gradient_match_quadrature(self):
        model, params = _setup()
        exact_val = _exact_var_q_r(model, params)
        exact_grad = jax.grad(lambda p: _exact_var_q_r(model, p))(params)
        mc_val, val_se, mc_grad, grad_se = _mc_value_and_grad(
            lambda k, p: model.recognition_conjugation_loss_at(k, p, X_OBS, N_MC),
            params,
        )
        # Dropping the score correction here would bias the prior/rho blocks
        # (the measure q depends on theta_Z, rho, and Theta).
        _assert_matches(exact_val, exact_grad, mc_val, val_se, mc_grad, grad_se)


class TestPriorConjugationLoss:
    """Var_p[r] with the deliberate direct-only gradient policy."""

    def test_value_and_gradient_match_frozen_measure_quadrature(self):
        model, params = _setup()
        exact_val = _exact_var_p_r(model, params, freeze_measure=False)
        exact_grad = jax.grad(
            lambda p: _exact_var_p_r(model, p, freeze_measure=True)
        )(params)
        mc_val, val_se, mc_grad, grad_se = _mc_value_and_grad(
            lambda k, p: model.prior_conjugation_loss(k, p, N_MC), params
        )
        _assert_matches(exact_val, exact_grad, mc_val, val_se, mc_grad, grad_se)

    def test_prior_block_gradient_is_exactly_zero(self):
        """r does not depend on theta_Z and the samples are sg'd, so the
        regularizer must exert exactly zero gradient on the prior."""
        model, params = _setup()
        grad = jax.grad(
            lambda p: model.prior_conjugation_loss(jax.random.PRNGKey(3), p, N_MC)
        )(params)
        prior_dim = model.prr_man.dim
        assert jnp.all(grad[:prior_dim] == 0.0)

    def test_dropped_score_term_lives_only_in_prior_block(self):
        """Quadrature consistency check on the design: the score term omitted
        by the direct-only policy affects theta_Z alone, and non-trivially."""
        model, params = _setup()
        g_full = jax.grad(lambda p: _exact_var_p_r(model, p, freeze_measure=False))(
            params
        )
        g_frozen = jax.grad(lambda p: _exact_var_p_r(model, p, freeze_measure=True))(
            params
        )
        diff = g_full - g_frozen
        prior_dim = model.prr_man.dim
        assert jnp.linalg.norm(diff[:prior_dim]) > 1e-3
        assert jnp.allclose(diff[prior_dim:], 0.0, atol=1e-12)


# --- models/graphical/variational.py: VariationalHierarchicalMixture ---


@dataclass(frozen=True)
class _ConcreteHarmonium(Harmonium[Binomials, Any]):
    """Harmonium with interaction restricted to the BaseLatent slot of the mixture."""

    _int_man: EmbeddedMap[Any, Binomials]

    @property
    @override
    def int_man(self) -> EmbeddedMap[Any, Binomials]:
        return self._int_man


@dataclass(frozen=True)
class _ConcreteHierarchicalMixture(
    VariationalHierarchicalMixture[Binomials, Bernoullis]
):
    _gen_hrm: Harmonium[Binomials, Any]

    @property
    @override
    def gen_hrm(self) -> Harmonium[Binomials, Any]:
        return self._gen_hrm


def _make_hierarchical_model() -> _ConcreteHierarchicalMixture:
    """Small instance: 6 Binomial(3) observables, 3 Bernoulli latents, 3 clusters."""
    obs_man = Binomials(6, 3)
    mix_man = CompleteMixture(Bernoullis(3), 3)
    int_man = EmbeddedMap(
        Rectangular(),
        ObservableEmbedding(mix_man),
        IdentityEmbedding(obs_man),
    )
    return _ConcreteHierarchicalMixture(_gen_hrm=_ConcreteHarmonium(int_man))


class TestVariationalHierarchicalMixture:
    def test_conjugation_parameters_zero_pad_structure(self):
        """The BaseLatent rho lands in the mixture's observable slot; the
        interaction and categorical slots are exactly zero."""
        model = _make_hierarchical_model()
        params = model.initialize(jax.random.PRNGKey(0))
        prior_p, lkl_p, _ = model.split_coords(params)
        rho = jnp.arange(1.0, model.cnj_man.dim + 1)
        params = model.join_coords(prior_p, lkl_p, rho)

        rho_full = model.conjugation_parameters(params)
        assert rho_full.shape == (model.prr_man.dim,)
        rho_y, rho_yk, rho_k = model.mix_man.split_coords(rho_full)
        assert jnp.array_equal(rho_y, rho)
        assert jnp.all(rho_yk == 0.0)
        assert jnp.all(rho_k == 0.0)

    def test_residual_vanishes_at_exact_conjugation(self):
        """With zero interaction the likelihood is exactly conjugate with
        rho = 0, and the +psi_X(theta_X) convention makes r identically zero
        --- to machine precision, not just statistically."""
        model = _make_hierarchical_model()
        key_init, key_z = jax.random.split(jax.random.PRNGKey(1))
        params = model.initialize(key_init)
        prior_p, lkl_p, _ = model.split_coords(params)
        obs_p, int_p = model.gen_hrm.lkl_fun_man.split_coords(lkl_p)
        lkl_zero = model.gen_hrm.lkl_fun_man.join_coords(
            obs_p, jnp.zeros_like(int_p)
        )
        params = model.join_coords(prior_p, lkl_zero, jnp.zeros(model.cnj_man.dim))

        z_samples = model.prr_man.sample(key_z, prior_p, 20)
        r_vals = jax.vmap(lambda z: model.conjugation_residual(params, z))(z_samples)
        assert jnp.all(r_vals == 0.0)

    def test_regression_reduces_prior_conjugation_loss(self):
        """regress_conjugation_parameters minimizes the sampled Var_p[r], so
        the fitted rho must beat rho = 0 on the prior conjugation loss."""
        model = _make_hierarchical_model()
        key_init, key_reg, key_loss = jax.random.split(jax.random.PRNGKey(2), 3)
        params = model.initialize(key_init)
        prior_p, lkl_p, _ = model.split_coords(params)

        rho_fit, r_squared, _, _ = regress_conjugation_parameters(
            model, key_reg, params, n_samples=1000
        )
        params_fit = model.join_coords(prior_p, lkl_p, rho_fit)

        loss_zero = model.prior_conjugation_loss(key_loss, params, 1000)
        loss_fit = model.prior_conjugation_loss(key_loss, params_fit, 1000)
        assert loss_fit < loss_zero
        # Small-init couplings make psi_X near-affine in s_Y, so the affine
        # correction should explain nearly all the residual variance.
        assert r_squared > 0.9
