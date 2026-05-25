"""Tests for geometry/exponential_family/variational.py.

Covers:
- ``VariationalHierarchical`` composing a flat Bernoullis-Bernoullis lower harmonium with a
  Bernoullis-Bernoullis-Categorical mixture upper harmonium:
  - Dimensions / split-join round-trip / initialize.
  - Joint sampling shape.
  - Joint log-density finiteness.
  - ELBO finiteness and gradient flow to every parameter slot
    (Upper params, lower likelihood, ρ_Y, MLP φ).
  - Per-level conjugation residuals.
"""

from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    Differentiable,
    EmbeddedMap,
    Harmonium,
    IdentityEmbedding,
    LinearMap,
    Manifold,
    Map,
    MultilayerPerceptron,
    ObservableEmbedding,
    Rectangular,
    VariationalConjugated,
    VariationalHierarchical,
    VariationalSymmetric,
)
from goal.models import (
    Bernoullis,
    CompleteMixture,
    VariationalHierarchicalMixture,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


### Concrete test classes ###


@dataclass(frozen=True)
class _ConcreteHarmonium[Observable: Differentiable, Latent: Differentiable](
    Harmonium[Observable, Latent]
):
    """Minimal concrete Harmonium for tests."""

    _int_man: EmbeddedMap[Latent, Observable]

    @property
    @override
    def int_man(self) -> LinearMap[Latent, Observable]:
        return self._int_man


@dataclass(frozen=True)
class _FlatVC[Observable: Differentiable, Latent: Differentiable](
    VariationalSymmetric[Observable, Latent, Latent]
):
    """Bivariate (flat) VariationalConjugated used as the lower component."""

    _gen_hrm: Harmonium[Observable, Latent]

    @property
    @override
    def gen_hrm(self) -> Harmonium[Observable, Latent]:
        return self._gen_hrm

    @property
    @override
    def lat_man(self) -> Latent:
        return self._gen_hrm.pst_man

    @property
    @override
    def cnj_man(self) -> Latent:
        return self._gen_hrm.pst_man


@dataclass(frozen=True)
class _ConcreteVHM[Observable: Differentiable, BaseLatent: Differentiable](
    VariationalHierarchicalMixture[Observable, BaseLatent]
):
    """Concrete VariationalHierarchicalMixture used as the upper component."""

    _gen_hrm: Harmonium[Observable, Any]

    @property
    @override
    def gen_hrm(self) -> Harmonium[Observable, Any]:
        return self._gen_hrm


@dataclass(frozen=True)
class _DeepHrm[
    Observable: Differentiable,
    Lower: VariationalConjugated[Any, Any, Any, Any],
    Upper: VariationalConjugated[Any, Any, Any, Any],
    LwrCnj: Manifold,
    RhoMap: Map[Any, Any],
](VariationalHierarchical[Observable, Lower, Upper, LwrCnj, RhoMap]):
    """Concrete VariationalHierarchical for tests."""

    _lwr_hrm: Lower
    _upr_hrm: Upper
    _rho_map: RhoMap

    @property
    @override
    def lwr_hrm(self) -> Lower:
        return self._lwr_hrm

    @property
    @override
    def upr_hrm(self) -> Upper:
        return self._upr_hrm

    @property
    @override
    def rho_map(self) -> RhoMap:
        return self._rho_map


def _make_flat_harmonium(obs_man: Any, lat_man: Any) -> _ConcreteHarmonium[Any, Any]:
    int_man = EmbeddedMap(
        Rectangular(),
        IdentityEmbedding(lat_man),
        IdentityEmbedding(obs_man),
    )
    return _ConcreteHarmonium(_int_man=int_man)


def _make_mixture_harmonium(
    obs_man: Any, base_lat_man: Any, n_categories: int
) -> _ConcreteHarmonium[Any, Any]:
    mix_man = CompleteMixture(base_lat_man, n_categories)
    int_man = EmbeddedMap(
        Rectangular(),
        ObservableEmbedding(mix_man),
        IdentityEmbedding(obs_man),
    )
    return _ConcreteHarmonium(_int_man=int_man)


def _make_model(
    n_x: int = 4, n_y: int = 3, n_z: int = 2, n_k: int = 3
) -> _DeepHrm[Any, Any, Any, Any, Any]:
    """Bernoullis X | Bernoullis Y | (Bernoullis Z, Categorical K)."""
    bx = Bernoullis(n_x)
    by = Bernoullis(n_y)
    bz = Bernoullis(n_z)

    lwr = _FlatVC(_gen_hrm=_make_flat_harmonium(bx, by))
    upr = _ConcreteVHM(_gen_hrm=_make_mixture_harmonium(by, bz, n_k))
    rho_map = MultilayerPerceptron(by, bz, hidden_dims=(), activation=jax.nn.tanh)
    return _DeepHrm(_lwr_hrm=lwr, _upr_hrm=upr, _rho_map=rho_map)


### Tests ###


class TestVariationalHierarchical:
    """Tests covering the new combinator end-to-end."""

    def test_dimensions(self) -> None:
        """Total dim is sum of slot dims."""
        model = _make_model()
        expected = (
            model.upr_hrm.dim
            + model.lwr_hrm.gen_hrm.lkl_fun_man.dim
            + model.lwr_hrm.cnj_man.dim
            + model.rho_map.dim
        )
        assert model.dim == expected

    def test_split_join_roundtrip(self) -> None:
        model = _make_model()
        params = model.initialize(jax.random.PRNGKey(0))
        upr, lkl, cnj = model.split_coords(params)
        rejoined = model.join_coords(upr, lkl, cnj)
        assert jnp.allclose(params, rejoined)

    def test_conjugation_split(self) -> None:
        """The combined conjugation slot splits into (ρ_Y, φ)."""
        model = _make_model()
        params = model.initialize(jax.random.PRNGKey(0))
        rho_y, phi = model.split_conjugation_params(params)
        assert rho_y.shape == (model.lwr_hrm.cnj_man.dim,)
        assert phi.shape == (model.rho_map.dim,)

    def test_sample_shape(self) -> None:
        """Joint samples have shape (n, x_dim + y_dim + z_dim + 1) where +1 is the categorical index."""
        n_x, n_y, n_z, n_k = 4, 3, 2, 3
        model = _make_model(n_x, n_y, n_z, n_k)
        params = model.initialize(jax.random.PRNGKey(0))
        samples = model.sample(jax.random.PRNGKey(1), params, n=5)
        expected_data_dim = n_x + n_y + n_z + 1
        del n_k  # n_k determines mixture structure but not data dim
        assert samples.shape == (5, expected_data_dim)

    def test_log_density_finite(self) -> None:
        model = _make_model()
        params = model.initialize(jax.random.PRNGKey(0))
        samples = model.sample(jax.random.PRNGKey(1), params, n=3)
        log_ps = jax.vmap(lambda s: model.log_density(params, s))(samples)
        assert jnp.all(jnp.isfinite(log_ps))

    def test_recognition_shape(self) -> None:
        """approximate_posterior_at returns a vector in upper-harmonium-shape (Upper.dim)."""
        model = _make_model()
        params = model.initialize(jax.random.PRNGKey(0))
        x = jnp.array([1.0, 0.0, 1.0, 1.0])
        q_params = model.approximate_posterior_at(params, x)
        assert q_params.shape == (model.upr_hrm.dim,)

    def test_elbo_finite(self) -> None:
        model = _make_model()
        params = model.initialize(jax.random.PRNGKey(0))
        x = jnp.array([1.0, 0.0, 1.0, 1.0])
        elbo = model.elbo_at(jax.random.PRNGKey(1), params, x, n_samples=4)
        assert jnp.isfinite(elbo)

    def test_mean_elbo_batches(self) -> None:
        model = _make_model()
        params = model.initialize(jax.random.PRNGKey(0))
        xs = jax.random.bernoulli(jax.random.PRNGKey(2), 0.5, shape=(6, 4)).astype(
            jnp.float64
        )
        m_elbo = model.mean_elbo(jax.random.PRNGKey(3), params, xs, n_samples=4)
        assert jnp.isfinite(m_elbo)
        assert m_elbo.shape == ()

    def test_elbo_gradient_flows_to_all_slots(self) -> None:
        """Each Triple slot (Upper params, lower lkl, conjugation pair) receives a finite gradient."""
        model = _make_model()
        params = model.initialize(jax.random.PRNGKey(0))
        x = jnp.array([1.0, 0.0, 1.0, 1.0])

        def loss(p: Array) -> Array:
            return -model.elbo_at(jax.random.PRNGKey(1), p, x, n_samples=8)

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads))
        g_upr, g_lkl, g_cnj = model.split_coords(grads)
        g_rho_y, g_phi = model.cnj_man.split_coords(g_cnj)
        # The forward pass touches all slots, so each slot should have some non-zero gradient.
        assert jnp.any(g_upr != 0)
        assert jnp.any(g_lkl != 0)
        assert jnp.any(g_rho_y != 0)
        # φ touches the recognition's ρ_Z(x); should be non-zero in general.
        assert jnp.any(g_phi != 0)

    def test_lower_residual_finite(self) -> None:
        """Per-level residual r_Y is finite at a sampled y."""
        model = _make_model()
        params = model.initialize(jax.random.PRNGKey(0))
        y = jnp.array([1.0, 0.0, 1.0])
        r_y = model.lower_conjugation_residual(params, y)
        assert jnp.isfinite(r_y)

    def test_upper_residual_finite(self) -> None:
        """Per-level residual r_Z(z; x) is finite at sampled (z, k)."""
        model = _make_model()
        params = model.initialize(jax.random.PRNGKey(0))
        x = jnp.array([1.0, 0.0, 1.0, 1.0])
        # (z, k): z is Bernoullis(2) sample, k is single integer index.
        zk = jnp.array([1.0, 0.0, 1.0])
        r_z = model.upper_conjugation_residual(params, zk, x)
        assert jnp.isfinite(r_z)

    def test_recognition_loss_finite(self) -> None:
        model = _make_model()
        params = model.initialize(jax.random.PRNGKey(0))
        x = jnp.array([1.0, 0.0, 1.0, 1.0])
        r_y_var = model.lower_recognition_conjugation_loss_at(
            jax.random.PRNGKey(1), params, x, n_samples=4
        )
        r_z_var = model.upper_recognition_conjugation_loss_at(
            jax.random.PRNGKey(2), params, x, n_samples=4
        )
        assert jnp.isfinite(r_y_var)
        assert jnp.isfinite(r_z_var)
        assert r_y_var >= 0
        assert r_z_var >= 0

    def test_initialize_from_sample(self) -> None:
        """initialize_from_sample uses sample data for lower observable biases."""
        model = _make_model()
        sample = jax.random.bernoulli(jax.random.PRNGKey(0), 0.3, shape=(20, 4)).astype(
            jnp.float64
        )
        params = model.initialize_from_sample(jax.random.PRNGKey(1), sample)
        assert params.shape == (model.dim,)
        assert jnp.all(jnp.isfinite(params))

    def test_elbo_divergence_partial_mc(self) -> None:
        """Partial-MC joint KL is finite and non-negative."""
        model = _make_model()
        params = model.initialize(jax.random.PRNGKey(0))
        x = jnp.array([1.0, 0.0, 1.0, 1.0])
        kl = model.elbo_divergence(jax.random.PRNGKey(1), params, x, n_samples=8)
        assert jnp.isfinite(kl)
        # KL >= 0 in population; finite-sample MC may dip slightly negative for
        # small n_samples but should be roughly non-negative.
        assert kl >= -1e-3

    def test_elbo_divergence_gradient_flow(self) -> None:
        """Gradient of partial-MC KL flows through every q-relevant parameter slot.

        $\\mathrm{KL}(q \\Vert p)$ depends on Upper params (both q and p priors share them),
        $\\rho_Y$ (shifts q's Y-bias), $\\phi$ (controls $\\rho_Z(x)$),
        and $\\Theta_{XY}$ in the lower likelihood (also shifts q's Y-bias via $s_X \\cdot \\Theta_{XY}$).
        Only $\\theta_X$ (the X-bias in the lower likelihood) is absent from the KL.
        """
        model = _make_model()
        params = model.initialize(jax.random.PRNGKey(0))
        x = jnp.array([1.0, 0.0, 1.0, 1.0])

        def loss(p: Array) -> Array:
            return model.elbo_divergence(jax.random.PRNGKey(1), p, x, n_samples=8)

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads))
        g_upr, g_lkl, g_cnj = model.split_coords(grads)
        g_rho_y, g_phi = model.cnj_man.split_coords(g_cnj)
        assert jnp.any(g_upr != 0)
        assert jnp.any(g_rho_y != 0)
        assert jnp.any(g_phi != 0)
        # Θ_XY portion of g_lkl should flow (shifts q via Δ_Y(x)).
        _, g_int = model.lwr_hrm.gen_hrm.lkl_fun_man.split_coords(g_lkl)
        assert jnp.any(g_int != 0)
