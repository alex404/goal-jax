"""Tests for ``VariationalGraphical`` in geometry/exponential_family/graphical.py.

The model is a three-level chain $x \\leftarrow y \\leftarrow z$ whose edges are exact or
learned by type. Three ground truths pin it down. An all-Gaussian chain has both edges
exact, so its ELBO must equal the log-marginal of the exact graphical harmonium it wraps,
for every sample. For Boltzmann middle layers the ELBO integrand must split as
$c(x) + r_Y + r^*_Z - r^X_Z$ against a brute-force $\\log p - \\log q$, an exact lower edge
must have a vanishing residual over all states of $y$, and the estimator's gradient must
match the gradient of the ELBO computed by enumerating $y$ and Gauss--Hermite quadrature
over a one-dimensional $z$, also with two samples per estimate. The inner conjugation
penalty and the two prior penalties, with their gradients, including the dependence of
their sampling distributions on the parameters, are checked against the same quadrature.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from goal.geometry import (
    DifferentiableGraphical,
    GraphicalHarmonium,
    Harmonium,
    IdentityEmbedding,
    MultilayerPerceptron,
    PositiveDefinite,
    VariationalGraphical,
)
from goal.models import (
    BoltzmannLGM,
    BoltzmannNormalHarmonium,
    ChordalBoltzmann,
    DiagonalBoltzmann,
    FullBoltzmann,
    NormalBoltzmannHarmonium,
    NormalLGM,
    full_normal,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class _ExactChain(DifferentiableGraphical[Any, Any]):
    """A lower harmonium attached to the observable node of a conjugated upper harmonium."""

    lwr_hrm: Harmonium[Any, Any]
    _pst_man: Any

    @property
    @override
    def attachments(self) -> tuple[tuple[Harmonium[Any, Any], tuple[int, ...]], ...]:
        return ((self.lwr_hrm, (0,)),)

    @property
    @override
    def pst_man(self) -> Any:
        return self._pst_man

    @property
    @override
    def pst_prr_emb(self) -> Any:
        return IdentityEmbedding(self._pst_man)


@dataclass(frozen=True)
class _Chain(GraphicalHarmonium[Any]):
    """A lower harmonium attached to the observable node of an upper harmonium."""

    lwr_hrm: Harmonium[Any, Any]
    _pst_man: Any

    @property
    @override
    def attachments(self) -> tuple[tuple[Harmonium[Any, Any], tuple[int, ...]], ...]:
        return ((self.lwr_hrm, (0,)),)

    @property
    @override
    def pst_man(self) -> Any:
        return self._pst_man


@dataclass(frozen=True)
class _Variational(VariationalGraphical):
    _gen_hrm: GraphicalHarmonium[Any]
    _inr_map: MultilayerPerceptron[Any, Any] | None

    @property
    @override
    def gen_hrm(self) -> GraphicalHarmonium[Any]:
        return self._gen_hrm

    @property
    @override
    def inr_map(self) -> MultilayerPerceptron[Any, Any] | None:
        return self._inr_map


PD = PositiveDefinite()


def _gaussian_chain() -> tuple[_ExactChain, _Variational]:
    chain = _ExactChain(NormalLGM(3, PD, 2, PD), NormalLGM(2, PD, 1, PD))
    return chain, _Variational(chain, None)


def _boltzmann_circuit(kind: str, n: int = 4, lat_dim: int = 2) -> _Variational:
    """$x$ Gaussian, $y$ Boltzmann of the given kind, $z$ Gaussian; the lower edge is exact for a full Boltzmann."""
    bol: Any
    lwr: Harmonium[Any, Any]
    if kind == "full":
        bol = FullBoltzmann(n)
        lwr = BoltzmannLGM(3, PD, n)
    else:
        bol = (
            DiagonalBoltzmann(n)
            if kind == "diagonal"
            else ChordalBoltzmann.from_edges(n, [(i, i + 1) for i in range(n - 1)])
        )
        lwr = NormalBoltzmannHarmonium(3, PD, bol)
    upr = BoltzmannNormalHarmonium(bol, lat_dim)
    mlp = MultilayerPerceptron(full_normal(lat_dim), bol, (6,), jax.nn.tanh)
    return _Variational(_Chain(lwr, upr), mlp)


def _perturbed(model: VariationalGraphical, seed: int) -> Array:
    """Initial parameters moved off the zero conjugation parameters and the zero inner bias."""
    params = model.initialize(jax.random.PRNGKey(seed), shape=0.3)
    noise = jax.random.normal(jax.random.PRNGKey(seed + 1), params.shape)
    return params + 0.1 * noise


X3 = jnp.array([0.3, -0.2, 0.5])


class TestLayout:
    """The learned slots exist exactly for the edges that are not conjugated."""

    def test_exact_chain_stores_no_conjugation(self) -> None:
        _, model = _gaussian_chain()
        _, _, _, rho_y, rho_z, inr = model.cmp_mans
        assert (rho_y.dim, rho_z.dim, inr.dim) == (0, 0, 0)

    @pytest.mark.parametrize("kind", ["chordal", "diagonal"])
    def test_learned_edges_store_their_conjugation(self, kind: str) -> None:
        model = _boltzmann_circuit(kind)
        _, _, _, rho_y, rho_z, inr = model.cmp_mans
        assert rho_y == model.mid_man
        assert rho_z == model.top_man
        assert inr == model.inr_map

    def test_exact_lower_edge_stores_no_rho_y(self) -> None:
        model = _boltzmann_circuit("full")
        _, _, _, rho_y, rho_z, _ = model.cmp_mans
        assert rho_y.dim == 0
        assert rho_z == model.top_man

    def test_split_join_round_trip(self) -> None:
        model = _boltzmann_circuit("chordal")
        params = _perturbed(model, 0)
        assert jnp.array_equal(model.join_coords(*model.split_coords(params)), params)


class TestExactChain:
    """With both edges exact, the ELBO is the log-marginal of the wrapped harmonium."""

    @pytest.mark.parametrize("pathwise_z", [False, True])
    def test_elbo_is_log_marginal(self, pathwise_z: bool) -> None:
        chain, model = _gaussian_chain()
        params = _perturbed(model, 2)
        log_px = chain.log_observable_density(model.to_harmonium(params), X3)
        key = jax.random.PRNGKey(3)
        elbo = model.elbo_at(key, params, X3, 5, pathwise_z)
        marginal = model.marginal_elbo_at(key, params, X3, 5, pathwise_z)
        assert jnp.allclose(elbo, log_px, atol=1e-10)
        assert jnp.allclose(marginal, log_px, atol=1e-10)

    def test_residuals_vanish(self) -> None:
        _, model = _gaussian_chain()
        params = _perturbed(model, 4)
        ys, zs = model.sample_posterior(jax.random.PRNGKey(5), params, X3, 6)
        r_y = jax.vmap(lambda y: model.lower_residual(params, y))(ys)
        r_z = jax.vmap(lambda z: model.upper_residual(params, z))(zs)
        r_inner = jax.vmap(lambda z: model.inner_residual(params, X3, z))(zs)
        assert jnp.allclose(r_y, 0.0, atol=1e-10)
        assert jnp.allclose(r_z, 0.0, atol=1e-10)
        assert jnp.allclose(r_inner, 0.0, atol=1e-10)

    def test_joint_is_the_harmoniums(self) -> None:
        chain, model = _gaussian_chain()
        params = _perturbed(model, 6)
        ys, zs = model.sample_posterior(jax.random.PRNGKey(7), params, X3, 4)
        hrm_params = model.to_harmonium(params)
        for y, z in zip(ys, zs, strict=True):
            xyz = jnp.concatenate([X3, y, z])
            assert jnp.allclose(
                model.log_density_joint(params, X3, y, z),
                chain.log_density(hrm_params, xyz),
                atol=1e-10,
            )

    @pytest.mark.parametrize("pathwise_z", [False, True])
    def test_gradient_is_log_marginal_gradient(self, pathwise_z: bool) -> None:
        """The estimator's gradient is unbiased for $\\nabla \\log p(x)$; its value is exact."""
        chain, model = _gaussian_chain()
        params = _perturbed(model, 8)
        g_true = jax.grad(
            lambda p: chain.log_observable_density(model.to_harmonium(p), X3)
        )(params)
        g_est = jax.grad(
            lambda p: model.elbo_at(jax.random.PRNGKey(9), p, X3, 20000, pathwise_z)
        )(params)
        rel = jnp.linalg.norm(g_est - g_true) / jnp.linalg.norm(g_true)
        assert rel < 0.05


class TestDecomposition:
    """$\\log p(x, y, z) - \\log q(y, z \\mid x) = c(x) + r_Y(y) + r^*_Z(z) - r^X_Z(z; x)$."""

    @pytest.mark.parametrize("kind", ["chordal", "diagonal", "full"])
    def test_integrand_splits(self, kind: str) -> None:
        model = _boltzmann_circuit(kind)
        params = _perturbed(model, 10)
        ys, zs = model.sample_posterior(jax.random.PRNGKey(11), params, X3, 8)
        c_x = model.conjugation_baseline(params, X3)
        for y, z in zip(ys, zs, strict=True):
            lhs = model.log_density_joint(params, X3, y, z) - model.log_q(
                params, X3, y, z
            )
            rhs = c_x + model.learning_signal(params, X3, y, z)
            assert jnp.allclose(lhs, rhs, atol=1e-10)


class TestExactLowerEdge:
    """A conjugated lower harmonium has $r_Y = 0$, so $y$ integrates out of the ELBO."""

    def test_lower_residual_vanishes_on_every_state(self) -> None:
        model = _boltzmann_circuit("full", n=4)
        params = _perturbed(model, 12)
        ys = jnp.array(list(itertools.product([0.0, 1.0], repeat=4)))
        r_y = jax.vmap(lambda y: model.lower_residual(params, y))(ys)
        assert jnp.allclose(r_y, 0.0, atol=1e-10)

    def test_marginal_elbo_matches_elbo(self) -> None:
        """At the same $z$ samples the two estimators agree, since $r_Y$ contributes nothing."""
        model = _boltzmann_circuit("full", n=4)
        params = _perturbed(model, 13)
        key = jax.random.PRNGKey(14)
        key_z, _ = jax.random.split(key)
        elbo = model.elbo_at(key, params, X3, 6, True)
        marginal = model.marginal_elbo_at(key_z, params, X3, 6, True)
        assert jnp.allclose(elbo, marginal, atol=1e-10)


class TestGradientUnbiased:
    """The estimator's gradient matches the ELBO's, computed by enumeration and quadrature."""

    @staticmethod
    def _quadrature_elbo(model: VariationalGraphical, params: Array, x: Array) -> Array:
        top = model.top_man
        q_top = model.approximate_posterior_top(params, x)
        mean, cov = top.split_mean_covariance(top.to_mean(q_top))
        sd = jnp.sqrt(top.cov_man.to_matrix(cov)[0, 0])
        nodes, weights = np.polynomial.hermite.hermgauss(60)
        zs = mean[0] + jnp.sqrt(2.0) * sd * jnp.asarray(nodes)
        ys = jnp.array(
            list(itertools.product([0.0, 1.0], repeat=model.mid_man.data_dim))
        )

        def at_z(z: Array) -> Array:
            z = jnp.atleast_1d(z)
            q_mid = model.posterior_mid_at(params, x, z)
            log_qy = jax.vmap(lambda y: model.mid_man.log_density(q_mid, y))(ys)
            f = jax.vmap(
                lambda y: (
                    model.log_density_joint(params, x, y, z)
                    - model.log_q(params, x, y, z)
                )
            )(ys)
            return jnp.sum(jnp.exp(log_qy) * f)

        return jnp.sum(jnp.asarray(weights) / jnp.sqrt(jnp.pi) * jax.vmap(at_z)(zs))

    @pytest.mark.parametrize("pathwise_z", [False, True])
    def test_matches_quadrature(self, pathwise_z: bool) -> None:
        model = _boltzmann_circuit("chordal", n=3, lat_dim=1)
        params = _perturbed(model, 15)
        x = jnp.array([0.4, -0.7, 0.1])
        g_true = jax.grad(lambda p: self._quadrature_elbo(model, p, x))(params)
        grad = jax.jit(jax.grad(lambda p, k: model.elbo_at(k, p, x, 20000, pathwise_z)))
        g_est = jnp.mean(
            jnp.stack([grad(params, jax.random.PRNGKey(20 + i)) for i in range(4)]),
            axis=0,
        )
        rel = jnp.linalg.norm(g_est - g_true) / jnp.linalg.norm(g_true)
        assert rel < 0.03

    @pytest.mark.parametrize("n_samples", [1, 2])
    @pytest.mark.parametrize("pathwise_z", [False, True])
    def test_small_sample_matches_quadrature(
        self, pathwise_z: bool, n_samples: int
    ) -> None:
        """Averaged over many keys, estimates from one or two samples are unbiased too."""
        model = _boltzmann_circuit("chordal", n=3, lat_dim=1)
        params = _perturbed(model, 15)
        x = jnp.array([0.4, -0.7, 0.1])
        g_true = jax.grad(lambda p: self._quadrature_elbo(model, p, x))(params)
        grad = jax.grad(lambda p, k: model.elbo_at(k, p, x, n_samples, pathwise_z))
        keys = jax.random.split(jax.random.PRNGKey(30), 40000)
        g_est = jnp.mean(jax.jit(jax.vmap(grad, in_axes=(None, 0)))(params, keys), 0)
        rel = jnp.linalg.norm(g_est - g_true) / jnp.linalg.norm(g_true)
        assert rel < 0.03


class TestInnerConjugationLoss:
    """$\\mathrm{Var}_{q(z \\mid x)}[r^X_Z]$ and its gradient, including the dependence of $q$ on the parameters, against quadrature over a one-dimensional $z$."""

    @staticmethod
    def _quadrature_variance(
        model: VariationalGraphical, params: Array, x: Array
    ) -> Array:
        top = model.top_man
        q_top = model.approximate_posterior_top(params, x)
        mean, cov = top.split_mean_covariance(top.to_mean(q_top))
        sd = jnp.sqrt(top.cov_man.to_matrix(cov)[0, 0])
        nodes, weights = np.polynomial.hermite.hermgauss(60)
        zs = mean[0] + jnp.sqrt(2.0) * sd * jnp.asarray(nodes)
        w = jnp.asarray(weights) / jnp.sqrt(jnp.pi)
        r = jax.vmap(lambda z: model.inner_residual(params, x, jnp.atleast_1d(z)))(zs)
        return jnp.sum(w * r**2) - jnp.sum(w * r) ** 2

    def test_matches_quadrature(self) -> None:
        model = _boltzmann_circuit("chordal", n=3, lat_dim=1)
        params = _perturbed(model, 16)
        x = jnp.array([0.4, -0.7, 0.1])
        v_true = self._quadrature_variance(model, params, x)
        g_true = jax.grad(lambda p: self._quadrature_variance(model, p, x))(params)

        def loss(p: Array, k: Array) -> Array:
            return model.inner_conjugation_loss_at(k, p, x, 4)

        keys = jax.random.split(jax.random.PRNGKey(40), 40000)
        vals, grads = jax.jit(jax.vmap(jax.value_and_grad(loss), in_axes=(None, 0)))(
            params, keys
        )
        assert jnp.allclose(jnp.mean(vals), v_true, rtol=0.03)
        rel = jnp.linalg.norm(jnp.mean(grads, 0) - g_true) / jnp.linalg.norm(g_true)
        assert rel < 0.05


class TestPriorConjugationLosses:
    """$\\mathrm{Var}_p[r_Y]$ and $\\mathrm{Var}_p[r^*_Z]$ and their gradients, including the dependence of the ancestral distribution on the parameters, against enumeration of $y$ and quadrature over a one-dimensional $z$."""

    @staticmethod
    def _quadrature_variances(
        model: VariationalGraphical, params: Array
    ) -> tuple[Array, Array]:
        top = model.top_man
        prr, upr_lkl, *_ = model.split_coords(params)
        mean, cov = top.split_mean_covariance(top.to_mean(prr))
        sd = jnp.sqrt(top.cov_man.to_matrix(cov)[0, 0])
        nodes, weights = np.polynomial.hermite.hermgauss(60)
        zs = mean[0] + jnp.sqrt(2.0) * sd * jnp.asarray(nodes)
        w = jnp.asarray(weights) / jnp.sqrt(jnp.pi)
        r_z = jax.vmap(lambda z: model.upper_residual(params, jnp.atleast_1d(z)))(zs)
        var_z = jnp.sum(w * r_z**2) - jnp.sum(w * r_z) ** 2

        ys = jnp.array(
            list(itertools.product([0.0, 1.0], repeat=model.mid_man.data_dim))
        )

        def p_y_given_z(z: Array) -> Array:
            y_params = model.upr_hrm.lkl_fun_man(
                upr_lkl, model.upr_hrm.pst_man.sufficient_statistic(jnp.atleast_1d(z))
            )
            return jnp.exp(
                jax.vmap(lambda y: model.mid_man.log_density(y_params, y))(ys)
            )

        p_y = jnp.sum(w[:, None] * jax.vmap(p_y_given_z)(zs), axis=0)
        r_y = jax.vmap(lambda y: model.lower_residual(params, y))(ys)
        var_y = jnp.sum(p_y * r_y**2) - jnp.sum(p_y * r_y) ** 2
        return var_y, var_z

    def test_matches_quadrature(self) -> None:
        model = _boltzmann_circuit("chordal", n=3, lat_dim=1)
        params = _perturbed(model, 17)
        v_true = jnp.stack(self._quadrature_variances(model, params))
        g_true = jax.jacobian(
            lambda p: jnp.stack(self._quadrature_variances(model, p))
        )(params)

        def losses(p: Array, k: Array) -> Array:
            return jnp.stack(model.prior_conjugation_losses(k, p, 4))

        keys = jax.random.split(jax.random.PRNGKey(50), 40000)
        vals = jax.jit(jax.vmap(losses, in_axes=(None, 0)))(params, keys)
        grads = jax.jit(jax.vmap(jax.jacobian(losses), in_axes=(None, 0)))(params, keys)
        # The residuals are heavy-tailed, so values and gradients are checked in
        # standard errors of the mean over keys
        n = keys.shape[0]
        se = jnp.std(vals, 0) / jnp.sqrt(n)
        assert jnp.all(jnp.abs(jnp.mean(vals, 0) - v_true) < 4 * se)
        g_se = jnp.std(grads, 0) / jnp.sqrt(n)
        assert jnp.all(jnp.abs(jnp.mean(grads, 0) - g_true) < 5 * g_se + 1e-10)
