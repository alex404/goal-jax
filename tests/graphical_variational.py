"""Tests for nested ``VariationalDifferentiable`` in geometry/exponential_family/variational.py.

A variational harmonium whose prior is itself variational nests to any depth, and its
underlying harmonium may be a graphical harmonium with several observable harmoniums.
Four ground truths pin it down. Gaussian models with every conjugation exact have an
ELBO equal to their log-marginal, computed in closed form from the quadratic joint
log-density; this holds for a two-level chain, a three-level chain, an exact harmonium
in its own joint coordinates as the prior, and graphical harmoniums with two and three
observable harmoniums on one latent. For Boltzmann and Gaussian levels mixed to depths
two to four, the ELBO integrand must split as $c(x) + \\sum r^0 - \\sum r^X$ against a
brute-force $\\log p - \\log q$, and exact levels must have vanishing residuals. The
estimator's gradient must match the gradient of the ELBO computed by enumerating the
Boltzmann levels and Gauss--Hermite quadrature over one-dimensional Gaussian levels, at
depth two (also from one or two samples per estimate) and depth three. The inner and
prior residual variances, with their gradients, including the dependence of their
sampling distributions on the parameters, are checked against the same quadrature.
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
    IdentityEmbedding,
    MultilayerPerceptron,
    PositiveDefinite,
    SubCliquesEmbedding,
    VariationalDifferentiable,
    VariationalPrior,
)
from goal.models import (
    BoltzmannLGM,
    BoltzmannNormalHarmonium,
    CanonicalCorrelationAnalysis,
    ChordalBoltzmann,
    DiagonalBoltzmann,
    FullBoltzmann,
    NormalBoltzmannHarmonium,
    NormalLGM,
    full_normal,
)
from goal.models.base.gaussian.generalized import Euclidean

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

PD = PositiveDefinite()


### Models ###


@dataclass(frozen=True)
class _Attached(VariationalDifferentiable[Any, Any, Any, Any]):
    """A harmonium whose posterior is node $0$ of its prior's root, with exact or learned conjugation parameters.

    Exact conjugation parameters are computed by the conjugated harmonium at any bias.
    Learned ones are a stored $\\rho^0$ and, when the model is nested, $\\rho^X$ from
    ``inr_map`` at the posterior bias; the stored coordinates are ``[rho0 | inner map]``.
    """

    hrm: Any
    prior: Any
    exact: bool
    inr_map: MultilayerPerceptron[Any, Any] | None

    @property
    @override
    def gen_hrm(self) -> Any:
        return self.hrm

    @property
    @override
    def prr_man(self) -> Any:
        return self.prior

    @property
    @override
    def pst_prr_emb(self) -> SubCliquesEmbedding:
        prior = self.prior
        shift_man: Any = (
            prior.shift_man if isinstance(prior, VariationalPrior) else prior
        )
        return SubCliquesEmbedding((0,), shift_man, self.hrm.pst_man)

    @property
    @override
    def cnj_man(self) -> Euclidean:
        if self.exact:
            return Euclidean(0)
        n_inner = 0 if self.inr_map is None else self.inr_map.dim
        return Euclidean(self.hrm.pst_man.dim + n_inner)

    @property
    @override
    def learned_conjugation(self) -> bool:
        return not self.exact

    @override
    def conjugation_parameters(self, params: Array, x: Array | None = None) -> Array:
        lkl, _, slot = self.split_coords(params)
        if self.exact:
            return self.pst_prr_emb.embed(self.hrm.conjugation_parameters(lkl))
        return self.pst_prr_emb.embed(slot[: self.hrm.pst_man.dim])

    @override
    def posterior_conjugation_parameters(self, params: Array, bias: Array) -> Array:
        lkl, _, slot = self.split_coords(params)
        if self.exact:
            _, int_params = self.hrm.lkl_fun_man.split_coords(lkl)
            lkl = self.hrm.lkl_fun_man.join_coords(bias, int_params)
            return self.pst_prr_emb.embed(self.hrm.conjugation_parameters(lkl))
        assert self.inr_map is not None
        return self.pst_prr_emb.embed(self.inr_map(slot[self.hrm.pst_man.dim :], bias))


@dataclass(frozen=True)
class _Fan(DifferentiableGraphical[Any, Any]):
    """Linear Gaussian models with the given observable dimensions, all on one Gaussian latent."""

    obs_dims: tuple[int, ...]
    lat_dim: int

    @property
    @override
    def obs_hrms_att_clqs(self) -> tuple[tuple[Any, tuple[int, ...]], ...]:
        return tuple((NormalLGM(d, PD, self.lat_dim, PD), (0,)) for d in self.obs_dims)

    @property
    @override
    def pst_man(self) -> Any:
        return full_normal(self.lat_dim)

    @property
    @override
    def pst_prr_emb(self) -> IdentityEmbedding[Any]:
        return IdentityEmbedding(full_normal(self.lat_dim))


@dataclass(frozen=True)
class _Graphical(VariationalDifferentiable[Any, Any, Any, Any]):
    """A conjugated graphical harmonium over a normal prior, with its conjugation parameters computed or learned; learned ones are stored in the prior's coordinates."""

    hrm: Any
    exact: bool

    @property
    @override
    def gen_hrm(self) -> Any:
        return self.hrm

    @property
    @override
    def prr_man(self) -> Any:
        return self.hrm.prr_man

    @property
    @override
    def pst_prr_emb(self) -> Any:
        return self.hrm.pst_prr_emb

    @property
    @override
    def cnj_man(self) -> Euclidean:
        return Euclidean(0 if self.exact else self.prr_man.dim)

    @property
    @override
    def learned_conjugation(self) -> bool:
        return not self.exact

    @override
    def conjugation_parameters(self, params: Array, x: Array | None = None) -> Array:
        lkl, _, slot = self.split_coords(params)
        return self.hrm.conjugation_parameters(lkl) if self.exact else slot


def _gaussian_chain(dims: tuple[int, ...]) -> Any:
    """An all-exact Gaussian chain with the given dimensions, root first."""
    model: Any = full_normal(dims[-1])
    for obs_dim, lat_dim in reversed(list(itertools.pairwise(dims))):
        model = _Attached(NormalLGM(obs_dim, PD, lat_dim, PD), model, True, None)
    return model


def _boltzmann(kind: str, n: int) -> Any:
    if kind == "full":
        return FullBoltzmann(n)
    if kind == "diagonal":
        return DiagonalBoltzmann(n)
    return ChordalBoltzmann.from_edges(n, [(i, i + 1) for i in range(n - 1)])


def _gaussian_boltzmann(obs_dim: int, bol: Any, prior: Any, lat_mlp: Any) -> Any:
    """A Gaussian observable over a Boltzmann latent: exact for a full Boltzmann, learned otherwise."""
    if isinstance(bol, FullBoltzmann):
        return _Attached(BoltzmannLGM(obs_dim, PD, bol.n_neurons), prior, True, None)
    return _Attached(NormalBoltzmannHarmonium(obs_dim, PD, bol), prior, False, lat_mlp)


def _circuit(kind: str, n: int = 4, lat_dim: int = 2) -> Any:
    """$x$ Gaussian, $y$ Boltzmann of the given kind, $z$ Gaussian; the lower level is exact for a full Boltzmann."""
    bol = _boltzmann(kind, n)
    mlp = MultilayerPerceptron(full_normal(lat_dim), bol, (6,), jax.nn.tanh)
    upper = _Attached(
        BoltzmannNormalHarmonium(bol, lat_dim), full_normal(lat_dim), False, mlp
    )
    return _gaussian_boltzmann(3, bol, upper, None)


def _stack(depth: int, kind: str) -> Any:
    """Alternating levels $x$ Gaussian, then Boltzmann, Gaussian, ... up to ``depth`` latent levels, every level learned except those over a full Boltzmann; the top is the last level's family."""
    fams: list[Any] = [full_normal(2)]
    for level in range(1, depth + 1):
        fams.append(_boltzmann(kind, 2) if level % 2 else full_normal(1))
    model: Any = fams[-1]
    for level in range(depth, 0, -1):
        obs, lat = fams[level - 1], fams[level]
        nested = level > 1
        mlp = MultilayerPerceptron(lat, obs, (4,), jax.nn.tanh) if nested else None
        if level % 2:
            model = _gaussian_boltzmann(obs.data_dim, lat, model, mlp)
        else:
            model = _Attached(
                BoltzmannNormalHarmonium(obs, lat.data_dim), model, False, mlp
            )
    return model


def _perturbed(model: Any, seed: int) -> Array:
    """Initial parameters moved off the zero conjugation parameters and the zero inner bias."""
    params = model.initialize(jax.random.PRNGKey(seed), shape=0.3)
    noise = jax.random.normal(jax.random.PRNGKey(seed + 1), params.shape)
    return params + 0.1 * noise


def _gaussian_log_marginal(model: Any, params: Array, x: Array) -> Array:
    """$\\log p(x)$ of a model whose joint log-density is quadratic, by integrating the latents in closed form."""
    n_obs, n_all = x.shape[0], model.data_dim

    def log_joint(v: Array) -> Array:
        return model.log_density(params, v)

    zero = jnp.zeros(n_all)
    c = log_joint(zero)
    b = jax.grad(log_joint)(zero)
    prec = -jax.hessian(log_joint)(zero)
    p_xx, p_ux, p_uu = prec[:n_obs, :n_obs], prec[n_obs:, :n_obs], prec[n_obs:, n_obs:]
    m = b[n_obs:] - p_ux @ x
    n_lat = n_all - n_obs
    return (
        c
        + b[:n_obs] @ x
        - 0.5 * x @ p_xx @ x
        + 0.5 * n_lat * jnp.log(2 * jnp.pi)
        - 0.5 * jnp.linalg.slogdet(p_uu)[1]
        + 0.5 * m @ jnp.linalg.solve(p_uu, m)
    )


def _fork(name: str, exact: bool) -> _Graphical:
    if name == "cca":
        return _Graphical(CanonicalCorrelationAnalysis(2, PD, 1, PD, 2, PD), exact)
    return _Graphical(_Fan((1, 1, 1), 2), exact)


X3 = jnp.array([0.3, -0.2, 0.5])


class TestLayout:
    """The stored conjugation coordinates exist exactly for the learned levels."""

    def test_exact_chain_stores_no_conjugation(self) -> None:
        model = _gaussian_chain((3, 2, 1))
        assert model.split_coords(model.zeros())[2].shape == (0,)
        assert model.prr_man.split_coords(model.prr_man.zeros())[2].shape == (0,)

    @pytest.mark.parametrize("kind", ["chordal", "diagonal"])
    def test_learned_levels_store_their_conjugation(self, kind: str) -> None:
        model = _circuit(kind)
        upper = model.prr_man
        assert model.cnj_man.dim == upper.obs_man.dim
        assert upper.cnj_man.dim == upper.prr_man.dim + upper.inr_map.dim

    def test_exact_lower_level_stores_nothing(self) -> None:
        model = _circuit("full")
        assert model.cnj_man.dim == 0

    def test_split_join_round_trip(self) -> None:
        model = _stack(4, "chordal")
        params = _perturbed(model, 0)
        assert jnp.array_equal(model.join_coords(*model.split_coords(params)), params)

    def test_data_dim(self) -> None:
        model = _stack(3, "chordal")
        assert model.data_dim == sum(model.level_dims)


class TestExact:
    """With every conjugation exact, the ELBO is the closed-form log-marginal."""

    @pytest.mark.parametrize("dims", [(3, 2, 1), (3, 2, 2, 1)])
    def test_elbo_is_log_marginal(self, dims: tuple[int, ...]) -> None:
        model = _gaussian_chain(dims)
        params = _perturbed(model, 2)
        log_px = _gaussian_log_marginal(model, params, X3)
        elbo = model.elbo_at(jax.random.PRNGKey(3), params, X3, 5)
        assert jnp.allclose(elbo, log_px, atol=1e-10)

    def test_gradient_is_log_marginal_gradient(self) -> None:
        model = _gaussian_chain((3, 2, 2, 1))
        params = _perturbed(model, 4)
        g_true = jax.grad(lambda p: _gaussian_log_marginal(model, p, X3))(params)
        g_est = jax.grad(lambda p: model.elbo_at(jax.random.PRNGKey(5), p, X3, 5))(
            params
        )
        assert jnp.allclose(g_est, g_true, atol=1e-8)

    def test_harmonium_prior(self) -> None:
        """An exact harmonium in its own joint coordinates as the prior."""
        prior = NormalLGM(2, PD, 1, PD)
        model = _Attached(NormalLGM(3, PD, 2, PD), prior, True, None)
        params = _perturbed(model, 6)
        log_px = _gaussian_log_marginal(model, params, X3)
        elbo = model.elbo_at(jax.random.PRNGKey(7), params, X3, 5)
        assert jnp.allclose(elbo, log_px, atol=1e-10)

    @pytest.mark.parametrize("name", ["cca", "fan"])
    def test_graphical_harmonium(self, name: str) -> None:
        """A graphical harmonium with two or three observable harmoniums as the underlying harmonium."""
        model = _fork(name, True)
        params = _perturbed(model, 8)
        x = jnp.array([0.3, -0.2, 0.5])
        log_px = _gaussian_log_marginal(model, params, x)
        elbo = model.elbo_at(jax.random.PRNGKey(9), params, x, 5)
        assert jnp.allclose(elbo, log_px, atol=1e-10)

    def test_residuals_vanish(self) -> None:
        model = _gaussian_chain((3, 2, 2, 1))
        params = _perturbed(model, 8)
        ws = model.sample_recognition(jax.random.PRNGKey(9), params, X3, 6)
        for w in ws:
            r0, rx = model.conjugation_residuals(params, w, X3)
            assert jnp.allclose(jnp.stack(r0 + rx), 0.0, atol=1e-10)


class TestDecomposition:
    """$\\log p(x, w) - \\log q(w \\mid x) = c(x) + \\sum r^0 - \\sum r^X$."""

    @staticmethod
    def _check(model: Any, params: Array, x: Array) -> None:
        ws = model.sample_recognition(jax.random.PRNGKey(11), params, x, 8)
        c_x = model.conjugation_baseline(params, x)
        for w in ws:
            lhs = model.log_density(
                params, jnp.concatenate([x, w])
            ) - model.recognition_log_density(params, x, w)
            r0, rx = model.conjugation_residuals(params, w, x)
            assert jnp.allclose(lhs, c_x + sum(r0) - sum(rx), atol=1e-10)

    @pytest.mark.parametrize("kind", ["chordal", "diagonal", "full"])
    def test_circuit(self, kind: str) -> None:
        model = _circuit(kind)
        self._check(model, _perturbed(model, 10), X3)

    @pytest.mark.parametrize("depth", [3, 4])
    @pytest.mark.parametrize("kind", ["chordal", "full"])
    def test_stack(self, depth: int, kind: str) -> None:
        model = _stack(depth, kind)
        self._check(model, _perturbed(model, 12), jnp.array([0.4, -0.3]))

    @pytest.mark.parametrize("name", ["cca", "fan"])
    def test_learned_graphical_harmonium(self, name: str) -> None:
        model = _fork(name, False)
        self._check(model, _perturbed(model, 13), jnp.array([0.3, -0.2, 0.5]))


class TestExactLevel:
    """An exact level has vanishing residuals on every latent state."""

    def test_lower_residual_vanishes_on_every_state(self) -> None:
        model = _circuit("full", n=4)
        params = _perturbed(model, 12)
        ys = jnp.array(list(itertools.product([0.0, 1.0], repeat=4)))
        r_y = jax.vmap(lambda y: model.conjugation_residual(params, y))(ys)
        assert jnp.allclose(r_y, 0.0, atol=1e-10)


### Quadrature ground truth ###


def _normal_nodes(family: Any, params: Array) -> tuple[Array, Array]:
    """Gauss--Hermite nodes and weights for a one-dimensional normal at the given natural parameters."""
    mean, cov = family.split_mean_covariance(family.to_mean(params))
    sd = jnp.sqrt(family.cov_man.to_matrix(cov)[0, 0])
    nodes, weights = np.polynomial.hermite.hermgauss(40)
    zs = (mean[0] + jnp.sqrt(2.0) * sd * jnp.asarray(nodes))[:, None]
    return zs, jnp.asarray(weights) / jnp.sqrt(jnp.pi)


def _states(family: Any) -> Array:
    return jnp.array(list(itertools.product([0.0, 1.0], repeat=family.data_dim)))


def _level_families(model: Any) -> list[Any]:
    """The family of each latent level, root first."""
    fams: list[Any] = []
    prior: Any = model.prr_man
    while isinstance(prior, VariationalDifferentiable):
        fams.append(prior.obs_man)
        prior = prior.prr_man
    fams.append(prior)
    return fams


def _expectation(
    families: list[Any], conditional: Any, f: Any, upto: int | None = None
) -> Array:
    """$\\mathbb E[f(w)]$ over a chain sampled from the top down by exact enumeration and quadrature.

    ``conditional(j, above)`` returns the natural parameters of level $j$ given the levels above it (``above`` holds levels $j + 1, \\ldots$, root first). Levels below ``upto`` are not integrated, and ``f`` receives only the integrated levels.
    """
    n = len(families)
    stop = 0 if upto is None else upto

    def recurse(j: int, above: list[Array]) -> Array:
        if j < stop:
            return f(above)
        fam, nat = families[j], conditional(j, above)
        if hasattr(fam, "split_mean_covariance"):
            pts, ws = _normal_nodes(fam, nat)
        else:
            pts = _states(fam)
            ws = jnp.exp(jax.vmap(lambda s: fam.log_density(nat, s))(pts))
        vals = jax.vmap(lambda pt: recurse(j - 1, [pt, *above]))(pts)
        return jnp.sum(ws * vals)

    return recurse(n - 1, [])


def _recognition_conditional(model: Any, params: Array, x: Array) -> Any:
    def conditional(j: int, above: list[Array]) -> Array:
        fams = _level_families(model)
        lower = [jnp.zeros(fam.data_dim) for fam in fams[:j]]
        w = jnp.concatenate([*lower, jnp.zeros(fams[j].data_dim), *above])
        return model.recognition_conditionals(params, x, w)[j]

    return conditional


def _quadrature_elbo(model: Any, params: Array, x: Array) -> Array:
    fams = _level_families(model)

    def f(levels: list[Array]) -> Array:
        w = jnp.concatenate(levels)
        return model.log_density(
            params, jnp.concatenate([x, w])
        ) - model.recognition_log_density(params, x, w)

    return _expectation(fams, _recognition_conditional(model, params, x), f)


def _variance(families: list[Any], conditional: Any, r: Any, upto: int) -> Array:
    mean = _expectation(families, conditional, r, upto)
    second = _expectation(families, conditional, lambda lv: r(lv) ** 2, upto)
    return second - mean**2


class TestGradientUnbiased:
    """The estimator's gradient matches the ELBO's, computed by enumeration and quadrature."""

    def test_circuit_matches_quadrature(self) -> None:
        model = _circuit("chordal", n=3, lat_dim=1)
        params = _perturbed(model, 15)
        x = jnp.array([0.4, -0.7, 0.1])
        g_true = jax.grad(lambda p: _quadrature_elbo(model, p, x))(params)
        grad = jax.jit(jax.grad(lambda p, k: model.elbo_at(k, p, x, 20000)))
        g_est = jnp.mean(
            jnp.stack([grad(params, jax.random.PRNGKey(20 + i)) for i in range(4)]),
            axis=0,
        )
        rel = jnp.linalg.norm(g_est - g_true) / jnp.linalg.norm(g_true)
        assert rel < 0.03

    @pytest.mark.parametrize("n_samples", [1, 2])
    def test_small_sample_matches_quadrature(self, n_samples: int) -> None:
        """Averaged over many keys, estimates from one or two samples are unbiased too."""
        model = _circuit("chordal", n=3, lat_dim=1)
        params = _perturbed(model, 15)
        x = jnp.array([0.4, -0.7, 0.1])
        g_true = jax.grad(lambda p: _quadrature_elbo(model, p, x))(params)
        grad = jax.grad(lambda p, k: model.elbo_at(k, p, x, n_samples))
        keys = jax.random.split(jax.random.PRNGKey(30), 40000)
        g_est = jnp.mean(jax.jit(jax.vmap(grad, in_axes=(None, 0)))(params, keys), 0)
        rel = jnp.linalg.norm(g_est - g_true) / jnp.linalg.norm(g_true)
        assert rel < 0.03

    def test_depth_three_matches_quadrature(self) -> None:
        """Gaussian $x$, Boltzmann $y$, Gaussian $z$, Boltzmann top; every level learned."""
        model = _stack(3, "diagonal")
        params = _perturbed(model, 16)
        x = jnp.array([0.4, -0.3])
        g_true = jax.grad(lambda p: _quadrature_elbo(model, p, x))(params)
        grad = jax.jit(jax.grad(lambda p, k: model.elbo_at(k, p, x, 20000)))
        g_est = jnp.mean(
            jnp.stack([grad(params, jax.random.PRNGKey(60 + i)) for i in range(4)]),
            axis=0,
        )
        rel = jnp.linalg.norm(g_est - g_true) / jnp.linalg.norm(g_true)
        assert rel < 0.03


class TestInnerResidualVariances:
    """$\\mathrm{Var}_q[r^X]$ and its gradient, including the dependence of $q$ on the parameters, against quadrature."""

    def test_matches_quadrature(self) -> None:
        model = _circuit("chordal", n=3, lat_dim=1)
        params = _perturbed(model, 16)
        x = jnp.array([0.4, -0.7, 0.1])
        fams = _level_families(model)

        def v_true(p: Array) -> Array:
            def r(levels: list[Array]) -> Array:
                (z,) = levels
                w = jnp.concatenate([jnp.zeros(fams[0].data_dim), z])
                return model.conjugation_residuals(p, w, x)[1][0]

            return _variance(fams, _recognition_conditional(model, p, x), r, 1)

        def loss(p: Array, k: Array) -> Array:
            (var,) = model.inner_residual_variances_at(k, p, x, 4)
            return var

        keys = jax.random.split(jax.random.PRNGKey(40), 40000)
        vals, grads = jax.jit(jax.vmap(jax.value_and_grad(loss), in_axes=(None, 0)))(
            params, keys
        )
        assert jnp.allclose(jnp.mean(vals), v_true(params), rtol=0.03)
        g_true = jax.grad(v_true)(params)
        rel = jnp.linalg.norm(jnp.mean(grads, 0) - g_true) / jnp.linalg.norm(g_true)
        assert rel < 0.05


class TestPriorResidualVariances:
    """$\\mathrm{Var}_p[r^0]$ per level and their gradients, including the dependence of the ancestral distribution on the parameters, against enumeration and quadrature."""

    def test_matches_quadrature(self) -> None:
        model = _circuit("chordal", n=3, lat_dim=1)
        params = _perturbed(model, 17)
        fams = _level_families(model)
        x = jnp.zeros(3)

        def generative(p: Array) -> Any:
            upper = model.prr_man
            p_upper = model.prior_params(p)

            def conditional(j: int, above: list[Array]) -> Array:
                if j == 1:
                    return upper.prior_params(p_upper)
                return upper.likelihood_at(p_upper, above[0])

            return conditional

        def v_true(p: Array) -> Array:
            def r(index: int, upto: int) -> Any:
                def f(levels: list[Array]) -> Array:
                    lower = [jnp.zeros(fam.data_dim) for fam in fams[:upto]]
                    w = jnp.concatenate([*lower, *levels])
                    return model.conjugation_residuals(p, w, x)[0][index]

                return f

            return jnp.stack(
                [
                    _variance(fams, generative(p), r(0, 0), 0),
                    _variance(fams, generative(p), r(1, 1), 1),
                ]
            )

        def losses(p: Array, k: Array) -> Array:
            return jnp.stack(model.prior_residual_variances(k, p, 4))

        keys = jax.random.split(jax.random.PRNGKey(50), 40000)
        vals = jax.jit(jax.vmap(losses, in_axes=(None, 0)))(params, keys)
        grads = jax.jit(jax.vmap(jax.jacobian(losses), in_axes=(None, 0)))(params, keys)
        # The residuals are heavy-tailed, so values and gradients are checked in
        # standard errors of the mean over keys
        n = keys.shape[0]
        se = jnp.std(vals, 0) / jnp.sqrt(n)
        assert jnp.all(jnp.abs(jnp.mean(vals, 0) - v_true(params)) < 4 * se)
        g_true = jax.jacobian(v_true)(params)
        g_se = jnp.std(grads, 0) / jnp.sqrt(n)
        assert jnp.all(jnp.abs(jnp.mean(grads, 0) - g_true) < 5 * g_se + 1e-10)
