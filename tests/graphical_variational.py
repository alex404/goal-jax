"""Tests for nested variational conjugation in geometry/exponential_family/variational.py.

A :class:`~goal.geometry.exponential_family.variational.VariationalConjugated` whose deep
model is again variational nests to any depth, and its graphical harmonium may attach
several harmoniums to one latent. Five ground truths pin it down. With exact conjugation
parameters the model is the graphical harmonium: its prior, log-partition function and
$c(x)$ are those of the conjugated harmonium, and Gaussian models have an ELBO equal to
their log-marginal, computed in closed form from the quadratic joint log-density; this
holds for two- and three-level chains, an exact harmonium as the prior, and graphical
harmoniums with two and three attached harmoniums on one latent. For Boltzmann and
Gaussian levels mixed to depths two to four, $\\log \\tilde p(x, w) - \\log q(w \\mid x)$
must equal $c(x)$ plus :meth:`elbo_residual`, and $\\log \\tilde p(x, w)$ the harmonium's
unnormalized log-density minus $\\tilde\\Psi$ plus the residuals; exact levels have vanishing
residuals. The estimator's
gradient must match the gradient of the ELBO computed by enumerating the Boltzmann levels
and Gauss--Hermite quadrature over one-dimensional Gaussian levels, at depth two (also
from one or two samples per estimate) and depth three. The recognition and prior
residual variances, with their gradients, including the dependence of their sampling
distributions on the parameters, are checked against the same quadrature.
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
    DifferentiableVariationalConjugated,
    GraphicalHarmonium,
    Harmonium,
    IdentityEmbedding,
    MultilayerPerceptron,
    PositiveDefinite,
    SubCliquesEmbedding,
    VariationalConjugated,
)
from goal.geometry.manifold.util import split_by_dims
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
class _Link(GraphicalHarmonium[Any]):
    """A harmonium whose posterior is attached to the leading nodes of a latent model: the whole of a single family, or the root of a harmonium."""

    lwr: Any
    dep: Any

    @property
    @override
    def pst_man(self) -> Any:
        return self.dep

    @property
    @override
    def obs_hrms_att_clqs(
        self,
    ) -> tuple[tuple[Harmonium[Any, Any], tuple[int, ...]], ...]:
        return ((self.lwr, tuple(range(self.lwr.pst_man.n_nodes))),)


def _conjugation(level: Any, lkl_params: Array, cnj_fun_params: Array) -> Array:
    """The conjugation parameters of a test level: exact, a stored constant, or ``inr_map`` of the observable bias; each attached harmonium's placed on its clique."""
    hrm = level.gen_hrm
    if level.inr_map is not None:
        (obs_hrm,), (att,) = hrm.obs_hrms, hrm.att_clqs
        theta_x, _ = hrm.lkl_fun_man.split_coords(lkl_params)
        rho = level.inr_map(cnj_fun_params, theta_x)
        return SubCliquesEmbedding(att, level.prr_man, obs_hrm.pst_man).embed(rho)
    stored = split_by_dims(cnj_fun_params, _stored_dims(level))
    rho = level.prr_man.zeros()
    for obs_hrm, att, hrm_lkl, rho_i in zip(
        hrm.obs_hrms, hrm.att_clqs, hrm.likelihood_functions(lkl_params), stored
    ):
        if level.exact:
            rho_i = obs_hrm.conjugation_parameters(hrm_lkl)
        rho = rho + SubCliquesEmbedding(att, level.prr_man, obs_hrm.pst_man).embed(
            rho_i
        )
    return rho


def _stored_dims(level: Any) -> tuple[int, ...]:
    if level.exact:
        return tuple(0 for _ in level.gen_hrm.obs_hrms)
    return tuple(obs_hrm.pst_man.dim for obs_hrm in level.gen_hrm.obs_hrms)


def _cnj_man(level: Any) -> Euclidean:
    if level.inr_map is not None:
        return Euclidean(level.inr_map.dim)
    return Euclidean(sum(_stored_dims(level)))


@dataclass(frozen=True)
class _ExactLevel(DifferentiableVariationalConjugated[Any, Any, Euclidean]):
    """A level over an exact prior family, with exact, constant or mapped conjugation parameters."""

    hrm: Any
    family: Any
    exact: bool
    inr_map: MultilayerPerceptron[Any, Any] | None

    @property
    @override
    def gen_hrm(self) -> Any:
        return self.hrm

    @property
    @override
    def pst_prr_emb(self) -> IdentityEmbedding[Any]:
        return IdentityEmbedding(self.family)

    @property
    @override
    def cnj_fun_man(self) -> Euclidean:
        return _cnj_man(self)

    @override
    def conjugation_parameters(self, lkl_params: Array, cnj_fun_params: Array) -> Array:
        return _conjugation(self, lkl_params, cnj_fun_params)


@dataclass(frozen=True)
class _NestedLevel(VariationalConjugated[Any, Euclidean]):
    """A level over a variational deep model, with exact, constant or mapped conjugation parameters."""

    hrm: Any
    deep: Any
    exact: bool
    inr_map: MultilayerPerceptron[Any, Any] | None

    @property
    @override
    def gen_hrm(self) -> Any:
        return self.hrm

    @property
    @override
    def dep_man(self) -> Any:
        return self.deep

    @property
    @override
    def pst_prr_emb(self) -> IdentityEmbedding[Any]:
        return IdentityEmbedding(self.deep.gen_hrm)

    @property
    @override
    def cnj_fun_man(self) -> Euclidean:
        return _cnj_man(self)

    @override
    def conjugation_parameters(self, lkl_params: Array, cnj_fun_params: Array) -> Array:
        return _conjugation(self, lkl_params, cnj_fun_params)


def _level(
    lwr: Any,
    deep: Any,
    exact: bool,
    mlp: Any,
) -> Any:
    """``lwr`` attached to the deep model: an exact family or another level."""
    if isinstance(deep, VariationalConjugated):
        return _NestedLevel(_Link(lwr, deep.gen_hrm), deep, exact, mlp)
    return _ExactLevel(_Link(lwr, deep), deep, exact, mlp)


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


def _gaussian_chain(dims: tuple[int, ...]) -> Any:
    """An all-exact Gaussian chain with the given dimensions, root first."""
    model: Any = full_normal(dims[-1])
    for obs_dim, lat_dim in reversed(list(itertools.pairwise(dims))):
        model = _level(NormalLGM(obs_dim, PD, lat_dim, PD), model, True, None)
    return model


def _boltzmann(kind: str, n: int) -> Any:
    if kind == "full":
        return FullBoltzmann(n)
    if kind == "diagonal":
        return DiagonalBoltzmann(n)
    return ChordalBoltzmann.from_edges(n, [(i, i + 1) for i in range(n - 1)])


def _gaussian_boltzmann(obs_dim: int, bol: Any, deep: Any, mlp: Any) -> Any:
    """A Gaussian observable over a Boltzmann latent: exact for a full Boltzmann, learned otherwise."""
    if isinstance(bol, FullBoltzmann):
        return _level(BoltzmannLGM(obs_dim, PD, bol.n_neurons), deep, True, None)
    return _level(NormalBoltzmannHarmonium(obs_dim, PD, bol), deep, False, mlp)


def _circuit(kind: str, n: int = 4, lat_dim: int = 2) -> Any:
    """$x$ Gaussian, $y$ Boltzmann of the given kind, $z$ Gaussian; the lower level is exact for a full Boltzmann, and $\\rho_Z$ is a map of the bias of $y$."""
    bol = _boltzmann(kind, n)
    mlp = MultilayerPerceptron(full_normal(lat_dim), bol, (6,), jax.nn.tanh)
    upper = _level(
        BoltzmannNormalHarmonium(bol, lat_dim), full_normal(lat_dim), False, mlp
    )
    return _gaussian_boltzmann(3, bol, upper, None)


def _stack(depth: int, kind: str) -> Any:
    """Alternating levels $x$ Gaussian, then Boltzmann, Gaussian, ... up to ``depth`` latent levels, every level learned except those over a full Boltzmann; deep levels map the bias of their observable to their conjugation parameters."""
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
            model = _level(
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


def _fork(name: str, exact: bool) -> _ExactLevel:
    hrm: Any = (
        CanonicalCorrelationAnalysis(2, PD, 1, PD, 2, PD)
        if name == "cca"
        else _Fan((1, 1, 1), 2)
    )
    return _ExactLevel(hrm, hrm.pst_man, exact, None)


X3 = jnp.array([0.3, -0.2, 0.5])


class TestLayout:
    """The stored conjugation function parameters exist exactly for the learned levels, and the rest is the graphical harmonium."""

    def test_exact_chain_stores_no_conjugation(self) -> None:
        model = _gaussian_chain((3, 2, 1))
        assert model.dim == model.gen_hrm.dim

    @pytest.mark.parametrize("kind", ["chordal", "diagonal"])
    def test_learned_levels_store_their_conjugation(self, kind: str) -> None:
        model = _circuit(kind)
        upper = model.dep_man
        assert model.cnj_fun_man.dim == model.gen_hrm.obs_hrms[0].pst_man.dim
        assert upper.cnj_fun_man.dim == upper.inr_map.dim
        assert (
            model.dim
            == model.gen_hrm.dim + model.cnj_fun_man.dim + upper.cnj_fun_man.dim
        )

    def test_one_conjugation_element_per_level(self) -> None:
        """Element $k$ of the conjugation tuple is level $k$, exact levels included."""
        model = _stack(4, "chordal")
        levels = [model]
        while isinstance(levels[-1].dep_man, VariationalConjugated):
            levels.append(levels[-1].dep_man)
        assert model.snd_man.elm_mans == tuple(level.cnj_fun_man for level in levels)
        params = _perturbed(model, 0)
        _, cnj_fun_tup_params = model.split_coords(params)
        _, *dep_cnj_fun_paramss = model.snd_man.split_coords(cnj_fun_tup_params)
        dep_params = model.conjugated_prior_params(params)
        _, dep_cnj_fun_tup_params = model.dep_man.split_coords(dep_params)
        assert jnp.array_equal(
            dep_cnj_fun_tup_params, jnp.concatenate(dep_cnj_fun_paramss)
        )

    def test_exact_lower_level_stores_nothing(self) -> None:
        model = _circuit("full")
        assert model.cnj_fun_man.dim == 0

    def test_split_join_round_trip(self) -> None:
        model = _stack(4, "chordal")
        params = _perturbed(model, 0)
        assert jnp.array_equal(model.join_coords(*model.split_coords(params)), params)

    def test_data_dim(self) -> None:
        model = _stack(3, "chordal")
        assert model.data_dim == model.gen_hrm.data_dim


class TestExact:
    """With every conjugation exact, the model is the graphical harmonium and the ELBO is its log-marginal."""

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
        """An exact harmonium in its own joint coordinates as the prior family."""
        model = _level(NormalLGM(3, PD, 2, PD), NormalLGM(2, PD, 1, PD), True, None)
        params = _perturbed(model, 6)
        log_px = _gaussian_log_marginal(model, params, X3)
        elbo = model.elbo_at(jax.random.PRNGKey(7), params, X3, 5)
        assert jnp.allclose(elbo, log_px, atol=1e-10)

    @pytest.mark.parametrize("name", ["cca", "fan"])
    def test_graphical_harmonium(self, name: str) -> None:
        """A graphical harmonium with two or three attached harmoniums."""
        model = _fork(name, True)
        params = _perturbed(model, 8)
        log_px = _gaussian_log_marginal(model, params, X3)
        elbo = model.elbo_at(jax.random.PRNGKey(9), params, X3, 5)
        assert jnp.allclose(elbo, log_px, atol=1e-10)

    @pytest.mark.parametrize("name", ["cca", "fan"])
    def test_matches_differentiable_graphical(self, name: str) -> None:
        """Prior, log-partition function and $c(x)$ equal those of the conjugated graphical harmonium."""
        model = _fork(name, True)
        params = _perturbed(model, 8)
        hrm_params, _ = model.split_coords(params)
        hrm = model.gen_hrm
        assert jnp.allclose(
            model.conjugated_prior_params(params), hrm.prior(hrm_params), atol=1e-12
        )
        assert jnp.allclose(
            model.conjugated_log_partition_function(params),
            hrm.log_partition_function(hrm_params),
            atol=1e-10,
        )
        assert jnp.allclose(
            model.conjugation_baseline(params, X3),
            hrm.log_observable_density(hrm_params, X3),
            atol=1e-10,
        )

    def test_residuals_vanish(self) -> None:
        model = _gaussian_chain((3, 2, 2, 1))
        params = _perturbed(model, 8)
        ws = model.sample_recognition(jax.random.PRNGKey(9), params, X3, 6)
        for w in ws:
            assert jnp.allclose(
                jnp.stack(
                    model.conjugation_residuals(params, jnp.concatenate([X3, w]))
                ),
                0.0,
                atol=1e-10,
            )
            assert jnp.allclose(model.elbo_residual(params, X3, w), 0.0, atol=1e-10)

    def test_zero_shift_conditioning(self) -> None:
        """Conditioning on an observation that adds nothing to the latent bias gives the prior: with a zero interaction, $q(w \\mid x) = \\tilde p(w)$."""
        model = _gaussian_chain((3, 2, 2, 1))
        params = _perturbed(model, 10)
        hrm_params, cnj_fun_tup_params = model.split_coords(params)
        obs_p, int_p, lat_p = model.gen_hrm.split_coords(hrm_params)
        zero_int = model.gen_hrm.join_coords(obs_p, jnp.zeros_like(int_p), lat_p)
        params = model.join_coords(zero_int, cnj_fun_tup_params)
        ws = model.sample_recognition(jax.random.PRNGKey(11), params, X3, 4)
        dep_prior = model.conjugated_prior_params(params)
        for w in ws:
            assert jnp.allclose(
                model.recognition_log_density(params, X3, w),
                model.dep_man.log_density(dep_prior, w),
                atol=1e-10,
            )


class TestDecomposition:
    """$\\log \\tilde p(x, w) - \\log q(w \\mid x) = c(x) + r(w)$ with :meth:`elbo_residual`, and $\\log \\tilde p(x, w) = \\theta \\cdot \\mathbf s(x, w) + \\log h(x, w) - \\tilde\\Psi(\\theta) + \\sum r(w)$."""

    @staticmethod
    def _check(model: Any, params: Array, x: Array) -> None:
        ws = model.sample_recognition(jax.random.PRNGKey(11), params, x, 8)
        c_x = model.conjugation_baseline(params, x)
        hrm_params, _ = model.split_coords(params)
        psi = model.conjugated_log_partition_function(params)
        for w in ws:
            xw = jnp.concatenate([x, w])
            log_p = model.log_density(params, xw)
            lhs = log_p - model.recognition_log_density(params, x, w)
            assert jnp.allclose(
                lhs, c_x + model.elbo_residual(params, x, w), atol=1e-10
            )
            unnormalized = jnp.dot(
                hrm_params, model.gen_hrm.sufficient_statistic(xw)
            ) + model.gen_hrm.log_base_measure(xw)
            residuals = sum(model.conjugation_residuals(params, xw))
            assert jnp.allclose(log_p, unnormalized - psi + residuals, atol=1e-10)

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
        self._check(model, _perturbed(model, 13), X3)


class TestExactLevel:
    """An exact level has vanishing residuals on every latent state."""

    def test_lower_residual_vanishes_on_every_state(self) -> None:
        model = _circuit("full", n=4)
        params = _perturbed(model, 12)
        ys = jnp.array(list(itertools.product([0.0, 1.0], repeat=4)))
        z = jnp.zeros(2)
        r_y = jax.vmap(
            lambda y: model.conjugation_residual(params, jnp.concatenate([y, z]))
        )(ys)
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


def _deep(model: Any) -> Any:
    deep = model.dep_man
    return deep if isinstance(deep, VariationalConjugated) else model.prr_man


def _levels(model: Any, params: Array) -> tuple[list[Any], list[Any]]:
    """The family of each level of a model at the given parameters, and the natural parameters of each level given the levels above it (root first); a model is an exact family or a level."""
    if not isinstance(model, VariationalConjugated):
        return [model], [lambda _above: params]
    fams, conds = _levels(_deep(model), model.conjugated_prior_params(params))
    obs = model.gen_hrm.obs_hrms[0].obs_man

    def own(above: list[Array]) -> Array:
        return model.likelihood_at(params, jnp.concatenate(above))

    return [obs, *fams], [own, *conds]


def _conditional(model: Any, params: Array) -> tuple[list[Any], Any]:
    """The families of a model's levels and ``conditional(j, above)``, for :func:`_expectation`."""
    fams, conds = _levels(model, params)

    def conditional(j: int, above: list[Array]) -> Array:
        return conds[j](above)

    return fams, conditional


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


def _quadrature_elbo(model: Any, params: Array, x: Array) -> Array:
    fams, conditional = _conditional(_deep(model), model.recognition_at(params, x))

    def f(levels: list[Array]) -> Array:
        w = jnp.concatenate(levels)
        return model.log_density(
            params, jnp.concatenate([x, w])
        ) - model.recognition_log_density(params, x, w)

    return _expectation(fams, conditional, f)


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


class TestRecognitionResidualVariances:
    """$\\mathrm{Var}_q[r^X]$ of the upper level and its gradient, including the dependence of $q$ on the parameters, against quadrature."""

    def test_matches_quadrature(self) -> None:
        model = _circuit("chordal", n=3, lat_dim=1)
        params = _perturbed(model, 16)
        x = jnp.array([0.4, -0.7, 0.1])
        upper = model.dep_man

        def v_true(p: Array) -> Array:
            dep_post = model.recognition_at(p, x)
            fams, conditional = _conditional(
                upper.prr_man, upper.conjugated_prior_params(dep_post)
            )

            def r(levels: list[Array]) -> Array:
                (z,) = levels
                return upper.conjugation_residual(dep_post, z)

            return _variance(fams, conditional, r, 0)

        def loss(p: Array, k: Array) -> Array:
            _, var = model.recognition_residual_variances_at(k, p, x, 4)
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
    """$\\mathrm{Var}_{\\tilde p}$ of each level's residual and their gradients, including the dependence of the ancestral distribution on the parameters, against enumeration and quadrature."""

    def test_matches_quadrature(self) -> None:
        model = _circuit("chordal", n=3, lat_dim=1)
        params = _perturbed(model, 17)
        n_x = model.obs_man.data_dim
        n_y = model.gen_hrm.obs_hrms[0].pst_man.data_dim

        def v_true(p: Array) -> Array:
            fams, conditional = _conditional(
                _deep(model), model.conjugated_prior_params(p)
            )

            def r_lower(levels: list[Array]) -> Array:
                return model.conjugation_residual(p, jnp.concatenate(levels))

            def r_upper(levels: list[Array]) -> Array:
                xw = jnp.concatenate([jnp.zeros(n_x + n_y), *levels])
                return model.conjugation_residuals(p, xw)[1]

            return jnp.stack(
                [
                    _variance(fams, conditional, r_lower, 0),
                    _variance(fams, conditional, r_upper, 1),
                ]
            )

        def losses(p: Array, k: Array) -> Array:
            return jnp.stack(model.conjugation_residual_variances(k, p, 4))

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
