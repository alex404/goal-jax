"""Tests for the base exponential families of ``models/base``, through the contract of ``geometry/exponential_family/base.py`` and its product combinators.

One case list covers every family. Each case is checked against independent ground truth: the density normalizes under enumeration, quadrature or importance sampling; the relative entropy matches its quadrature; sample averages of the sufficient statistic match the mean parameters, and samples lie in the support. Analytic families also satisfy the natural-mean round trip and the Legendre identity $\\phi(\\eta) = \\eta \\cdot \\theta - \\psi(\\theta)$. Closed forms that pin each family's conventions follow, one class per family.
"""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scipy import special, stats

from goal.geometry import (
    Analytic,
    Differentiable,
    DifferentiableProduct,
    PositiveDefinite,
)
from goal.models import (
    Bernoulli,
    Bernoullis,
    Binomial,
    Binomials,
    Categorical,
    CoMPoisson,
    Dirichlet,
    Normal,
    Poisson,
    Poissons,
    VonMises,
    VonMisesProduct,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-5
ATOL = 1e-7
N_SAMPLES = 50_000

type Quadrature = Callable[[], tuple[Array, Array]]


# Quadrature rules: points $x_i$ and weights $w_i$ with $\sum_i w_i f(x_i) \approx \int f$.


def lattice(*sizes: int) -> Quadrature:
    """Every integer point of the box $\\{0, \\ldots, n_1 - 1\\} \\times \\cdots$, with unit weights."""

    def rule() -> tuple[Array, Array]:
        points = np.array(list(itertools.product(*(range(n) for n in sizes))))
        return jnp.asarray(points), jnp.ones(len(points))

    return rule


def torus(dim: int, n: int = 128) -> Quadrature:
    """Rectangle rule on $[-\\pi, \\pi)^d$, spectrally accurate for smooth periodic integrands."""

    def rule() -> tuple[Array, Array]:
        angles = jnp.linspace(-jnp.pi, jnp.pi, n, endpoint=False)
        grid = jnp.stack(jnp.meshgrid(*([angles] * dim), indexing="ij"), axis=-1)
        points = grid.reshape(-1, dim)
        return points, jnp.full(len(points), (2 * jnp.pi / n) ** dim)

    return rule


def line(lo: float, hi: float, n: int = 4001) -> Quadrature:
    """Trapezoid rule on $[lo, hi]$."""

    def rule() -> tuple[Array, Array]:
        points = jnp.linspace(lo, hi, n)
        weights = jnp.full(n, (hi - lo) / (n - 1)).at[jnp.array([0, -1])].mul(0.5)
        return points[:, None], weights

    return rule


def simplex(k: int, n: int = 200_000) -> Quadrature:
    """Importance sampling on the simplex from the uniform Dirichlet, whose density is $(k - 1)!$."""

    def rule() -> tuple[Array, Array]:
        points = jax.random.dirichlet(jax.random.PRNGKey(0), jnp.ones(k), (n,))
        return points, jnp.full(n, 1 / (n * math.factorial(k - 1)))

    return rule


# Support predicates over a batch of samples.


def in_range(hi: float) -> Callable[[Array], Array]:
    """Integers in $\\{0, \\ldots, hi\\}$; ``hi = inf`` for counts."""

    def pred(xs: Array) -> Array:
        return jnp.all((xs >= 0) & (xs <= hi) & (xs == jnp.round(xs)))

    return pred


def finite(xs: Array) -> Array:
    return jnp.all(jnp.isfinite(xs))


def on_simplex(xs: Array) -> Array:
    return jnp.all(xs > 0) & jnp.allclose(jnp.sum(xs, axis=1), 1.0, atol=1e-6)


@dataclass(frozen=True)
class Case:
    name: str
    model: Differentiable
    params: tuple[float, ...]
    """Natural parameters of the reference distribution $p$."""
    other: tuple[float, ...]
    """Natural parameters of a second distribution $q$, for the relative entropy."""
    quadrature: Quadrature
    quad_atol: float
    support: Callable[[Array], Array]
    sample_atol: float


CASES = [
    Case(
        "bernoulli", Bernoulli(), (0.8,), (-0.5,), lattice(2), 1e-7, in_range(1), 0.01
    ),
    Case(
        "bernoullis",
        Bernoullis(3),
        (0.8, -0.3, 1.5),
        (0.0, 0.5, -1.0),
        lattice(2, 2, 2),
        1e-7,
        in_range(1),
        0.01,
    ),
    Case(
        "categorical",
        Categorical(5),
        (0.5, -0.2, 1.0, -1.0),
        (0.0, 0.3, -0.5, 0.8),
        lattice(5),
        1e-7,
        in_range(4),
        0.01,
    ),
    Case(
        "binomial", Binomial(10), (0.5,), (-0.4,), lattice(11), 1e-7, in_range(10), 0.02
    ),
    Case(
        "binomials",
        Binomials(3, 4),
        (0.5, -0.5, 1.0),
        (0.0, 0.2, -0.8),
        lattice(5, 5, 5),
        1e-7,
        in_range(4),
        0.02,
    ),
    Case(
        "poisson", Poisson(), (1.1,), (0.4,), lattice(60), 1e-7, in_range(jnp.inf), 0.02
    ),
    Case(
        "poissons",
        Poissons(3),
        (0.5, 1.0, 1.5),
        (0.8, 0.2, 1.0),
        lattice(30, 30, 30),
        1e-7,
        in_range(jnp.inf),
        0.03,
    ),
    # Mode 3; overdispersed ($\nu = 0.7$, geometric envelope) and underdispersed
    # ($\nu = 1.5$, Poisson envelope) exercise both branches of the sampler.
    Case(
        "com_poisson_over",
        CoMPoisson(),
        (0.7 * math.log(3.0), -0.7),
        (0.5, -1.0),
        lattice(200),
        1e-7,
        in_range(jnp.inf),
        0.03,
    ),
    Case(
        "com_poisson_under",
        CoMPoisson(),
        (1.5 * math.log(3.0), -1.5),
        (0.5, -1.0),
        lattice(200),
        1e-7,
        in_range(jnp.inf),
        0.03,
    ),
    Case(
        "von_mises", VonMises(), (1.4, 1.4), (-0.5, 0.3), torus(1), 1e-7, finite, 0.01
    ),
    Case(
        "von_mises_product",
        VonMisesProduct(2),
        (1.4, 1.4, -0.6, 0.2),
        (0.0, 1.0, 0.5, -0.5),
        torus(2),
        1e-7,
        finite,
        0.01,
    ),
    Case(
        "dirichlet",
        Dirichlet(3),
        (2.0, 3.0, 4.0),
        (3.0, 2.0, 2.0),
        simplex(3),
        0.02,
        on_simplex,
        0.02,
    ),
    # Mean 0.5 and variance 2: $\theta = (\mu / \sigma^2, -1 / (2 \sigma^2))$.
    Case(
        "normal",
        Normal(1, PositiveDefinite()),
        (0.25, -0.25),
        (-0.5, -0.5),
        line(-20.0, 20.0),
        1e-7,
        finite,
        0.1,
    ),
]

ANALYTIC = [c for c in CASES if isinstance(c.model, Analytic)]


def ids(cases: list[Case]) -> list[str]:
    return [c.name for c in cases]


def quadrature_mean(case: Case, params: Array, f: Callable[[Array], Array]) -> Array:
    """$\\int p(x) f(x) dx$ under the case's quadrature rule."""
    xs, ws = case.quadrature()
    densities = jax.vmap(case.model.density, in_axes=(None, 0))(params, xs)
    values = jax.vmap(f)(xs)
    return jnp.sum(ws * densities.reshape(len(xs)) * values.reshape(len(xs)))


class TestExponentialFamily:
    """The ``Differentiable`` contract, against independent ground truth."""

    @pytest.mark.parametrize("case", CASES, ids=ids(CASES))
    def test_density_normalizes(self, case: Case) -> None:
        params = jnp.array(case.params)
        total = quadrature_mean(case, params, lambda _: jnp.array(1.0))
        assert jnp.allclose(total, 1.0, atol=case.quad_atol)

    @pytest.mark.parametrize("case", CASES, ids=ids(CASES))
    def test_relative_entropy_matches_quadrature(self, case: Case) -> None:
        """$D(p \\| q) = \\int p \\log(p / q)$, and $D(p \\| p) = 0$."""
        model = case.model
        p, q = jnp.array(case.params), jnp.array(case.other)
        expected = quadrature_mean(
            case, p, lambda x: model.log_density(p, x) - model.log_density(q, x)
        )
        assert jnp.allclose(model.relative_entropy(p, q), expected, atol=case.quad_atol)
        assert jnp.allclose(model.relative_entropy(p, p), 0.0, atol=ATOL)

    @pytest.mark.parametrize("case", CASES, ids=ids(CASES))
    def test_sample_statistics_match_mean_parameters(self, case: Case) -> None:
        model = case.model
        params = jnp.array(case.params)
        samples = model.sample(jax.random.PRNGKey(1), params, N_SAMPLES)
        assert samples.shape == (N_SAMPLES, model.data_dim)
        assert case.support(samples)
        assert jnp.allclose(
            model.average_sufficient_statistic(samples),
            model.to_mean(params),
            atol=case.sample_atol,
        )

    @pytest.mark.parametrize("case", ANALYTIC, ids=ids(ANALYTIC))
    def test_natural_mean_round_trip(self, case: Case) -> None:
        model = case.model
        assert isinstance(model, Analytic)
        params = jnp.array(case.params)
        recovered = model.to_natural(model.to_mean(params))
        assert jnp.allclose(recovered, params, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("case", ANALYTIC, ids=ids(ANALYTIC))
    def test_negative_entropy_is_legendre_dual(self, case: Case) -> None:
        """The closed-form $\\phi(\\eta)$ equals $\\eta \\cdot \\theta - \\psi(\\theta)$ at $\\eta = \\nabla \\psi(\\theta)$."""
        model = case.model
        assert isinstance(model, Analytic)
        params = jnp.array(case.params)
        means = model.to_mean(params)
        expected = jnp.dot(means, params) - model.log_partition_function(params)
        assert jnp.allclose(
            model.negative_entropy(means), expected, rtol=RTOL, atol=ATOL
        )


class TestParameterValidity:
    """Families that override ``initialize`` and ``check_natural_parameters`` agree with each other, and reject a point outside the domain."""

    @pytest.mark.parametrize(
        ("model", "invalid"),
        [
            (CoMPoisson(), (1.0, 0.5)),  # $\nu < 0$
            (Dirichlet(3), (2.0, -1.0, 1.0)),  # $\alpha_2 < 0$
            (Normal(1, PositiveDefinite()), (0.0, 0.5)),  # negative precision
        ],
        ids=["com_poisson", "dirichlet", "normal"],
    )
    def test_initialize_is_valid(
        self, model: Differentiable, invalid: tuple[float, ...]
    ) -> None:
        for i in range(10):
            params = model.initialize(jax.random.PRNGKey(i))
            assert model.check_natural_parameters(params)
        assert not model.check_natural_parameters(jnp.array(invalid))


class TestProduct:
    """A product family is the independent product of its replicates, stored replicate by replicate."""

    @pytest.mark.parametrize(
        ("model", "params"),
        [
            (Bernoullis(3), (0.8, -0.3, 1.5)),
            (Binomials(3, 4), (0.5, -0.5, 1.0)),
            (Poissons(3), (0.5, 1.0, 1.5)),
            (VonMisesProduct(2), (1.4, 1.4, -0.6, 0.2)),
        ],
        ids=["bernoullis", "binomials", "poissons", "von_mises_product"],
    )
    def test_factorizes(
        self, model: DifferentiableProduct[Differentiable], params: tuple[float, ...]
    ) -> None:
        theta = jnp.array(params)
        rep, n = model.rep_man, model.n_reps
        rep_params = theta.reshape(n, rep.dim)
        x = model.sample(jax.random.PRNGKey(0), theta, 1)[0]
        rep_xs = x.reshape(n, rep.data_dim)

        psi = sum(rep.log_partition_function(rep_params[i]) for i in range(n))
        log_p = sum(rep.log_density(rep_params[i], rep_xs[i]) for i in range(n))
        assert jnp.allclose(
            model.log_partition_function(theta), psi, rtol=RTOL, atol=ATOL
        )
        assert jnp.allclose(model.log_density(theta, x), log_p, rtol=RTOL, atol=ATOL)


class TestBinomial:
    @pytest.mark.parametrize("n_trials", [1, 5, 10])
    def test_log_partition_is_n_softplus(self, n_trials: int) -> None:
        model = Binomial(n_trials)
        for theta in [-2.0, 0.0, 2.0]:
            params = jnp.array([theta])
            expected = n_trials * np.logaddexp(0.0, theta)
            assert jnp.allclose(
                model.log_partition_function(params), expected, rtol=RTOL, atol=ATOL
            )

    def test_one_trial_is_bernoulli(self) -> None:
        binomial, bernoulli = Binomial(1), Bernoulli()
        for theta in [-2.0, 0.0, 2.0]:
            params = jnp.array([theta])
            for x in [0.0, 1.0]:
                assert jnp.allclose(
                    binomial.log_density(params, jnp.array([x])),
                    bernoulli.log_density(params, jnp.array([x])),
                    rtol=RTOL,
                    atol=ATOL,
                )


class TestCategorical:
    def test_statistic_is_one_hot_with_category_zero_as_reference(self) -> None:
        model = Categorical(5)
        stats_ = jax.vmap(model.sufficient_statistic)(jnp.arange(5)[:, None])
        expected = jnp.concatenate([jnp.zeros((1, 4)), jnp.eye(4)])
        assert jnp.allclose(stats_, expected)

    @pytest.mark.parametrize("n", [2, 5])
    def test_uniform_negative_entropy_is_minus_log_n(self, n: int) -> None:
        model = Categorical(n)
        means = model.to_mean(jnp.zeros(n - 1))
        assert jnp.allclose(
            model.negative_entropy(means), -np.log(n), rtol=RTOL, atol=ATOL
        )


class TestCoMPoisson:
    def test_statistic_is_count_and_log_factorial(self) -> None:
        model = CoMPoisson()
        for x in [0, 1, 5, 20]:
            expected = jnp.array([x, special.gammaln(x + 1)])
            assert jnp.allclose(
                model.sufficient_statistic(jnp.array([float(x)])),
                expected,
                rtol=RTOL,
                atol=ATOL,
            )

    @pytest.mark.parametrize("rate", [0.5, 3.0, 20.0])
    def test_unit_dispersion_is_poisson(self, rate: float) -> None:
        """At $\\theta_2 = -1$ the density is Poisson with rate $e^{\\theta_1}$, and so is the mean."""
        model = CoMPoisson()
        params = jnp.array([math.log(rate), -1.0])
        xs = jnp.arange(60.0)[:, None]
        densities = jax.vmap(model.density, in_axes=(None, 0))(params, xs)
        expected = stats.poisson.pmf(np.arange(60), rate)
        assert jnp.allclose(densities.ravel(), expected, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(model.to_mean(params)[0], rate, rtol=RTOL, atol=ATOL)


class TestVonMises:
    @pytest.mark.parametrize("kappa", [0.0, 0.5, 2.0, 10.0])
    def test_density_matches_bessel_normalizer(self, kappa: float) -> None:
        """$p(x) = e^{\\kappa \\cos(x - \\mu)} / (2 \\pi I_0(\\kappa))$; uniform $1 / 2\\pi$ at $\\kappa = 0$."""
        model = VonMises()
        mu = 1.0
        params = model.join_mean_concentration(mu, kappa)
        xs = np.linspace(-np.pi, np.pi, 50)
        densities = jax.vmap(model.density, in_axes=(None, 0))(params, jnp.asarray(xs))
        expected = np.exp(kappa * np.cos(xs - mu)) / (2 * np.pi * special.i0(kappa))
        assert jnp.allclose(densities, expected, rtol=RTOL, atol=ATOL)

    def test_mode_is_mean_direction(self) -> None:
        model = VonMises()
        mu = jnp.pi / 3
        params = model.join_mean_concentration(mu, 3.0)
        xs = jnp.linspace(-jnp.pi, jnp.pi, 3601)
        densities = jax.vmap(model.density, in_axes=(None, 0))(params, xs)
        assert jnp.allclose(xs[jnp.argmax(densities)], mu, atol=2 * jnp.pi / 3600)

    def test_mean_concentration_chart_round_trip(self) -> None:
        model = VonMises()
        for mu, kappa in [(0.0, 1.0), (jnp.pi / 4, 2.0), (-2.5, 0.3), (3.0, 10.0)]:
            mu_hat, kappa_hat = model.split_mean_concentration(
                model.join_mean_concentration(mu, kappa)
            )
            assert jnp.allclose(mu_hat, mu, rtol=RTOL, atol=ATOL)
            assert jnp.allclose(kappa_hat, kappa, rtol=RTOL, atol=ATOL)


class TestDirichlet:
    def test_log_partition_is_log_beta(self) -> None:
        alpha = np.array([3.0, 7.0, 5.0])
        expected = special.gammaln(alpha).sum() - special.gammaln(alpha.sum())
        assert jnp.allclose(
            Dirichlet(3).log_partition_function(jnp.asarray(alpha)),
            expected,
            rtol=RTOL,
            atol=ATOL,
        )

    def test_mean_is_digamma_difference(self) -> None:
        """$\\mathbb E[\\log x_i] = \\digamma(\\alpha_i) - \\digamma(\\sum_j \\alpha_j)$."""
        alpha = np.array([3.0, 7.0, 5.0])
        expected = special.digamma(alpha) - special.digamma(alpha.sum())
        assert jnp.allclose(
            Dirichlet(3).to_mean(jnp.asarray(alpha)), expected, rtol=RTOL, atol=ATOL
        )
