"""Tests for models/base/gaussian/normal.py.

Verifies parameter conversions, fitting, log-density vs scipy, representation
consistency, precision/covariance inversion, whitening, and sampling for
Normal distributions across Scale, Diagonal, and PositiveDefinite representations.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jax.scipy import stats

from goal.geometry import Diagonal, PositiveDefinite, Scale
from goal.models import FullNormal, Normal

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-5
ATOL = 1e-7

SAMPLE_MEAN = jnp.array([1.0, -0.5])
SAMPLE_COV = jnp.array([[2.0, 1.0], [1.0, 1.0]])
SAMPLE_SIZE = 1000


@pytest.fixture
def ground_truth() -> tuple[FullNormal, Array]:
    """Ground truth normal model and mean parameters."""
    model = Normal(SAMPLE_MEAN.shape[0], PositiveDefinite())
    cov_params = model.cov_man.from_matrix(SAMPLE_COV)
    means = model.join_mean_covariance(SAMPLE_MEAN, cov_params)
    return model, means


@pytest.fixture
def samples(ground_truth: tuple[FullNormal, Array]) -> Array:
    """Sample data from ground truth distribution."""
    model, means = ground_truth
    natural_params = model.to_natural(means)
    return model.sample(jax.random.PRNGKey(0), natural_params, SAMPLE_SIZE)


class TestNormalFitting:
    """Test fitting and log-density evaluation across representations."""

    @pytest.mark.parametrize("rep", [Scale(), Diagonal(), PositiveDefinite()])
    def test_fit_and_evaluate(
        self, rep: Scale | Diagonal | PositiveDefinite, samples: Array
    ) -> None:
        """Fit model, verify round-trip and log-density matches scipy."""
        model = Normal(SAMPLE_MEAN.shape[0], rep)

        means = model.average_sufficient_statistic(samples)
        params = model.to_natural(means)

        assert model.check_natural_parameters(params)
        recovered = model.to_mean(params)
        assert jnp.allclose(recovered, means, rtol=RTOL, atol=ATOL)

        log_density = model.average_log_density(params, samples)
        mean, cov = model.split_mean_covariance(means)
        scipy_ll = jnp.mean(
            jax.vmap(stats.multivariate_normal.logpdf, in_axes=(0, None, None))(
                samples, mean, model.cov_man.to_matrix(cov)
            )
        )
        assert jnp.allclose(log_density, scipy_ll, rtol=RTOL, atol=ATOL)


class TestNormalRepresentations:
    """Test embedded representations give same log-density."""

    def test_representation_consistency(self, samples: Array) -> None:
        """Isotropic embedded into Diagonal and PositiveDefinite gives same results."""
        dim = SAMPLE_MEAN.shape[0]
        iso = Normal(dim, Scale())
        dia = Normal(dim, Diagonal())
        pod = Normal(dim, PositiveDefinite())

        iso_means = iso.average_sufficient_statistic(samples)
        iso_natural = iso.to_natural(iso_means)
        dia_natural = iso.embed_rep(dia, iso_natural)
        pod_natural = iso.embed_rep(pod, iso_natural)

        iso_lpf = iso.log_partition_function(iso_natural)
        dia_lpf = dia.log_partition_function(dia_natural)
        pod_lpf = pod.log_partition_function(pod_natural)
        assert jnp.allclose(iso_lpf, dia_lpf, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(dia_lpf, pod_lpf, rtol=RTOL, atol=ATOL)

        iso_ll = iso.average_log_density(iso_natural, samples)
        dia_ll = dia.average_log_density(dia_natural, samples)
        pod_ll = pod.average_log_density(pod_natural, samples)
        assert jnp.allclose(iso_ll, dia_ll, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(dia_ll, pod_ll, rtol=RTOL, atol=ATOL)


class TestNormalPrecision:
    """Test precision matrix is covariance inverse."""

    def test_precision_is_covariance_inverse(
        self, ground_truth: tuple[FullNormal, Array]
    ) -> None:
        model, means = ground_truth
        natural_params = model.to_natural(means)

        _, precision = model.split_location_precision(natural_params)
        _, covariance = model.split_mean_covariance(means)

        prec_dense = model.cov_man.to_matrix(precision)
        cov_dense = model.cov_man.to_matrix(covariance)

        assert jnp.allclose(prec_dense, jnp.linalg.inv(cov_dense), rtol=RTOL, atol=ATOL)
        assert jnp.allclose(
            prec_dense @ cov_dense, jnp.eye(model.data_dim), rtol=RTOL, atol=ATOL
        )


class TestNormalWhitening:
    """Test whitening transformation."""

    def test_whiten_self(self, ground_truth: tuple[FullNormal, Array]) -> None:
        """Whitening relative to itself gives standard normal."""
        model, means = ground_truth
        gt_mean, _ = model.split_mean_covariance(means)

        whitened = model.relative_whiten(means, means)
        w_mean, w_cov = model.split_mean_covariance(whitened)

        assert jnp.allclose(w_mean, jnp.zeros_like(gt_mean), rtol=RTOL, atol=ATOL)
        assert jnp.allclose(
            model.cov_man.to_matrix(w_cov), jnp.eye(model.data_dim), rtol=RTOL, atol=ATOL
        )

    def test_whiten_different(self, ground_truth: tuple[FullNormal, Array]) -> None:
        """Whitening a different distribution matches manual Cholesky computation."""
        model, means = ground_truth
        gt_mean, gt_cov = model.split_mean_covariance(means)
        gt_cov_mat = model.cov_man.to_matrix(gt_cov)

        test_mean = jnp.array([3.0, -2.0])
        test_cov_mat = jnp.array([[1.5, 0.3], [0.3, 2.0]])
        test_params = model.join_mean_covariance(
            test_mean, model.cov_man.from_matrix(test_cov_mat)
        )

        whitened_test = model.relative_whiten(test_params, means)
        wt_mean, wt_cov = model.split_mean_covariance(whitened_test)

        chol = jnp.linalg.cholesky(gt_cov_mat)
        expected_mean = jax.scipy.linalg.solve_triangular(
            chol, test_mean - gt_mean, lower=True
        )
        temp = jax.scipy.linalg.solve_triangular(chol, test_cov_mat, lower=True)
        expected_cov = jax.scipy.linalg.solve_triangular(chol, temp.T, lower=True).T

        assert jnp.allclose(wt_mean, expected_mean, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(
            model.cov_man.to_matrix(wt_cov), expected_cov, rtol=RTOL, atol=ATOL
        )


class TestNormalNegativeEntropy:
    """Test negative entropy computation."""

    def test_legendre_identity(self) -> None:
        """negative_entropy = -psi + dot(theta, eta) for FullNormal."""
        model = Normal(2, PositiveDefinite())
        cov_params = model.cov_man.from_matrix(jnp.array([[2.0, 0.5], [0.5, 1.0]]))
        means = model.join_mean_covariance(jnp.array([1.0, -0.5]), cov_params)
        params = model.to_natural(means)
        ne = model.negative_entropy(means)
        expected = -model.log_partition_function(params) + jnp.dot(params, means)
        assert jnp.allclose(ne, expected, rtol=RTOL, atol=ATOL)

    def test_known_formula(self) -> None:
        """EF negative entropy: -0.5 * (log|Sigma| + d)."""
        model = Normal(2, PositiveDefinite())
        cov_mat = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        cov_params = model.cov_man.from_matrix(cov_mat)
        means = model.join_mean_covariance(jnp.array([1.0, -0.5]), cov_params)
        ne = model.negative_entropy(means)
        expected = -0.5 * (jnp.log(jnp.linalg.det(cov_mat)) + 2)
        assert jnp.allclose(ne, expected, rtol=RTOL, atol=ATOL)


class TestNormalSampling:
    """Test sampling produces correct statistics."""

    @pytest.mark.parametrize("rep", [Scale(), Diagonal(), PositiveDefinite()])
    def test_sampling_statistics(self, rep: Scale | Diagonal | PositiveDefinite) -> None:
        """Empirical sufficient statistics match to_mean at 50k samples."""
        dim = 2
        model = Normal(dim, rep)

        if isinstance(rep, Scale):
            cov_params = jnp.array([1.5])
        elif isinstance(rep, Diagonal):
            cov_params = jnp.array([1.5, 2.0])
        else:
            cov_params = model.cov_man.from_matrix(jnp.array([[1.5, 0.3], [0.3, 2.0]]))

        mean = jnp.array([1.0, -0.5])
        means = model.join_mean_covariance(mean, cov_params)
        params = model.to_natural(means)

        samples = model.sample(jax.random.PRNGKey(42), params, 50_000)
        assert samples.shape == (50_000, dim)

        # Data-space mean: tight tolerance (SE ~ sqrt(2/50000) ~ 0.006)
        assert jnp.allclose(jnp.mean(samples, axis=0), mean, atol=0.03)

        # Full sufficient statistics including second moments (higher variance)
        empirical_means = model.average_sufficient_statistic(samples)
        theoretical_means = model.to_mean(params)
        assert jnp.allclose(empirical_means, theoretical_means, atol=0.1)
