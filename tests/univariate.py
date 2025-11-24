"""Tests for univariate exponential family models.

Tests cover:
1. Base tests for all Differentiable models:
   - Density integration
   - Sufficient statistic convergence

2. Additional tests for Analytic models:
   - Parameter recovery through to_mean/to_natural cycle
   - Relative entropy with itself (should be zero)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import override

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import Analytic, Differentiable, Natural, Point, PositiveDefinite
from goal.models import (
    Categorical,
    CoMPoisson,
    Normal,
    Poisson,
    VonMises,
)

# Configure logging
logger = logging.getLogger(__name__)

# Test parameters
N_TRIALS = 10  # Number of trials for statistical tests
SAMPLE_SIZES = [10, 100, 1000]  # Sample sizes for convergence tests
SEED = 0  # Base random seed


@dataclass
class UnivariateStats:
    """Statistics for testing exponential family models."""

    model_name: str
    total_mass: list[float]  # Should be close to 1 for each trial
    mse_progression: list[list[float]]  # Should decrease within each trial
    param_recovery_error: list[float] | None  # Only for Analytic models
    relative_entropy_error: list[float] | None = (
        None  # Self-relative entropy should be 0
    )


### Tests of differentiable models ###


@dataclass
class DifferentiableUnivariateTest[M: Differentiable](ABC):
    """Base test harness defining the testing protocol."""

    model: M
    rng_seed: int = SEED
    n_trials: int = N_TRIALS
    sample_sizes: list[int] = field(default_factory=lambda: SAMPLE_SIZES)

    @abstractmethod
    def run_integration_test(self, params: Point[Natural, M]) -> Array:
        """Run numeric integration test for this model type."""
        ...

    def run_sufficient_stats_test(
        self, params: Point[Natural, M], samples: Array, n_samples: int
    ) -> float:
        """Test convergence of sufficient statistics."""
        true_mean_params = self.model.to_mean(params)
        batch = samples[:n_samples]
        est_mean_params = self.model.average_sufficient_statistic(batch)
        return float(jnp.mean((est_mean_params.array - true_mean_params.array) ** 2))

    def run_parameter_recovery_test(self, params: Point[Natural, M]) -> float | None:
        """Test parameter recovery through mean coordinates."""
        params = params
        return None

    def run_trial(self, trial_idx: int) -> tuple[float, list[float], float | None]:
        """Run a single complete trial."""
        key = jax.random.PRNGKey(self.rng_seed + trial_idx)
        keys = jax.random.split(key, 3)

        # Initialize
        params = self.model.initialize(keys[0])
        valid = bool(self.model.check_natural_parameters(params))

        logger.info(
            "Natural parameter check for %s: %s", type(self.model).__name__, valid
        )
        assert valid, (
            f"Invalid natural parameters after initialization for {type(self.model).__name__}"
        )

        # Integration test
        total_mass = self.run_integration_test(params).item()

        # Sufficient statistics test
        max_samples = max(self.sample_sizes)
        samples = self.model.sample(keys[1], params, max_samples)
        mse_progression = [
            self.run_sufficient_stats_test(params, samples, n)
            for n in self.sample_sizes
        ]

        # Parameter recovery test
        param_recovery_error = self.run_parameter_recovery_test(params)

        return total_mass, mse_progression, param_recovery_error

    def run_all_trials(self) -> UnivariateStats:
        """Run all trials and collect statistics."""
        results = [self.run_trial(i) for i in range(self.n_trials)]
        total_masses, mse_progs, recovery_errors = zip(*results)

        return UnivariateStats(
            model_name=type(self.model).__name__,
            total_mass=list(total_masses),
            mse_progression=list(mse_progs),
            param_recovery_error=(
                list(recovery_errors) if isinstance(self.model, Analytic) else None
            ),
        )


@dataclass
class CoMPoissonTest[M: CoMPoisson](DifferentiableUnivariateTest[M]):
    """Test harness for count-based distributions."""

    @override
    def run_integration_test(self, params: Point[Natural, M]) -> Array:
        """Compute total probability mass for count-based distributions."""
        max_k = 500  # Could be made adaptive based on parameters
        xs = jnp.arange(max_k)
        densities = jax.vmap(self.model.density, in_axes=(None, 0))(params, xs)
        return jnp.sum(densities)


@dataclass
class VonMisesTest[M: VonMises](DifferentiableUnivariateTest[M]):
    """Test harness for VonMises distribution."""

    @override
    def run_integration_test(self, params: Point[Natural, M]) -> Array:
        """Compute total probability mass for von Mises distribution."""
        xs = jnp.linspace(-jnp.pi, jnp.pi, 1000)
        densities = jax.vmap(self.model.density, in_axes=(None, 0))(params, xs)
        return jnp.sum(densities) * (2 * jnp.pi / 1000)


### Tests of analytic models ###


@dataclass
class AnalyticUnivariateTest[M: Analytic](DifferentiableUnivariateTest[M], ABC):
    """Base test harness defining the testing protocol."""

    @override
    def run_parameter_recovery_test(self, params: Point[Natural, M]) -> float | None:
        """Test parameter recovery through mean coordinates."""
        mean_params = self.model.to_mean(params)
        recovered_params = self.model.to_natural(mean_params)
        return float(jnp.mean((recovered_params.array - params.array) ** 2))

    def run_relative_entropy_tests(self) -> list[float]:
        """Run relative entropy tests across multiple trials."""
        results: list[float] = []
        for i in range(self.n_trials):
            key = jax.random.PRNGKey(self.rng_seed + i)
            params = self.model.initialize(key)
            mean_params = self.model.to_mean(params)
            re = self.model.relative_entropy(mean_params, params)
            results.append(float(re))
        return results

    @override
    def run_all_trials(self) -> UnivariateStats:
        """Run all trials and collect statistics."""
        base_stats = super().run_all_trials()
        relative_entropy_error = self.run_relative_entropy_tests()
        return replace(base_stats, relative_entropy_error=relative_entropy_error)


@dataclass
class CategoricalTest[M: Categorical](AnalyticUnivariateTest[M]):
    """Test harness for Categorical distribution."""

    @override
    def run_integration_test(self, params: Point[Natural, M]) -> Array:
        """Compute total probability mass for categorical distribution."""
        xs = jnp.arange(self.model.n_categories)
        densities = jax.vmap(self.model.density, in_axes=(None, 0))(params, xs)
        return jnp.sum(densities)


@dataclass
class PoissonTest[M: Poisson](AnalyticUnivariateTest[M]):
    """Test harness for count-based distributions."""

    @override
    def run_integration_test(self, params: Point[Natural, M]) -> Array:
        """Compute total probability mass for count-based distributions."""
        max_k = 500  # Could be made adaptive based on parameters
        xs = jnp.arange(max_k)
        densities = jax.vmap(self.model.density, in_axes=(None, 0))(params, xs)
        return jnp.sum(densities)


@dataclass
class NormalTest[M: Normal](AnalyticUnivariateTest[M]):
    """Test harness for Normal distribution."""

    @override
    def run_integration_test(self, params: Point[Natural, M]) -> Array:
        """Compute total probability mass for normal distribution."""
        mean, var = self.model.split_mean_covariance(self.model.to_mean(params))
        std = jnp.sqrt(var.array)
        xs = jnp.linspace(mean.array - 6 * std, mean.array + 6 * std, 1000)
        densities = jax.vmap(self.model.density, in_axes=(None, 0))(params, xs)
        return jnp.sum(densities) * (12 * std / 1000)


### Test main ###


def analyze_stats(stats: UnivariateStats) -> None:
    """Analyze and log model test results."""
    logger.info(f"\nResults for {stats.model_name}:")

    # Analyze density integration
    mean_mass = jnp.mean(jnp.array(stats.total_mass))
    std_mass = jnp.std(jnp.array(stats.total_mass))
    logger.info(f"Total probability mass: {mean_mass:.4f} ± {std_mass:.4f}")

    # Analyze MSE progression
    mse_array = jnp.array(stats.mse_progression)
    mean_mses = jnp.mean(mse_array, axis=0)
    std_mses = jnp.std(mse_array, axis=0)

    logger.info("MSE progression:")
    for n, mean_mse, std_mse in zip(SAMPLE_SIZES, mean_mses, std_mses):
        logger.info(f"  n={n}: {mean_mse:.4e} ± {std_mse:.4e}")

    # Check if MSE consistently decreases
    decreases = [
        all(trial_mses[i] > trial_mses[i + 1] for i in range(len(SAMPLE_SIZES) - 1))
        for trial_mses in stats.mse_progression
    ]
    decrease_rate = sum(decreases) / len(decreases)
    logger.info(f"MSE decreased consistently in {decrease_rate:.1%} of trials")

    # Analyze parameter recovery if available
    if stats.param_recovery_error is not None:
        errors = jnp.array(stats.param_recovery_error)
        mean_error = jnp.mean(errors)
        std_error = jnp.std(errors)
        logger.info(f"Parameter recovery MSE: {mean_error:.4e} ± {std_error:.4e}")

    # Analyze relative entropy if available
    if stats.relative_entropy_error is not None:
        errors = jnp.array(stats.relative_entropy_error)
        mean_error = jnp.mean(errors)
        std_error = jnp.std(errors)
        logger.info(f"Self relative entropy: {mean_error:.4e} ± {std_error:.4e}")
        # Check for non zero
        assert jnp.all(errors >= -1e-6), (
            f"{stats.model_name} has negative relative entropy values: {errors}"
        )


### Test assertions ###


def assert_univariate_stats(stats: UnivariateStats) -> None:
    """Assert statistical properties of univariate model."""
    # Probability mass should integrate to 1
    assert jnp.allclose(jnp.mean(jnp.array(stats.total_mass)), 1.0, rtol=1e-2), (
        f"{stats.model_name} total probability mass deviates significantly from 1"
    )

    # Check parameter bounds

    # MSE should decrease with sample size
    mse_array = jnp.array(stats.mse_progression)
    mean_mses = jnp.mean(mse_array, axis=0)
    assert all(mean_mses[i] > mean_mses[i + 1] for i in range(len(mean_mses) - 1)), (
        f"{stats.model_name} MSE did not consistently decrease"
    )

    # Parameter recovery should be accurate if available
    if stats.param_recovery_error is not None:
        assert jnp.mean(jnp.array(stats.param_recovery_error)) < 1e-5, (
            f"{stats.model_name} parameter recovery error too high"
        )

    # Check relative entropy
    if stats.relative_entropy_error is not None:
        relative_entropies = jnp.array(stats.relative_entropy_error)
        assert jnp.all(relative_entropies >= -1e-6), (
            f"{stats.model_name} has negative relative entropy values: {relative_entropies}"
        )
        assert jnp.mean(jnp.abs(relative_entropies)) < 1e-5, (
            f"{stats.model_name} relative entropy with itself not close to zero"
        )


### Tests ###


def test_categorical(caplog: pytest.LogCaptureFixture) -> None:
    """Test categorical distribution."""
    caplog.set_level(logging.INFO)
    model = Categorical(n_categories=5)
    test = CategoricalTest(model)
    stats = test.run_all_trials()
    analyze_stats(stats)
    assert_univariate_stats(stats)


def test_poisson(caplog: pytest.LogCaptureFixture) -> None:
    """Test Poisson distribution."""
    caplog.set_level(logging.INFO)
    model = Poisson()
    test = PoissonTest(model)
    stats = test.run_all_trials()
    analyze_stats(stats)
    assert_univariate_stats(stats)


def test_normal(caplog: pytest.LogCaptureFixture) -> None:
    """Test normal distribution."""
    caplog.set_level(logging.INFO)
    model = Normal(1, PositiveDefinite())
    test = NormalTest(model)
    stats = test.run_all_trials()
    analyze_stats(stats)
    assert_univariate_stats(stats)


def test_von_mises(caplog: pytest.LogCaptureFixture) -> None:
    """Test von Mises distribution."""
    caplog.set_level(logging.INFO)
    model = VonMises()
    test = VonMisesTest(model)
    stats = test.run_all_trials()
    analyze_stats(stats)
    assert_univariate_stats(stats)


def test_com_poisson(caplog: pytest.LogCaptureFixture) -> None:
    """Test COM-Poisson distribution."""
    caplog.set_level(logging.INFO)
    model = CoMPoisson()
    test = CoMPoissonTest(model)
    stats = test.run_all_trials()
    analyze_stats(stats)
    assert_univariate_stats(stats)


if __name__ == "__main__":
    # This allows running the tests directly with python
    _ = pytest.main([__file__])
