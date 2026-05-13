"""Kalman filter and its linear Gaussian transition.

Provides ``KalmanFilter`` as the canonical linear Gaussian state-space model, along with the ``LinearGaussianTransition`` and ``NormalEmission`` building blocks it composes.
"""

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import PositiveDefinite
from ...geometry.exponential_family.dynamical import (
    AnalyticLatentProcess,
    AnalyticTransition,
)
from ..base.gaussian.normal import FullNormal, full_normal
from ..harmonium.lgm import NormalAnalyticLGM

# =============================================================================
# Type aliases
# =============================================================================

NormalEmission = NormalAnalyticLGM[PositiveDefinite]
"""Linear Gaussian emission $p(x_t \\mid z_t) = N(C z_t, R)$ for a Kalman filter. ``obs_dim`` and ``lat_dim`` may differ."""


# =============================================================================
# Linear Gaussian Transition
# =============================================================================


@dataclass(frozen=True)
class LinearGaussianTransition(AnalyticTransition[FullNormal]):
    """Linear Gaussian transition $p(z_t \\mid z_{t-1})$ on a single ``FullNormal`` state.

    Composes a ``NormalAnalyticLGM[PositiveDefinite]`` as its kernel (when ``obs_dim == lat_dim``, the LGM's observable side ``Normal[PositiveDefinite]`` and latent side ``FullNormal`` are the same manifold). ``AnalyticTransition.__call__`` derives the predict map from the kernel.
    """

    kernel: NormalAnalyticLGM[PositiveDefinite]


def create_linear_gaussian_transition(lat_dim: int) -> LinearGaussianTransition:
    """Create a linear Gaussian transition for a state of dimension ``lat_dim``."""
    kernel = NormalAnalyticLGM(
        obs_dim=lat_dim, obs_rep=PositiveDefinite(), lat_dim=lat_dim
    )
    return LinearGaussianTransition(kernel=kernel)


# =============================================================================
# Kalman Filter
# =============================================================================


@dataclass(frozen=True)
class KalmanFilter(AnalyticLatentProcess[FullNormal, FullNormal]):
    """Linear Gaussian state-space model.

    The model is

    .. math::
        z_0 \\sim N(\\mu_0, \\Sigma_0), \\quad
        z_t = A z_{t-1} + w_t, \\, w_t \\sim N(0, Q), \\quad
        x_t = C z_t + v_t, \\, v_t \\sim N(0, R)

    Composed of a ``FullNormal`` prior over $z_0$, a ``NormalEmission`` for $p(x_t \\mid z_t)$, and a ``LinearGaussianTransition`` for $p(z_t \\mid z_{t-1})$.
    """

    obs_dim: int
    """Dimension of observations."""

    _lat_dim: int
    """Dimension of latent state."""

    @property
    def lat_dim(self) -> int:
        return self._lat_dim

    @property
    @override
    def lat_man(self) -> FullNormal:
        return full_normal(self._lat_dim)

    @property
    @override
    def ems_hrm(self) -> NormalEmission:
        return NormalAnalyticLGM(
            obs_dim=self.obs_dim, obs_rep=PositiveDefinite(), lat_dim=self._lat_dim
        )

    @property
    @override
    def trn_map(self) -> LinearGaussianTransition:
        return create_linear_gaussian_transition(self._lat_dim)

    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize prior, emission, and transition parameters with small random noise."""
        keys = jax.random.split(key, 3)
        prior_params = self.lat_man.initialize(keys[0], location, shape)
        ems_full = self.ems_hrm.initialize(keys[1], location, shape)
        ems_params = self.ems_hrm.likelihood_function(ems_full)
        trns_params = self.trn_map.initialize(keys[2], location, shape)
        return self.join_coords(prior_params, ems_params, trns_params)

    def from_standard(
        self,
        transition_matrix: Array,
        process_noise: Array,
        emission_matrix: Array,
        observation_noise: Array,
        prior_mean: Array,
        prior_covariance: Array,
    ) -> Array:
        """Construct natural parameters from the standard LDS parameterization $(A, Q, C, R, \\mu_0, \\Sigma_0)$.

        Builds the emission and transition harmoniums via the precision-weighted-interaction encoding, then composes them with the prior into a ``KalmanFilter`` Triple.
        """

        def _build_likelihood(
            hrm: NormalAnalyticLGM[PositiveDefinite],
            weight_matrix: Array,
            noise_covariance: Array,
        ) -> Array:
            """Build the harmonium likelihood (obs bias + interaction) from $(C, R)$."""
            om = hrm.obs_man
            zero_mean = jnp.zeros(om.data_dim)
            cov_coords = om.cov_man.from_matrix(noise_covariance)
            obs_params = om.to_natural(om.join_mean_covariance(zero_mean, cov_coords))
            obs_prec = om.split_location_precision(obs_params)[1]
            dns_prec = om.cov_man.to_matrix(obs_prec)

            int_mat = hrm.int_man.from_matrix(dns_prec @ weight_matrix)
            return hrm.lkl_fun_man.join_coords(obs_params, int_mat)

        ems_params = _build_likelihood(
            self.ems_hrm, emission_matrix, observation_noise
        )
        trns_params = _build_likelihood(
            self.trn_map.kernel, transition_matrix, process_noise
        )

        lm = self.lat_man
        prior_cov_coords = lm.cov_man.from_matrix(prior_covariance)
        prior_params = lm.to_natural(
            lm.join_mean_covariance(prior_mean, prior_cov_coords)
        )
        return self.join_coords(prior_params, ems_params, trns_params)


def create_kalman_filter(obs_dim: int, lat_dim: int) -> KalmanFilter:
    """Create a Kalman filter model."""
    return KalmanFilter(obs_dim=obs_dim, _lat_dim=lat_dim)
