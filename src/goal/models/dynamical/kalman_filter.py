"""Kalman filter: linear Gaussian state-space model.

Composes a ``NormalAnalyticLGM[PositiveDefinite]`` emission with an ``AnalyticTransition[FullNormal]`` whose kernel is a square ``NormalAnalyticLGM[PositiveDefinite]``.
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


@dataclass(frozen=True)
class KalmanFilter(AnalyticLatentProcess[FullNormal, FullNormal]):
    """Linear Gaussian state-space model.

    The model is

    .. math::
        z_0 \\sim N(\\mu_0, \\Sigma_0), \\quad
        z_t = A z_{t-1} + w_t, \\, w_t \\sim N(0, Q), \\quad
        x_t = C z_t + v_t, \\, v_t \\sim N(0, R)
    """

    obs_dim: int
    lat_dim: int

    @property
    @override
    def lat_man(self) -> FullNormal:
        return full_normal(self.lat_dim)

    @property
    @override
    def ems_hrm(self) -> NormalAnalyticLGM[PositiveDefinite]:
        return NormalAnalyticLGM(
            obs_dim=self.obs_dim, obs_rep=PositiveDefinite(), lat_dim=self.lat_dim
        )

    @property
    @override
    def trn_map(self) -> AnalyticTransition[FullNormal]:
        kernel = NormalAnalyticLGM(
            obs_dim=self.lat_dim, obs_rep=PositiveDefinite(), lat_dim=self.lat_dim
        )
        return AnalyticTransition(kernel=kernel)

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
        """Construct natural parameters from the standard LDS parameterization $(A, Q, C, R, \\mu_0, \\Sigma_0)$."""
        lm = self.lat_man
        ems_hrm = self.ems_hrm
        trn_kernel = NormalAnalyticLGM(
            obs_dim=self.lat_dim, obs_rep=PositiveDefinite(), lat_dim=self.lat_dim
        )
        prior_params = lm.to_natural(
            lm.join_mean_covariance(
                prior_mean, lm.cov_man.from_matrix(prior_covariance)
            )
        )
        ems_params = ems_hrm.likelihood_from_loadings(
            emission_matrix,
            jnp.zeros(self.obs_dim),
            ems_hrm.obs_man.cov_man.from_matrix(observation_noise),
        )
        trns_params = trn_kernel.likelihood_from_loadings(
            transition_matrix,
            jnp.zeros(self.lat_dim),
            trn_kernel.obs_man.cov_man.from_matrix(process_noise),
        )
        return self.join_coords(prior_params, ems_params, trns_params)
