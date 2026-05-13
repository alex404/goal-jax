"""Dynamical model leaves: typed wrappers for state-space models.

This module exports:

- ``LinearGaussianTransition`` --- analytic Gaussian transition, smoothing-ready.
- ``CategoricalKernel`` / ``CategoricalTransition`` --- the Cat-Cat conjugated harmonium and its analytic transition for HMMs.
- ``CategoricalEmission`` / ``NormalEmission`` --- emission harmoniums (kernels only; not transitions).
- ``KalmanFilter`` --- linear Gaussian state-space model.
- ``HiddenMarkovModel`` --- discrete state-space model.

For an MLP-based hybrid filter, plug a ``MultilayerPerceptron[L, L]`` directly into a ``LatentProcess`` --- it satisfies the ``Transition[L]`` alias (``Map[L, L]``) without further wrapping.
"""

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import (
    AnalyticConjugated,
    EmbeddedMap,
    IdentityEmbedding,
    PositiveDefinite,
    Rectangular,
)
from ...geometry.exponential_family.dynamical import (
    AnalyticLatentProcess,
    AnalyticTransition,
)
from ..base.categorical import Categorical
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
# Categorical Conjugated Harmonium
# =============================================================================


@dataclass(frozen=True)
class _CategoricalConjugated(AnalyticConjugated[Categorical, Categorical]):
    """Conjugated harmonium between two Categoricals, parameterized by a ``LinearMap`` over the categoricals' natural-parameter spaces.

    The interaction matrix encodes log conditional probabilities. Conjugation parameters are $\\rho_k = \\Psi(\\theta_x + W_k) - \\Psi(\\theta_x)$ where $\\Psi$ is the Categorical log-partition. ``CategoricalKernel`` and ``CategoricalEmission`` are thin wrappers that fix the shape (square vs rectangular).
    """

    _obs_man: Categorical
    _lat_man: Categorical

    @property
    @override
    def lat_man(self) -> Categorical:
        return self._lat_man

    @property
    @override
    def obs_man(self) -> Categorical:
        return self._obs_man

    @property
    @override
    def int_man(self) -> EmbeddedMap[Categorical, Categorical]:
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(self._lat_man),
            IdentityEmbedding(self._obs_man),
        )

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        obs_bias, int_mat = self.lkl_fun_man.split_coords(lkl_params)
        psi_base = self.obs_man.log_partition_function(obs_bias)
        int_matrix = int_mat.reshape((self.obs_man.dim, self.lat_man.dim))

        def compute_rho_k(col_k: Array) -> Array:
            return self.obs_man.log_partition_function(obs_bias + col_k) - psi_base

        return jax.vmap(compute_rho_k)(int_matrix.T)

    @override
    def to_natural_likelihood(self, means: Array) -> Array:
        obs_means, int_means, lat_means = self.split_coords(means)
        obs_probs = self.obs_man.to_probs(obs_means)
        lat_probs = self.lat_man.to_probs(lat_means)
        n_obs = self.obs_man.n_categories
        n_states = self.lat_man.n_categories
        int_matrix = int_means.reshape((self.obs_man.dim, self.lat_man.dim))

        joint_probs = jnp.zeros((n_obs, n_states))
        joint_probs = joint_probs.at[1:, 1:].set(int_matrix)
        col0_nonref = obs_probs[1:] - jnp.sum(int_matrix, axis=1)
        joint_probs = joint_probs.at[1:, 0].set(col0_nonref)
        joint_probs = joint_probs.at[0, 1:].set(
            lat_probs[1:] - jnp.sum(int_matrix, axis=0)
        )
        joint_probs = joint_probs.at[0, 0].set(lat_probs[0] - jnp.sum(col0_nonref))

        eps = 1e-10
        cond_probs = joint_probs / (lat_probs[None, :] + eps)

        cond_given_lat0 = cond_probs[:, 0]
        obs_nat = jnp.log(cond_given_lat0[1:] + eps) - jnp.log(cond_given_lat0[0] + eps)

        def compute_int_col(j: int) -> Array:
            cond_j = cond_probs[:, j]
            nat_j = jnp.log(cond_j[1:] + eps) - jnp.log(cond_j[0] + eps)
            return nat_j - obs_nat

        int_nat = jnp.stack([compute_int_col(j) for j in range(1, n_states)], axis=1)

        return self.lkl_fun_man.join_coords(obs_nat, int_nat.reshape(-1))


@dataclass(frozen=True)
class CategoricalKernel(_CategoricalConjugated):
    """Symmetric Cat-Cat conjugated harmonium kernel: $p(z_t \\mid z_{t-1})$ with both states ``Categorical(n_{\\rm states})``."""

    def __init__(self, n_states: int):
        cat = Categorical(n_states)
        super().__init__(_obs_man=cat, _lat_man=cat)

    @property
    def n_states(self) -> int:
        return self._lat_man.n_categories


@dataclass(frozen=True)
class CategoricalTransition(AnalyticTransition[Categorical]):
    """HMM transition: a ``CategoricalKernel`` lifted into a ``Map[Categorical, Categorical]``."""

    kernel: CategoricalKernel


def create_categorical_transition(n_states: int) -> CategoricalTransition:
    """Create a categorical transition with ``n_states`` states."""
    return CategoricalTransition(kernel=CategoricalKernel(n_states=n_states))


@dataclass(frozen=True)
class CategoricalEmission(_CategoricalConjugated):
    """HMM emission harmonium: $p(x_t \\mid z_t)$ with ``Categorical(n_{\\rm obs})`` observations and ``Categorical(n_{\\rm states})`` states.

    Asymmetric: the number of observation categories and latent states can differ. This is a kernel only; not a ``Transition``.
    """

    def __init__(self, n_obs: int, n_states: int):
        super().__init__(_obs_man=Categorical(n_obs), _lat_man=Categorical(n_states))

    @property
    def n_obs(self) -> int:
        return self._obs_man.n_categories

    @property
    def n_states(self) -> int:
        return self._lat_man.n_categories


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
    def emsn_hrm(self) -> NormalEmission:
        return NormalAnalyticLGM(
            obs_dim=self.obs_dim, obs_rep=PositiveDefinite(), lat_dim=self._lat_dim
        )

    @property
    @override
    def transition(self) -> LinearGaussianTransition:
        return create_linear_gaussian_transition(self._lat_dim)

    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize prior, emission, and transition parameters with small random noise."""
        keys = jax.random.split(key, 3)
        prior_params = self.lat_man.initialize(keys[0], location, shape)
        emsn_params = self.emsn_hrm.initialize(keys[1], location, shape)
        trns_params = self.transition.initialize(keys[2], location, shape)
        return self.join_coords(prior_params, emsn_params, trns_params)

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

        emsn_lkl = _build_likelihood(
            self.emsn_hrm, emission_matrix, observation_noise
        )
        # Emission slot still stores a full harmonium; pair the likelihood with a
        # standard-normal prior block.
        z_standard = self.emsn_hrm.lat_man.to_natural(
            self.emsn_hrm.lat_man.standard_normal()
        )
        emsn_params = self.emsn_hrm.join_conjugated(emsn_lkl, z_standard)
        # Transition slot stores only the likelihood.
        trns_params = _build_likelihood(
            self.transition.kernel, transition_matrix, process_noise
        )

        lm = self.lat_man
        prior_cov_coords = lm.cov_man.from_matrix(prior_covariance)
        prior_params = lm.to_natural(
            lm.join_mean_covariance(prior_mean, prior_cov_coords)
        )
        return self.join_coords(prior_params, emsn_params, trns_params)


def create_kalman_filter(obs_dim: int, lat_dim: int) -> KalmanFilter:
    """Create a Kalman filter model."""
    return KalmanFilter(obs_dim=obs_dim, _lat_dim=lat_dim)


# =============================================================================
# Hidden Markov Model
# =============================================================================


@dataclass(frozen=True)
class HiddenMarkovModel(AnalyticLatentProcess[Categorical, Categorical]):
    """Discrete state-space model.

    The model is

    .. math::
        z_0 \\sim {\\rm Cat}(\\pi_0), \\quad
        z_t \\mid z_{t-1} \\sim {\\rm Cat}(A[z_{t-1}, :]), \\quad
        x_t \\mid z_t \\sim {\\rm Cat}(B[z_t, :])

    Composed of a ``Categorical`` prior over $z_0$, a ``CategoricalEmission`` for $p(x_t \\mid z_t)$, and a ``CategoricalTransition`` for $p(z_t \\mid z_{t-1})$.
    """

    n_obs: int
    _n_states: int

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    @override
    def lat_man(self) -> Categorical:
        return Categorical(self._n_states)

    @property
    @override
    def emsn_hrm(self) -> CategoricalEmission:
        return CategoricalEmission(n_obs=self.n_obs, n_states=self._n_states)

    @property
    @override
    def transition(self) -> CategoricalTransition:
        return create_categorical_transition(self._n_states)

    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize prior, emission, and transition parameters with small random noise."""
        keys = jax.random.split(key, 3)
        prior_params = self.lat_man.initialize(keys[0], location, shape)
        emsn_params = self.emsn_hrm.initialize(keys[1], location, shape)
        trns_params = self.transition.initialize(keys[2], location, shape)
        return self.join_coords(prior_params, emsn_params, trns_params)


def create_hidden_markov_model(n_obs: int, n_states: int) -> HiddenMarkovModel:
    """Create a discrete hidden Markov model."""
    return HiddenMarkovModel(n_obs=n_obs, _n_states=n_states)


__all__ = [
    "CategoricalEmission",
    "CategoricalKernel",
    "CategoricalTransition",
    "HiddenMarkovModel",
    "KalmanFilter",
    "LinearGaussianTransition",
    "NormalEmission",
    "create_categorical_transition",
    "create_hidden_markov_model",
    "create_kalman_filter",
    "create_linear_gaussian_transition",
]
