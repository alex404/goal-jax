"""Discrete hidden Markov model and its categorical building blocks.

Provides ``HiddenMarkovModel`` as the discrete state-space model, along with the ``CategoricalKernel``/``CategoricalTransition``/``CategoricalEmission`` harmoniums it composes.
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
    Rectangular,
)
from ...geometry.exponential_family.dynamical import (
    AnalyticLatentProcess,
    AnalyticTransition,
)
from ..base.categorical import Categorical

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
    def ems_hrm(self) -> CategoricalEmission:
        return CategoricalEmission(n_obs=self.n_obs, n_states=self._n_states)

    @property
    @override
    def trn_map(self) -> CategoricalTransition:
        return create_categorical_transition(self._n_states)

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


def create_hidden_markov_model(n_obs: int, n_states: int) -> HiddenMarkovModel:
    """Create a discrete hidden Markov model."""
    return HiddenMarkovModel(n_obs=n_obs, _n_states=n_states)
