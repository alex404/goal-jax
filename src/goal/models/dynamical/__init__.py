"""Type aliases for common dynamical models.

All inference algorithms are generic on LatentProcess - there are no
Kalman-specific or HMM-specific algorithm implementations. KalmanFilter
and HiddenMarkovModel are just type aliases.

This module provides:
- KalmanFilter: Linear Gaussian state-space model

Example usage:
    # Create a Kalman filter
    kf = create_kalman_filter(obs_dim=2, lat_dim=3)
    params = kf.initialize(key)

    # Sample a trajectory
    observations, latents = sample_latent_process(kf, params, key, n_steps=100)

    # Filter and smooth
    filtered, log_lik = conjugated_filtering(kf, params, observations)
    smoothed = conjugated_smoothing(kf, params, observations)

    # EM learning
    params = latent_process_expectation_maximization(kf, params, observations_batch)
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
    LatentProcess,
    conjugated_filtering,
    conjugated_smoothing,
    conjugated_smoothing0,
    latent_process_expectation_maximization,
    latent_process_expectation_step,
    latent_process_expectation_step_batch,
    latent_process_log_density,
    latent_process_log_observable_density,
    sample_latent_process,
)
from ..base.categorical import Categorical
from ..base.gaussian.normal import FullNormal, full_normal
from ..harmonium.lgm import GeneralizedGaussianLocationEmbedding

# =============================================================================
# Kalman Filter Implementation
# =============================================================================


@dataclass(frozen=True)
class NormalTransition(AnalyticConjugated[FullNormal, FullNormal]):
    """Transition harmonium for Kalman filter: p(z_t | z_{t-1}).

    Models linear Gaussian transitions:
        z_t = A * z_{t-1} + w_t,  w_t ~ N(0, Q)

    This is a symmetric conjugated harmonium where both observable (z_{t-1})
    and latent (z_t) are FullNormal distributions.
    """

    lat_dim: int
    """Dimension of the latent state."""

    @property
    @override
    def lat_man(self) -> FullNormal:
        return full_normal(self.lat_dim)

    @property
    @override
    def obs_man(self) -> FullNormal:
        # In the transition model, the "observable" is z_{t-1}
        return full_normal(self.lat_dim)

    @property
    def int_obs_emb(self) -> GeneralizedGaussianLocationEmbedding[FullNormal]:
        return GeneralizedGaussianLocationEmbedding(self.obs_man)

    @property
    def int_lat_emb(self) -> GeneralizedGaussianLocationEmbedding[FullNormal]:
        return GeneralizedGaussianLocationEmbedding(self.lat_man)

    @property
    @override
    def int_man(self) -> EmbeddedMap[FullNormal, FullNormal]:
        return EmbeddedMap(
            Rectangular(),
            self.int_lat_emb,
            self.int_obs_emb,
        )

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute conjugation parameters for Normal-Normal transition.

        Same formula as LGM since both are linear Gaussian.
        """
        obs_cov_man = self.obs_man.cov_man
        obs_bias, int_mat = self.lkl_fun_man.split_coords(lkl_params)
        obs_loc, obs_prec = self.obs_man.split_location_precision(obs_bias)

        # Intermediate computations
        obs_sigma = obs_cov_man.inverse(obs_prec)
        obs_mean = obs_cov_man(obs_sigma, obs_loc)

        with self.int_man as im:
            int_mat_trn = im.transpose(int_mat)
            rho_mean = im.trn_man.rep.matvec(
                im.trn_man.matrix_shape, int_mat_trn, obs_mean
            )

            # Change of basis for covariance contribution
            int_mat_dense = im.to_matrix(int_mat)
            obs_sigma_dense = obs_cov_man.to_matrix(obs_sigma)
            rho_cov_dense = int_mat_dense.T @ obs_sigma_dense @ int_mat_dense
            rho_shape = self.lat_man.cov_man.from_matrix(-rho_cov_dense)

        return self.lat_man.join_location_precision(rho_mean, rho_shape)

    @override
    def to_natural_likelihood(self, means: Array) -> Array:
        """Convert mean parameters to natural likelihood parameters."""
        # Get relevant manifolds
        ocm = self.obs_man.cov_man
        lcm = self.lat_man.cov_man
        im = self.int_man

        # Deconstruct parameters
        obs_means, int_means, lat_means = self.split_coords(means)
        obs_mean, obs_cov = self.obs_man.split_mean_covariance(obs_means)
        lat_mean, lat_cov = self.lat_man.split_mean_covariance(lat_means)

        # Interaction covariance
        int_cov = int_means - im.rep.outer_product(obs_mean, lat_mean)

        # Construct precisions
        lat_prs = lcm.inverse(lat_cov)

        # Change of basis
        int_cov_dense = im.to_matrix(int_cov)
        lat_prs_dense = lcm.to_matrix(lat_prs)
        cob_dense = int_cov_dense @ lat_prs_dense @ int_cov_dense.T
        shaped_cob = ocm.from_matrix(cob_dense)

        obs_prs = ocm.inverse(obs_cov - shaped_cob)

        # Compute interaction in natural coords
        obs_prs_dense = ocm.to_matrix(obs_prs)
        int_params_dense = -obs_prs_dense @ im.to_matrix(int_cov) @ lat_prs_dense
        int_params = im.from_matrix(int_params_dense)

        # Compute observable location
        obs_loc0 = ocm(obs_prs, obs_mean)
        obs_loc1 = im.rep.matvec(im.matrix_shape, int_params, lat_mean)
        obs_loc = obs_loc0 - obs_loc1

        obs_params = self.obs_man.join_location_precision(obs_loc, obs_prs)
        return self.lkl_fun_man.join_coords(obs_params, int_params)


@dataclass(frozen=True)
class NormalEmission(AnalyticConjugated[FullNormal, FullNormal]):
    """Emission harmonium for Kalman filter: p(x_t | z_t).

    Models linear Gaussian emissions:
        x_t = C * z_t + v_t,  v_t ~ N(0, R)

    This is essentially an LGM but both observable and latent use full covariance.
    """

    obs_dim: int
    """Dimension of observations."""

    _lat_dim: int
    """Dimension of latent state."""

    @property
    @override
    def lat_man(self) -> FullNormal:
        return full_normal(self._lat_dim)

    @property
    @override
    def obs_man(self) -> FullNormal:
        return full_normal(self.obs_dim)

    @property
    def int_obs_emb(self) -> GeneralizedGaussianLocationEmbedding[FullNormal]:
        return GeneralizedGaussianLocationEmbedding(self.obs_man)

    @property
    def int_lat_emb(self) -> GeneralizedGaussianLocationEmbedding[FullNormal]:
        return GeneralizedGaussianLocationEmbedding(self.lat_man)

    @property
    @override
    def int_man(self) -> EmbeddedMap[FullNormal, FullNormal]:
        return EmbeddedMap(
            Rectangular(),
            self.int_lat_emb,
            self.int_obs_emb,
        )

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute conjugation parameters for Normal emission."""
        obs_cov_man = self.obs_man.cov_man
        obs_bias, int_mat = self.lkl_fun_man.split_coords(lkl_params)
        obs_loc, obs_prec = self.obs_man.split_location_precision(obs_bias)

        obs_sigma = obs_cov_man.inverse(obs_prec)
        obs_mean = obs_cov_man(obs_sigma, obs_loc)

        with self.int_man as im:
            int_mat_trn = im.transpose(int_mat)
            rho_mean = im.trn_man.rep.matvec(
                im.trn_man.matrix_shape, int_mat_trn, obs_mean
            )

            int_mat_dense = im.to_matrix(int_mat)
            obs_sigma_dense = obs_cov_man.to_matrix(obs_sigma)
            rho_cov_dense = int_mat_dense.T @ obs_sigma_dense @ int_mat_dense
            rho_shape = self.lat_man.cov_man.from_matrix(-rho_cov_dense)

        return self.lat_man.join_location_precision(rho_mean, rho_shape)

    @override
    def to_natural_likelihood(self, means: Array) -> Array:
        """Convert mean parameters to natural likelihood parameters."""
        ocm = self.obs_man.cov_man
        lcm = self.lat_man.cov_man
        im = self.int_man

        obs_means, int_means, lat_means = self.split_coords(means)
        obs_mean, obs_cov = self.obs_man.split_mean_covariance(obs_means)
        lat_mean, lat_cov = self.lat_man.split_mean_covariance(lat_means)

        int_cov = int_means - im.rep.outer_product(obs_mean, lat_mean)

        lat_prs = lcm.inverse(lat_cov)

        int_cov_dense = im.to_matrix(int_cov)
        lat_prs_dense = lcm.to_matrix(lat_prs)
        cob_dense = int_cov_dense @ lat_prs_dense @ int_cov_dense.T
        shaped_cob = ocm.from_matrix(cob_dense)

        obs_prs = ocm.inverse(obs_cov - shaped_cob)

        obs_prs_dense = ocm.to_matrix(obs_prs)
        int_params_dense = -obs_prs_dense @ im.to_matrix(int_cov) @ lat_prs_dense
        int_params = im.from_matrix(int_params_dense)

        obs_loc0 = ocm(obs_prs, obs_mean)
        obs_loc1 = im.rep.matvec(im.matrix_shape, int_params, lat_mean)
        obs_loc = obs_loc0 - obs_loc1

        obs_params = self.obs_man.join_location_precision(obs_loc, obs_prs)
        return self.lkl_fun_man.join_coords(obs_params, int_params)


@dataclass(frozen=True)
class KalmanFilter(LatentProcess[FullNormal, FullNormal]):
    """Kalman Filter: Linear Gaussian state-space model.

    The model is:
        z_0 ~ N(mu_0, Sigma_0)
        z_t = A * z_{t-1} + w_t,  w_t ~ N(0, Q)
        x_t = C * z_t + v_t,      v_t ~ N(0, R)

    This is a LatentProcess with:
    - Prior: FullNormal for p(z_0)
    - Emission: NormalEmission for p(x_t | z_t)
    - Transition: NormalTransition for p(z_t | z_{t-1})
    """

    obs_dim: int
    """Dimension of observations."""

    _lat_dim: int
    """Dimension of latent state."""

    @property
    @override
    def lat_man(self) -> FullNormal:
        return full_normal(self._lat_dim)

    @property
    @override
    def obs_man(self) -> FullNormal:
        return full_normal(self.obs_dim)

    @property
    @override
    def emsn_hrm(self) -> NormalEmission:
        return NormalEmission(self.obs_dim, self._lat_dim)

    @property
    @override
    def trns_hrm(self) -> NormalTransition:
        return NormalTransition(self._lat_dim)

    def initialize(
        self,
        key: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Array:
        """Initialize Kalman filter parameters randomly."""
        keys = jax.random.split(key, 3)
        prior_params = self.lat_man.initialize(keys[0], location, shape)
        emsn_params = self.emsn_hrm.initialize(keys[1], location, shape)
        trns_params = self.trns_hrm.initialize(keys[2], location, shape)
        return self.join_coords(prior_params, emsn_params, trns_params)


def create_kalman_filter(obs_dim: int, lat_dim: int) -> KalmanFilter:
    """Create a Kalman filter model structure.

    Args:
        obs_dim: Dimension of observations
        lat_dim: Dimension of latent state

    Returns:
        KalmanFilter model
    """
    return KalmanFilter(obs_dim=obs_dim, _lat_dim=lat_dim)


# =============================================================================
# Hidden Markov Model Implementation
# =============================================================================


@dataclass(frozen=True)
class CategoricalTransition(AnalyticConjugated[Categorical, Categorical]):
    """Transition harmonium for HMM: p(z_t | z_{t-1}).

    Models discrete state transitions where both current and previous states
    are Categorical distributions. The interaction matrix encodes transition
    log-probabilities.

    For Categorical-Categorical harmoniums, the conjugation parameters are:
        ρₖ = Ψ(θ_x + W_k) - Ψ(θ_x)

    where:
    - θ_x = observable bias (natural parameters)
    - W_k = k-th column of interaction matrix
    - Ψ = log-partition function: log(1 + Σ exp(θᵢ))
    """

    n_states: int
    """Number of latent states."""

    @property
    @override
    def lat_man(self) -> Categorical:
        return Categorical(self.n_states)

    @property
    @override
    def obs_man(self) -> Categorical:
        # In the transition model, the "observable" is z_{t-1}
        return Categorical(self.n_states)

    @property
    @override
    def int_man(self) -> EmbeddedMap[Categorical, Categorical]:
        """Interaction manifold with identity embeddings on both sides."""
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(self.lat_man),
            IdentityEmbedding(self.obs_man),
        )

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute conjugation parameters: ρₖ = Ψ(θ_x + W_k) - Ψ(θ_x).

        For each latent category k, we compute how the log-partition function
        changes when we add the k-th column of the interaction matrix to the
        observable bias.
        """
        obs_bias, int_mat = self.lkl_fun_man.split_coords(lkl_params)

        # Base log-partition: Ψ(θ_x)
        psi_base = self.obs_man.log_partition_function(obs_bias)

        # Reshape interaction matrix: (obs_dim, lat_dim) = (n-1, n-1)
        # For Categorical, dim = n_categories - 1
        obs_dim = self.obs_man.dim
        lat_dim = self.lat_man.dim
        int_matrix = int_mat.reshape((obs_dim, lat_dim))

        # For each latent category k (k = 1, ..., n-1), compute ρₖ
        # The 0-th category (reference) contributes ρ₀ = 0 implicitly
        def compute_rho_k(col_k: Array) -> Array:
            """Compute ρₖ = Ψ(θ_x + W_k) - Ψ(θ_x) for a single column."""
            return self.obs_man.log_partition_function(obs_bias + col_k) - psi_base

        # Apply to each column (lat_dim columns)
        return jax.vmap(compute_rho_k)(int_matrix.T)  # Shape: (lat_dim,)

    @override
    def to_natural_likelihood(self, means: Array) -> Array:
        """Convert mean parameters to natural likelihood parameters.

        For EM, we need to recover the likelihood parameters from expected
        sufficient statistics. For Categorical, this involves computing
        natural parameters from probability ratios.
        """
        obs_means, int_means, lat_means = self.split_coords(means)

        # Get probabilities from means
        obs_probs = self.obs_man.to_probs(obs_means)
        lat_probs = self.lat_man.to_probs(lat_means)

        # Joint probabilities (outer product gives p(obs, lat))
        # int_means represents E[s_obs(x) ⊗ s_lat(z)] = P(obs=i, lat=j) for i,j > 0
        obs_dim = self.obs_man.dim
        lat_dim = self.lat_man.dim
        int_matrix = int_means.reshape((obs_dim, lat_dim))

        # Reconstruct full joint probability table
        # P(obs=i, lat=j) for i,j = 0, ..., n-1
        joint_probs = jnp.zeros((self.n_states, self.n_states))
        # Interior: directly from means (for i,j > 0)
        joint_probs = joint_probs.at[1:, 1:].set(int_matrix)
        # First row (obs=0): P(obs=0, lat=j) = P(lat=j) - sum over obs>0
        for j in range(self.n_states):
            if j == 0:
                joint_probs = joint_probs.at[0, 0].set(
                    lat_probs[0] - jnp.sum(int_matrix[:, 0] if lat_dim > 0 else 0.0)
                )
            else:
                joint_probs = joint_probs.at[0, j].set(
                    lat_probs[j] - jnp.sum(int_matrix[:, j - 1])
                )
        # First column (lat=0): P(obs=i, lat=0) = P(obs=i) - sum over lat>0
        for i in range(1, self.n_states):
            joint_probs = joint_probs.at[i, 0].set(
                obs_probs[i] - jnp.sum(int_matrix[i - 1, :])
            )

        # Conditional p(obs | lat) = joint / p(lat)
        # Add small epsilon for numerical stability
        eps = 1e-10
        cond_probs = joint_probs / (lat_probs[None, :] + eps)

        # Convert to natural parameters
        # θ_obs[i] = log(p(obs=i | lat=0) / p(obs=0 | lat=0)) for the bias
        # θ_int[i,j] = log(p(obs=i | lat=j) / p(obs=i | lat=0)) - log(p(obs=0 | lat=j) / p(obs=0 | lat=0))
        # This is the log-linear parameterization

        # Observable bias: natural params from conditional p(obs | lat=0)
        cond_given_lat0 = cond_probs[:, 0]
        obs_nat = jnp.log(cond_given_lat0[1:] + eps) - jnp.log(cond_given_lat0[0] + eps)

        # Interaction: for each lat category j > 0
        def compute_int_col(j: int) -> Array:
            cond_j = cond_probs[:, j]
            # log(p(obs=i|lat=j)/p(obs=0|lat=j)) - log(p(obs=i|lat=0)/p(obs=0|lat=0))
            nat_j = jnp.log(cond_j[1:] + eps) - jnp.log(cond_j[0] + eps)
            return nat_j - obs_nat

        int_nat = jnp.stack(
            [compute_int_col(j) for j in range(1, self.n_states)], axis=1
        )

        return self.lkl_fun_man.join_coords(obs_nat, int_nat.reshape(-1))


@dataclass(frozen=True)
class CategoricalEmission(AnalyticConjugated[Categorical, Categorical]):
    """Emission harmonium for HMM: p(x_t | z_t).

    Models discrete emissions where both observable and latent states are
    Categorical distributions.
    """

    n_obs: int
    """Number of observable categories."""

    _n_states: int
    """Number of latent states."""

    @property
    @override
    def lat_man(self) -> Categorical:
        return Categorical(self._n_states)

    @property
    @override
    def obs_man(self) -> Categorical:
        return Categorical(self.n_obs)

    @property
    @override
    def int_man(self) -> EmbeddedMap[Categorical, Categorical]:
        """Interaction manifold with identity embeddings on both sides."""
        return EmbeddedMap(
            Rectangular(),
            IdentityEmbedding(self.lat_man),
            IdentityEmbedding(self.obs_man),
        )

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute conjugation parameters: ρₖ = Ψ(θ_x + W_k) - Ψ(θ_x)."""
        obs_bias, int_mat = self.lkl_fun_man.split_coords(lkl_params)

        psi_base = self.obs_man.log_partition_function(obs_bias)

        obs_dim = self.obs_man.dim
        lat_dim = self.lat_man.dim
        int_matrix = int_mat.reshape((obs_dim, lat_dim))

        def compute_rho_k(col_k: Array) -> Array:
            return self.obs_man.log_partition_function(obs_bias + col_k) - psi_base

        return jax.vmap(compute_rho_k)(int_matrix.T)

    @override
    def to_natural_likelihood(self, means: Array) -> Array:
        """Convert mean parameters to natural likelihood parameters."""
        obs_means, int_means, lat_means = self.split_coords(means)

        obs_probs = self.obs_man.to_probs(obs_means)
        lat_probs = self.lat_man.to_probs(lat_means)

        obs_dim = self.obs_man.dim
        lat_dim = self.lat_man.dim
        int_matrix = int_means.reshape((obs_dim, lat_dim))

        # Reconstruct full joint probability table
        joint_probs = jnp.zeros((self.n_obs, self._n_states))
        joint_probs = joint_probs.at[1:, 1:].set(int_matrix)

        for j in range(self._n_states):
            if j == 0:
                joint_probs = joint_probs.at[0, 0].set(
                    lat_probs[0] - jnp.sum(int_matrix[:, 0] if lat_dim > 0 else 0.0)
                )
            else:
                joint_probs = joint_probs.at[0, j].set(
                    lat_probs[j] - jnp.sum(int_matrix[:, j - 1])
                )

        for i in range(1, self.n_obs):
            joint_probs = joint_probs.at[i, 0].set(
                obs_probs[i] - jnp.sum(int_matrix[i - 1, :])
            )

        eps = 1e-10
        cond_probs = joint_probs / (lat_probs[None, :] + eps)

        cond_given_lat0 = cond_probs[:, 0]
        obs_nat = jnp.log(cond_given_lat0[1:] + eps) - jnp.log(cond_given_lat0[0] + eps)

        def compute_int_col(j: int) -> Array:
            cond_j = cond_probs[:, j]
            nat_j = jnp.log(cond_j[1:] + eps) - jnp.log(cond_j[0] + eps)
            return nat_j - obs_nat

        int_nat = jnp.stack(
            [compute_int_col(j) for j in range(1, self._n_states)], axis=1
        )

        return self.lkl_fun_man.join_coords(obs_nat, int_nat.reshape(-1))


@dataclass(frozen=True)
class HiddenMarkovModel(LatentProcess[Categorical, Categorical]):
    """Discrete Hidden Markov Model.

    The model is:
        z_0 ~ Categorical(π_0)
        z_t | z_{t-1} ~ Categorical(A[z_{t-1}, :])
        x_t | z_t ~ Categorical(B[z_t, :])

    where:
    - π_0 is the initial state distribution
    - A is the transition probability matrix
    - B is the emission probability matrix

    This is a LatentProcess with:
    - Prior: Categorical for p(z_0)
    - Emission: CategoricalEmission for p(x_t | z_t)
    - Transition: CategoricalTransition for p(z_t | z_{t-1})
    """

    n_obs: int
    """Number of observable categories."""

    _n_states: int
    """Number of latent states."""

    @property
    @override
    def lat_man(self) -> Categorical:
        return Categorical(self._n_states)

    @property
    @override
    def obs_man(self) -> Categorical:
        return Categorical(self.n_obs)

    @property
    @override
    def emsn_hrm(self) -> CategoricalEmission:
        return CategoricalEmission(self.n_obs, self._n_states)

    @property
    @override
    def trns_hrm(self) -> CategoricalTransition:
        return CategoricalTransition(self._n_states)

    def initialize(
        self,
        key: Array,
        location: float = 0.0,
        shape: float = 0.1,
    ) -> Array:
        """Initialize HMM parameters randomly.

        Creates parameters close to uniform distributions with small noise.
        """
        keys = jax.random.split(key, 3)
        prior_params = self.lat_man.initialize(keys[0], location, shape)
        emsn_params = self.emsn_hrm.initialize(keys[1], location, shape)
        trns_params = self.trns_hrm.initialize(keys[2], location, shape)
        return self.join_coords(prior_params, emsn_params, trns_params)


def create_hmm(n_obs: int, n_states: int) -> HiddenMarkovModel:
    """Create a Hidden Markov Model.

    Args:
        n_obs: Number of observable categories
        n_states: Number of latent states

    Returns:
        HiddenMarkovModel
    """
    return HiddenMarkovModel(n_obs=n_obs, _n_states=n_states)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CategoricalEmission",
    "CategoricalTransition",
    "HiddenMarkovModel",
    "KalmanFilter",
    "LatentProcess",
    "NormalEmission",
    "NormalTransition",
    "conjugated_filtering",
    "conjugated_smoothing",
    "conjugated_smoothing0",
    "create_hmm",
    "create_kalman_filter",
    "latent_process_expectation_maximization",
    "latent_process_expectation_step",
    "latent_process_expectation_step_batch",
    "latent_process_log_density",
    "latent_process_log_observable_density",
    "sample_latent_process",
]
