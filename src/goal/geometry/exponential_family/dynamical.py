"""Dynamical models for state-space inference.

This module implements generic latent process models that unify Kalman filters
and Hidden Markov Models under a single conjugated harmonium framework.

The key insight is that both Kalman filters and HMMs are instances of the same
generic LatentProcess structure:

    LatentProcess = (Prior, Emission Harmonium, Transition Harmonium)

where:
- Prior: p(z_0) - initial state distribution
- Emission: p(x_t | z_t) - a conjugated harmonium over (observable, latent)
- Transition: p(z_t | z_{t-1}) - a conjugated harmonium over (z_{t-1}, z_t)

All inference algorithms (filtering, smoothing, EM) are generic on LatentProcess.
There are no Kalman-specific or HMM-specific algorithm implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.combinators import Triple
from .base import Analytic, Differentiable
from .harmonium import AnalyticConjugated

# Type variables for generic functions
Obs = TypeVar("Obs", bound=Differentiable)
Lat = TypeVar("Lat", bound=Analytic)


@dataclass(frozen=True)
class LatentProcess[O: Differentiable, L: Analytic](
    Triple[L, AnalyticConjugated[O, L], AnalyticConjugated[L, L]],
    ABC,
):
    """State-space model as a triple of (prior, emission, transition).

    A LatentProcess defines a joint distribution over sequences:
        p(x_{1:T}, z_{0:T}) = p(z_0) * prod_{t=1}^T p(x_t | z_t) * p(z_t | z_{t-1})

    Components (via Triple):
    - fst_man (L): Prior manifold for p(z_0)
    - snd_man (AnalyticConjugated[O, L]): Emission harmonium for p(x_t | z_t)
    - trd_man (AnalyticConjugated[L, L]): Transition harmonium for p(z_t | z_{t-1})

    The split_coords/join_coords methods from Triple give us:
        params -> (prior_params, emission_params, transition_params)

    Type Parameters:
        O: Observable exponential family (must be Differentiable)
        L: Latent exponential family (must be Analytic for EM)
    """

    # Contract: Subclasses must define these properties

    @property
    @abstractmethod
    def lat_man(self) -> L:
        """Manifold of latent variables."""

    @property
    @abstractmethod
    def obs_man(self) -> O:
        """Manifold of observable variables."""

    @property
    @abstractmethod
    def emsn_hrm(self) -> AnalyticConjugated[O, L]:
        """Emission harmonium for p(x_t | z_t)."""

    @property
    @abstractmethod
    def trns_hrm(self) -> AnalyticConjugated[L, L]:
        """Transition harmonium for p(z_t | z_{t-1})."""

    # Triple interface implementation

    @property
    @override
    def fst_man(self) -> L:
        """Prior manifold for p(z_0)."""
        return self.lat_man

    @property
    @override
    def snd_man(self) -> AnalyticConjugated[O, L]:
        """Emission harmonium manifold."""
        return self.emsn_hrm

    @property
    @override
    def trd_man(self) -> AnalyticConjugated[L, L]:
        """Transition harmonium manifold."""
        return self.trns_hrm


# =============================================================================
# Convenience Functions
# =============================================================================


def split_latent_process(
    process: LatentProcess[Obs, Lat],
    params: Array,
) -> tuple[Array, Array, Array]:
    """Split params into (prior, emission, transition).

    Alias for process.split_coords(params) for API compatibility with Haskell.

    Args:
        process: The latent process model
        params: Full parameter array

    Returns:
        Tuple of (prior_params, emission_params, transition_params)
    """
    return process.split_coords(params)


def join_latent_process(
    process: LatentProcess[Obs, Lat],
    prior: Array,
    emission: Array,
    transition: Array,
) -> Array:
    """Join (prior, emission, transition) into full params.

    Alias for process.join_coords(...) for API compatibility with Haskell.

    Args:
        process: The latent process model
        prior: Prior parameters for p(z_0)
        emission: Emission harmonium parameters
        transition: Transition harmonium parameters

    Returns:
        Full parameter array
    """
    return process.join_coords(prior, emission, transition)


# =============================================================================
# Harmonium Operations
# =============================================================================


def transpose_harmonium(
    hrm: AnalyticConjugated[Lat, Lat],
    params: Array,
) -> Array:
    """Transpose harmonium: swap observable/latent roles.

    For a symmetric conjugated harmonium over (A, B), returns parameters
    for the transposed harmonium over (B, A).

    This is critical for the backward pass - it converts p(z_t | z_{t-1})
    into backward message form p(z_{t-1} | z_t).

    Args:
        hrm: A symmetric conjugated harmonium
        params: Natural parameters

    Returns:
        Transposed natural parameters
    """
    obs_params, int_params, lat_params = hrm.split_coords(params)
    # Transpose the interaction matrix
    int_params_t = hrm.int_man.transpose(int_params)
    # Swap observable and latent
    return hrm.join_coords(lat_params, int_params_t, obs_params)


def join_conjugated_harmonium(
    hrm: AnalyticConjugated[Obs, Lat],
    lkl_params: Array,
    lat_params: Array,
) -> Array:
    """Join likelihood with latent params (subtracting conjugation offset).

    Inverts split_conjugated: given likelihood parameters and latent (prior)
    parameters in natural coordinates, reconstructs the full harmonium params.

    Args:
        hrm: Analytic conjugated harmonium
        lkl_params: Likelihood function parameters
        lat_params: Latent (prior) natural parameters

    Returns:
        Full harmonium natural parameters
    """
    return hrm.join_conjugated(lkl_params, lat_params)


def split_conjugated_harmonium(
    hrm: AnalyticConjugated[Obs, Lat],
    params: Array,
) -> tuple[Array, Array]:
    """Split into likelihood and latent params (adding conjugation offset).

    Returns the likelihood function parameters and the prior parameters
    (latent params plus conjugation parameters).

    Args:
        hrm: Analytic conjugated harmonium
        params: Full harmonium natural parameters

    Returns:
        Tuple of (likelihood_params, prior_params)
    """
    return hrm.split_conjugated(params)


# =============================================================================
# Sampling
# =============================================================================


def sample_latent_process(
    process: LatentProcess[Obs, Lat],
    params: Array,
    key: Array,
    n_steps: int,
) -> tuple[Array, Array]:
    """Sample a trajectory from the latent process.

    Generates a sequence of observations and latent states:
        z_0 ~ p(z_0)
        for t = 1, ..., n_steps:
            z_t ~ p(z_t | z_{t-1})
            x_t ~ p(x_t | z_t)

    Args:
        process: The latent process model
        params: Natural parameters
        key: PRNG key
        n_steps: Number of time steps

    Returns:
        (observations, latents): Arrays of shape (n_steps, obs_dim) and
                                 (n_steps+1, lat_dim) where latents includes z_0
    """
    prior_params, emsn_params, trns_params = process.split_coords(params)

    # Split key for initial state and sequence
    key_init, key_seq = jax.random.split(key)

    # Sample initial latent state z_0
    z_0 = process.lat_man.sample(key_init, prior_params, n=1)[0]

    # Get likelihood and prior from transition harmonium
    trns_lkl_params, _ = process.trns_hrm.split_conjugated(trns_params)
    emsn_lkl_params, _ = process.emsn_hrm.split_conjugated(emsn_params)

    def step(
        carry: tuple[Array, Array], _: int
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        z_prev, key = carry
        key_trans, key_emit, key_next = jax.random.split(key, 3)

        # Sample z_t ~ p(z_t | z_{t-1})
        # Compute sufficient statistics of z_prev before passing to likelihood
        s_z_prev = process.lat_man.sufficient_statistic(z_prev)
        z_params = process.trns_hrm.lkl_fun_man(trns_lkl_params, s_z_prev)
        z_t = process.lat_man.sample(key_trans, z_params, n=1)[0]

        # Sample x_t ~ p(x_t | z_t)
        # Compute sufficient statistics of z_t before passing to likelihood
        s_z_t = process.lat_man.sufficient_statistic(z_t)
        x_params = process.emsn_hrm.lkl_fun_man(emsn_lkl_params, s_z_t)
        x_t = process.obs_man.sample(key_emit, x_params, n=1)[0]

        return (z_t, key_next), (x_t, z_t)

    # Run the scan
    _, (observations, latents_after_0) = jax.lax.scan(
        step, (z_0, key_seq), None, length=n_steps
    )

    # Prepend z_0 to latents
    latents = jnp.concatenate([z_0[None, :], latents_after_0], axis=0)

    return observations, latents


# =============================================================================
# Forward Filtering
# =============================================================================


def conjugated_forward_step(
    emsn_hrm: AnalyticConjugated[Obs, Lat],
    trns_hrm: AnalyticConjugated[Lat, Lat],
    emsn_params: Array,
    trns_params: Array,
    prior: Array,
    obs: Array,
) -> tuple[Array, Array]:
    """Single forward filtering step: predict + update.

    Computes p(z_t | x_{1:t}) from p(z_{t-1} | x_{1:t-1}) and x_t.

    Algorithm:
    1. Prediction: p(z_t | x_{1:t-1}) via transition
       - Join prior with transition to get joint p(z_{t-1}, z_t | x_{1:t-1})
       - Marginalize out z_{t-1} to get predicted state

    2. Update: p(z_t | x_{1:t}) via emission likelihood
       - Compute posterior given observation x_t

    Args:
        emsn_hrm: Emission harmonium
        trns_hrm: Transition harmonium
        emsn_params: Emission harmonium parameters
        trns_params: Transition harmonium parameters
        prior: Prior p(z_{t-1} | x_{1:t-1}) in natural coords
        obs: Observation x_t

    Returns:
        Tuple of (filtered, log_likelihood_contribution):
        - filtered: p(z_t | x_{1:t}) in natural coordinates
        - log_likelihood_contribution: log p(x_t | x_{1:t-1})
    """
    # Prediction step: compute p(z_t | x_{1:t-1})
    # Following Haskell: snd . splitConjugatedHarmonium . transposeHarmonium
    #                    $ joinConjugatedHarmonium trns prr
    trns_lkl_params, _ = trns_hrm.split_conjugated(trns_params)
    predicted = trns_hrm.join_conjugated(trns_lkl_params, prior)
    transposed = transpose_harmonium(trns_hrm, predicted)
    _, predicted_prior = trns_hrm.split_conjugated(transposed)

    # Update step: condition on observation x_t
    # Get the posterior function from emission harmonium
    emsn_lkl_params, _ = emsn_hrm.split_conjugated(emsn_params)
    emsn_params_with_pred = emsn_hrm.join_conjugated(emsn_lkl_params, predicted_prior)

    # Compute posterior p(z_t | x_t, x_{1:t-1}) = p(z_t | x_{1:t})
    filtered = emsn_hrm.posterior_at(emsn_params_with_pred, obs)

    # Compute log p(x_t | x_{1:t-1}) for log-likelihood
    log_lik = emsn_hrm.log_observable_density(emsn_params_with_pred, obs)

    return filtered, log_lik


def conjugated_filtering(
    process: LatentProcess[Obs, Lat],
    params: Array,
    observations: Array,
) -> tuple[Array, Array]:
    """Forward filtering over observation sequence.

    Computes the filtering distributions p(z_t | x_{1:t}) for all t.

    Args:
        process: The latent process model
        params: Natural parameters
        observations: Shape (T, obs_dim)

    Returns:
        Tuple of (filtered, log_likelihood):
        - filtered: Filtering distributions in natural coords, shape (T, lat_dim)
        - log_likelihood: Total log p(x_{1:T})
    """
    prior_params, emsn_params, trns_params = process.split_coords(params)

    def step(
        carry: tuple[Array, Array], obs: Array
    ) -> tuple[tuple[Array, Array], Array]:
        prior, total_ll = carry
        filtered, log_lik = conjugated_forward_step(
            process.emsn_hrm,
            process.trns_hrm,
            emsn_params,
            trns_params,
            prior,
            obs,
        )
        return (filtered, total_ll + log_lik), filtered

    (_, log_likelihood), filtered_seq = jax.lax.scan(
        step, (prior_params, jnp.array(0.0)), observations
    )

    return filtered_seq, log_likelihood


# =============================================================================
# Backward Smoothing
# =============================================================================


def conjugated_smoothing0(
    process: LatentProcess[Obs, Lat],
    params: Array,
    observations: Array,
) -> tuple[Array, Array, Array]:
    """Low-level smoothing returning smoothed z_0, smoothed sequence, and joints.

    Performs forward-backward smoothing.

    Args:
        process: The latent process model
        params: Natural parameters
        observations: Shape (T, obs_dim)

    Returns:
        Tuple of (smoothed_z0, smoothed, joints):
        - smoothed_z0: Shape (lat_dim,) - p(z_0 | x_{1:T})
        - smoothed: Shape (T, lat_dim) - p(z_t | x_{1:T}) for t = 1, ..., T
        - joints: Shape (T, joint_dim) - mean params of p(z_{t-1}, z_t | x_{1:T}) for t = 1, ..., T
    """
    prior_params, emsn_params, trns_params = process.split_coords(params)
    trns_hrm = process.trns_hrm
    emsn_hrm = process.emsn_hrm
    lat_man = process.lat_man

    # Get likelihood params
    trns_lkl_params, _ = trns_hrm.split_conjugated(trns_params)
    emsn_lkl_params, _ = emsn_hrm.split_conjugated(emsn_params)

    def forward_step(
        prior: Array, obs: Array
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        """Forward step: filter and return (filtered, predicted_prior, prior_input)."""
        # Prediction step: p(z_t | x_{1:t-1}) via transition
        predicted = trns_hrm.join_conjugated(trns_lkl_params, prior)
        transposed_pred = transpose_harmonium(trns_hrm, predicted)
        _, predicted_prior = trns_hrm.split_conjugated(transposed_pred)

        # Update step: p(z_t | x_{1:t}) via emission
        emsn_with_pred = emsn_hrm.join_conjugated(emsn_lkl_params, predicted_prior)
        filtered = emsn_hrm.posterior_at(emsn_with_pred, obs)

        return filtered, (filtered, predicted_prior, prior)

    # Forward pass - save filtered, predicted, and prior for each step
    _, (filtered_seq, predicted_seq, _) = jax.lax.scan(
        forward_step, prior_params, observations
    )

    def backward_step(smoothed_next: Array, filtered: Array) -> tuple[Array, Array]:
        """Backward step: compute smoothed estimate from filtered and future smoothed."""
        # Transposed likelihood for backward message
        joint_hrm_params = trns_hrm.join_conjugated(trns_lkl_params, filtered)
        transposed = transpose_harmonium(trns_hrm, joint_hrm_params)
        trns_t_lkl, _ = trns_hrm.split_conjugated(transposed)

        # Backward message: join transposed likelihood with smoothed next
        backward_joint = trns_hrm.join_conjugated(trns_t_lkl, smoothed_next)
        backward_hrm = transpose_harmonium(trns_hrm, backward_joint)
        _, smoothed_curr = trns_hrm.split_conjugated(backward_hrm)

        return smoothed_curr, smoothed_curr

    # Initialize with last filtered (smoothed[T] = filtered[T])
    last_filtered = filtered_seq[-1]

    # Run backward scan over all but the last time step
    _, smoothed_rest = jax.lax.scan(
        backward_step,
        last_filtered,
        filtered_seq[:-1],
        reverse=True,
    )

    # Concatenate: smoothed_rest has T-1 elements, add last_filtered
    smoothed_seq = jnp.concatenate([smoothed_rest, last_filtered[None, :]], axis=0)

    # Compute smoothed z_0 by one more backward step
    joint_hrm_0 = trns_hrm.join_conjugated(trns_lkl_params, prior_params)
    transposed_0 = transpose_harmonium(trns_hrm, joint_hrm_0)
    trns_t_lkl_0, _ = trns_hrm.split_conjugated(transposed_0)

    backward_joint_0 = trns_hrm.join_conjugated(trns_t_lkl_0, smoothed_seq[0])
    backward_hrm_0 = transpose_harmonium(trns_hrm, backward_joint_0)
    _, smoothed_z0 = trns_hrm.split_conjugated(backward_hrm_0)

    # Compute two-slice joint mean parameters p(z_{t-1}, z_t | x_{1:T})
    #
    # The formula for the two-slice joint (ξ) is:
    # ξ(z_{t-1}, z_t) ∝ filtered[t-1](z_{t-1}) * A(z_{t-1}, z_t) * correction(z_t)
    #
    # where:
    # - filtered[t-1] = p(z_{t-1} | x_{1:t-1})
    # - A(z_{t-1}, z_t) = p(z_t | z_{t-1}) from the transition model
    # - correction(z_t) = smoothed[t](z_t) / predicted[t](z_t)
    #
    # The mean parameters of the joint are:
    # - obs_mean = E[s(z_{t-1})] = smoothed z_{t-1} mean params
    # - lat_mean = E[s(z_t)] = smoothed z_t mean params
    # - int_mean = E[s(z_{t-1}) ⊗ s(z_t)] = cross term computed from ξ

    # All smoothed z's including z_0
    all_smoothed = jnp.concatenate([smoothed_z0[None, :], smoothed_seq], axis=0)

    # All priors: prior_params (for z_0), then filtered[0] to filtered[T-2]
    all_priors = jnp.concatenate([prior_params[None, :], filtered_seq[:-1]], axis=0)

    def compute_joint_means(data: tuple[Array, Array, Array, Array]) -> Array:
        """Compute two-slice joint mean params using the xi formula.

        The cross term E[s(z_{t-1}) ⊗ s(z_t)] is computed by:
        1. Building the joint natural params properly
        2. Using posterior_statistics to compute expectations

        For the two-slice joint, we need to compute the cross term by
        constructing a harmonium that represents:
        p(z_{t-1}, z_t | x_{1:T}) ∝ prior(z_{t-1}) * trans(z_t|z_{t-1}) * corr(z_t)
        """
        prior_nat, smoothed_nat, predicted_nat, smoothed_prev_nat = data

        # Mean parameters for marginals (from smoothed distributions)
        obs_mean = lat_man.to_mean(smoothed_prev_nat)  # smoothed z_{t-1}
        lat_mean = lat_man.to_mean(smoothed_nat)  # smoothed z_t

        # Compute the cross term E[s(z_{t-1}) ⊗ s(z_t)] using the ξ formula.
        #
        # The two-slice posterior ξ satisfies:
        # ξ(z_{t-1}, z_t) ∝ α(z_{t-1}) * A(z_{t-1}, z_t) * β(z_t)
        #
        # where:
        # - α(z_{t-1}) = filtered[t-1] = p(z_{t-1} | x_{1:t-1})
        # - A(z_{t-1}, z_t) = p(z_t | z_{t-1})
        # - β(z_t) ∝ p(x_t:T | z_t) = smoothed[t] / predicted[t] (up to normalization)
        #
        # We compute this by constructing a joint harmonium with:
        # - obs params = prior_nat (encodes α)
        # - int params = transition interaction (encodes A structure)
        # - lat params = transition base + backward correction (encodes A * β)

        # The transition likelihood params encode p(z_t | z_{t-1})
        # Split into obs_bias (base rate for z_t) and interaction
        obs_bias, int_params = trns_hrm.lkl_fun_man.split_coords(trns_lkl_params)

        # The backward contribution in natural params gives the log-ratio of β
        backward_nat = smoothed_nat - predicted_nat

        # Build the joint natural params
        # The harmonium structure is:
        # log p(z_{t-1}, z_t) = θ_obs · s(z_{t-1}) + θ_int · s(z_{t-1}) ⊗ s(z_t)
        #                     + θ_lat · s(z_t) - ψ
        #
        # For the two-slice joint:
        # - θ_obs = prior_nat (the filtered prior on z_{t-1})
        # - θ_int = transition interaction
        # - θ_lat = obs_bias + backward_nat (base rate + backward correction)

        joint_nat = trns_hrm.join_coords(prior_nat, int_params, obs_bias + backward_nat)

        # Compute mean params of this joint
        joint_mean = trns_hrm.to_mean(joint_nat)

        # Extract the cross term
        _, int_mean, _ = trns_hrm.split_coords(joint_mean)

        # Return with smoothed marginals for obs/lat (which are correct)
        return trns_hrm.join_coords(obs_mean, int_mean, lat_mean)

    # Smoothed z_{t-1} for each transition: smoothed_z0, smoothed[0], ..., smoothed[T-2]
    smoothed_prev_seq = all_smoothed[:-1]

    joints = jax.vmap(compute_joint_means)(
        (all_priors, smoothed_seq, predicted_seq, smoothed_prev_seq)
    )

    return smoothed_z0, smoothed_seq, joints


def conjugated_smoothing(
    process: LatentProcess[Obs, Lat],
    params: Array,
    observations: Array,
) -> Array:
    """Forward-backward smoothing (RTS smoother).

    Returns smoothed marginal distributions for z_1 through z_T only.

    Args:
        process: The latent process model
        params: Natural parameters
        observations: Shape (T, obs_dim)

    Returns:
        Smoothed distributions in natural coords, shape (T, lat_dim)
    """
    _, smoothed, _ = conjugated_smoothing0(process, params, observations)
    return smoothed


# =============================================================================
# Log-Likelihood
# =============================================================================


def latent_process_log_observable_density(
    process: LatentProcess[Obs, Lat],
    params: Array,
    observations: Array,
) -> Array:
    """Compute log p(x_{1:T}) via forward filtering.

    This is the marginal log-likelihood integrating out latent states.

    Args:
        process: The latent process model
        params: Natural parameters
        observations: Shape (T, obs_dim)

    Returns:
        Scalar log-likelihood log p(x_{1:T})
    """
    _, log_likelihood = conjugated_filtering(process, params, observations)
    return log_likelihood


def latent_process_log_density(
    process: LatentProcess[Obs, Lat],
    params: Array,
    observations: Array,
    latents: Array,
) -> Array:
    """Compute log p(x_{1:T}, z_{0:T}) - joint log density.

    Useful for evaluating sampled trajectories.

    Args:
        process: The latent process model
        params: Natural parameters
        observations: Shape (T, obs_dim)
        latents: Shape (T+1, lat_dim) including z_0

    Returns:
        Scalar joint log-density
    """
    prior_params, emsn_params, trns_params = process.split_coords(params)
    lat_man = process.lat_man

    # Log prior: log p(z_0)
    log_prior = lat_man.log_density(prior_params, latents[0])

    # Log transitions: sum of log p(z_t | z_{t-1})
    trns_lkl_params, _ = process.trns_hrm.split_conjugated(trns_params)

    def log_transition(z_prev: Array, z_curr: Array) -> Array:
        # Compute sufficient statistics before passing to likelihood
        s_z_prev = lat_man.sufficient_statistic(z_prev)
        z_params = process.trns_hrm.lkl_fun_man(trns_lkl_params, s_z_prev)
        return lat_man.log_density(z_params, z_curr)

    log_transitions = jnp.sum(jax.vmap(log_transition)(latents[:-1], latents[1:]))

    # Log emissions: sum of log p(x_t | z_t)
    emsn_lkl_params, _ = process.emsn_hrm.split_conjugated(emsn_params)

    def log_emission(z: Array, x: Array) -> Array:
        # Compute sufficient statistics before passing to likelihood
        s_z = lat_man.sufficient_statistic(z)
        x_params = process.emsn_hrm.lkl_fun_man(emsn_lkl_params, s_z)
        return process.obs_man.log_density(x_params, x)

    # Latents[1:] corresponds to z_1, ..., z_T which generate x_1, ..., x_T
    log_emissions = jnp.sum(jax.vmap(log_emission)(latents[1:], observations))

    return log_prior + log_transitions + log_emissions


# =============================================================================
# EM Learning
# =============================================================================


def latent_process_expectation_step(
    process: LatentProcess[Obs, Lat],
    params: Array,
    observations: Array,
) -> tuple[Array, Array, Array]:
    """E-step: compute expected sufficient statistics for a single sequence.

    Computes the expected sufficient statistics under the posterior
    distribution p(z_{0:T} | x_{1:T}).

    Args:
        process: The latent process
        params: Current natural parameters
        observations: Shape (T, obs_dim)

    Returns:
        Mean parameters for (prior, emission, transition)
    """
    # Get smoothed distributions and joint distributions
    smoothed_z0, smoothed, joints = conjugated_smoothing0(process, params, observations)
    lat_man = process.lat_man

    # E-step for prior: E[s(z_0) | x_{1:T}]
    # Use the properly computed smoothed z_0
    prior_means = lat_man.to_mean(smoothed_z0)

    # E-step for emission: E[s(x_t, z_t) | x_{1:T}] averaged over t
    # For each t, we need E[s_obs(x_t), s_lat(z_t), s_obs(x_t) ⊗ s_lat(z_t)]
    _, emsn_params, _ = process.split_coords(params)

    def emission_stats(x: Array, z_params: Array) -> Array:
        emsn_lkl_params, _ = process.emsn_hrm.split_conjugated(emsn_params)
        hrm_params = process.emsn_hrm.join_conjugated(emsn_lkl_params, z_params)
        return process.emsn_hrm.posterior_statistics(hrm_params, x)

    emsn_stats = jax.vmap(emission_stats)(observations, smoothed)
    emsn_means = jnp.mean(emsn_stats, axis=0)

    # E-step for transition: E[s(z_{t-1}, z_t) | x_{1:T}] averaged over t
    # joints are already in mean params from conjugated_smoothing0
    trns_means = jnp.mean(joints, axis=0)

    return prior_means, emsn_means, trns_means


def latent_process_expectation_step_batch(
    process: LatentProcess[Obs, Lat],
    params: Array,
    observations_batch: Array,
) -> tuple[Array, Array, Array]:
    """E-step: compute expected sufficient statistics over batch of sequences.

    Args:
        process: The latent process
        params: Current natural parameters
        observations_batch: Shape (N, T, obs_dim) - N sequences of length T

    Returns:
        Mean parameters for (prior, emission, transition) averaged over batch
    """

    # Apply E-step to each sequence
    def single_estep(obs_seq: Array) -> tuple[Array, Array, Array]:
        return latent_process_expectation_step(process, params, obs_seq)

    prior_means, emsn_means, trns_means = jax.vmap(single_estep)(observations_batch)

    # Average over batch
    return (
        jnp.mean(prior_means, axis=0),
        jnp.mean(emsn_means, axis=0),
        jnp.mean(trns_means, axis=0),
    )


def latent_process_expectation_maximization(
    process: LatentProcess[Obs, Lat],
    params: Array,
    observations_batch: Array,
) -> Array:
    """Full EM step: E-step then M-step via to_natural.

    Args:
        process: The latent process
        params: Current natural parameters
        observations_batch: Shape (N, T, obs_dim) - N sequences

    Returns:
        Updated natural parameters
    """
    # E-step
    prior_means, emsn_means, trns_means = latent_process_expectation_step_batch(
        process, params, observations_batch
    )

    # M-step: convert mean parameters to natural parameters
    prior_params = process.lat_man.to_natural(prior_means)
    emsn_params = process.emsn_hrm.to_natural(emsn_means)
    trns_params = process.trns_hrm.to_natural(trns_means)

    return process.join_coords(prior_params, emsn_params, trns_params)
