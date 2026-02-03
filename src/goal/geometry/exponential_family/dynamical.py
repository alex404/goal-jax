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
) -> tuple[Array, Array]:
    """Low-level smoothing returning joint distributions.

    Performs forward-backward smoothing following the Haskell implementation.

    The algorithm works by:
    1. Forward pass: compute filtered states and transposed likelihoods
    2. Backward pass: use transposed likelihoods to propagate smoothed estimates

    Args:
        process: The latent process model
        params: Natural parameters
        observations: Shape (T, obs_dim)

    Returns:
        Tuple of (smoothed, joints):
        - smoothed: Shape (T, lat_dim) - p(z_t | x_{1:T})
        - joints: Shape (T-1, joint_dim) - p(z_t, z_{t+1} | x_{1:T})
    """
    prior_params, emsn_params, trns_params = process.split_coords(params)
    trns_hrm = process.trns_hrm
    emsn_hrm = process.emsn_hrm

    # Get likelihood params
    trns_lkl_params, _ = trns_hrm.split_conjugated(trns_params)
    emsn_lkl_params, _ = emsn_hrm.split_conjugated(emsn_params)

    def forward_step(prior: Array, obs: Array) -> tuple[Array, tuple[Array, Array]]:
        """Forward step: filter and compute transposed likelihood for backward pass.

        Uses the same convention as conjugated_forward_step:
        - 'prior' is p(z_{t-1} | x_{1:t-1}), the previous filtered state
        - We first predict p(z_t | x_{1:t-1}), then update to get p(z_t | x_{1:t})

        The carry is the filtered state, matching conjugated_filtering.
        """
        # Prediction step: p(z_t | x_{1:t-1}) via transition
        predicted = trns_hrm.join_conjugated(trns_lkl_params, prior)
        transposed_pred = transpose_harmonium(trns_hrm, predicted)
        _, predicted_prior = trns_hrm.split_conjugated(transposed_pred)

        # Update step: p(z_t | x_{1:t}) via emission
        emsn_with_pred = emsn_hrm.join_conjugated(emsn_lkl_params, predicted_prior)
        filtered = emsn_hrm.posterior_at(emsn_with_pred, obs)

        # Compute transposed likelihood for backward pass
        joint_hrm_params = trns_hrm.join_conjugated(trns_lkl_params, filtered)
        transposed = transpose_harmonium(trns_hrm, joint_hrm_params)
        trns_t_lkl, _ = trns_hrm.split_conjugated(transposed)

        # Carry is the filtered state (matching conjugated_filtering)
        return filtered, (filtered, trns_t_lkl)

    # Forward pass
    _, (filtered_seq, trns_t_lkls) = jax.lax.scan(
        forward_step, prior_params, observations
    )

    def backward_step(
        smoothed_next: Array, forward_data: tuple[Array, Array]
    ) -> tuple[Array, tuple[Array, Array]]:
        """Backward step: compute smoothed estimate using transposed likelihood.

        Following Haskell:
            hrm = transposeHarmonium $ joinConjugatedHarmonium trns' smth
            bwd = snd $ splitConjugatedHarmonium hrm
        """
        _, trns_t_lkl = forward_data

        # Join transposed likelihood with smoothed next state
        joint_hrm_params = trns_hrm.join_conjugated(trns_t_lkl, smoothed_next)

        # Transpose back
        hrm = transpose_harmonium(trns_hrm, joint_hrm_params)

        # Split to get smoothed current state
        _, smoothed_curr = trns_hrm.split_conjugated(hrm)

        # Joint distribution for EM (join original transition with smoothed current)
        joint_params = trns_hrm.join_conjugated(trns_lkl_params, smoothed_curr)

        return smoothed_curr, (smoothed_curr, joint_params)

    # Initialize with last filtered (smoothed[T] = filtered[T])
    last_filtered = filtered_seq[-1]

    # Prepare forward data for backward pass (exclude last time step)
    forward_data = (filtered_seq[:-1], trns_t_lkls[:-1])

    # Run backward scan
    _, (smoothed_rest, joints) = jax.lax.scan(
        backward_step,
        last_filtered,
        forward_data,
        reverse=True,
    )

    # Concatenate smoothed sequence
    smoothed = jnp.concatenate([smoothed_rest, last_filtered[None, :]], axis=0)

    return smoothed, joints


def conjugated_smoothing(
    process: LatentProcess[Obs, Lat],
    params: Array,
    observations: Array,
) -> Array:
    """Forward-backward smoothing (RTS smoother).

    Returns smoothed marginal distributions only.

    Args:
        process: The latent process model
        params: Natural parameters
        observations: Shape (T, obs_dim)

    Returns:
        Smoothed distributions in natural coords, shape (T, lat_dim)
    """
    smoothed, _ = conjugated_smoothing0(process, params, observations)
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
    smoothed, joints = conjugated_smoothing0(process, params, observations)
    lat_man = process.lat_man

    # E-step for prior: E[s(z_0)]
    prior_means = lat_man.to_mean(smoothed[0])

    # E-step for emission: E[s(x_t, z_t)] averaged over t
    # For each t, we need E[s_obs(x_t), s_lat(z_t), s_obs(x_t) ⊗ s_lat(z_t)]
    _, emsn_params, _ = process.split_coords(params)

    def emission_stats(x: Array, z_params: Array) -> Array:
        emsn_lkl_params, _ = process.emsn_hrm.split_conjugated(emsn_params)
        hrm_params = process.emsn_hrm.join_conjugated(emsn_lkl_params, z_params)
        return process.emsn_hrm.posterior_statistics(hrm_params, x)

    emsn_stats = jax.vmap(emission_stats)(observations, smoothed)
    emsn_means = jnp.mean(emsn_stats, axis=0)

    # E-step for transition: E[s(z_{t-1}, z_t)] averaged over t
    # Use the joint distributions from smoothing
    trns_means = jnp.mean(
        jax.vmap(process.trns_hrm.to_mean)(joints),
        axis=0,
    )

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
