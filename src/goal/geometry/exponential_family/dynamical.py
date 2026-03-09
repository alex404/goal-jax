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
from dataclasses import dataclass, replace
from typing import Any, Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.combinators import Triple
from .base import Analytic, Differentiable, Gibbs
from .graphical import AnalyticHierarchical, SymmetricHierarchical
from .harmonium import (
    AnalyticConjugated,
    DifferentiableConjugated,
    Harmonium,
    SymmetricConjugated,
)


@dataclass(frozen=True)
class MarkovProcess[L: Gibbs](
    Harmonium[L, Any],
    ABC,
):
    """Fully observed Markov chain built recursively as a harmonium.

    The recursion is driven by a one-step transition harmonium ``trns_hrm`` over
    the state family ``L``.  Data layout is chronological
    ``[x_0, x_1, \\ldots, x_T]`` where ``obs = x_0`` (single state) and
    ``lat = nxt\\_mrk(x_1 \\ldots x_T)`` for ``T > 1``.

    - ``n_steps == 1``: the chain is exactly ``trns_hrm``
    - ``n_steps > 1``: the latent side is the same chain with one fewer
      transition, and the interaction embeds via the sub-chain's observable slot

    The ``Posterior`` type parameter of ``Harmonium`` is ``Any`` because the
    latent manifold is recursive: for ``n_steps > 1`` it is another
    ``MarkovProcess`` (i.e. ``Self``), while for ``n_steps == 1`` it is ``L``.
    Python's type system cannot express this runtime-dependent self-referential
    type, so ``Any`` is used throughout the hierarchy wherever this recursive
    latent type appears.
    """

    # Fields

    n_steps: int
    trns_hrm: Harmonium[L, L]

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError("MarkovProcess requires n_steps >= 1.")
        if self.trns_hrm.obs_man != self.trns_hrm.pst_man:
            raise TypeError(
                "MarkovProcess requires a one-step self-transition harmonium."
            )

    # Overrides

    @property
    @override
    def obs_man(self) -> L:
        return self.trns_hrm.obs_man

    # Methods

    @property
    def nxt_mrk(self) -> Self:
        """The recursive sub-chain with one fewer step."""
        if self.n_steps == 1:
            raise ValueError("A one-step MarkovProcess has no sub-chain.")
        return replace(self, n_steps=self.n_steps - 1)


@dataclass(frozen=True)
class ConjugatedMarkovProcess[L: Differentiable](
    MarkovProcess[L],
    SymmetricHierarchical[SymmetricConjugated[L, L], Any],
    ABC,
):
    """Recursive symmetric conjugated Markov chain.

    Inherits from ``SymmetricHierarchical`` with ``lwr_hrm = trns_hrm`` and
    ``upr_hrm = nxt_mrk`` (the sub-chain). The ``UpperHarmonium`` type
    parameter is ``Any`` for the same recursive reason as ``MarkovProcess``:
    the upper harmonium is another ``ConjugatedMarkovProcess`` (``Self``),
    which cannot be expressed as a static generic bound.
    """

    trns_hrm: SymmetricConjugated[L, L]

    # Bridge properties: map hierarchy to Markov process

    @property
    @override
    def lwr_hrm(self) -> SymmetricConjugated[L, L]:
        return self.trns_hrm

    @property
    @override
    def upr_hrm(self) -> Any:
        return self.nxt_mrk

    # n_steps==1 base case overrides (n_steps>1 delegates to SymmetricHierarchical)

    @property
    @override
    def lat_man(self) -> Any:
        if self.n_steps == 1:
            return self.trns_hrm.lat_man
        return super().lat_man

    @property
    @override
    def int_man(self) -> Any:
        if self.n_steps == 1:
            return self.trns_hrm.int_man
        return super().int_man

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        if self.n_steps == 1:
            return self.trns_hrm.conjugation_parameters(lkl_params)
        return super().conjugation_parameters(lkl_params)


@dataclass(frozen=True)
class DifferentiableMarkovProcess[L: Differentiable](
    ConjugatedMarkovProcess[L],
    DifferentiableConjugated[L, Any, Any],
    ABC,
):
    """Recursive differentiable conjugated Markov chain.

    Both ``Posterior`` and ``Prior`` type parameters of
    ``DifferentiableConjugated`` are ``Any`` because they are the recursive
    sub-chain type (see ``MarkovProcess``).
    """


@dataclass(frozen=True)
class AnalyticMarkovProcess[L: Analytic](
    DifferentiableMarkovProcess[L],
    AnalyticHierarchical[AnalyticConjugated[L, L], Any],
    ABC,
):
    """Recursive analytic Markov chain.

    The ``UpperHarmonium`` parameter of ``AnalyticHierarchical`` is ``Any``
    for the same recursive reason (see ``MarkovProcess``).
    """

    trns_hrm: AnalyticConjugated[L, L]

    @property
    @override
    def lwr_hrm(self) -> AnalyticConjugated[L, L]:
        return self.trns_hrm

    @override
    def to_natural_likelihood(self, means: Array) -> Array:
        """Convert recursive mean coordinates to likelihood natural parameters."""
        if self.n_steps == 1:
            return self.trns_hrm.to_natural_likelihood(means)
        return super().to_natural_likelihood(means)


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

    # Methods — mirror harmonium API

    def initialize_from_sample(
        self, key: Array, observations_batch: Array, shape: float = 0.1
    ) -> Array:
        """Initialize from observation data.

        Uses sample statistics to set emission biases, and initializes
        prior and transition with small random parameters.

        Args:
            observations_batch: Shape (N, T, obs_dim) or (T, obs_dim).
        """
        keys = jax.random.split(key, 3)
        # Flatten to (N*T, obs_dim) for emission initialization
        flat_obs = observations_batch.reshape(-1, self.obs_man.data_dim)
        prior_params = self.lat_man.initialize(keys[0], shape=shape)
        emsn_params = self.emsn_hrm.initialize_from_sample(
            keys[1], flat_obs, shape=shape
        )
        trns_params = self.trns_hrm.initialize(keys[2], shape=shape)
        return self.join_coords(prior_params, emsn_params, trns_params)

    def sample(self, key: Array, params: Array, n_steps: int) -> tuple[Array, Array]:
        """Sample trajectories (observations, latents) from the joint.

        Generates a sequence of observations and latent states:
            z_0 ~ p(z_0)
            for t = 1, ..., n_steps:
                z_t ~ p(z_t | z_{t-1})
                x_t ~ p(x_t | z_t)

        Returns:
            (observations, latents): Arrays of shape (n_steps, obs_dim) and
                                     (n_steps+1, lat_dim) where latents includes z_0
        """
        prior_params, emsn_params, trns_params = self.split_coords(params)

        # Split key for initial state and sequence
        key_init, key_seq = jax.random.split(key)

        # Sample initial latent state z_0
        z_0 = self.lat_man.sample(key_init, prior_params, n=1)[0]

        # Get likelihood and prior from transition harmonium
        trns_lkl_params, _ = self.trns_hrm.split_conjugated(trns_params)
        emsn_lkl_params, _ = self.emsn_hrm.split_conjugated(emsn_params)

        def step(
            carry: tuple[Array, Array], _: int
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            z_prev, key = carry
            key_trans, key_emit, key_next = jax.random.split(key, 3)

            s_z_prev = self.lat_man.sufficient_statistic(z_prev)
            z_params = self.trns_hrm.lkl_fun_man(trns_lkl_params, s_z_prev)
            z_t = self.lat_man.sample(key_trans, z_params, n=1)[0]

            s_z_t = self.lat_man.sufficient_statistic(z_t)
            x_params = self.emsn_hrm.lkl_fun_man(emsn_lkl_params, s_z_t)
            x_t = self.obs_man.sample(key_emit, x_params, n=1)[0]

            return (z_t, key_next), (x_t, z_t)

        _, (observations, latents_after_0) = jax.lax.scan(
            step, (z_0, key_seq), None, length=n_steps
        )

        latents = jnp.concatenate([z_0[None, :], latents_after_0], axis=0)
        return observations, latents

    def filtering(self, params: Array, observations: Array) -> tuple[Array, Array]:
        """Forward filtering: p(z_t | x_{1:t}) for all t.

        Mirrors DifferentiableConjugated.log_observable_density by computing
        the marginal likelihood as a byproduct.

        Returns (filtered_seq, log_likelihood).
        """
        prior_params, emsn_params, trns_params = self.split_coords(params)
        emsn_hrm = self.emsn_hrm
        trns_hrm = self.trns_hrm

        def step(
            carry: tuple[Array, Array], obs: Array
        ) -> tuple[tuple[Array, Array], Array]:
            prior, total_ll = carry
            filtered, log_lik = _conjugated_forward_step(
                emsn_hrm, trns_hrm, emsn_params, trns_params, prior, obs,
            )
            return (filtered, total_ll + log_lik), filtered

        (_, log_likelihood), filtered_seq = jax.lax.scan(
            step, (prior_params, jnp.array(0.0)), observations
        )

        return filtered_seq, log_likelihood

    def log_observable_density(self, params: Array, observations: Array) -> Array:
        """Log marginal likelihood log p(x_{1:T}) via forward filtering.

        Mirrors DifferentiableConjugated.log_observable_density.
        """
        _, log_likelihood = self.filtering(params, observations)
        return log_likelihood

    def log_density(self, params: Array, observations: Array, latents: Array) -> Array:
        """Joint log-density log p(x_{1:T}, z_{0:T})."""
        prior_params, emsn_params, trns_params = self.split_coords(params)
        lat_man = self.lat_man

        log_prior = lat_man.log_density(prior_params, latents[0])

        trns_lkl_params, _ = self.trns_hrm.split_conjugated(trns_params)

        def log_transition(z_prev: Array, z_curr: Array) -> Array:
            s_z_prev = lat_man.sufficient_statistic(z_prev)
            z_params = self.trns_hrm.lkl_fun_man(trns_lkl_params, s_z_prev)
            return lat_man.log_density(z_params, z_curr)

        log_transitions = jnp.sum(
            jax.vmap(log_transition)(latents[:-1], latents[1:])
        )

        emsn_lkl_params, _ = self.emsn_hrm.split_conjugated(emsn_params)

        def log_emission(z: Array, x: Array) -> Array:
            s_z = lat_man.sufficient_statistic(z)
            x_params = self.emsn_hrm.lkl_fun_man(emsn_lkl_params, s_z)
            return self.obs_man.log_density(x_params, x)

        log_emissions = jnp.sum(jax.vmap(log_emission)(latents[1:], observations))

        return log_prior + log_transitions + log_emissions

    def smoothing(
        self, params: Array, observations: Array
    ) -> tuple[Array, Array, Array]:
        """Forward-backward smoothing.

        Returns (smoothed_z0, smoothed_seq, joints):
        - smoothed_z0: Shape (lat_dim,) - p(z_0 | x_{1:T})
        - smoothed_seq: Shape (T, lat_dim) - p(z_t | x_{1:T}) for t = 1, ..., T
        - joints: Shape (T, joint_dim) - mean params of p(z_{t-1}, z_t | x_{1:T})
        """
        prior_params, emsn_params, trns_params = self.split_coords(params)
        trns_hrm = self.trns_hrm
        emsn_hrm = self.emsn_hrm

        trns_lkl_params, _ = trns_hrm.split_conjugated(trns_params)
        emsn_lkl_params, _ = emsn_hrm.split_conjugated(emsn_params)

        def forward_step(prior: Array, obs: Array) -> tuple[Array, Array]:
            predicted = trns_hrm.join_conjugated(trns_lkl_params, prior)
            transposed_pred = transpose_harmonium(trns_hrm, predicted)
            _, predicted_prior = trns_hrm.split_conjugated(transposed_pred)

            emsn_with_pred = emsn_hrm.join_conjugated(
                emsn_lkl_params, predicted_prior
            )
            filtered = emsn_hrm.posterior_at(emsn_with_pred, obs)
            return filtered, filtered

        _, filtered_seq = jax.lax.scan(forward_step, prior_params, observations)

        def backward_step(
            smoothed_next: Array, filtered: Array
        ) -> tuple[Array, tuple[Array, Array]]:
            joint_hrm_params = trns_hrm.join_conjugated(trns_lkl_params, filtered)
            transposed = transpose_harmonium(trns_hrm, joint_hrm_params)
            trns_t_lkl, _ = trns_hrm.split_conjugated(transposed)

            backward_joint = trns_hrm.join_conjugated(trns_t_lkl, smoothed_next)
            backward_hrm = transpose_harmonium(trns_hrm, backward_joint)
            _, smoothed_curr = trns_hrm.split_conjugated(backward_hrm)

            return smoothed_curr, (smoothed_curr, backward_hrm)

        last_filtered = filtered_seq[-1]

        _, (smoothed_rest, backward_hrms) = jax.lax.scan(
            backward_step, last_filtered, filtered_seq[:-1], reverse=True,
        )

        smoothed_seq = jnp.concatenate(
            [smoothed_rest, last_filtered[None, :]], axis=0
        )

        joint_hrm_0 = trns_hrm.join_conjugated(trns_lkl_params, prior_params)
        transposed_0 = transpose_harmonium(trns_hrm, joint_hrm_0)
        trns_t_lkl_0, _ = trns_hrm.split_conjugated(transposed_0)

        backward_joint_0 = trns_hrm.join_conjugated(trns_t_lkl_0, smoothed_seq[0])
        backward_hrm_0 = transpose_harmonium(trns_hrm, backward_joint_0)
        _, smoothed_z0 = trns_hrm.split_conjugated(backward_hrm_0)

        all_backward_hrms = jnp.concatenate(
            [backward_hrm_0[None, :], backward_hrms], axis=0
        )

        joints = jax.vmap(trns_hrm.to_mean)(all_backward_hrms)

        return smoothed_z0, smoothed_seq, joints

    def posterior_statistics(self, params: Array, observations: Array) -> Array:
        """E-step for a single sequence: expected sufficient statistics E[s | x_{1:T}].

        Returns concatenated mean params in Triple layout [prior | emission | transition],
        mirroring DifferentiableConjugated.posterior_statistics.
        """
        smoothed_z0, smoothed, joints = self.smoothing(params, observations)
        lat_man = self.lat_man
        emsn_hrm = self.emsn_hrm

        prior_means = lat_man.to_mean(smoothed_z0)

        def emission_stats(x: Array, z_nat: Array) -> Array:
            obs_stats = emsn_hrm.obs_man.sufficient_statistic(x)
            lat_means = lat_man.to_mean(z_nat)
            int_means = emsn_hrm.int_man.outer_product(obs_stats, lat_means)
            return emsn_hrm.join_coords(obs_stats, int_means, lat_means)

        emsn_means = jnp.mean(
            jax.vmap(emission_stats)(observations, smoothed), axis=0
        )

        trns_means = jnp.mean(joints, axis=0)

        return self.join_coords(prior_means, emsn_means, trns_means)

    def mean_posterior_statistics(
        self, params: Array, observations_batch: Array
    ) -> Array:
        """Batched E-step: averaged expected sufficient statistics.

        Mirrors DifferentiableConjugated.mean_posterior_statistics.
        """
        all_stats = jax.vmap(lambda obs: self.posterior_statistics(params, obs))(
            observations_batch
        )
        return jnp.mean(all_stats, axis=0)

    def to_natural(self, means: Array) -> Array:
        """Convert Triple-layout mean params to natural params.

        Delegates to each component's to_natural independently.
        Mirrors AnalyticConjugated.to_natural.
        """
        prior_means, emsn_means, trns_means = self.split_coords(means)
        return self.join_coords(
            self.lat_man.to_natural(prior_means),
            self.emsn_hrm.to_natural(emsn_means),
            self.trns_hrm.to_natural(trns_means),
        )

    def expectation_maximization(
        self, params: Array, observations_batch: Array
    ) -> Array:
        """Full EM step: E-step then M-step.

        Mirrors AnalyticConjugated.expectation_maximization:
            means = self.mean_posterior_statistics(params, observations_batch)
            return self.to_natural(means)
        """
        means = self.mean_posterior_statistics(params, observations_batch)
        return self.to_natural(means)


# =============================================================================
# Harmonium Operations
# =============================================================================


def transpose_harmonium[Lat: Analytic](
    hrm: AnalyticConjugated[Lat, Lat],
    params: Array,
) -> Array:
    """Transpose harmonium: swap observable/latent roles.

    For a symmetric conjugated harmonium over (A, B), returns parameters
    for the transposed harmonium over (B, A).

    This is critical for the backward pass - it converts p(z_t | z_{t-1})
    into backward message form p(z_{t-1} | z_t).
    """
    obs_params, int_params, lat_params = hrm.split_coords(params)
    int_params_t = hrm.int_man.transpose(int_params)
    return hrm.join_coords(lat_params, int_params_t, obs_params)


def _conjugated_forward_step[Obs: Differentiable, Lat: Analytic](
    emsn_hrm: AnalyticConjugated[Obs, Lat],
    trns_hrm: AnalyticConjugated[Lat, Lat],
    emsn_params: Array,
    trns_params: Array,
    prior: Array,
    obs: Array,
) -> tuple[Array, Array]:
    """Single forward filtering step: predict + update.

    Computes p(z_t | x_{1:t}) from p(z_{t-1} | x_{1:t-1}) and x_t.
    """
    trns_lkl_params, _ = trns_hrm.split_conjugated(trns_params)
    predicted = trns_hrm.join_conjugated(trns_lkl_params, prior)
    transposed = transpose_harmonium(trns_hrm, predicted)
    _, predicted_prior = trns_hrm.split_conjugated(transposed)

    emsn_lkl_params, _ = emsn_hrm.split_conjugated(emsn_params)
    emsn_params_with_pred = emsn_hrm.join_conjugated(emsn_lkl_params, predicted_prior)

    filtered = emsn_hrm.posterior_at(emsn_params_with_pred, obs)
    log_lik = emsn_hrm.log_observable_density(emsn_params_with_pred, obs)

    return filtered, log_lik


# =============================================================================
# Homogeneous Markov Processes
# =============================================================================


@dataclass(frozen=True)
class HomogeneousMarkovProcess[L: Differentiable](Differentiable, ABC):
    """Homogeneous fully-observed Markov chain as a flat exponential family.

    Stores a single copy of transition harmonium parameters. All operations
    use scan/vmap instead of Python recursion, giving O(1) Python overhead.

    The chain models an undirected Markov random field on a path graph with
    ``n_steps`` edges and ``n_steps + 1`` nodes. Natural parameters are a
    single copy of the transition harmonium's parameters
    ``[obs_params, int_params, lat_params]``, and the sufficient statistic is
    the sum of pairwise transition sufficient statistics.

    In the transition harmonium's convention, ``obs`` is the left node and
    ``lat`` is the right node of each edge. The leftmost node receives only
    ``obs_params`` bias, interior nodes receive ``obs_params + lat_params``,
    and the rightmost node receives only ``lat_params``.
    """

    # Fields

    n_steps: int
    trns_hrm: SymmetricConjugated[L, L]

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError("HomogeneousMarkovProcess requires n_steps >= 1.")

    # Methods

    @property
    def state_man(self) -> L:
        """State manifold (same as trns_hrm.obs_man and trns_hrm.lat_man)."""
        return self.trns_hrm.obs_man

    def reshape_trajectory(self, trajectory: Array) -> Array:
        """Reshape flat trajectory to ``(n_steps + 1, state_data_dim)``."""
        return trajectory.reshape(self.n_steps + 1, self.state_man.data_dim)

    # Overrides

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Delegate to the transition harmonium's initialization."""
        return self.trns_hrm.initialize(key, location, shape)

    @property
    @override
    def dim(self) -> int:
        return self.trns_hrm.dim

    @property
    @override
    def data_dim(self) -> int:
        return (self.n_steps + 1) * self.state_man.data_dim

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Sum of pairwise transition sufficient statistics via vmap."""
        states = self.reshape_trajectory(x)
        pairs = jnp.concatenate([states[:-1], states[1:]], axis=-1)
        pair_stats = jax.vmap(self.trns_hrm.sufficient_statistic)(pairs)
        return jnp.sum(pair_stats, axis=0)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Sum of per-state base measures."""
        states = self.reshape_trajectory(x)
        return jnp.sum(jax.vmap(self.state_man.log_base_measure)(states))

    @override
    def log_partition_function(self, params: Array) -> Array:
        """Compute log partition via forward scan (transfer matrix method).

        Integrates states left-to-right using conjugation parameters. The
        leftmost state has bias ``obs_params``, interior states accumulate
        ``obs_params + lat_params + rho``, and the rightmost state gets
        ``lat_params + rho``.
        """
        obs_params, int_params, lat_params = self.trns_hrm.split_coords(params)
        lkl_fun_man = self.trns_hrm.lkl_fun_man

        def compute_rho(f: Array) -> Array:
            lkl = lkl_fun_man.join_coords(f, int_params)
            return self.trns_hrm.conjugation_parameters(lkl)

        def interior_step(f: Array, _: None) -> tuple[Array, Array]:
            rho = compute_rho(f)
            psi = self.state_man.log_partition_function(f)
            f_next = obs_params + lat_params + rho
            return f_next, psi

        f_init = obs_params
        f_last, psi_terms = jax.lax.scan(
            interior_step, f_init, None, length=self.n_steps - 1
        )

        rho_final = compute_rho(f_last)
        psi_last = self.state_man.log_partition_function(f_last)
        psi_final = self.state_man.log_partition_function(lat_params + rho_final)

        return jnp.sum(psi_terms) + psi_last + psi_final

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Sample trajectories via forward-filtering backward-sampling.

        Computes forward messages (marginal natural params for each state),
        then samples backward from rightmost to leftmost.
        """
        obs_params, int_params, lat_params = self.trns_hrm.split_coords(params)
        lkl_fun_man = self.trns_hrm.lkl_fun_man

        def compute_rho(f: Array) -> Array:
            lkl = lkl_fun_man.join_coords(f, int_params)
            return self.trns_hrm.conjugation_parameters(lkl)

        # Compute forward messages
        def fwd_step(f: Array, _: None) -> tuple[Array, Array]:
            rho = compute_rho(f)
            f_next = obs_params + lat_params + rho
            return f_next, f

        f_init = obs_params
        f_last, f_interior = jax.lax.scan(
            fwd_step, f_init, None, length=self.n_steps - 1
        )

        # f_interior has shape (n_steps-1, dim): forward messages for
        # states 0 through n_steps-2 (the scan outputs f before transitioning).
        # f_last is the forward message for state n_steps-1.
        rho_final = compute_rho(f_last)
        f_rightmost = lat_params + rho_final

        # All forward messages ordered left-to-right: shape (n_steps+1, dim)
        all_f = jnp.concatenate([f_interior, f_last[None], f_rightmost[None]])

        def sample_one(key: Array) -> Array:
            # Sample rightmost state from marginal
            key_last, key_rest = jax.random.split(key)
            x_last = self.state_man.sample(key_last, all_f[-1], n=1)[0]

            # Backward sampling: x_{k} | x_{k+1}
            def bwd_step(
                x_right: Array, carry: tuple[Array, Array]
            ) -> tuple[Array, Array]:
                f_k, key_k = carry
                s_right = self.state_man.sufficient_statistic(x_right)
                cond_params = f_k + self.trns_hrm.int_man(int_params, s_right)
                x_k = self.state_man.sample(key_k, cond_params, n=1)[0]
                return x_k, x_k

            keys_bwd = jax.random.split(key_rest, self.n_steps)
            # Reverse the forward messages (exclude rightmost) and keys
            f_reversed = all_f[:-1][::-1]
            keys_reversed = keys_bwd[::-1]

            _, states_reversed = jax.lax.scan(
                bwd_step, x_last, (f_reversed, keys_reversed)
            )
            # states_reversed is in reverse order (x_{n-1}, ..., x_0)
            states_fwd = states_reversed[::-1]
            return jnp.concatenate([states_fwd.reshape(-1), x_last])

        keys = jax.random.split(key, n)
        return jax.vmap(sample_one)(keys)


@dataclass(frozen=True)
class AnalyticHomogeneousMarkovProcess[L: Analytic](
    HomogeneousMarkovProcess[L], Analytic, ABC
):
    """Homogeneous Markov chain with analytic mean-to-natural conversion.

    Adds ``to_natural`` (via Newton iteration on the log-partition Hessian)
    and ``negative_entropy``, enabling closed-form EM.
    """

    trns_hrm: AnalyticConjugated[L, L]

    # TODO: Can't we get rid of this and just rely on the derivative of the negative_entropy?
    @override
    def to_natural(self, means: Array) -> Array:
        """Invert ``to_mean`` via Newton-Raphson on the log-partition function.

        Uses the transition harmonium's ``to_natural`` as an initial guess
        (exact for ``n_steps == 1``) and refines with Newton steps.
        """
        init = self.trns_hrm.to_natural(means)

        def newton_step(params: Array, _: None) -> tuple[Array, None]:
            residual = self.to_mean(params) - means
            H = jax.hessian(self.log_partition_function)(params)
            delta = jnp.linalg.solve(H, residual)
            return params - delta, None

        result, _ = jax.lax.scan(newton_step, init, None, length=10)
        return result

    @override
    def negative_entropy(self, means: Array) -> Array:
        params = self.to_natural(means)
        return jnp.dot(params, means) - self.log_partition_function(params)


# =============================================================================
# Homogeneous / Recursive Conversion
# =============================================================================


def expand_homogeneous_params[L: Analytic](
    hom: AnalyticHomogeneousMarkovProcess[L],
    hom_params: Array,
) -> Array:
    """Expand homogeneous params into recursive MarkovProcess params.

    In the homogeneous chain, interior states get bias ``obs_params + lat_params``
    while the leftmost gets ``obs_params`` and rightmost gets ``lat_params``.
    The recursive chain assigns biases hierarchically: the outermost level's
    obs gets ``obs_params``, and all inner levels' obs must be ``obs_params +
    lat_params`` to match the homogeneous interior bias. The innermost lat
    remains ``lat_params``.
    """
    obs_params, int_params, lat_params = hom.trns_hrm.split_coords(hom_params)
    outer_prefix = jnp.concatenate([obs_params, int_params])
    inner_prefix = jnp.concatenate([obs_params + lat_params, int_params])
    inner_tail = hom.trns_hrm.join_coords(
        obs_params + lat_params, int_params, lat_params
    )

    if hom.n_steps == 1:
        return hom_params

    # [outer_prefix, inner_prefix * (n_steps - 2), inner_tail]
    inner_repeated = jnp.tile(inner_prefix, hom.n_steps - 2)
    return jnp.concatenate([outer_prefix, inner_repeated, inner_tail])
