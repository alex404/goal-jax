"""Dynamical exponential families: transition abstractions and latent processes.

This module contains:

- ``Transition[L]`` --- a parameterized predict map on belief natural parameters, the headline primitive for filtering. Stateless or stateful; gradient-friendly for BPTT.
- ``AnalyticTransition[L]`` --- a Transition backed by a ``SymmetricConjugated`` harmonium kernel, so the predict step is derived analytically and the same parameters support smoothing and exact EM in later phases.
- ``LatentProcess[O, L]`` / ``AnalyticLatentProcess[O, L]`` --- state-space models composing a prior, a conjugated emission, and a transition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import Manifold
from ..manifold.combinators import Triple
from .base import Analytic, Differentiable
from .harmonium import (
    AnalyticConjugated,
    SymmetricConjugated,
)


@dataclass(frozen=True)
class Transition[L: Differentiable](Manifold, ABC):
    """A parameterized predict map on a latent manifold's natural parameters.

    Given the natural parameters of the current belief, returns the natural parameters of the predicted belief. The contract is just enough to drive the filter scan: the LatentProcess does ``predicted = transition.predict(params, belief)`` and feeds ``predicted`` into a conjugate observation update. Subclasses choose how to parameterize the predict map; ``AnalyticTransition`` derives it from a harmonium kernel, while ``MLPTransition`` (defined in the model layer) wraps a feedforward network.
    """

    # Contract

    @property
    @abstractmethod
    def lat_man(self) -> L:
        """The latent manifold whose natural parameters we predict."""

    @abstractmethod
    def predict(self, params: Array, belief: Array) -> Array:
        """Predict next belief from current belief."""


def transpose_harmonium[L: Differentiable](
    hrm: SymmetricConjugated[L, L],
    params: Array,
) -> Array:
    """Transpose a symmetric harmonium: swap observable/latent roles.

    For a symmetric conjugated harmonium over $(L, L)$, returns parameters for the same harmonium with the obs and lat slots swapped and the interaction matrix transposed. Used for the backward step of smoothing.
    """
    obs_params, int_params, lat_params = hrm.split_coords(params)
    int_params_t = hrm.int_man.transpose(int_params)
    return hrm.join_coords(lat_params, int_params_t, obs_params)


@dataclass(frozen=True)
class AnalyticTransition[L: Analytic](Transition[L]):
    """A ``Transition`` backed by an ``AnalyticConjugated`` harmonium kernel.

    The transition's parameters are exactly the kernel harmonium's natural parameters; ``predict`` is implemented via the kernel's ``split_conjugated`` / ``join_conjugated`` and the interaction matrix's ``transpose``.

    Composition (kernel as a field) rather than multiple inheritance keeps the class hierarchy clean for both the symmetric-Gaussian case (kernel is a ``NormalAnalyticLGM``) and the categorical case (kernel is a custom ``AnalyticConjugated[Categorical, Categorical]``). Subclasses may narrow the ``kernel`` field type.

    Because the kernel is the underlying joint distribution over $(z_{t-1}, z_t)$, smoothing and exact EM are available in later phases without any additional storage.
    """

    kernel: AnalyticConjugated[L, L]

    @property
    @override
    def lat_man(self) -> L:
        return self.kernel.lat_man

    @property
    @override
    def dim(self) -> int:
        return self.kernel.dim

    @override
    def predict(self, params: Array, belief: Array) -> Array:
        kernel = self.kernel
        lkl_params, _ = kernel.split_conjugated(params)
        joined = kernel.join_conjugated(lkl_params, belief)
        transposed = transpose_harmonium(kernel, joined)
        _, predicted = kernel.split_conjugated(transposed)
        return predicted


@dataclass(frozen=True)
class LatentProcess[
    O: Differentiable,
    L: Differentiable,
](
    Triple[L, SymmetricConjugated[O, L], Transition[L]],
    ABC,
):
    """A state-space model: prior over $z_0$, conjugated emission, and predict transition.

    Parameters are a ``Triple`` layout ``[prior_params, emission_params, transition_params]``. The base class supports BPTT-friendly filtering and observation-marginal log-density --- both work for any ``Transition`` (linear, MLP, RNN). Subclasses with an ``AnalyticTransition`` (i.e. ``AnalyticLatentProcess``) gain joint sampling, smoothing, and exact EM.

    Concrete subclasses narrow ``emsn_hrm`` and ``transition`` return types covariantly.
    """

    # Contract

    @property
    @abstractmethod
    def lat_man(self) -> L:
        """The latent manifold (prior over $z_0$ lives here)."""

    @property
    @abstractmethod
    def emsn_hrm(self) -> SymmetricConjugated[O, L]:
        """The emission harmonium $p(x_t \\mid z_t)$."""

    @property
    @abstractmethod
    def transition(self) -> Transition[L]:
        """The transition operator $z_{t-1} \\mapsto z_t$ (predict map)."""

    # Triple slots

    @property
    @override
    def fst_man(self) -> L:
        return self.lat_man

    @property
    @override
    def snd_man(self) -> SymmetricConjugated[O, L]:
        return self.emsn_hrm

    @property
    @override
    def trd_man(self) -> Transition[L]:
        return self.transition

    # Methods

    def filter(self, params: Array, observations: Array) -> tuple[Array, Array]:
        """Forward filter: returns ``(beliefs, log_likelihood)``.

        Iterates ``predict`` then conjugate-update over the observations sequence. Differentiable end-to-end: ``jax.grad`` of ``-log_likelihood`` w.r.t. (prior, emission, transition) params is BPTT through the scan.
        """
        prior_params, emsn_params, trns_params = self.split_coords(params)
        emsn_hrm = self.emsn_hrm
        transition = self.transition
        emsn_lkl_params, _ = emsn_hrm.split_conjugated(emsn_params)

        def step(
            carry: tuple[Array, Array], obs: Array
        ) -> tuple[tuple[Array, Array], Array]:
            belief, total_ll = carry
            predicted = transition.predict(trns_params, belief)
            emsn_with_pred = emsn_hrm.join_conjugated(emsn_lkl_params, predicted)
            posterior = emsn_hrm.posterior_at(emsn_with_pred, obs)
            log_lik = emsn_hrm.log_observable_density(emsn_with_pred, obs)
            return (posterior, total_ll + log_lik), posterior

        (_, log_likelihood), beliefs = jax.lax.scan(
            step,
            (prior_params, jnp.array(0.0)),
            observations,
        )
        return beliefs, log_likelihood

    def log_observable_density(self, params: Array, observations: Array) -> Array:
        """Log marginal $\\log p(x_{1:T})$ via forward filtering."""
        _, log_likelihood = self.filter(params, observations)
        return log_likelihood


@dataclass(frozen=True)
class AnalyticLatentProcess[
    O: Differentiable,
    L: Analytic,
](
    LatentProcess[O, L],
    ABC,
):
    """A LatentProcess with an analytic transition kernel: smoothing, joint sampling, and exact EM are available.

    The ``AnalyticTransition`` kernel provides $p(z_t, z_{t-1})$ as a symmetric conjugated harmonium, so the backward smoothing pass and the closed-form M-step both work directly.
    """

    @property
    @abstractmethod
    @override
    def emsn_hrm(self) -> AnalyticConjugated[O, L]:
        """Emission harmonium with analytic mean-to-natural conversion."""

    @property
    @abstractmethod
    @override
    def transition(self) -> AnalyticTransition[L]:
        """Transition with an analytic harmonium kernel for smoothing/EM."""

    # Methods

    def sample(
        self, key: Array, params: Array, n_steps: int
    ) -> tuple[Array, Array]:
        """Sample a trajectory $(x_{1:T}, z_{0:T})$ from the joint.

        Returns ``(observations, latents)`` of shapes ``(n_steps, obs_data_dim)`` and ``(n_steps + 1, lat_data_dim)`` respectively. Latents include $z_0$.
        """
        prior_params, emsn_params, trns_params = self.split_coords(params)
        kernel = self.transition.kernel
        emsn_hrm = self.emsn_hrm
        trns_lkl_params, _ = kernel.split_conjugated(trns_params)
        emsn_lkl_params, _ = emsn_hrm.split_conjugated(emsn_params)

        key_init, key_seq = jax.random.split(key)
        z_0 = self.lat_man.sample(key_init, prior_params, n=1)[0]

        def step(
            carry: tuple[Array, Array], _: None
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            z_prev, key = carry
            key_trans, key_emit, key_next = jax.random.split(key, 3)

            s_z_prev = kernel.lat_man.sufficient_statistic(z_prev)
            z_t_params = kernel.lkl_fun_man(trns_lkl_params, s_z_prev)
            z_t = kernel.obs_man.sample(key_trans, z_t_params, n=1)[0]

            s_z_t = emsn_hrm.pst_man.sufficient_statistic(z_t)
            x_t_params = emsn_hrm.lkl_fun_man(emsn_lkl_params, s_z_t)
            x_t = emsn_hrm.obs_man.sample(key_emit, x_t_params, n=1)[0]

            return (z_t, key_next), (x_t, z_t)

        _, (observations, latents_after_0) = jax.lax.scan(
            step, (z_0, key_seq), None, length=n_steps
        )
        latents = jnp.concatenate([z_0[None, :], latents_after_0], axis=0)
        return observations, latents

    def log_density(
        self, params: Array, observations: Array, latents: Array
    ) -> Array:
        """Log joint density $\\log p(x_{1:T}, z_{0:T})$."""
        prior_params, emsn_params, trns_params = self.split_coords(params)
        kernel = self.transition.kernel
        emsn_hrm = self.emsn_hrm
        lat_man = self.lat_man

        log_prior = lat_man.log_density(prior_params, latents[0])

        trns_lkl_params, _ = kernel.split_conjugated(trns_params)
        emsn_lkl_params, _ = emsn_hrm.split_conjugated(emsn_params)

        def log_transition(z_prev: Array, z_curr: Array) -> Array:
            s_z_prev = kernel.lat_man.sufficient_statistic(z_prev)
            z_params = kernel.lkl_fun_man(trns_lkl_params, s_z_prev)
            return kernel.obs_man.log_density(z_params, z_curr)

        log_transitions = jnp.sum(
            jax.vmap(log_transition)(latents[:-1], latents[1:])
        )

        def log_emission(z: Array, x: Array) -> Array:
            s_z = emsn_hrm.pst_man.sufficient_statistic(z)
            x_params = emsn_hrm.lkl_fun_man(emsn_lkl_params, s_z)
            return emsn_hrm.obs_man.log_density(x_params, x)

        log_emissions = jnp.sum(
            jax.vmap(log_emission)(latents[1:], observations)
        )
        return log_prior + log_transitions + log_emissions

    def smooth(
        self, params: Array, observations: Array
    ) -> tuple[Array, Array, Array]:
        """Forward-backward smoothing.

        Returns ``(smoothed_z0, smoothed_seq, joints)``:

        - ``smoothed_z0`` (lat_dim,): natural params of $p(z_0 \\mid x_{1:T})$.
        - ``smoothed_seq`` (T, lat_dim): natural params of $p(z_t \\mid x_{1:T})$ for $t=1,\\ldots,T$.
        - ``joints`` (T, joint_dim): mean params of $p(z_{t-1}, z_t \\mid x_{1:T})$ for $t=1,\\ldots,T$.
        """
        prior_params, emsn_params, trns_params = self.split_coords(params)
        kernel = self.transition.kernel
        emsn_hrm = self.emsn_hrm

        trns_lkl_params, _ = kernel.split_conjugated(trns_params)
        emsn_lkl_params, _ = emsn_hrm.split_conjugated(emsn_params)

        def forward_step(prior: Array, obs: Array) -> tuple[Array, Array]:
            predicted = kernel.join_conjugated(trns_lkl_params, prior)
            transposed_pred = transpose_harmonium(kernel, predicted)
            _, predicted_prior = kernel.split_conjugated(transposed_pred)

            emsn_with_pred = emsn_hrm.join_conjugated(emsn_lkl_params, predicted_prior)
            filtered = emsn_hrm.posterior_at(emsn_with_pred, obs)
            return filtered, filtered

        _, filtered_seq = jax.lax.scan(forward_step, prior_params, observations)

        def backward_step(
            smoothed_next: Array, filtered: Array
        ) -> tuple[Array, tuple[Array, Array]]:
            joint_hrm_params = kernel.join_conjugated(trns_lkl_params, filtered)
            transposed = transpose_harmonium(kernel, joint_hrm_params)
            trns_t_lkl, _ = kernel.split_conjugated(transposed)

            backward_joint = kernel.join_conjugated(trns_t_lkl, smoothed_next)
            backward_hrm = transpose_harmonium(kernel, backward_joint)
            _, smoothed_curr = kernel.split_conjugated(backward_hrm)

            return smoothed_curr, (smoothed_curr, backward_hrm)

        last_filtered = filtered_seq[-1]

        _, (smoothed_rest, backward_hrms) = jax.lax.scan(
            backward_step,
            last_filtered,
            filtered_seq[:-1],
            reverse=True,
        )

        smoothed_seq = jnp.concatenate(
            [smoothed_rest, last_filtered[None, :]], axis=0
        )

        joint_hrm_0 = kernel.join_conjugated(trns_lkl_params, prior_params)
        transposed_0 = transpose_harmonium(kernel, joint_hrm_0)
        trns_t_lkl_0, _ = kernel.split_conjugated(transposed_0)

        backward_joint_0 = kernel.join_conjugated(trns_t_lkl_0, smoothed_seq[0])
        backward_hrm_0 = transpose_harmonium(kernel, backward_joint_0)
        _, smoothed_z0 = kernel.split_conjugated(backward_hrm_0)

        all_backward_hrms = jnp.concatenate(
            [backward_hrm_0[None, :], backward_hrms], axis=0
        )

        joints = jax.vmap(kernel.to_mean)(all_backward_hrms)

        return smoothed_z0, smoothed_seq, joints

    def posterior_statistics(
        self, params: Array, observations: Array
    ) -> Array:
        """E-step for a single trajectory: expected sufficient statistics in mean coordinates."""
        smoothed_z0, smoothed, joints = self.smooth(params, observations)
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
        """Batched E-step: averaged expected sufficient statistics over trajectories."""
        all_stats = jax.vmap(
            lambda obs: self.posterior_statistics(params, obs)
        )(observations_batch)
        return jnp.mean(all_stats, axis=0)

    def to_natural(self, means: Array) -> Array:
        """Convert mean parameters to natural parameters.

        Delegates to each component (prior, emission, transition kernel) independently.
        """
        prior_means, emsn_means, trns_means = self.split_coords(means)
        return self.join_coords(
            self.lat_man.to_natural(prior_means),
            self.emsn_hrm.to_natural(emsn_means),
            self.transition.kernel.to_natural(trns_means),
        )

    def expectation_maximization(
        self, params: Array, observations_batch: Array
    ) -> Array:
        """One full EM step: compute the expected sufficient statistics and convert to natural parameters."""
        means = self.mean_posterior_statistics(params, observations_batch)
        return self.to_natural(means)
