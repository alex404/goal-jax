"""Dynamical exponential families: transition abstractions and latent processes.

This module contains:

- ``AnalyticTransition[L]`` --- a ``Map[L, L]`` backed by an ``AnalyticConjugated`` harmonium kernel, so predict is derived analytically and the same parameters support smoothing and exact EM in later phases.
- ``LatentProcess[O, L]`` / ``AnalyticLatentProcess[O, L]`` --- state-space models composing a prior, a conjugated emission, and a transition. The transition slot is any ``Map[L, L]``; a ``MultilayerPerceptron[L, L]`` plugs in directly for hybrid filters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.combinators import Triple
from ..manifold.map import AffineMap, Map
from .base import Analytic, Differentiable
from .harmonium import (
    AnalyticConjugated,
    SymmetricConjugated,
)


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
class AnalyticTransition[L: Analytic](Map[L, L]):
    """A ``Map[L, L]`` backed by an ``AnalyticConjugated`` harmonium kernel.

    The transition's parameters are the kernel's *likelihood* natural parameters (``kernel.lkl_fun_man.dim`` = ``obs_man.dim + int_man.dim``), not the full harmonium. The kernel's own latent-prior block is structurally redundant for a transition --- the conjugate predict step always supplies the prior from the incoming belief --- so it is omitted from storage. ``__call__`` rebuilds a joint harmonium on the fly via ``join_conjugated`` to apply the transpose-and-marginalize update.

    Composition (kernel as a field) rather than multiple inheritance keeps the class hierarchy clean for both the symmetric-Gaussian case (kernel is a ``NormalAnalyticLGM``) and the categorical case (kernel is a custom ``AnalyticConjugated[Categorical, Categorical]``). Subclasses may narrow the ``kernel`` field type.

    Because the kernel encodes the joint distribution over $(z_{t-1}, z_t)$ when paired with a belief, smoothing and exact EM are available without any additional storage; the M-step uses ``kernel.to_natural_likelihood`` so the result lands in this likelihood-only parameter space.
    """

    kernel: AnalyticConjugated[L, L]

    @property
    @override
    def dom_man(self) -> L:
        return self.kernel.lat_man

    @property
    @override
    def cod_man(self) -> L:
        return self.kernel.lat_man

    @property
    @override
    def dim(self) -> int:
        return self.kernel.lkl_fun_man.dim

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        kernel = self.kernel
        joined = kernel.join_conjugated(f_coords, v_coords)
        transposed = transpose_harmonium(kernel, joined)
        _, predicted = kernel.split_conjugated(transposed)
        return predicted

    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize the transition's likelihood natural parameters.

        Generates a full kernel harmonium initialization and extracts its likelihood part, so the obs-bias / interaction-matrix scaling conventions of ``Harmonium.initialize`` are inherited verbatim.
        """
        full_params = self.kernel.initialize(key, location, shape)
        return self.kernel.likelihood_function(full_params)


@dataclass(frozen=True)
class LatentProcess[
    O: Differentiable,
    L: Differentiable,
](
    Triple[L, AffineMap[L, O], Map[L, L]],
    ABC,
):
    """A state-space model: prior over $z_0$, conjugated emission, and predict transition.

    Parameters are a ``Triple`` layout ``[prior_params, emission_likelihood_params, transition_params]``. Both compound slots store only the conditional (likelihood) natural parameters --- ``ems_hrm.lkl_fun_man.dim`` for the emission and (for ``AnalyticTransition``) ``kernel.lkl_fun_man.dim`` for the transition --- so the parameter vector matches the model's actual degrees of freedom; structurally redundant prior blocks of the underlying harmoniums are not stored. The base class supports BPTT-friendly filtering and observation-marginal log-density --- both work for any ``Map[L, L]`` (linear, MLP, RNN). Subclasses with an ``AnalyticTransition`` (i.e. ``AnalyticLatentProcess``) gain joint sampling, smoothing, and exact EM.

    Concrete subclasses narrow ``ems_hrm`` and ``trn_map`` return types covariantly.
    """

    # Contract

    @property
    @abstractmethod
    def lat_man(self) -> L:
        """The latent manifold (prior over $z_0$ lives here)."""

    @property
    @abstractmethod
    def ems_hrm(self) -> SymmetricConjugated[O, L]:
        """The emission harmonium $p(x_t \\mid z_t)$.

        Returns the full symmetric conjugated harmonium used by forward methods (``join_conjugated``, ``posterior_at``, ``log_observable_density``) and by EM. The stored slot is just its ``lkl_fun_man`` --- see ``snd_man``.
        """

    @property
    @abstractmethod
    def trn_map(self) -> Map[L, L]:
        """The transition operator $z_{t-1} \\mapsto z_t$ (predict map)."""

    # Triple slots

    @property
    @override
    def fst_man(self) -> L:
        return self.lat_man

    @property
    @override
    def snd_man(self) -> AffineMap[L, O]:
        """The emission *likelihood* manifold (``ems_hrm.lkl_fun_man``), sized to the bias + interaction parameters only."""
        return self.ems_hrm.lkl_fun_man

    @property
    @override
    def trd_man(self) -> Map[L, L]:
        return self.trn_map

    # Methods

    def filter(self, params: Array, observations: Array) -> tuple[Array, Array]:
        """Forward filter: returns ``(beliefs, log_likelihood)``.

        Iterates ``predict`` then conjugate-update over the observations sequence. Differentiable end-to-end: ``jax.grad`` of ``-log_likelihood`` w.r.t. (prior, emission, transition) params is BPTT through the scan.
        """
        prior_params, ems_lkl_params, trns_params = self.split_coords(params)
        ems_hrm = self.ems_hrm
        trn_map = self.trn_map

        def step(
            carry: tuple[Array, Array], obs: Array
        ) -> tuple[tuple[Array, Array], Array]:
            belief, total_ll = carry
            predicted = trn_map(trns_params, belief)
            ems_with_pred = ems_hrm.join_conjugated(ems_lkl_params, predicted)
            posterior = ems_hrm.posterior_at(ems_with_pred, obs)
            log_lik = ems_hrm.log_observable_density(ems_with_pred, obs)
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
    def ems_hrm(self) -> AnalyticConjugated[O, L]:
        """Emission harmonium with analytic mean-to-natural conversion."""

    @property
    @abstractmethod
    @override
    def trn_map(self) -> AnalyticTransition[L]:
        """Transition with an analytic harmonium kernel for smoothing/EM."""

    # Methods

    def sample(self, key: Array, params: Array, n_steps: int) -> tuple[Array, Array]:
        """Sample a trajectory $(x_{1:T}, z_{0:T})$ from the joint.

        Returns ``(observations, latents)`` of shapes ``(n_steps, obs_data_dim)`` and ``(n_steps + 1, lat_data_dim)`` respectively. Latents include $z_0$.
        """
        prior_params, ems_lkl_params, trns_lkl_params = self.split_coords(params)
        kernel = self.trn_map.kernel
        ems_hrm = self.ems_hrm

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

            s_z_t = ems_hrm.pst_man.sufficient_statistic(z_t)
            x_t_params = ems_hrm.lkl_fun_man(ems_lkl_params, s_z_t)
            x_t = ems_hrm.obs_man.sample(key_emit, x_t_params, n=1)[0]

            return (z_t, key_next), (x_t, z_t)

        _, (observations, latents_after_0) = jax.lax.scan(
            step, (z_0, key_seq), None, length=n_steps
        )
        latents = jnp.concatenate([z_0[None, :], latents_after_0], axis=0)
        return observations, latents

    def log_density(self, params: Array, observations: Array, latents: Array) -> Array:
        """Log joint density $\\log p(x_{1:T}, z_{0:T})$."""
        prior_params, ems_lkl_params, trns_lkl_params = self.split_coords(params)
        kernel = self.trn_map.kernel
        ems_hrm = self.ems_hrm
        lat_man = self.lat_man

        log_prior = lat_man.log_density(prior_params, latents[0])

        def log_transition(z_prev: Array, z_curr: Array) -> Array:
            s_z_prev = kernel.lat_man.sufficient_statistic(z_prev)
            z_params = kernel.lkl_fun_man(trns_lkl_params, s_z_prev)
            return kernel.obs_man.log_density(z_params, z_curr)

        log_transitions = jnp.sum(jax.vmap(log_transition)(latents[:-1], latents[1:]))

        def log_emission(z: Array, x: Array) -> Array:
            s_z = ems_hrm.pst_man.sufficient_statistic(z)
            x_params = ems_hrm.lkl_fun_man(ems_lkl_params, s_z)
            return ems_hrm.obs_man.log_density(x_params, x)

        log_emissions = jnp.sum(jax.vmap(log_emission)(latents[1:], observations))
        return log_prior + log_transitions + log_emissions

    def smooth(self, params: Array, observations: Array) -> tuple[Array, Array, Array]:
        """Forward-backward smoothing.

        Returns ``(smoothed_z0, smoothed_seq, joints)``:

        - ``smoothed_z0`` (lat_dim,): natural params of $p(z_0 \\mid x_{1:T})$.
        - ``smoothed_seq`` (T, lat_dim): natural params of $p(z_t \\mid x_{1:T})$ for $t=1,\\ldots,T$.
        - ``joints`` (T, joint_dim): mean params of $p(z_{t-1}, z_t \\mid x_{1:T})$ for $t=1,\\ldots,T$.
        """
        prior_params, ems_lkl_params, trns_lkl_params = self.split_coords(params)
        kernel = self.trn_map.kernel
        ems_hrm = self.ems_hrm

        def predict_transpose(prior: Array) -> tuple[Array, Array]:
            """Returns ``(transposed_transition_likelihood, predicted_next_prior)``."""
            joined = kernel.join_conjugated(trns_lkl_params, prior)
            transposed = transpose_harmonium(kernel, joined)
            return kernel.split_conjugated(transposed)

        def forward_step(prior: Array, obs: Array) -> tuple[Array, Array]:
            _, predicted_prior = predict_transpose(prior)
            ems_with_pred = ems_hrm.join_conjugated(ems_lkl_params, predicted_prior)
            filtered = ems_hrm.posterior_at(ems_with_pred, obs)
            return filtered, filtered

        _, filtered_seq = jax.lax.scan(forward_step, prior_params, observations)

        # Treat z_0's "filtered" belief as the prior itself (no observation), so the
        # backward scan covers z_0 -> z_T uniformly with no boundary special-case.
        filtered_with_prior = jnp.concatenate(
            [prior_params[None, :], filtered_seq], axis=0
        )

        def backward_step(
            smoothed_next: Array, filtered: Array
        ) -> tuple[Array, tuple[Array, Array]]:
            trns_t_lkl, _ = predict_transpose(filtered)
            backward_joint = kernel.join_conjugated(trns_t_lkl, smoothed_next)
            backward_hrm = transpose_harmonium(kernel, backward_joint)
            _, smoothed_curr = kernel.split_conjugated(backward_hrm)
            return smoothed_curr, (smoothed_curr, backward_hrm)

        last_filtered = filtered_with_prior[-1]

        _, (smoothed_rest, backward_hrms) = jax.lax.scan(
            backward_step,
            last_filtered,
            filtered_with_prior[:-1],
            reverse=True,
        )

        smoothed_full = jnp.concatenate(
            [smoothed_rest, last_filtered[None, :]], axis=0
        )
        smoothed_z0 = smoothed_full[0]
        smoothed_seq = smoothed_full[1:]
        joints = jax.vmap(kernel.to_mean)(backward_hrms)

        return smoothed_z0, smoothed_seq, joints

    def posterior_statistics(self, params: Array, observations: Array) -> Array:
        """E-step for a single trajectory: expected sufficient statistics in mean coordinates.

        Layout is ``[prior_means(lat_man.dim) | ems_means(ems_hrm.dim) | trns_means(kernel.dim)]``. Both compound slots hold the **full harmonium** mean (emission joint over $(x_t, z_t)$ and transition joint over $(z_{t-1}, z_t)$) --- not the likelihood-only natural-parameter layout --- because each M-step ``to_natural_likelihood`` requires the joint sufficient statistics. ``to_natural`` consumes this layout and produces the smaller natural-parameter vector.
        """
        smoothed_z0, smoothed, joints = self.smooth(params, observations)
        lat_man = self.lat_man
        ems_hrm = self.ems_hrm

        prior_means = lat_man.to_mean(smoothed_z0)

        def emission_stats(x: Array, z_nat: Array) -> Array:
            obs_stats = ems_hrm.obs_man.sufficient_statistic(x)
            lat_means = lat_man.to_mean(z_nat)
            int_means = ems_hrm.int_man.outer_product(obs_stats, lat_means)
            return ems_hrm.join_coords(obs_stats, int_means, lat_means)

        ems_means = jnp.mean(jax.vmap(emission_stats)(observations, smoothed), axis=0)

        trns_means = jnp.mean(joints, axis=0)
        return jnp.concatenate([prior_means, ems_means, trns_means])

    def mean_posterior_statistics(
        self, params: Array, observations_batch: Array
    ) -> Array:
        """Batched E-step: averaged expected sufficient statistics over trajectories.

        Output uses the same full-joint transition layout as ``posterior_statistics`` --- average first, then call ``to_natural`` to invert. Averaging *after* the nonlinear M-step would not correspond to a valid EM update.
        """
        all_stats = jax.vmap(lambda obs: self.posterior_statistics(params, obs))(
            observations_batch
        )
        return jnp.mean(all_stats, axis=0)

    def to_natural(self, means: Array) -> Array:
        """M-step: convert expected sufficient statistics to natural parameters.

        ``means`` follows the ``posterior_statistics`` layout (full harmonium mean in both the emission and transition slots); the result follows the model's natural-parameter layout (likelihood-only slots, sized ``ems_hrm.lkl_fun_man.dim`` and ``kernel.lkl_fun_man.dim``). The prior slot uses the standard ``to_natural``; the emission and transition both use ``to_natural_likelihood`` since their kernel prior blocks aren't stored.
        """
        lat_dim = self.lat_man.dim
        ems_dim = self.ems_hrm.dim
        prior_means = means[:lat_dim]
        ems_means = means[lat_dim : lat_dim + ems_dim]
        trns_means = means[lat_dim + ems_dim :]
        return self.join_coords(
            self.lat_man.to_natural(prior_means),
            self.ems_hrm.to_natural_likelihood(ems_means),
            self.trn_map.kernel.to_natural_likelihood(trns_means),
        )

    def expectation_maximization(
        self, params: Array, observations_batch: Array
    ) -> Array:
        """One full EM step: compute the expected sufficient statistics and convert to natural parameters."""
        means = self.mean_posterior_statistics(params, observations_batch)
        return self.to_natural(means)
