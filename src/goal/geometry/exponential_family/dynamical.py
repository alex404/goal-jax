"""Dynamical exponential families: homogeneous Markov chains and transition abstractions.

This module contains:

- ``HomogeneousMarkovProcess[L]`` / ``AnalyticHomogeneousMarkovProcess[L]`` --- fully observed Markov chains on a path graph, as flat exponential families.
- ``Transition[L]`` --- a parameterized predict map on belief natural parameters, the headline primitive for filtering. Stateless or stateful; gradient-friendly for BPTT.
- ``AnalyticTransition[L]`` --- a Transition backed by a ``SymmetricConjugated`` harmonium kernel, so the predict step is derived analytically and the same parameters support smoothing and exact EM in later phases.

State-space models with hidden latents (Kalman filter, HMM, hybrid filters) compose a ``DifferentiableConjugated`` emission with a ``Transition`` and live in later phases.
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
class HomogeneousMarkovProcess[L: Differentiable](Differentiable, ABC):
    """Homogeneous fully-observed Markov chain as a flat exponential family.

    The chain models an undirected Markov random field on a path graph with ``n_steps`` edges and ``n_steps + 1`` nodes, parameterized by a single one-step transition harmonium ``trns_hrm``. Natural parameters are a single copy of the harmonium's parameters ``[obs_params, int_params, lat_params]``, and the sufficient statistic of a trajectory is the sum of the per-edge transition sufficient statistics.

    In the transition harmonium's convention, ``obs`` is the left node and ``lat`` is the right node of each edge. The leftmost node receives only ``obs_params`` bias, interior nodes receive ``obs_params + lat_params``, and the rightmost node receives only ``lat_params``.

    All operations use ``jax.lax.scan`` / ``jax.vmap`` rather than Python recursion, giving constant Python overhead in ``n_steps``.
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
        """State manifold (same as ``trns_hrm.obs_man`` and ``trns_hrm.lat_man``)."""
        return self.trns_hrm.obs_man

    def reshape_trajectory(self, trajectory: Array) -> Array:
        """Reshape a flat trajectory to ``(n_steps + 1, state_data_dim)``."""
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
        """Sum of pairwise transition sufficient statistics via ``jax.vmap``."""
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
        """Compute log partition via forward scan (transfer-matrix method).

        Integrates states left-to-right using conjugation parameters. The leftmost state has bias ``obs_params``, interior states accumulate ``obs_params + lat_params + rho``, and the rightmost state gets ``lat_params + rho``.
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

        Computes forward messages (marginal natural params for each state), then samples backward from the rightmost to leftmost node.
        """
        obs_params, int_params, lat_params = self.trns_hrm.split_coords(params)
        lkl_fun_man = self.trns_hrm.lkl_fun_man

        def compute_rho(f: Array) -> Array:
            lkl = lkl_fun_man.join_coords(f, int_params)
            return self.trns_hrm.conjugation_parameters(lkl)

        def fwd_step(f: Array, _: None) -> tuple[Array, Array]:
            rho = compute_rho(f)
            f_next = obs_params + lat_params + rho
            return f_next, f

        f_init = obs_params
        f_last, f_interior = jax.lax.scan(
            fwd_step, f_init, None, length=self.n_steps - 1
        )

        rho_final = compute_rho(f_last)
        f_rightmost = lat_params + rho_final

        # Forward messages ordered left-to-right: shape (n_steps + 1, dim)
        all_f = jnp.concatenate([f_interior, f_last[None], f_rightmost[None]])

        def sample_one(key: Array) -> Array:
            key_last, key_rest = jax.random.split(key)
            x_last = self.state_man.sample(key_last, all_f[-1], n=1)[0]

            def bwd_step(
                x_right: Array, carry: tuple[Array, Array]
            ) -> tuple[Array, Array]:
                f_k, key_k = carry
                s_right = self.state_man.sufficient_statistic(x_right)
                cond_params = f_k + self.trns_hrm.int_man(int_params, s_right)
                x_k = self.state_man.sample(key_k, cond_params, n=1)[0]
                return x_k, x_k

            keys_bwd = jax.random.split(key_rest, self.n_steps)
            f_reversed = all_f[:-1][::-1]
            keys_reversed = keys_bwd[::-1]

            _, states_reversed = jax.lax.scan(
                bwd_step, x_last, (f_reversed, keys_reversed)
            )
            states_fwd = states_reversed[::-1]
            return jnp.concatenate([states_fwd.reshape(-1), x_last])

        keys = jax.random.split(key, n)
        return jax.vmap(sample_one)(keys)


@dataclass(frozen=True)
class AnalyticHomogeneousMarkovProcess[L: Analytic](
    HomogeneousMarkovProcess[L], Analytic, ABC
):
    """Homogeneous Markov chain with analytic mean-to-natural conversion.

    Adds ``to_natural`` (via Newton iteration on the log-partition Hessian) and ``negative_entropy``, enabling closed-form EM.
    """

    trns_hrm: AnalyticConjugated[L, L]

    @override
    def to_natural(self, means: Array) -> Array:
        """Invert ``to_mean`` via Newton-Raphson on the log-partition function.

        Uses the transition harmonium's ``to_natural`` as an initial guess (exact for ``n_steps == 1``) and refines with Newton steps.
        """
        init = self.trns_hrm.to_natural(means)

        def newton_step(params: Array, _: None) -> tuple[Array, None]:
            residual = self.to_mean(params) - means
            hessian = jax.hessian(self.log_partition_function)(params)
            delta = jnp.linalg.solve(hessian, residual)
            return params - delta, None

        result, _ = jax.lax.scan(newton_step, init, None, length=10)
        return result

    @override
    def negative_entropy(self, means: Array) -> Array:
        params = self.to_natural(means)
        return jnp.dot(params, means) - self.log_partition_function(params)


@dataclass(frozen=True)
class Transition[L: Differentiable](Manifold, ABC):
    """A parameterized predict map on a latent manifold's natural parameters.

    Given the natural parameters of the current belief and an auxiliary state, returns the natural parameters of the predicted belief and the next auxiliary state. Supports both stateless maps (``init_aux`` returns an empty array) and stateful maps (an RNN's hidden state).

    The contract is just enough to drive the filter scan: the LatentProcess does ``predicted, aux = transition.predict(params, belief, aux)`` and feeds ``predicted`` into a conjugate observation update. Subclasses choose how to parameterize the predict map; ``AnalyticTransition`` derives it from a harmonium kernel, while ``MLPTransition`` (defined in the model layer) wraps a feedforward network.
    """

    # Contract

    @property
    @abstractmethod
    def lat_man(self) -> L:
        """The latent manifold whose natural parameters we predict."""

    @abstractmethod
    def init_aux(self, key: Array) -> Array:
        """Initialize the auxiliary state. Returns ``jnp.zeros(0)`` for stateless transitions."""

    @abstractmethod
    def predict(
        self, params: Array, belief: Array, aux: Array
    ) -> tuple[Array, Array]:
        """Predict next belief from current belief and auxiliary state.

        Returns ``(predicted_belief_natural_params, new_aux)``.
        """


@dataclass(frozen=True)
class AnalyticTransition[L: Analytic](Transition[L]):
    """A ``Transition`` backed by an ``AnalyticConjugated`` harmonium kernel.

    The transition's parameters are exactly the kernel harmonium's natural parameters; ``predict`` is implemented via the kernel's ``split_conjugated`` / ``join_conjugated`` and the interaction matrix's ``transpose``. Stateless: ``init_aux`` returns an empty array.

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
    def init_aux(self, key: Array) -> Array:
        return jnp.zeros(0)

    @override
    def predict(
        self, params: Array, belief: Array, aux: Array
    ) -> tuple[Array, Array]:
        kernel = self.kernel
        lkl_params, _ = kernel.split_conjugated(params)
        joined = kernel.join_conjugated(lkl_params, belief)
        obs_p, int_p, lat_p = kernel.split_coords(joined)
        int_p_t = kernel.int_man.transpose(int_p)
        transposed = kernel.join_coords(lat_p, int_p_t, obs_p)
        _, predicted = kernel.split_conjugated(transposed)
        return predicted, aux


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

    def filter(
        self,
        params: Array,
        observations: Array,
        aux_init: Array | None = None,
    ) -> tuple[Array, Array]:
        """Forward filter: returns ``(beliefs, log_likelihood)``.

        Iterates ``predict`` then conjugate-update over the observations sequence. Differentiable end-to-end: ``jax.grad`` of ``-log_likelihood`` w.r.t. (prior, emission, transition) params is BPTT through the scan.

        ``aux_init`` defaults to ``transition.init_aux(jax.random.PRNGKey(0))`` for stateless transitions; pass an explicit value for stateful transitions where the initial state matters.
        """
        prior_params, emsn_params, trns_params = self.split_coords(params)
        emsn_hrm = self.emsn_hrm
        transition = self.transition
        emsn_lkl_params, _ = emsn_hrm.split_conjugated(emsn_params)

        if aux_init is None:
            aux_init = transition.init_aux(jax.random.PRNGKey(0))

        def step(
            carry: tuple[Array, Array, Array], obs: Array
        ) -> tuple[tuple[Array, Array, Array], Array]:
            belief, aux, total_ll = carry
            predicted, new_aux = transition.predict(trns_params, belief, aux)
            emsn_with_pred = emsn_hrm.join_conjugated(emsn_lkl_params, predicted)
            posterior = emsn_hrm.posterior_at(emsn_with_pred, obs)
            log_lik = emsn_hrm.log_observable_density(emsn_with_pred, obs)
            return (posterior, new_aux, total_ll + log_lik), posterior

        (_, _, log_likelihood), beliefs = jax.lax.scan(
            step,
            (prior_params, aux_init, jnp.array(0.0)),
            observations,
        )
        return beliefs, log_likelihood

    def log_observable_density(
        self,
        params: Array,
        observations: Array,
        aux_init: Array | None = None,
    ) -> Array:
        """Log marginal $\\log p(x_{1:T})$ via forward filtering."""
        _, log_likelihood = self.filter(params, observations, aux_init)
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
