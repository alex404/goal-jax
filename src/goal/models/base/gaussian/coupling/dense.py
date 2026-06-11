"""Dense-coupling kernels: enumeration and Gibbs sampling for ``FullBoltzmann``.

Free functions over the packed upper-triangular parameter vector of a dense
symmetric coupling matrix on $n$ binary units --- the algorithmic backend of
:class:`~goal.models.base.gaussian.boltzmann.CouplingMatrix`, mirroring how
:mod:`~goal.models.base.gaussian.coupling.sum_product` backs the chordal
classes. :func:`dense_log_partition` is exact via enumeration over all $2^n$
states; :func:`dense_sample` is approximate via Gibbs sweeps with burn-in and
thinning.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ...categorical import Bernoulli


def dense_states(n: int) -> Array:
    """All $2^n$ binary states via bit manipulation, LSB-first."""
    idx = jnp.arange(1 << n, dtype=jnp.uint32)
    return ((idx[:, None] >> jnp.arange(n)) & 1).astype(jnp.float32)


def dense_log_partition(n: int, params: Array) -> Array:
    """Exact $\\log Z$ via enumeration over all $2^n$ states.

    Energies as a dense quadratic form: scattering the packed parameters into
    a symmetric matrix $M$ with $M_{ii} = \\theta_{ii}$ and $M_{ij} = M_{ji} =
    \\theta_{ij} / 2$ gives $x^T M x = \\sum_i \\theta_{ii} x_i + \\sum_{i<j}
    \\theta_{ij} x_i x_j$ on binary states.
    """
    rows, cols = np.triu_indices(n)
    m = jnp.zeros((n, n)).at[rows, cols].set(params).at[cols, rows].add(params) / 2.0
    states = dense_states(n)
    energies = jnp.einsum("si,ij,sj->s", states, m, states)
    return jax.scipy.special.logsumexp(energies)


def dense_unit_conditional_prob(
    n: int, params: Array, state: Array, unit_idx: Array | int
) -> Array:
    """$P(x_k = 1 \\mid x_{\\setminus k})$ for a single unit --- $O(n)$.

    The conditional logit is $\\theta_{kk} + \\sum_{j \\neq k} \\theta_{kj}
    x_j$, extracted from the packed upper-triangular storage without building
    the full sufficient statistics.
    """
    k = jnp.asarray(unit_idx)

    # Flat index for (i, j) with i <= j in upper-triangular storage:
    # i * n - i * (i + 1) / 2 + j.
    def get_triu_idx(i: Array, j: Array) -> Array:
        return i * n - i * (i + 1) // 2 + j

    # For each j, the parameter \theta_kj (symmetric: \theta_kj = \theta_jk):
    # element (k, j) lives in row k when j >= k, in row j otherwise.
    j = jnp.arange(n)
    idx = jnp.where(j >= k, get_triu_idx(k, j), get_triu_idx(j, k))
    theta_k = params[idx]

    # Logit = \theta_kk + sum_{j != k} \theta_kj x_j. The dot term includes
    # the j = k contribution \theta_kk x_k, and \theta_kk (1 - x_k) tops it
    # up to the full \theta_kk --- so the result is independent of the
    # current x_k without masking the state array.
    energy_diff = theta_k[k] * (1 - state[k]) + jnp.dot(theta_k, state)
    return Bernoulli().to_mean(jnp.atleast_1d(energy_diff))[0]


def dense_gibbs_step(n: int, params: Array, state: Array, key: Array) -> Array:
    """Single Gibbs sweep updating all $n$ units in random order."""
    perm_key, update_key = jax.random.split(key)
    perm = jax.random.permutation(perm_key, n)

    def update_unit(state: Array, unit_idx: Array) -> tuple[Array, None]:
        prob = dense_unit_conditional_prob(n, params, state, unit_idx)
        subkey = jax.random.fold_in(update_key, unit_idx)
        new_val = jax.random.bernoulli(subkey, prob).astype(state.dtype)
        return state.at[unit_idx].set(new_val), None

    final_state, _ = jax.lax.scan(update_unit, state, perm)
    return final_state


def dense_sample(
    n: int,
    params: Array,
    key: Array,
    n_samples: int = 1,
    n_burnin: int = 1000,
    n_thin: int = 10,
) -> Array:
    """Approximate i.i.d. samples via Gibbs sweeps with burn-in and thinning."""
    init_key, sample_key = jax.random.split(key)
    init_state = jax.random.bernoulli(init_key, 0.5, shape=(n,)).astype(jnp.float32)

    def burn_step(state: Array, step: Array) -> tuple[Array, None]:
        subkey = jax.random.fold_in(sample_key, step)
        return dense_gibbs_step(n, params, state, subkey), None

    burned_state, _ = jax.lax.scan(burn_step, init_state, jnp.arange(n_burnin))

    def sample_with_thinning(state: Array, step: Array) -> tuple[Array, Array]:
        def thin_step(i: int, current_state: Array) -> Array:
            subkey = jax.random.fold_in(sample_key, n_burnin + step * n_thin + i)
            return dense_gibbs_step(n, params, current_state, subkey)

        final_state = jax.lax.fori_loop(0, n_thin, thin_step, state)
        return final_state, final_state

    _, samples = jax.lax.scan(sample_with_thinning, burned_state, jnp.arange(n_samples))
    return samples
