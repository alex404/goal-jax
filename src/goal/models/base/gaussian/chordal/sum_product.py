"""Sum-product inference on a junction tree for chordal binary graphical models.

Two public functions: ``jt_log_partition`` computes $\\log Z$ via a single collect
pass (leaves to root), and ``jt_sample`` draws one exact ancestral sample via
collect + walk-down. Both take a static :class:`JunctionTree` and a parameter
vector with layout ``[diagonal (n,), off-diagonal (n_chordal_edges,)]``.

Implementation: everything inside the inference is shape-fixed and runs through
``jax.lax.scan`` over the precomputed static topology arrays on the
:class:`JunctionTree`. There are no Python loops over cliques or tree edges, so
JIT compile time is constant in ``n_cliques`` (only depends on
``max_clique_size`` and ``max_separator_size``). Inference cost remains
$O(n \\cdot 2^{w+1})$ where $w$ is the treewidth.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .junction_tree import JunctionTree

### Helpers ###


def _segment_logsumexp(values: Array, segment_ids: Array, num_segments: int) -> Array:
    """``logsumexp`` over groups defined by ``segment_ids``.

    Numerically stable: subtracts the per-segment max before exponentiating.
    Padded clique states (carrying ``-inf`` values) get absorbed into segments
    whose max is finite, so the result is correct as long as every output
    segment has at least one finite contribution — guaranteed by construction of
    the separator-state map in :meth:`JunctionTree.collect_step_arrays`.
    """
    seg_max = jax.ops.segment_max(values, segment_ids, num_segments=num_segments)
    shifted = values - seg_max[segment_ids]
    seg_sum = jax.ops.segment_sum(
        jnp.exp(shifted), segment_ids, num_segments=num_segments
    )
    return seg_max + jnp.log(seg_sum)


### Clique potentials ###


def _clique_potentials(
    jt: JunctionTree, diag_params: Array, off_diag_params: Array
) -> Array:
    """Per-clique log-potential array of shape ``(n_cliques, n_clique_states)``.

    Computed by vmapping a single-clique potential builder over the leading
    clique axis. Padded states ($s \\geq 2^{|C|}$) are set to $-\\infty$ so they
    fall out of the subsequent ``logsumexp``-based collect pass.
    """
    bits = jnp.asarray(jt.bits_table)  # (n_clique_states, max_clique_size)
    bias_idx, bias_pos = jt.owned_bias_arrays
    edge_idx, edge_pi, edge_pj = jt.owned_edge_arrays
    bias_idx = jnp.asarray(bias_idx)
    bias_pos = jnp.asarray(bias_pos)
    edge_idx = jnp.asarray(edge_idx)
    edge_pi = jnp.asarray(edge_pi)
    edge_pj = jnp.asarray(edge_pj)
    state_mask = jnp.asarray(jt.clique_state_mask)

    # Pad diag/off-diag params with a leading zero so sentinel -1 gathers a 0
    # contribution; equivalent to mask-and-zero but cheaper (one gather, no mask).
    diag_padded = jnp.concatenate([jnp.array([0.0]), diag_params])
    if off_diag_params.size > 0:
        off_padded = jnp.concatenate([jnp.array([0.0]), off_diag_params])
    else:
        off_padded = jnp.array([0.0])

    def one_clique(
        b_idx: Array, b_pos: Array, e_idx: Array, e_i: Array, e_j: Array
    ) -> Array:
        # +1 shift so sentinel -1 lands on the leading zero.
        b_vals = diag_padded[b_idx + 1]  # (max_owned_biases,)
        # bits[:, b_pos] has shape (n_clique_states, max_owned_biases).
        phi_b = jnp.sum(bits[:, b_pos] * b_vals[None, :], axis=1)
        e_vals = off_padded[e_idx + 1]  # (max_owned_edges,)
        phi_e = jnp.sum(
            bits[:, e_i] * bits[:, e_j] * e_vals[None, :], axis=1
        )
        return phi_b + phi_e

    phi_all = jax.vmap(one_clique)(bias_idx, bias_pos, edge_idx, edge_pi, edge_pj)
    return jnp.where(state_mask, phi_all, -jnp.inf)


### Collect pass ###


def _collect(
    jt: JunctionTree, diag_params: Array, off_diag_params: Array
) -> Array:
    """Run the leaves-to-root collect pass via ``lax.scan``.

    Returns the per-clique log-totals (initial potential plus all incoming child
    messages) as an array of shape ``(n_cliques, n_clique_states)``. Padded
    clique states carry ``-inf``.
    """
    totals0 = _clique_potentials(jt, diag_params, off_diag_params)
    if not jt.collect_order:
        return totals0

    child_idx_np, parent_idx_np, sep_maps_np = jt.collect_step_arrays
    child_idx = jnp.asarray(child_idx_np)
    parent_idx = jnp.asarray(parent_idx_np)
    sep_maps = jnp.asarray(sep_maps_np)  # (n_tree_edges, 2, n_clique_states)
    n_sep_states = jt.n_separator_states

    def step(totals: Array, edge_info: tuple[Array, Array, Array]) -> tuple[Array, None]:
        child, parent, sep_pair = edge_info
        c_to_sep = sep_pair[0]  # (n_clique_states,)
        p_to_sep = sep_pair[1]
        phi_child = totals[child]
        msg = _segment_logsumexp(phi_child, c_to_sep, n_sep_states)
        msg_broadcast = msg[p_to_sep]
        new_totals = totals.at[parent].add(msg_broadcast)
        return new_totals, None

    totals_final, _ = jax.lax.scan(step, totals0, (child_idx, parent_idx, sep_maps))
    return totals_final


### Public API ###


def jt_log_partition(
    jt: JunctionTree, diag_params: Array, off_diag_params: Array
) -> Array:
    """Compute $\\log Z$ for the chordal Boltzmann distribution by sum-product."""
    totals = _collect(jt, diag_params, off_diag_params)
    return jax.scipy.special.logsumexp(totals[jt.root])


def jt_sample(
    jt: JunctionTree,
    diag_params: Array,
    off_diag_params: Array,
    key: Array,
) -> Array:
    """Draw a single exact sample from the chordal Boltzmann distribution.

    Pipeline: collect pass to get per-clique log-totals, then walk down the
    rooted tree in pre-order sampling the root clique unconditionally and each
    child clique conditional on the values already drawn for the separator with
    its parent. Both passes run as ``lax.scan``s with shape-fixed bodies.
    """
    totals = _collect(jt, diag_params, off_diag_params)

    ws = jt.walk_step_arrays
    clique_idx_arr = jnp.asarray(ws["clique_idx"])
    sep_map_arr = jnp.asarray(ws["sep_map"])
    sep_global_vars_arr = jnp.asarray(ws["sep_global_vars"])
    sep_size_arr = jnp.asarray(ws["sep_size"])
    clique_global_vars_arr = jnp.asarray(ws["clique_global_vars"])
    clique_size_arr = jnp.asarray(ws["clique_size"])

    n = jt.n_neurons
    m = jt.max_clique_size
    state_mask_arr = jnp.asarray(jt.clique_state_mask)
    bits_arr = jnp.asarray(jt.bits_table)
    m_range = jnp.arange(m, dtype=jnp.int32)

    init_x = jnp.zeros(n, dtype=jnp.float32)
    init_set_mask = jnp.zeros(n, dtype=jnp.bool_)
    init_carry = (init_x, init_set_mask, key)

    def step(
        carry: tuple[Array, Array, Array],
        step_info: tuple[Array, Array, Array, Array, Array, Array],
    ) -> tuple[tuple[Array, Array, Array], None]:
        x, set_mask, key = carry
        (
            c_idx,
            sep_map,
            sep_global_vars,
            sep_size,
            clique_global_vars,
            clique_size,
        ) = step_info
        logits = totals[c_idx]  # (n_clique_states,)
        # Mask to states consistent with already-sampled separator values
        # (no-op for the root, where sep_size == 0 and target stays 0).
        sep_mask = m_range < sep_size  # (max_clique_size,)
        sep_vals = x[sep_global_vars].astype(jnp.int32)  # (max_clique_size,)
        # Pack only valid separator bits.
        target = jnp.sum(
            jnp.where(sep_mask, sep_vals, 0) << m_range,
            dtype=jnp.int32,
        )
        consistent = (sep_map == target) | (sep_size == 0)
        valid = state_mask_arr[c_idx]
        logits = jnp.where(consistent & valid, logits, -jnp.inf)

        key, sub = jax.random.split(key)
        s = jax.random.categorical(sub, logits)
        # Decode sampled state into the clique's variables, writing only
        # entries that are not yet set (separator entries are already set).
        sampled_bits = bits_arr[s].astype(jnp.float32)  # (max_clique_size,)
        # Sequential update over clique-local positions via fori_loop. Sequential
        # semantics avoid the duplicate-index race that a vectorised
        # ``.at[target_vars].set`` would have when padded slots collide on the
        # default index 0.
        def body(i: Array, args: tuple[Array, Array]) -> tuple[Array, Array]:
            x_cur, mask_cur = args
            v = clique_global_vars[i]
            in_clique = i < clique_size
            already_set = mask_cur[v]
            new_val = jnp.where(
                in_clique & ~already_set, sampled_bits[i], x_cur[v]
            )
            x_new = x_cur.at[v].set(new_val)
            mask_new = mask_cur.at[v].set(mask_cur[v] | in_clique)
            return x_new, mask_new

        x, set_mask = jax.lax.fori_loop(0, m, body, (x, set_mask))
        return (x, set_mask, key), None

    (final_x, _, _), _ = jax.lax.scan(
        step,
        init_carry,
        (
            clique_idx_arr,
            sep_map_arr,
            sep_global_vars_arr,
            sep_size_arr,
            clique_global_vars_arr,
            clique_size_arr,
        ),
    )
    return final_x
