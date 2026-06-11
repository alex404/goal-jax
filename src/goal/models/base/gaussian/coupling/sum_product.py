"""Sum-product inference on a junction tree for chordal binary graphical models.

Four public functions, all taking static topology and a parameter vector with
layout ``[diagonal (n,), off-diagonal (n_chordal_edges,)]``: ``jt_log_partition``
computes $\\log Z$ via a single sequential collect pass (leaves to root) and
``jt_sample`` draws one exact ancestral sample via collect + walk-down, both for
any junction tree; ``chain_log_partition`` and ``chain_sample`` are their
:class:`ChainTree` counterparts built on ``associative_scan`` ($O(\\log n)$
depth --- up to two orders of magnitude faster on accelerators for long chains
with small separators; measured ~100x for $\\nabla \\log Z$ at $n \\approx
1000$).

Implementation: all static index tables (binary state tables, separator-state
maps, ownership and traversal arrays) are built with numpy at trace time from
the junction-tree topology --- under ``jax.jit`` this runs once per compilation,
so the compilation cache is the only cache. The parameter-dependent arithmetic
runs through ``jax.lax.scan`` over those tables, so the staged program is not
unrolled over cliques (compile time still grows with the embedded table sizes,
but not with program length); inference cost is $O(n \\cdot 2^{w+1})$ where $w$
is the treewidth.

State encoding is LSB-first throughout: bit $p$ of a clique state is the value
of the clique's $p$-th node (in sorted node order), and separator states pack
the separator's bits in the same convention.
"""

from __future__ import annotations

from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .junction_tree import ChainTree, JunctionTree

### Static table builders (numpy, run at trace time) ###


def _bits(m: int) -> np.ndarray:
    """``(2^m, m)`` table: row $s$ is the LSB-first binary expansion of $s$."""
    states = np.arange(1 << m, dtype=np.int64)
    return ((states[:, None] >> np.arange(m)) & 1).astype(np.int32)


def _sep_state_map(
    clique: tuple[int, ...], separator: tuple[int, ...], m: int
) -> np.ndarray:
    """Map each (padded) clique state to its separator state.

    For a real state $s < 2^{|C|}$ the separator state packs the bits of $s$ at
    the separator nodes' local positions, LSB-first. Padded states ($s \\geq
    2^{|C|}$) map to separator state $0$, which always has at least one real
    contribution, so segment reductions never see an all-padded segment.
    """
    result = np.zeros(1 << m, dtype=np.int32)
    if not separator:
        return result
    states = np.arange(1 << len(clique), dtype=np.int64)
    sep_state = np.zeros_like(states)
    for i, v in enumerate(separator):
        sep_state |= ((states >> clique.index(v)) & 1) << i
    result[: 1 << len(clique)] = sep_state.astype(np.int32)
    return result


def _ownership_tables(
    jt: JunctionTree,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Padded per-clique parameter-ownership tables.

    Returns ``(bias_idx, bias_pos, edge_idx, edge_pi, edge_pj)``; the ``*_idx``
    arrays index into the parameter vector with sentinel ``-1`` marking padded
    slots, and the ``*_pos`` / ``*_p{i,j}`` arrays give clique-local positions.
    """
    nc, m = jt.n_cliques, jt.max_clique_size
    cliques = jt.cliques

    bias_idx = np.full((nc, m), -1, dtype=np.int32)
    bias_pos = np.zeros((nc, m), dtype=np.int32)
    slots = [0] * nc
    for v, c in enumerate(jt.bias_owner):
        bias_idx[c, slots[c]] = v
        bias_pos[c, slots[c]] = cliques[c].index(v)
        slots[c] += 1

    owned: list[list[tuple[int, int, int]]] = [[] for _ in range(nc)]
    edge_owner = jt.edge_owner
    for k, (vi, vj) in enumerate(jt.chordal_edges):
        c = edge_owner[k]
        owned[c].append((k, cliques[c].index(vi), cliques[c].index(vj)))
    width = max((len(o) for o in owned), default=0)
    edge_idx = np.full((nc, width), -1, dtype=np.int32)
    edge_pi = np.zeros((nc, width), dtype=np.int32)
    edge_pj = np.zeros((nc, width), dtype=np.int32)
    for c, edges_in_c in enumerate(owned):
        for slot, (k, pi, pj) in enumerate(edges_in_c):
            edge_idx[c, slot] = k
            edge_pi[c, slot] = pi
            edge_pj[c, slot] = pj
    return bias_idx, bias_pos, edge_idx, edge_pi, edge_pj


def _state_mask(jt: JunctionTree) -> np.ndarray:
    """``(n_cliques, 2^m)`` bool: True iff the state is a real assignment for the clique."""
    sizes = np.array([len(c) for c in jt.cliques], dtype=np.int64)
    states = np.arange(1 << jt.max_clique_size, dtype=np.int64)
    return states[None, :] < (1 << sizes[:, None])


### Traced computations ###


def _segment_logsumexp(values: Array, segment_ids: Array, num_segments: int) -> Array:
    """``logsumexp`` over groups defined by ``segment_ids``.

    Numerically stable: subtracts the per-segment max before exponentiating.
    Padded states carry ``-inf`` and are absorbed into segments whose max is
    finite --- guaranteed by the separator-state map construction.
    """
    seg_max = jax.ops.segment_max(values, segment_ids, num_segments=num_segments)
    shifted = values - seg_max[segment_ids]
    seg_sum = jax.ops.segment_sum(
        jnp.exp(shifted), segment_ids, num_segments=num_segments
    )
    return seg_max + jnp.log(seg_sum)


def _clique_potentials(
    jt: JunctionTree, diag_params: Array, off_diag_params: Array
) -> Array:
    """Per-clique log-potential array of shape ``(n_cliques, 2^m)``.

    Each owned bias / coupling contributes to the states where its node(s) are
    on. Padded states are set to $-\\infty$ so they fall out of the
    ``logsumexp``-based collect pass.
    """
    bits = jnp.asarray(_bits(jt.max_clique_size))
    bias_idx, bias_pos, edge_idx, edge_pi, edge_pj = map(
        jnp.asarray, _ownership_tables(jt)
    )
    mask = jnp.asarray(_state_mask(jt))

    # Leading-zero padding so the sentinel -1 gathers a zero contribution.
    diag_padded = jnp.concatenate([jnp.zeros(1), diag_params])
    off_padded = jnp.concatenate([jnp.zeros(1), off_diag_params])

    def one_clique(
        b_idx: Array, b_pos: Array, e_idx: Array, e_i: Array, e_j: Array
    ) -> Array:
        phi_b = jnp.sum(bits[:, b_pos] * diag_padded[b_idx + 1][None, :], axis=1)
        phi_e = jnp.sum(
            bits[:, e_i] * bits[:, e_j] * off_padded[e_idx + 1][None, :], axis=1
        )
        return phi_b + phi_e

    phi = jax.vmap(one_clique)(bias_idx, bias_pos, edge_idx, edge_pi, edge_pj)
    return jnp.where(mask, phi, -jnp.inf)


def _collect(jt: JunctionTree, diag_params: Array, off_diag_params: Array) -> Array:
    """Leaves-to-root collect pass via ``lax.scan``.

    Returns per-clique log-totals (own potential plus all incoming child
    messages), shape ``(n_cliques, 2^m)``; padded states carry ``-inf``. Each
    step marginalizes the child's total onto the separator, $m_{C \\to P}(x_S)
    = \\log \\sum_{x_{C \\setminus S}} \\exp(\\phi_C(x_C) + \\text{incoming})$,
    and broadcasts it into the parent.
    """
    totals = _clique_potentials(jt, diag_params, off_diag_params)
    order = jt.collect_order
    if not order:
        return totals

    m = jt.max_clique_size
    seps = [jt.separator(c, p) for c, p in order]
    n_sep_states = 1 << max(len(s) for s in seps)
    child_idx = jnp.asarray(np.array([c for c, _ in order], dtype=np.int32))
    parent_idx = jnp.asarray(np.array([p for _, p in order], dtype=np.int32))
    sep_maps = jnp.asarray(
        np.stack(
            [
                np.stack(
                    [
                        _sep_state_map(jt.cliques[c], sep, m),
                        _sep_state_map(jt.cliques[p], sep, m),
                    ]
                )
                for (c, p), sep in zip(order, seps)
            ]
        )
    )

    def step(
        totals: Array, edge_info: tuple[Array, Array, Array]
    ) -> tuple[Array, None]:
        child, parent, sep_pair = edge_info
        msg = _segment_logsumexp(totals[child], sep_pair[0], n_sep_states)
        return totals.at[parent].add(msg[sep_pair[1]]), None

    totals, _ = jax.lax.scan(step, totals, (child_idx, parent_idx, sep_maps))
    return totals


### Chain (path decomposition) kernel ###

_LOG_ZERO = -1e30
"""Finite stand-in for $-\\infty$ in transfer matrices. Empty separator-state
segments (heterogeneous separator sizes along the path) would otherwise put
true $-\\infty$ rows into the matrices, and autodiff through an all-$-\\infty$
``logsumexp`` produces NaN gradients; clamping keeps values exact (the weight
$e^{-10^{30}}$ is exactly $0$ in floating point) and gradients clean.

This is not a stability floor in the sense the library excludes from core
(cf. CLAUDE.md): it changes no value for any parameters, carries no tunable
policy, and the slots it touches are structurally empty (topology-determined
constants with zero parameter dependence, whose true gradient contribution is
zero). It exists to make autodiff agree with the exact mathematical gradient,
which is finite; only the differentiated kernel needs it ---
:func:`chain_sample` builds the same matrices unclamped."""


def chain_log_partition(
    chain: ChainTree, diag_params: Array, off_diag_params: Array
) -> Array:
    """$\\log Z$ along a path-shaped clique tree via ``associative_scan``.

    Mathematically, the collect recurrence along a path is a product of
    transfer matrices in the log-semiring: with $T_k[t_{\\text{out}},
    t_{\\text{in}}] = \\log \\sum_{s : \\text{out}(s) = t_{\\text{out}},\\,
    \\text{in}(s) = t_{\\text{in}}} \\exp \\phi_k(s)$, the message recurrence is
    the ``logsumexp``-matvec $m_k = T_k \\otimes m_{k-1}$. Matrix composition
    is associative, so the product runs in $O(\\log K)$ depth instead of the
    $O(K)$ sequential scan --- the win on accelerators, where the sequential
    collect is latency-bound.

    Each combine costs $(2^{\\lvert S \\rvert})^3$ versus the sequential
    scan's $2^{\\lvert S \\rvert + 1}$ per step, so the log-depth trade pays
    off for separator sizes up to roughly 6; beyond that prefer
    :func:`jt_log_partition`.
    """
    order = chain.clique_path
    phi = _clique_potentials(chain, diag_params, off_diag_params)
    k_total = len(order)
    if k_total == 1:
        return jax.scipy.special.logsumexp(phi[order[0]])

    m = chain.max_clique_size
    cliques = chain.cliques
    seps = [chain.separator(a, b) for a, b in pairwise(order)]
    n_sep_states = 1 << max(len(s) for s in seps)
    out_maps = [
        _sep_state_map(cliques[order[k]], seps[k], m) for k in range(k_total - 1)
    ]
    in_maps = [
        _sep_state_map(cliques[order[k]], seps[k - 1], m) for k in range(1, k_total)
    ]

    msg = _segment_logsumexp(
        phi[order[0]], jnp.asarray(out_maps[0]), n_sep_states
    )
    msg = jnp.maximum(msg, _LOG_ZERO)

    if k_total > 2:
        # T_k as a 2D segment reduction: segment id = out * S + in.
        ids = np.stack(
            [out_maps[k] * n_sep_states + in_maps[k - 1] for k in range(1, k_total - 1)]
        )
        phi_mid = phi[jnp.asarray(np.array(order[1:-1], dtype=np.int32))]
        transfer = jax.vmap(
            lambda p, i: _segment_logsumexp(
                p, i, n_sep_states * n_sep_states
            ).reshape(n_sep_states, n_sep_states)
        )(phi_mid, jnp.asarray(ids))
        transfer = jnp.maximum(transfer, _LOG_ZERO)

        def logmm(left: Array, right: Array) -> Array:
            # Earlier matrix `left`, later `right`; composed = right after left.
            return jax.scipy.special.logsumexp(
                right[..., :, :, None] + left[..., None, :, :], axis=-2
            )

        prod = jax.lax.associative_scan(logmm, transfer)[-1]
        msg = jax.scipy.special.logsumexp(prod + msg[None, :], axis=-1)
        msg = jnp.maximum(msg, _LOG_ZERO)

    return jax.scipy.special.logsumexp(
        phi[order[-1]] + msg[jnp.asarray(in_maps[-1])]
    )


def chain_sample(
    chain: ChainTree,
    diag_params: Array,
    off_diag_params: Array,
    key: Array,
) -> Array:
    """Draw one exact sample from a chain Boltzmann distribution at $O(\\log n)$ depth.

    Parallel forward-filtering backward-sampling: two ``associative_scan``s
    replace the sequential collect + walk-down of :func:`jt_sample`.

    Mathematically, the backward messages $\\beta_k(t)$ (log-mass downstream of
    clique $k$ given separator state $t$) are suffix products of the transfer
    matrices. The ancestral pass then exploits that each clique's conditional
    draw depends on the past only through the incoming separator state: a
    shared Gumbel vector per clique resolves the draw for *every possible*
    incoming state at once, defining a random map $F_k : s_{k-1} \\mapsto s_k$
    over the $2^{\\lvert S \\rvert}$ separator states. The Gumbels are
    independent across cliques and the realized chain consumes each map at a
    point independent of that clique's Gumbels, so composing the maps --- an
    associative operation --- realizes the joint sample exactly.

    Only the Gumbels and final gathers depend on ``key``: under ``vmap`` over
    keys the messages and conditional logits are computed once across draws.
    """
    order = chain.clique_path
    phi = _clique_potentials(chain, diag_params, off_diag_params)
    k_total = len(order)
    m = chain.max_clique_size
    n = chain.n_nodes
    bits = jnp.asarray(_bits(m))

    # Scatter table: clique-local position -> node index; padded slots hit the
    # sentinel row n and are dropped. Overlapping writes agree by construction
    # (shared nodes are separator-consistent), so write order is irrelevant.
    var_idx = np.full((k_total, m), n, dtype=np.int32)
    for k, c in enumerate(order):
        var_idx[k, : len(chain.cliques[c])] = chain.cliques[c]

    phi_path = phi[jnp.asarray(np.array(order, dtype=np.int32))]  # (K, 2^m)

    if k_total == 1:
        vals = bits[jax.random.categorical(key, phi_path[0])].astype(jnp.float32)
        x = jnp.zeros(n + 1, dtype=jnp.float32).at[jnp.asarray(var_idx[0])].set(vals)
        return x[:n]

    cliques = chain.cliques
    seps = [chain.separator(a, b) for a, b in pairwise(order)]
    n_sep_states = 1 << max(len(s) for s in seps)
    out_maps = np.stack(
        [_sep_state_map(cliques[order[k]], seps[k], m) for k in range(k_total - 1)]
    )
    in_maps = np.stack(
        [_sep_state_map(cliques[order[k]], seps[k - 1], m) for k in range(1, k_total)]
    )

    # Backward messages beta_k (k = 0..K-2) via suffix products of the
    # transfer matrices M_k[t_in, t_out].
    last_vec = _segment_logsumexp(
        phi_path[-1], jnp.asarray(in_maps[-1]), n_sep_states
    )
    if k_total > 2:
        ids = np.stack(
            [in_maps[j] * n_sep_states + out_maps[j + 1] for j in range(k_total - 2)]
        )
        transfer = jax.vmap(
            lambda p, i: _segment_logsumexp(
                p, i, n_sep_states * n_sep_states
            ).reshape(n_sep_states, n_sep_states)
        )(phi_path[1:-1], jnp.asarray(ids))

        def logmm(acc: Array, elem: Array) -> Array:
            # Log-semiring product elem . acc. Under reverse=True the first
            # argument accumulates the higher-index product, so this yields
            # suffix[j] = M_j . M_{j+1} . ... --- lowest index leftmost.
            return jax.scipy.special.logsumexp(
                elem[..., :, :, None] + acc[..., None, :, :], axis=-2
            )

        suffix = jax.lax.associative_scan(logmm, transfer, reverse=True)
        betas_front = jax.scipy.special.logsumexp(
            suffix + last_vec[None, None, :], axis=-1
        )
        betas = jnp.concatenate([betas_front, last_vec[None, :]])  # (K-1, S)
    else:
        betas = last_vec[None, :]

    # Conditional logits for every clique k and every incoming separator state
    # t: phi_k(s) + beta_k(out_k(s)), masked to in_k(s) == t. Clique 0 has no
    # incoming separator: its in-map row is zero, so conditioning on t = 0
    # recovers the unconditional draw; the last clique has no beta term.
    in_ext = jnp.asarray(
        np.concatenate([np.zeros((1, 1 << m), dtype=np.int32), in_maps])
    )
    out_ext = jnp.asarray(
        np.concatenate([out_maps, np.zeros((1, 1 << m), dtype=np.int32)])
    )
    betas_ext = jnp.concatenate([betas, jnp.zeros((1, n_sep_states))])
    beta_term = jnp.take_along_axis(betas_ext, out_ext, axis=1)  # (K, 2^m)
    t_range = jnp.arange(n_sep_states)
    logits = jnp.where(
        in_ext[:, None, :] == t_range[None, :, None],
        (phi_path + beta_term)[:, None, :],
        -jnp.inf,
    )  # (K, S, 2^m)

    # Shared Gumbels resolve the draw for all hypothetical separator states at
    # once; sampled[k, t] is the clique state drawn given incoming state t, and
    # maps[k, t] the separator state it hands to clique k + 1.
    gumbels = jax.random.gumbel(key, (k_total, 1 << m))
    sampled = jnp.argmax(logits + gumbels[:, None, :], axis=-1)  # (K, S)
    maps = jnp.take_along_axis(out_ext, sampled, axis=1)  # (K, S)

    def compose(first: Array, second: Array) -> Array:
        # (second after first)[t] = second[first[t]]
        return jnp.take_along_axis(second, first, axis=-1)

    # Realized separator sequence: t_{-1} = 0, t_k = F_k(t_{k-1}).
    prefix = jax.lax.associative_scan(compose, maps[:-1])  # (K-1, S)
    t_prev = jnp.concatenate([jnp.zeros(1, dtype=prefix.dtype), prefix[:, 0]])
    states = jnp.take_along_axis(sampled, t_prev[:, None], axis=1)[:, 0]  # (K,)

    vals = bits[states].astype(jnp.float32).reshape(-1)
    x = jnp.zeros(n + 1, dtype=jnp.float32).at[jnp.asarray(var_idx).reshape(-1)].set(vals)
    return x[:n]


### Public API ###


def jt_log_partition(
    jt: JunctionTree, diag_params: Array, off_diag_params: Array
) -> Array:
    """Compute $\\log Z$ for the chordal Boltzmann distribution by sum-product.

    Runs the sequential leaves-to-root collect, correct for any junction tree.
    For path-shaped clique trees, :func:`chain_log_partition` computes the
    same value at $O(\\log n)$ depth.
    """
    totals = _collect(jt, diag_params, off_diag_params)
    return jax.scipy.special.logsumexp(totals[jt.root])


def jt_sample(
    jt: JunctionTree,
    diag_params: Array,
    off_diag_params: Array,
    key: Array,
) -> Array:
    """Draw a single exact sample from the chordal Boltzmann distribution.

    Collect pass for per-clique log-totals, then a pre-order walk down the
    rooted tree: the root clique is sampled from its full categorical, and each
    child clique from its total masked to states consistent with the values
    already drawn for the separator with its parent (the running-intersection
    property makes that conditioning sufficient).

    The same property makes the write pattern static: when the walk visits a
    clique, its already-assigned nodes are exactly that parent separator (any
    node shared with an earlier-visited clique lies on the tree path to it,
    which passes through the parent). Each step therefore writes the
    statically known complement, and every node is written exactly once --- by
    the first pre-order clique that contains it.
    """
    totals = _collect(jt, diag_params, off_diag_params)

    n, m = jt.n_nodes, jt.max_clique_size
    pre = jt.pre_order
    parent = jt.parent_of
    nc = jt.n_cliques

    clique_idx = np.zeros(nc, dtype=np.int32)
    sep_map = np.zeros((nc, 1 << m), dtype=np.int32)
    sep_vars = np.zeros((nc, m), dtype=np.int32)
    sep_size = np.zeros(nc, dtype=np.int32)
    # Nodes this step writes (clique minus parent separator): node index, with
    # sentinel n for padded slots (dropped after the scan), and the node's
    # clique-local position for bit extraction.
    new_vars = np.full((nc, m), n, dtype=np.int32)
    new_pos = np.zeros((nc, m), dtype=np.int32)
    for step_i, c in enumerate(pre):
        cl = jt.cliques[c]
        clique_idx[step_i] = c
        sep: tuple[int, ...] = ()
        if parent[c] >= 0:
            sep = jt.separator(c, parent[c])
            sep_size[step_i] = len(sep)
            sep_vars[step_i, : len(sep)] = sep
            sep_map[step_i] = _sep_state_map(cl, sep, m)
        slot = 0
        for p, v in enumerate(cl):
            if v not in sep:
                new_vars[step_i, slot] = v
                new_pos[step_i, slot] = p
                slot += 1

    state_mask = jnp.asarray(_state_mask(jt))
    bits = jnp.asarray(_bits(m))
    m_range = jnp.arange(m, dtype=jnp.int32)

    def step(
        carry: tuple[Array, Array],
        info: tuple[Array, Array, Array, Array, Array, Array],
    ) -> tuple[tuple[Array, Array], None]:
        x, key = carry
        c_idx, smap, svars, ssize, nvars, npos = info
        # Mask to states consistent with already-sampled separator values
        # (no-op for the root, where ssize == 0 and target stays 0).
        valid_sep = m_range < ssize
        sep_vals = jnp.where(valid_sep, x[svars].astype(jnp.int32), 0)
        target = jnp.sum(sep_vals << m_range, dtype=jnp.int32)
        consistent = (smap == target) | (ssize == 0)
        logits = jnp.where(consistent & state_mask[c_idx], totals[c_idx], -jnp.inf)

        key, sub = jax.random.split(key)
        sampled = bits[jax.random.categorical(sub, logits)]
        x = x.at[nvars].set(sampled[npos].astype(x.dtype))
        return (x, key), None

    init = (jnp.zeros(n + 1, dtype=jnp.float32), key)
    (x, _), _ = jax.lax.scan(
        step,
        init,
        (
            jnp.asarray(clique_idx),
            jnp.asarray(sep_map),
            jnp.asarray(sep_vars),
            jnp.asarray(sep_size),
            jnp.asarray(new_vars),
            jnp.asarray(new_pos),
        ),
    )
    return x[:n]
