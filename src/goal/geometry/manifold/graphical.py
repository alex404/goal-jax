"""Clique manifolds: parameter layouts indexed by the cliques of a graph.

A clique is one parameter block acting multilinearly on its member nodes: it pairs against
one vector per member and returns a scalar. Arity is the only thing that varies: 1 is a
linear functional, 2 a matrix contraction, 3+ a tensor contraction. So a :class:`LinearClique`
*is* a multilinear form over its members --- which is why this module sits above ``map.py``.

A :class:`LinearCliques` manifold lays its coordinates out over a tuple of such cliques.
It is *linear* because every one of those blocks is a multilinear form: the graph's bare
combinatorics, with no parameters attached, are
:class:`~goal.geometry.algebra.clique.CliqueSet`. :class:`LevelCliques` is the recursive
case, decomposing into the three spans of one level ascent; :class:`CliqueProduct` is the
disjoint union that a multi-root model's root span needs. A leaf family is not a linear
clique manifold at all --- :func:`span_blocks` reads any manifold without a clique
structure as a single opaque node.

**Nodes partition two ways; cliques partition three ways.** The nodes split into root and
non-root; the cliques split into those lying wholly in the root set, those crossing, and
those lying wholly above. A crossing clique holds nodes from both sides --- there is no
third group of *nodes*. See
:attr:`~goal.geometry.algebra.clique.CliqueSet.cross_cliques`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from math import prod
from typing import Any, Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ..algebra.clique import CliqueSet
from .base import Manifold
from .combinators import Pair, Tuple
from .embedding import LinearEmbedding
from .map import MultilinearMap

### Linear Cliques ###


@dataclass(frozen=True)
class LinearClique(Manifold):
    """One clique: a multilinear interaction among its member nodes.

    Mathematically, a clique $c$ holds a parameter tensor $\\Theta_c$ contracted against one
    vector per member,

    .. math::
        \\langle \\Theta_c, \\bigotimes_{i \\in c} v_i \\rangle,

    so its parameters inhabit the dense multilinear :attr:`form` over the subspaces its
    members contribute. Arity is a degree, not a kind: a one-member block and a coupling
    are both cliques, and neither needs its own type.

    :attr:`members` are in the frame of whatever reports the clique. A node index is not a
    property of the manifold occupying it --- a ``Normal`` is the same ``Normal`` wherever
    it sits --- so renumbering is :meth:`shifted` and relabelling is :meth:`on`, both
    explicit and both cheap.
    """

    # Fields

    members: tuple[int, ...]
    """Nodes this clique couples, ascending and distinct, in its reporter's frame."""

    axes: tuple[int, ...]
    """Factor dimensions, one per member.

    A one-member block has one factor, a pairwise coupling two, a three-way interaction
    three. This is what lets a caller address one axis of a block --- restricting one
    member's factor while leaving the others joined.

    A subclass with a better source for these --- selectors that already know their own
    dimensions --- redeclares this ``init=False`` and fills it in
    :meth:`__post_init__`, so the two cannot drift.
    """

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("a clique must name at least one node")
        if tuple(sorted(set(self.members))) != self.members:
            msg = f"members {self.members} must be ascending and distinct"
            raise ValueError(msg)
        if len(self.axes) != len(self.members):
            msg = f"clique {self.members} has {len(self.members)} members"
            raise ValueError(f"{msg} but {len(self.axes)} axes {self.axes}")

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return prod(self.axes)

    # Methods

    @property
    def form(self) -> MultilinearMap:
        """The dense multilinear form this clique's parameters inhabit."""
        return MultilinearMap(self.axes)

    def shifted(self, offset: int) -> Self:
        """The same clique renumbered into an outer frame."""
        return replace(self, members=tuple(i + offset for i in self.members))

    def on(self, members: tuple[int, ...]) -> Self:
        """The same clique relabelled onto the nodes its owner says it couples."""
        return replace(self, members=members)


### Clique Layouts ###


@dataclass(frozen=True)
class LinearCliques(Manifold, ABC):
    """A manifold whose parameters are laid out over the cliques of a graph.

    Each block is a multilinear form over its clique's members, which is what the *linear*
    names: strip the parameters away and what remains is a
    :class:`~goal.geometry.algebra.clique.CliqueSet`.

    :attr:`clique_blocks` is the primitive: the :class:`LinearClique` objects whose parameters this
    manifold concatenates, in storage order. Block sizes and factor shapes are read off
    them, and :meth:`clique_index` looks a clique up by its members, so an offset and the
    members it belongs to always come from the same object. :class:`LevelCliques` is the
    recursive case and :class:`CliqueProduct` the disjoint union; the recursion terminates
    wherever a span stops being a clique manifold.

    Layout order is *storage* order --- the order the blocks occupy in the flat coordinate
    vector. Nothing at runtime requires it to match the graph's
    :attr:`~goal.geometry.algebra.clique.CliqueSet.canonical_cliques`, because
    :meth:`clique_offsets` and :meth:`clique_index` both read the blocks' own members. It
    does match for every model the library ships, and ``tests/graphical.py`` enforces that
    over all of them; a model whose declaration order diverges is a bug in the model, not
    a case this class handles.
    """

    # Contract

    @property
    @abstractmethod
    def clq_set(self) -> CliqueSet:
        """The graph this manifold is defined on."""

    @property
    @abstractmethod
    def clique_blocks(self) -> tuple[LinearClique, ...]:
        """One record per parameter block, in storage order."""

    # Methods

    @property
    def clique_members(self) -> tuple[tuple[int, ...], ...]:
        """Which nodes each block couples, in storage order."""
        return tuple(block.members for block in self.clique_blocks)

    @property
    def clique_dims(self) -> tuple[int, ...]:
        """Dimension of each clique's parameter block, in storage order."""
        return tuple(block.dim for block in self.clique_blocks)

    @property
    def clique_axes(self) -> tuple[tuple[int, ...], ...]:
        """Factor dimensions of each clique's block, parallel to :attr:`clique_dims`."""
        return tuple(block.axes for block in self.clique_blocks)

    def clique_offsets(self) -> tuple[int, ...]:
        """Start of each clique's block in the flat parameter vector."""
        offsets: list[int] = []
        running = 0
        for block_dim in self.clique_dims:
            offsets.append(running)
            running += block_dim
        return tuple(offsets)

    def clique_index(self, members: tuple[int, ...]) -> int:
        """Layout position of the clique on exactly ``members``.

        Raises:
            ValueError: if no clique covers exactly those nodes, which is the structural
                condition a coupling into this manifold needs --- there has to be a single
                block holding those members jointly.
        """
        wanted = tuple(sorted(members))
        layout = self.clique_members
        for i, clique in enumerate(layout):
            if clique == wanted:
                return i
        msg = f"no clique on {wanted}; this manifold has "
        raise ValueError(f"{msg}{layout}")

    # Methods

    def split_cliques(self, coords: Array) -> tuple[Array, ...]:
        """Split coordinates into per-clique blocks, in layout order."""
        return _split_by_dims(coords, self.clique_dims)

    def join_cliques(self, *blocks: Array) -> Array:
        """Concatenate per-clique blocks back into coordinates.

        Raises:
            ValueError: if the number of blocks or any block's size disagrees with
                :attr:`clique_dims`. Concatenation would otherwise accept wrongly-sized
                blocks and produce a coordinate vector of the wrong length.
        """
        dims = self.clique_dims
        if len(blocks) != len(dims):
            raise ValueError(f"expected {len(dims)} clique blocks, got {len(blocks)}")
        for i, (block, block_dim) in enumerate(zip(blocks, dims, strict=True)):
            if block.shape != (block_dim,):
                msg = f"clique block {i} has shape {block.shape}"
                raise ValueError(f"{msg}, expected ({block_dim},)")
        return jnp.concatenate(blocks)

    def cut(self, far_node: int) -> CliqueCut:
        """Re-view the layout with ``far_node`` split off instead of the root nodes.

        The re-rooting isomorphism: the same coordinates, regrouped as
        ``(near | crossing | far)`` where the near side is everything not touching
        ``far_node``, the far side is that node's own block, and the crossing cliques become
        one matrix whose rows follow the near side's clique order. For a mixture of
        harmoniums cut at its category node, the near side is exactly the base harmonium's
        parameter vector, which is what makes the mixture view and the graph view two
        readings of one array.

        Only integers are involved, so this is a block permutation and applies to any
        coordinate system alike. Cutting **one** node is what makes the row assignment
        total: two
        crossing cliques sharing a near part $P$ would both be $P \\cup \\{far\\}$, hence
        the same clique, so distinct crossing cliques always land on distinct rows.

        The view exists only when **every crossing clique couples ``far_node`` in full**,
        so that they share one column axis --- that is, when its factor in each crossing
        clique is the node's whole span, not a subspace of it. A mixture's category node is
        coupled in full; a linear Gaussian model's interaction reaches only a subspace of
        the node it couples to, and has no such view. So this is not a generalization of
        :meth:`~LevelCliques.split_level`, whose cross span couples the root to the
        *boundary* and may address only a subspace of it.

        Reads the layout, not the graph: ``far_node`` is matched against
        :attr:`clique_members`, so the indices it computes and the dimensions it selects
        come from the same blocks. Consulting
        :attr:`~goal.geometry.algebra.clique.CliqueSet.canonical_cliques` here would pair
        positions from one order with sizes from another.

        Raises:
            ValueError: if ``far_node`` is not a node of the graph, or is its only node; if
                it carries no block of its own, so the cut has no columns; if a crossing
                clique's near part is not itself a clique, so it has no row block; or if a
                crossing clique does not couple the whole far side.
        """
        cliques = self.clique_members
        dims = self.clique_dims
        n_nodes = self.clq_set.n_nodes
        if not 0 <= far_node < n_nodes:
            msg = f"far_node {far_node} is not one of the graph's {n_nodes} nodes"
            raise ValueError(msg)
        if n_nodes == 1:
            raise ValueError("cutting the only node off leaves no near side")

        near_idx = tuple(i for i, c in enumerate(cliques) if far_node not in c)
        far_idx = tuple(i for i, c in enumerate(cliques) if c == (far_node,))
        cross_idx = tuple(
            i for i in range(len(cliques)) if i not in near_idx and i not in far_idx
        )
        if not far_idx:
            msg = f"far_node {far_node} has no block of its own"
            raise ValueError(f"{msg}: the cut would have no columns")

        n_cols = sum(dims[i] for i in far_idx)
        near_at = {cliques[i]: pos for pos, i in enumerate(near_idx)}

        cross_rows: list[int] = []
        for i in cross_idx:
            near_part = tuple(j for j in cliques[i] if j != far_node)
            if near_part not in near_at:
                msg = f"clique {cliques[i]} crosses the cut, near part {near_part}"
                raise ValueError(
                    f"{msg} is not itself a clique, so it has no row block"
                )
            pos = near_at[near_part]
            if dims[i] != dims[near_idx[pos]] * n_cols:
                want = f"{dims[near_idx[pos]]} x {n_cols}"
                msg = f"clique {cliques[i]} has dimension {dims[i]}, not {want}"
                raise ValueError(f"{msg}: it does not couple the whole far side")
            cross_rows.append(pos)

        return CliqueCut(
            clique_dims=dims,
            near_idx=near_idx,
            cross_idx=cross_idx,
            far_idx=far_idx,
            cross_rows=tuple(cross_rows),
            n_cols=n_cols,
        )


@dataclass(frozen=True)
class LevelCliques[Root: Manifold, Cross: Manifold, Deep: Manifold](
    LinearCliques, Tuple, ABC
):
    """Clique manifold laid out as the three spans of one level ascent.

    Coordinates are stored as ``[root | cross | deep]``: the parameters carried by the root
    nodes, the interactions joining the root nodes to the rest of the graph, and everything
    above. The deep span is laid out exactly as the manifold on ``clq_set.ascend_level()``
    lays itself out, so the same split applies again one level up --- recursion over the
    graph is a sequence of these.

    Unlike ``Pair`` and ``Triple``, the components are not arbitrary: the graph says what
    each one is. A span may hold several cliques --- ``deep`` always does past depth two ---
    which is why the three spans are named rather than the individual blocks.
    """

    # Contract

    @property
    @abstractmethod
    def root_man(self) -> Root:
        """Manifold of the parameters carried by the root nodes."""

    @property
    @abstractmethod
    def cross_man(self) -> Cross:
        """Manifold of the interactions joining the root nodes to the rest of the graph."""

    @property
    @abstractmethod
    def deep_man(self) -> Deep:
        """Manifold of everything above the root level."""

    @property
    @abstractmethod
    def cross_blocks(self) -> tuple[LinearClique, ...]:
        """The cliques joining the root nodes to the rest, in this level's frame.

        The cross span is a bare parameter manifold --- a model supplies its interaction as
        a map --- so which nodes each of its blocks couples is the part only the model
        knows. Node ``0`` is the first root; the deep span's nodes start at
        :func:`span_nodes` of the root span.
        """

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """Total dimension is the sum of the three span dimensions."""
        return self.root_man.dim + self.cross_man.dim + self.deep_man.dim

    @property
    @override
    def clq_set(self) -> CliqueSet:
        """The graph glued from the spans' graphs.

        The root span supplies the root nodes and the deep span the rest, shifted past
        them; :attr:`cross_blocks` supplies the cliques joining the two. Levels are
        distances in the glued graph, so how the deep span roots *itself* is discarded ---
        which is what makes a fork
        (:class:`~goal.models.graphical.mixture.CompleteMixtureOfHarmoniums`) depth two
        rather than depth three.
        """
        n_roots = span_nodes(self.root_man)
        return CliqueSet(
            n_nodes=n_roots + span_nodes(self.deep_man),
            n_roots=n_roots,
            cliques=self.clique_members,
        )

    @property
    @override
    def clique_blocks(self) -> tuple[LinearClique, ...]:
        """The spans' blocks, concatenated in storage order.

        The root span's blocks are already in this level's frame, and
        :attr:`cross_blocks` reports its own. The deep span's keep the order and the
        relative labels the deep manifold itself gives them, shifted past the root nodes ---
        so however the glued graph reroots, a deep block still says which nodes it couples.
        """
        offset = span_nodes(self.root_man)
        return (
            span_blocks(self.root_man)
            + self.cross_blocks
            + tuple(b.shifted(offset) for b in span_blocks(self.deep_man))
        )

    @override
    def split_coords(self, coords: Array) -> tuple[Array, Array, Array]:
        """Split coordinates into the root, cross, and deep spans."""
        root_dim = self.root_man.dim
        cross_dim = self.cross_man.dim
        return (
            coords[:root_dim],
            coords[root_dim : root_dim + cross_dim],
            coords[root_dim + cross_dim :],
        )

    @override
    def join_coords(self, *components: Array) -> Array:
        """Concatenate the root, cross, and deep spans."""
        if len(components) != 3:
            raise ValueError(f"expected 3 spans, got {len(components)}")
        return jnp.concatenate(components)

    # Methods

    def split_level(self, coords: Array) -> tuple[Array, Array, Array]:
        """Split off one level: the root span, the cross span, and the deep span.

        The domain-facing name for :meth:`split_coords`. A graph of depth one has an empty
        cross and deep span.
        """
        return self.split_coords(coords)

    def join_level(self, root: Array, cross: Array, deep: Array) -> Array:
        """Concatenate the root, cross, and deep spans."""
        return self.join_coords(root, cross, deep)


@dataclass(frozen=True)
class CliqueProduct[Fst: Manifold, Snd: Manifold](LinearCliques, Pair[Fst, Snd], ABC):
    """Two groups of nodes side by side, with no clique joining them.

    The graph is the disjoint union of the components' graphs, the second component's
    indices shifted past the first, and the parameter layout is the components'
    concatenated. Every node is a root: nothing links the two sides, so a non-root node
    would have no path from the root set at all.

    This is the shape a **multi-root** model's root span takes --- two root spans coupled
    to one shared node above, as in probabilistic CCA. A level whose root span is one of these has as many
    root nodes as the product has components, which is what lets its crossing cliques fan
    out to more than one of them.
    """

    # Overrides

    @property
    @override
    def clq_set(self) -> CliqueSet:
        n_nodes = span_nodes(self.fst_man) + span_nodes(self.snd_man)
        return CliqueSet(n_nodes, n_nodes, self.clique_members)

    @property
    @override
    def clique_blocks(self) -> tuple[LinearClique, ...]:
        offset = span_nodes(self.fst_man)
        return span_blocks(self.fst_man) + tuple(
            b.shifted(offset) for b in span_blocks(self.snd_man)
        )


### Clique Block Embeddings ###


def _map_axis(tensor: Array, axis: int, fn: Any) -> Array:
    """Apply a vector function along one axis of a tensor, replacing that axis."""
    moved = jnp.moveaxis(tensor, axis, 0)
    trailing = moved.shape[1:]
    columns = moved.reshape(moved.shape[0], -1)
    mapped = jax.vmap(fn, in_axes=1, out_axes=1)(columns)
    return jnp.moveaxis(mapped.reshape((-1, *trailing)), 0, axis)


@dataclass(frozen=True)
class CliqueBlockEmbedding[Ambient: LinearCliques](
    LinearEmbedding[MultilinearMap, Ambient]
):
    """Addresses one clique's block of a :class:`LinearCliques` manifold, one selector per member.

    Say which nodes you are coupling and how much of each one's coordinates you want, and
    this finds the block holding those members *jointly* and restricts it one axis at a
    time. One embedding per clique, at any arity.

    Restricting a joint block axis by axis is not the same as composing per-member
    restrictions, and the difference is the point: the joint block is a tensor that need
    not factor across its members, so a coupling to several at once must select from that
    tensor rather than combine separate per-member ones.

    The ambient manifold must have a clique on exactly these members --- there has to be a
    single block holding them jointly. :meth:`LinearCliques.clique_index` raises when it
    does not, which is the structural condition higher arity needs.
    """

    # Fields

    _amb_man: Ambient
    """The clique manifold holding the block."""

    members: tuple[int, ...]
    """Nodes coupled, in the ambient manifold's frame, ascending."""

    selectors: tuple[LinearEmbedding[Any, Any], ...]
    """One selector per member, in the same order as :attr:`members`."""

    def __post_init__(self) -> None:
        if len(self.members) != len(self.selectors):
            msg = f"{len(self.members)} members but {len(self.selectors)} selectors"
            raise ValueError(msg)
        if tuple(sorted(self.members)) != self.members:
            raise ValueError(f"members {self.members} must be ascending")

    # Overrides

    @property
    @override
    def amb_man(self) -> Ambient:
        return self._amb_man

    @property
    @override
    def sub_man(self) -> MultilinearMap:
        """The selected sub-tensor: one factor per member."""
        return MultilinearMap(tuple(sel.sub_man.dim for sel in self.selectors))

    @override
    def project(self, coords: Array) -> Array:
        start, block_dim, axes = self._block()
        out = coords[start : start + block_dim].reshape(axes)
        for axis, sel in enumerate(self.selectors):
            out = _map_axis(out, axis, sel.project)
        return out.reshape(-1)

    @override
    def embed(self, coords: Array) -> Array:
        start, block_dim, _ = self._block()
        out = coords.reshape(self.sub_man.sub_dims)
        for axis, sel in enumerate(self.selectors):
            out = _map_axis(out, axis, sel.embed)
        full = jnp.zeros(self.amb_man.dim)
        return full.at[start : start + block_dim].set(out.reshape(-1))

    # Private

    def _block(self) -> tuple[int, int, tuple[int, ...]]:
        """Start, size, and factor dimensions of the block this addresses."""
        index = self.amb_man.clique_index(self.members)
        return (
            self.amb_man.clique_offsets()[index],
            self.amb_man.clique_dims[index],
            self.amb_man.clique_axes[index],
        )


### Cut Views ###


def span_nodes(span: Manifold) -> int:
    """How many graph nodes one span of a :class:`LevelCliques` contributes.

    A span that declares a clique structure says so itself; anything else is a single node.
    That fallback is what lets a leaf family occupy a node without knowing it does, and what
    lets a model carry a span whose parameters are not a clique decomposition --- a learned
    variational correction, for instance.
    """
    return span.clq_set.n_nodes if isinstance(span, LinearCliques) else 1


def span_blocks(span: Manifold) -> tuple[LinearClique, ...]:
    """The cliques contributed by one span, in that span's own frame.

    Matching :func:`span_nodes`: a span without a clique structure is one opaque clique over
    one node.
    """
    if isinstance(span, LinearCliques):
        return span.clique_blocks
    return (LinearClique((0,), (span.dim,)),)


@dataclass(frozen=True)
class CliqueCut:
    """A clique manifold's layout, re-viewed as bipartite across a node cut.

    Produced by :meth:`LinearCliques.cut`. Holds only integers, so the re-view is a block
    permutation and nothing else --- which is why it applies to any coordinate system
    alike.
    """

    # Fields

    clique_dims: tuple[int, ...]
    """Block sizes of the manifold being re-viewed, in its layout order."""

    near_idx: tuple[int, ...]
    """Layout positions of the cliques lying wholly on the near side."""

    cross_idx: tuple[int, ...]
    """Layout positions of the cliques crossing the cut."""

    far_idx: tuple[int, ...]
    """Layout positions of the cliques lying wholly on the far side."""

    cross_rows: tuple[int, ...]
    """For each crossing clique, the position *within* ``near_idx`` of its near part."""

    n_cols: int
    """Width of the crossing matrix: the far side's total dimension."""

    # Methods

    def project(self, coords: Array) -> tuple[Array, Array, Array]:
        """Split coordinates into ``(near, cross, far)`` under this cut.

        ``cross`` is the crossing matrix ravelled row-major, its rows following the near
        side's clique order.
        """
        blocks = _split_by_dims(coords, self.clique_dims)
        near = jnp.concatenate([blocks[i] for i in self.near_idx])
        far = jnp.concatenate([blocks[i] for i in self.far_idx])
        row_source = {row: pos for pos, row in enumerate(self.cross_rows)}
        rows: list[Array] = []
        for pos, i in enumerate(self.near_idx):
            height = self.clique_dims[i]
            if pos in row_source:
                block = blocks[self.cross_idx[row_source[pos]]]
                rows.append(block.reshape(height, self.n_cols))
            else:
                rows.append(jnp.zeros((height, self.n_cols)))
        return near, jnp.vstack(rows).ravel(), far

    def join(self, near: Array, cross: Array, far: Array) -> Array:
        """Reassemble coordinates from a ``(near, cross, far)`` view."""
        near_dims = tuple(self.clique_dims[i] for i in self.near_idx)
        near_blocks = _split_by_dims(near, near_dims)
        far_blocks = _split_by_dims(
            far, tuple(self.clique_dims[i] for i in self.far_idx)
        )
        cross_matrix = cross.reshape(sum(near_dims), self.n_cols)

        offsets: list[int] = []
        running = 0
        for block_dim in near_dims:
            offsets.append(running)
            running += block_dim

        blocks: list[Array | None] = [None] * len(self.clique_dims)
        for pos, i in enumerate(self.near_idx):
            blocks[i] = near_blocks[pos]
        for pos, i in enumerate(self.far_idx):
            blocks[i] = far_blocks[pos]
        for pos, i in enumerate(self.cross_idx):
            row = self.cross_rows[pos]
            start = offsets[row]
            blocks[i] = cross_matrix[start : start + near_dims[row], :].ravel()

        return jnp.concatenate([b for b in blocks if b is not None])


def _split_by_dims(coords: Array, dims: tuple[int, ...]) -> tuple[Array, ...]:
    """Split a flat array into consecutive slices of the given sizes.

    Raises:
        ValueError: if ``coords`` is not one-dimensional of length ``sum(dims)``. Slicing
            past the end is silent in JAX, so without this a short vector yields empty
            trailing blocks and a long one drops coordinates.
    """
    total = sum(dims)
    if coords.ndim != 1 or coords.shape[0] != total:
        msg = f"expected a flat array of {total} coordinates"
        raise ValueError(f"{msg}, got shape {coords.shape}")
    out: list[Array] = []
    offset = 0
    for block_dim in dims:
        out.append(coords[offset : offset + block_dim])
        offset += block_dim
    return tuple(out)


### Span Embeddings ###


@dataclass(frozen=True)
class RootEmbedding[
    Sub: LevelCliques[Any, Any, Any],
    Ambient: LevelCliques[Any, Any, Any],
](LinearEmbedding[Sub, Ambient]):
    """Embeds one clique manifold into another over the same graph, transforming only the root span.

    Use this when two models share a graph but parameterize the root nodes differently ---
    one restricting its root span to a submanifold of the other's. The cross and deep spans
    pass through unchanged, so a difference deeper in
    the graph is expressed by nesting: the deep manifolds are themselves clique manifolds
    related by their own ``RootEmbedding``.

    Mathematically, for span coordinates $(r, c, d)$: ``embed`` maps $(r, c, d) \\mapsto
    (\\phi(r), c, d)$ and ``project`` maps $(r, c, d) \\mapsto (\\pi(r), c, d)$, where
    $\\phi$ and $\\pi$ are the root embedding's own maps.
    """

    # Fields

    root_emb: LinearEmbedding[Any, Any]
    """Embedding of the restricted root manifold into the full one."""

    _sub_man: Sub
    """The clique manifold with the restricted root span."""

    _amb_man: Ambient
    """The clique manifold with the full root span."""

    def __post_init__(self) -> None:
        if self.sub_man.clq_set != self.amb_man.clq_set:
            raise ValueError(
                f"sub and ambient manifolds must share a clique set: {self.sub_man.clq_set} vs {self.amb_man.clq_set}"
            )
        for name, sub_span, amb_span in (
            ("cross", self.sub_man.cross_man, self.amb_man.cross_man),
            ("deep", self.sub_man.deep_man, self.amb_man.deep_man),
        ):
            if sub_span.dim != amb_span.dim:
                msg = f"{name} spans differ: {sub_span.dim} vs {amb_span.dim}"
                raise ValueError(f"{msg}; only the root span may be transformed")
        for name, span, emb_man in (
            ("sub", self.sub_man.root_man, self.root_emb.sub_man),
            ("ambient", self.amb_man.root_man, self.root_emb.amb_man),
        ):
            if span.dim != emb_man.dim:
                msg = f"{name} root span has dimension {span.dim}"
                raise ValueError(f"{msg}, but root_emb expects {emb_man.dim}")

    # Overrides

    @property
    @override
    def sub_man(self) -> Sub:
        return self._sub_man

    @property
    @override
    def amb_man(self) -> Ambient:
        return self._amb_man

    @override
    def project(self, coords: Array) -> Array:
        root, cross, deep = self.amb_man.split_level(coords)
        return self.sub_man.join_level(self.root_emb.project(root), cross, deep)

    @override
    def embed(self, coords: Array) -> Array:
        root, cross, deep = self.sub_man.split_level(coords)
        return self.amb_man.join_level(self.root_emb.embed(root), cross, deep)
