"""Cliques over exponential families: one clique, one inner product, and the layouts over them.

An :class:`EFClique` is a multilinear form on a graph position together with what that form
*pairs against*: the tensor product of its member nodes' sufficient statistics, each
restricted to the sub-statistic this clique actually couples. The selectors carry both
facts at once --- they name the sub-statistics, and their dimensions *are* the form's
factors, so arity is one fact rather than several that can disagree.

:class:`LinearCliques` lays a coordinate vector out over a tuple of such cliques,
:class:`LevelCliques` is the recursive case over one level of a graph, and
:class:`CliqueProduct` the disjoint union a multi-root model's root span needs. The graph's
bare combinatorics, with no parameters attached, are
:class:`~goal.geometry.algebra.clique.Cliques`.

Mathematically, a clique $c$ contributes exactly one term to the log-density,

.. math::
    \\langle \\theta_c, \\bigotimes_{i \\in c} \\pi_i(\\mathbf s_i(x_i)) \\rangle,

with $\\pi_i$ the selector for member $i$. Arity is the only thing that varies: at $|c| = 1$
this is a bias, at $|c| = 2$ a matrix contraction, and beyond that a higher-order tensor
contraction. A bias and a coupling differ in arity, not in kind.

**Why the layouts live here and not below.** A clique's factor dimensions are decided by
which sub-statistics it couples, and a sub-statistic is an exponential-family notion. A
layout layer that could not see selectors would have to recover the factors from something
else, which is exactly the reconstruction this module exists to make unnecessary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from math import prod
from typing import Any, Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ..algebra.clique import CliqueCut, Cliques, cut_indices
from ..manifold.base import Manifold
from ..manifold.combinators import Pair, Tuple
from ..manifold.embedding import IdentityEmbedding, LinearEmbedding
from ..manifold.map import EmbeddedMap, LinearMap, MultilinearMap
from .base import ExponentialFamily


@dataclass(frozen=True)
class EFClique(Manifold):
    """One clique: a multilinear interaction among its member exponential-family nodes.

    Each member contributes a **selector** --- a
    :class:`~goal.geometry.manifold.embedding.LinearEmbedding` from the sub-statistic this
    clique couples into that node's full statistic. One selector per member, most of them
    identities, and nothing else deciding which coordinates the block addresses.

    Mathematically, the clique holds a parameter tensor $\\Theta_c$ contracted against one
    vector per member, $\\langle \\Theta_c, \\bigotimes_{i \\in c} v_i \\rangle$, so its
    parameters inhabit the dense multilinear :attr:`form` over the subspaces its members
    contribute. Arity is a degree, not a kind: a one-member block and a coupling are both
    cliques, and neither needs its own type.

    At arity 2 with selectors ``(cod_emb, dom_emb)`` this reproduces
    :class:`~goal.geometry.manifold.map.EmbeddedMap` exactly, block for block ---
    :meth:`tensor` is its ``outer_product`` and :meth:`contract` is its application and
    transposed application. The gain is that arity is no longer capped at 2.

    :attr:`members` are in the frame of whatever reports the clique. A node index is not a
    property of the family occupying it --- a ``Normal`` is the same ``Normal`` wherever it
    sits --- so relabelling is :meth:`on` and renumbering :meth:`shifted`, both explicit
    and both cheap.
    """

    # Fields

    members: tuple[int, ...]
    """Nodes this clique couples, ascending and distinct, in its reporter's frame."""

    selectors: tuple[LinearEmbedding[Any, ExponentialFamily], ...]
    """One selector per member, in the same order: sub-statistic into full statistic."""

    # Overrides

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("a clique must name at least one node")
        if tuple(sorted(set(self.members))) != self.members:
            msg = f"members {self.members} must be ascending and distinct"
            raise ValueError(msg)
        if len(self.axes) != len(self.members):
            msg = f"clique {self.members} has {len(self.members)} members"
            raise ValueError(f"{msg} but {len(self.axes)} axes {self.axes}")

    @property
    @override
    def dim(self) -> int:
        return prod(self.axes)

    # Methods

    @property
    def axes(self) -> tuple[int, ...]:
        """One factor per member: the dimension of the sub-statistic it selects.

        Reading the factors off the selectors is what makes arity a single fact rather
        than several that can disagree --- ``len(members) == len(selectors) == len(axes)``
        holds by construction.
        """
        return tuple(sel.sub_man.dim for sel in self.selectors)

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

    @property
    def node_mans(self) -> tuple[ExponentialFamily, ...]:
        """The member nodes' manifolds, in clique order."""
        return tuple(sel.amb_man for sel in self.selectors)

    def tensor(self, *node_stats: Array) -> Array:
        """Pair one full statistic per member into this clique's parameter block.

        Each argument is a member's *full* sufficient statistic; the selectors restrict
        them before the outer product is taken.

        **What the arguments must be.** Each is that member's statistic *on its own*, so
        this builds the block as a product of marginals. That is exact when every member is
        observed --- one data point, all statistics deterministic --- which is the
        :meth:`sufficient_statistic` case. It is *not* how the block looks when two or more
        members are latent: there the block is $\\mathbb E[\\bigotimes_i \\mathbf s_i]$
        jointly, and expectation does not pass through a tensor product. For that case the
        latent members' joint block comes from the level above and the selectors are
        contracted into it --- see :meth:`partial_contract`.
        """
        selected = [
            sel.project(stats)
            for sel, stats in zip(self.selectors, node_stats, strict=True)
        ]
        return self.form.tensor(*selected)

    def contract(self, params: Array, keep: int, *node_stats: Array) -> Array:
        """Contract every member but ``keep``, returning parameters on that node.

        ``node_stats`` supplies one full statistic per contracted member, in ascending
        member order. The result is embedded back into the kept node's full parameter
        space, so it can be added to that node's bias directly. At arity 2 this is the
        likelihood map (``keep`` the codomain) or the posterior map (``keep`` the domain).
        """
        others = [pos for pos in range(len(self.members)) if pos != keep]
        selected = [
            self.selectors[pos].project(stats)
            for pos, stats in zip(others, node_stats, strict=True)
        ]
        return self.selectors[keep].embed(self.form.contract(params, keep, *selected))

    def partial_contract(
        self, params: Array, keep: tuple[int, ...], *node_stats: Array
    ) -> Array:
        """Contract several members at once, leaving a joint tensor over ``keep``.

        The general form of :meth:`contract`, and the one a level split needs: the members
        on the near side of a cut are contracted individually against their own statistics,
        and the members on the far side are left *joined*, because their expectations do
        not factor. ``node_stats`` supplies one full statistic per contracted member, in
        ascending member order; ``keep`` names the surviving members, also in ascending
        order.

        The result is a flat tensor over the kept members' *selected* dimensions, in clique
        order, with no embedding applied --- placing it is the caller's job, since where it
        goes depends on which manifold holds the joint block for those members.
        """
        dropped = [pos for pos in range(len(self.members)) if pos not in keep]
        selected = [
            self.selectors[pos].project(stats)
            for pos, stats in zip(dropped, node_stats, strict=True)
        ]
        out = self.form.to_tensor(params)
        # Descending order so that contracting one axis does not shift the next.
        for axis, vector in sorted(zip(dropped, selected), key=lambda p: -p[0]):
            out = jnp.tensordot(out, vector, axes=([axis], [0]))
        return out.reshape(-1)

    def select_joint(self, keep: tuple[int, ...], joint_block: Array) -> Array:
        """Restrict a joint block over the ``keep`` members to this clique's sub-statistics.

        ``joint_block`` is $\\mathbb E[\\bigotimes_{i \\in keep} \\mathbf s_i]$ as the level
        above stores it --- a flat tensor over those members' full statistic dimensions,
        in ascending member order. Each of this clique's selectors is contracted into the
        corresponding axis, giving the sub-tensor this clique actually couples.

        This is the operation :meth:`tensor` cannot do: it never forms a marginal, so it
        stays correct when the kept members are dependent.
        """
        out = joint_block.reshape(
            tuple(self.selectors[pos].amb_man.dim for pos in keep)
        )
        for axis, pos in enumerate(keep):
            out = _map_axis(out, axis, self.selectors[pos].project)
        return out.reshape(-1)

    def sufficient_statistic(self, *xs: Array) -> Array:
        """This clique's contribution to the joint sufficient statistic.

        Takes one data point per member and returns the block, which is what makes this an
        exponential-family clique rather than a bare multilinear form.
        """
        stats = [
            man.sufficient_statistic(x)
            for man, x in zip(self.node_mans, xs, strict=True)
        ]
        return self.tensor(*stats)


### Clique Layouts ###


@dataclass(frozen=True)
class LinearCliques(Manifold, ABC):
    """A manifold whose parameters are laid out over the cliques of a graph.

    Each block is a multilinear form over its clique's members, which is what the *linear*
    names: strip the parameters away and what remains is a
    :class:`~goal.geometry.algebra.clique.Cliques`.

    :attr:`clique_blocks` is the primitive: the :class:`EFClique` objects whose parameters this
    manifold concatenates, in storage order. Block sizes and factor shapes are read off
    them, and :meth:`clique_index` looks a clique up by its members, so an offset and the
    members it belongs to always come from the same object. :class:`LevelCliques` is the
    recursive case and
    :class:`~goal.geometry.exponential_family.clique.CliqueProduct` the disjoint union; the
    recursion terminates wherever a span stops being a clique manifold.

    Layout order is *storage* order --- the order the blocks occupy in the flat coordinate
    vector. Nothing at runtime requires it to match the graph's
    :attr:`~goal.geometry.algebra.clique.Cliques.canonical_cliques`, because
    :meth:`clique_offsets` and :meth:`clique_index` both read the blocks' own members. It
    does match for every model the library ships, and ``tests/graphical.py`` enforces that
    over all of them; a model whose declaration order diverges is a bug in the model, not
    a case this class handles.
    """

    # Contract

    @property
    @abstractmethod
    def clq_set(self) -> Cliques:
        """The graph this manifold is defined on."""

    @property
    @abstractmethod
    def clique_blocks(self) -> tuple[EFClique, ...]:
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

        Reads the layout, not the graph: ``far_node`` is matched against
        :attr:`clique_members`, so the positions computed and the dimensions selected come
        from the same blocks. Consulting
        :attr:`~goal.geometry.algebra.clique.Cliques.canonical_cliques` here would pair
        positions from one order with sizes from another. The index computation itself, and
        every condition the view requires, is
        :func:`~goal.geometry.algebra.clique.cut_indices`.
        """
        return cut_indices(
            self.clique_members,
            self.clique_dims,
            far_node,
            self.clq_set.n_nodes,
        )


@dataclass(frozen=True)
class LevelCliques[Root: ExponentialFamily, Cross: Manifold, Deep: ExponentialFamily](
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
    def cross_blocks(self) -> tuple[EFClique, ...]:
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
    def clq_set(self) -> Cliques:
        """The graph glued from the spans' graphs.

        The root span supplies the root nodes and the deep span the rest, shifted past
        them; :attr:`cross_blocks` supplies the cliques joining the two. Levels are
        distances in the glued graph, so how the deep span roots *itself* is discarded ---
        which is what makes a fork
        (:class:`~goal.models.graphical.mixture.CompleteMixtureOfHarmoniums`) depth two
        rather than depth three.
        """
        n_roots = span_nodes(self.root_man)
        return Cliques(
            n_nodes=n_roots + span_nodes(self.deep_man),
            n_roots=n_roots,
            cliques=self.clique_members,
        )

    @property
    @override
    def clique_blocks(self) -> tuple[EFClique, ...]:
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
class CliqueProduct[Fst: ExponentialFamily, Snd: ExponentialFamily](
    LinearCliques, Pair[Fst, Snd], ABC
):
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
    def clq_set(self) -> Cliques:
        n_nodes = span_nodes(self.fst_man) + span_nodes(self.snd_man)
        return Cliques(n_nodes, n_nodes, self.clique_members)

    @property
    @override
    def clique_blocks(self) -> tuple[EFClique, ...]:
        offset = span_nodes(self.fst_man)
        return span_blocks(self.fst_man) + tuple(
            b.shifted(offset) for b in span_blocks(self.snd_man)
        )


### Spans ###


def span_nodes(span: Manifold) -> int:
    """How many graph nodes one span of a :class:`LevelCliques` contributes.

    A span that declares a clique structure says so itself; anything else is a single node.
    That fallback is what lets a leaf family occupy a node without knowing it does.
    """
    return span.clq_set.n_nodes if isinstance(span, LinearCliques) else 1


def span_blocks(span: ExponentialFamily) -> tuple[EFClique, ...]:
    """The cliques contributed by one span, in that span's own frame.

    Matching :func:`span_nodes`: a span without a clique structure is one clique over one
    node, selecting that node's statistic in full. This is where the recursion terminates,
    and taking an :class:`~goal.geometry.exponential_family.base.ExponentialFamily` is what
    makes the terminal case a real clique rather than a bare pair of numbers --- it knows
    its node's family, so it can produce a sufficient statistic like any other.
    """
    if isinstance(span, LinearCliques):
        return span.clique_blocks
    return (EFClique(members=(0,), selectors=(IdentityEmbedding(span),)),)


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
    EFClique, LinearEmbedding[MultilinearMap, Ambient]
):
    """An :class:`EFClique` that addresses a block of *another* manifold's layout.

    Same clique, different ownership: an ``EFClique`` on its own describes a block a model
    holds, while this one names a block held by ``_amb_man`` and reads or writes it in
    place. Say which nodes you are coupling and how much of each one's coordinates you
    want, and :meth:`project` finds the block holding those members *jointly* and restricts
    it one axis at a time; :meth:`embed` is the adjoint, scattering back into a zero
    ambient vector.

    Restricting a joint block axis by axis is not the same as composing per-member
    restrictions, and the difference is the point: the joint block is a tensor that need
    not factor across its members, so a coupling to several at once must select from that
    tensor rather than combine separate per-member ones. :meth:`EFClique.select_joint` is
    the same walk over a block supplied by the level above rather than read out of an
    ambient layout --- there the axis sizes come from the selectors, here from
    :attr:`~goal.geometry.exponential_family.clique.LinearCliques.clique_axes`, which is why the two
    stay separate methods over one :func:`_map_axis`.

    The ambient manifold must have a clique on exactly these members --- there has to be a
    single block holding them jointly.
    :meth:`~goal.geometry.exponential_family.clique.LinearCliques.clique_index` raises when it does
    not, which is the structural condition higher arity needs.
    """

    # Fields

    _amb_man: Ambient
    """The clique manifold holding the block. Last, so that a clique's own fields lead."""

    # Overrides

    @property
    @override
    def amb_man(self) -> Ambient:
        return self._amb_man

    @property
    @override
    def sub_man(self) -> MultilinearMap:
        """The selected sub-tensor --- this clique's own multilinear form."""
        return self.form

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
        out = coords.reshape(self.axes)
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


def block_clique(block: LinearMap[Any, Any], members: tuple[int, ...]) -> EFClique:
    """The clique one interaction block already is, on the nodes it is said to couple.

    A block states which coordinates it addresses through its two embeddings, and those
    *are* the clique's selectors: the codomain embedding is the first, and the domain
    embedding supplies the rest --- one selector if it addresses a single node, several if
    it addresses a joint block over a group of them. That second case is what makes a
    three-way interaction arity three: its domain is the joint statistic of two latent
    nodes, held by the level above as one block, so it contributes one factor per node
    rather than one for the pair.

    ``members`` has to be passed in. Arity is readable from the block, but *which* nodes it
    couples is not: blocks of one interaction share a domain and a codomain, so only the
    model knows.

    Raises:
        TypeError: if the block carries no embeddings, so there is nothing to read
            selectors off.
        ValueError: if ``members`` and the selectors read off the block disagree in number
            --- raised by :class:`EFClique` itself.
    """
    if not isinstance(block, EmbeddedMap):
        msg = f"{type(block).__name__} carries no embeddings to read selectors off"
        raise TypeError(msg)
    dom = block.dom_emb
    tail = dom.selectors if isinstance(dom, CliqueBlockEmbedding) else (dom,)
    return EFClique(members=members, selectors=(block.cod_emb, *tail))


### Cut Views ###


def project_cut(cut: CliqueCut, coords: Array) -> tuple[Array, Array, Array]:
    """Split a clique layout's coordinates into ``(near, cross, far)`` under ``cut``.

    ``cross`` is the crossing matrix ravelled row-major, its rows following the near side's
    clique order. A near clique with no crossing block above it still gets a row band, zero
    filled, so that the matrix has one row block per near clique and the far side is one
    shared column axis.

    Everything about *which* block goes where is decided by
    :func:`~goal.geometry.algebra.clique.cut_indices`; this is only the array work.
    """
    blocks = _split_by_dims(coords, cut.clique_dims)
    near = jnp.concatenate([blocks[i] for i in cut.near_idx])
    far = jnp.concatenate([blocks[i] for i in cut.far_idx])
    row_source = {row: pos for pos, row in enumerate(cut.cross_rows)}
    rows: list[Array] = []
    for pos, i in enumerate(cut.near_idx):
        height = cut.clique_dims[i]
        if pos in row_source:
            block = blocks[cut.cross_idx[row_source[pos]]]
            rows.append(block.reshape(height, cut.n_cols))
        else:
            rows.append(jnp.zeros((height, cut.n_cols)))
    return near, jnp.vstack(rows).ravel(), far


def join_cut(cut: CliqueCut, near: Array, cross: Array, far: Array) -> Array:
    """Reassemble a clique layout's coordinates from a ``(near, cross, far)`` view.

    The inverse of :func:`project_cut` on the blocks the cut names. Row bands that
    :func:`project_cut` zero filled are dropped rather than written back, so the round trip
    is the identity on coordinates but not on an arbitrary crossing matrix.
    """
    near_dims = tuple(cut.clique_dims[i] for i in cut.near_idx)
    near_blocks = _split_by_dims(near, near_dims)
    far_blocks = _split_by_dims(far, tuple(cut.clique_dims[i] for i in cut.far_idx))
    cross_matrix = cross.reshape(sum(near_dims), cut.n_cols)

    offsets: list[int] = []
    running = 0
    for block_dim in near_dims:
        offsets.append(running)
        running += block_dim

    blocks: list[Array | None] = [None] * len(cut.clique_dims)
    for pos, i in enumerate(cut.near_idx):
        blocks[i] = near_blocks[pos]
    for pos, i in enumerate(cut.far_idx):
        blocks[i] = far_blocks[pos]
    for pos, i in enumerate(cut.cross_idx):
        row = cut.cross_rows[pos]
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
                f"sub and ambient must share a clique set: {self.sub_man.clq_set} vs {self.amb_man.clq_set}"
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
