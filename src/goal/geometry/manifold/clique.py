"""Manifolds whose parameters are laid out over the cliques of a graph.

The clique-indexed sibling of :mod:`goal.geometry.manifold.combinators`. Where ``Pair`` and
``Triple`` concatenate their components' coordinates, a :class:`LinearClique` takes their
*tensor product*, and a :class:`LinearCliques` lays a coordinate vector out over a tuple of
those --- one per clique of a graph, so the manifold and the graph it is defined on are the
same object rather than the manifold holding a reference to one.

A clique is a container of manifolds together with a rep-backed form over their product.
Arity is a degree, not a kind: at arity 1 it is a bias, at arity 2 a matrix, beyond that a
higher-order tensor. It is *not* a map: a form has $2^n$ conditional readings and privileges
none, and picking one needs to know what manifold a caller actually holds, which is a fact
about the graph. Whoever wants a map builds one from the form and the paths this module
derives --- see :meth:`LevelCliques.cross_paths`.

**Where a clique sits is not part of it.** Node numbering is a fact about position in a
graph rather than about what occupies a position, so a partition numbers its own nodes from zero
and can be reused at any depth; the containing layout pairs each form with the nodes it
couples, and renumbers what it receives. :attr:`LinearCliques.placements` is that pairing.
:class:`LevelCliques` is the recursive case, storing its coordinates as the three partitions of
one level ascent, and :class:`CliqueProduct` is the disjoint union, which is what a
multi-root model's root partition is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from typing import Any, Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ..algebra.clique import Cliques
from ..algebra.matrix import MatrixRep, Rectangular
from .base import Manifold
from .combinators import Pair, Tuple
from .cut import CliqueCut
from .embedding import IdentityEmbedding, LinearEmbedding
from .util import split_by_dims

### Linear Cliques ###


@dataclass(frozen=True)
class LinearClique(Manifold):
    """A coupling of nodes: one embedding per node, and a form over what they select.

    What a :class:`LinearCliques` is laid out over, and where its name comes from. Each
    :class:`~goal.geometry.manifold.embedding.LinearEmbedding` goes from the sub-space this
    coupling uses *into that node's own manifold*, and the parameters are the tensor over
    those sub-spaces. That is the whole of it: a clique says which nodes it couples and how
    much of each one, and nothing about where those nodes sit or how anyone reaches them.

    One embedding, one **axis**. Axes are positions, ``0`` to ``arity - 1``; which nodes
    they sit at is the layout's business, assigned by :attr:`LinearCliques.placements`, and each
    embedding's ``amb_man`` is the manifold the layout must find there --- see
    :attr:`LinearCliques.node_mans`.

    Mathematically, for axes of dimensions $(d_0, \\ldots, d_{n-1})$ the parameters are a
    tensor $\\Theta \\in \\mathbb R^{d_0 \\times \\cdots \\times d_{n-1}}$, stored under the
    **canonical fold**: axis $0$ is the rows and the rest are the columns, flattened in
    order, so :attr:`matrix_shape` is $(d_0, \\prod_{i > 0} d_i)$. At arity 2 that is the
    ordinary matrix shape; at arity 1 the empty product is 1 and the form is a column, which
    is exactly a bias.

    Every conditional reading is :meth:`partial_contract` over a choice of which axes to
    keep --- there are $2^n$ of them and the class privileges none. Materializing one as a
    :class:`~goal.geometry.manifold.map.LinearMap` needs to know what manifold the caller
    holds, which only a layout does, so no such reading lives here.
    """

    # Fields

    rep: MatrixRep
    """The matrix representation strategy for this clique's form."""

    node_embs: tuple[LinearEmbedding[Any, Any], ...]
    """One embedding per axis: the sub-space this coupling uses, inside that node's manifold."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """What the representation stores, which for a structured ``rep`` is less than the tensor holds."""
        return self.rep.num_params(self.matrix_shape)

    # Methods

    @property
    def node_mans(self) -> tuple[Manifold, ...]:
        """The manifold each axis expects at its node, in axis order.

        What a layout checks its cliques against --- see :attr:`LinearCliques.node_mans`.
        An embedding goes from a sub-space *into a node*, never into whatever larger
        manifold a caller happens to hold, which is what makes the check possible.
        """
        return tuple(emb.amb_man for emb in self.node_embs)

    @property
    def sub_mans(self) -> tuple[Manifold, ...]:
        """The manifold each axis is over --- the sub-space its embedding picks out."""
        return tuple(emb.sub_man for emb in self.node_embs)

    @property
    def sub_dims(self) -> tuple[int, ...]:
        """Dimension of each axis --- the shape of the parameter tensor."""
        return tuple(man.dim for man in self.sub_mans)

    @property
    def arity(self) -> int:
        """Number of axes."""
        return len(self.node_embs)

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """The canonical fold: axis 0 is the rows, the rest the flattened columns.

        At arity 1 the column product is empty, hence 1, and the form is a column vector.
        """
        dims = self.sub_dims
        return (dims[0], prod(dims[1:]))

    def to_tensor(self, params: Array) -> Array:
        """View flat parameters as a tensor of shape :attr:`sub_dims`.

        Raises:
            ValueError: unless the representation stores every entry, since a structured
                ``rep`` holds fewer parameters than the tensor has entries.
        """
        self._require_dense()
        return params.reshape(self.sub_dims)

    def from_tensor(self, tensor: Array) -> Array:
        """Flatten a tensor of shape :attr:`sub_dims` into parameters."""
        self._require_dense()
        return tensor.reshape(-1)

    def to_matrix(self, params: Array) -> Array:
        """Unpack flat parameters into a dense 2D matrix under the canonical fold."""
        return self.rep.to_matrix(self.matrix_shape, params)

    def from_matrix(self, matrix: Array) -> Array:
        """Pack a dense 2D matrix under the canonical fold into flat parameters."""
        return self.rep.from_matrix(matrix)

    def tensor(self, *node_coords: Array) -> Array:
        """Outer product of one node vector per axis, as flat parameters.

        Each argument is a point of that axis's node manifold, which its embedding restricts
        before the product is taken. Because it multiplies the axes together it builds a
        product of marginals: exact when every node is observed, and not what the parameters
        look like when two or more are latent, where the joint expectation does not
        factorize. For that case see :meth:`select_joint`.
        """
        if len(node_coords) != self.arity:
            raise ValueError(f"expected {self.arity} axes, got {len(node_coords)}")
        out = self.node_embs[0].project(node_coords[0])
        for emb, coords in zip(self.node_embs[1:], node_coords[1:], strict=True):
            out = jnp.tensordot(out, emb.project(coords), axes=0)
        return out.reshape(-1)

    def contract(self, params: Array, keep: int, *node_coords: Array) -> Array:
        """Contract every axis but ``keep``, embedded back into that node's coordinates.

        ``node_coords`` supplies one node vector per contracted axis, in ascending axis
        order.
        """
        if not 0 <= keep < self.arity:
            msg = f"keep must be in 0..{self.arity - 1}, got {keep}"
            raise ValueError(msg)
        contracted = self.partial_contract(params, (keep,), *node_coords)
        return self.node_embs[keep].embed(contracted)

    def partial_contract(
        self, params: Array, keep: tuple[int, ...], *node_coords: Array
    ) -> Array:
        """Contract several axes at once, leaving a joint tensor over ``keep``.

        The general conditional reading, and the one a level split needs: nodes on the near
        side of a cut are contracted against their own coordinates, and nodes on the far
        side are left *joined*, because their expectations do not factorize. ``keep`` names
        the surviving axes and ``node_coords`` supplies one node vector per contracted axis,
        both in ascending axis order. The result is flat over the kept axes' *selected*
        dimensions, with no embedding applied.
        """
        dropped = [axis for axis in range(self.arity) if axis not in keep]
        if len(node_coords) != len(dropped):
            raise ValueError(f"expected {len(dropped)} axes, got {len(node_coords)}")
        out = self.to_tensor(params)
        # Descending order so that contracting one axis does not shift the next.
        for axis, coords in sorted(zip(dropped, node_coords), key=lambda p: -p[0]):
            projected = self.node_embs[axis].project(coords)
            out = jnp.tensordot(out, projected, axes=([axis], [0]))
        return out.reshape(-1)

    def reorder(self, params: Array, keep: tuple[int, ...]) -> Array:
        """Permute parameters so the ``keep`` axes lead, the rest following in order.

        The coordinate half of re-viewing a form: which axes are its rows and which its
        columns. At arity 2 with ``keep=(1,)`` this is transposition; above it, the general
        axis permutation.
        """
        rest = tuple(i for i in range(self.arity) if i not in keep)
        return jnp.transpose(self.to_tensor(params), keep + rest).reshape(-1)

    def select_joint(self, keep: tuple[int, ...], joint: Array) -> Array:
        """Restrict a joint form over the ``keep`` axes to this clique's sub-spaces.

        ``joint`` is a tensor over those axes' *node* dimensions, flat, in ascending axis
        order --- their joint expectation as a layout stores it. Each embedding is
        contracted into the corresponding axis. This is the operation :meth:`tensor` cannot
        do: it never forms a marginal, so it stays correct when the kept nodes are
        dependent.
        """
        embs = self.node_embs
        out = joint.reshape(tuple(embs[axis].amb_man.dim for axis in keep))
        for position, axis in enumerate(keep):
            out = map_axis(out, position, embs[axis].project)
        return out.reshape(-1)

    def embed_joint(self, keep: tuple[int, ...], selected: Array) -> Array:
        """The adjoint of :meth:`select_joint`: back out to the nodes' full dimensions."""
        embs = self.node_embs
        out = selected.reshape(tuple(self.sub_dims[axis] for axis in keep))
        for position, axis in enumerate(keep):
            out = map_axis(out, position, embs[axis].embed)
        return out.reshape(-1)

    # Private

    def _require_dense(self) -> None:
        if self.rep.num_params(self.matrix_shape) != prod(self.sub_dims):
            msg = f"{type(self.rep).__name__} stores fewer parameters than the tensor"
            raise ValueError(f"{msg} of shape {self.sub_dims} has entries")


def node_clique(node_man: Manifold) -> LinearClique:
    """The clique holding a manifold whole at one node: arity one, a bias.

    Where the recursion in :meth:`LinearCliques.placements_of` bottoms out --- what a
    partition with no clique structure of its own contributes --- and what a level supplies
    when it holds a structured partition as a single node rather than expanding it.
    """
    return LinearClique(Rectangular(), (IdentityEmbedding(node_man),))


### Clique Embeddings ###


def map_axis(tensor: Array, axis: int, fn: Any) -> Array:
    """Apply a vector function along one axis of a tensor, replacing that axis."""
    moved = jnp.moveaxis(tensor, axis, 0)
    trailing = moved.shape[1:]
    columns = moved.reshape(moved.shape[0], -1)
    mapped = jax.vmap(fn, in_axes=1, out_axes=1)(columns)
    return jnp.moveaxis(mapped.reshape((-1, *trailing)), 0, axis)


@dataclass(frozen=True)
class CliqueEmbedding[Ambient: LinearCliques](LinearEmbedding[LinearClique, Ambient]):
    """The inclusion of one clique of a layout into that layout: pure addressing.

    Say which nodes you want and this finds the form holding them *jointly* --- ``project``
    slices its coordinates out, ``embed`` scatters them back into a zero ambient vector.
    Nothing is restricted on the way: the result is the clique's own form, over the *full*
    coordinates of the nodes it couples. Which sub-space of those a coupling actually uses
    is that coupling's :attr:`LinearClique.node_embs`, kept apart so that an embedding's
    ``amb_man`` is always a node's manifold and a layout can check it.

    That separation is why a joint address stays correct: a form over several nodes is a
    tensor that need not factorize across them, so reaching them together is a slice, not a
    composition of per-node reaches. That is the whole point of this class: for a single
    node an offset would do, but a coupling that reaches *several* nodes at once --- what
    raises a crossing clique's arity above two --- can only get at them jointly, and only a
    layout holds them that way.

    Build one with :meth:`LinearCliques.clique_emb` rather than directly. The ambient
    manifold must have a clique on exactly those nodes;
    :meth:`LinearCliques.clique_index` raises when it does not, which is the structural
    condition higher arity needs.
    """

    # Fields

    _members: tuple[int, ...]
    """Nodes this addresses, ascending and distinct, in the ambient manifold's frame."""

    _amb_man: Ambient
    """The clique manifold holding the form."""

    # Overrides

    @property
    @override
    def amb_man(self) -> Ambient:
        return self._amb_man

    @property
    @override
    def sub_man(self) -> LinearClique:
        """The form this addresses --- the layout's own, not a copy."""
        return self.amb_man.clique_forms[self._index]

    @override
    def project(self, coords: Array) -> Array:
        start, size = self._placement()
        return coords[start : start + size]

    @override
    def embed(self, coords: Array) -> Array:
        start, size = self._placement()
        full = jnp.zeros(self.amb_man.dim)
        return full.at[start : start + size].set(coords)

    # Methods

    @property
    def members(self) -> tuple[int, ...]:
        """The nodes this addresses, in the ambient manifold's frame."""
        return self._members

    # Private

    @property
    def _index(self) -> int:
        return self.amb_man.clique_index(self._members)

    def _placement(self) -> tuple[int, int]:
        """Start and size of the clique this addresses."""
        index = self._index
        return self.amb_man.clique_offsets()[index], self.amb_man.clique_dims[index]


### Clique Layouts ###


def validate_placement(members: tuple[int, ...], arity: int) -> None:
    """The two rules pairing a form with nodes has to satisfy.

    **One axis per node.** A form knows how many axes it has and a layout knows which nodes
    it couples; this is the only place the two meet, so it is the only place they can
    disagree.

    **Distinct nodes, ascending.** Node order is axis order, so a clique on a given node
    set has one spelling and only one. Without that,
    :meth:`LinearCliques.clique_index` could not find a clique by naming its nodes, and one
    node set could sit in a layout twice under two spellings.

    Node *labels* are free --- they need not start at zero, run contiguously, or ascend with
    level. It is only their order **within a clique** that is fixed, and only because it is
    what pairs them with the form's axes.

    Raises:
        ValueError: if the number of nodes and the arity disagree, or the nodes repeat or
            descend.
    """
    if len(members) != arity:
        msg = f"clique {members} names {len(members)} nodes"
        raise ValueError(f"{msg} but its form has arity {arity}")
    if tuple(sorted(set(members))) != members:
        raise ValueError(f"clique {members} must name distinct nodes, ascending")


def shift_placements(
    placements: tuple[tuple[tuple[int, ...], LinearClique], ...], offset: int
) -> tuple[tuple[tuple[int, ...], LinearClique], ...]:
    """Renumber a partition's cliques into an outer frame.

    A partition numbers its own nodes from zero, so whatever contains it moves them past the
    labels already in use. Only the address changes --- the forms are untouched, because
    where a coupling sits is not part of what it couples.
    """
    return tuple((tuple(i + offset for i in ms), form) for ms, form in placements)


class LinearCliques(Cliques, Manifold, ABC):
    """A manifold whose parameters are laid out over the cliques of a graph.

    Each clique carries a linear form over its nodes, which is what the *linear* names. The
    manifold **is** its graph: :attr:`cliques` reads the cover off :attr:`placements`, so
    there is no second description to disagree with the first.

    :attr:`placements` is the primitive: one ``(members, form)`` pair per clique, in storage
    order. Sizes and axis shapes come from the forms and the cover from the nodes, so an
    offset and the nodes it belongs to always come from the same record.
    :class:`LevelCliques` is the recursive case and :class:`CliqueProduct` the disjoint
    union; the recursion terminates wherever a partition stops being a clique manifold.

    Layout order is *storage* order --- the order the forms occupy in the flat coordinate
    vector. Nothing at runtime requires it to match
    :attr:`~goal.geometry.algebra.clique.Cliques.canonical_cliques`, because
    :meth:`clique_offsets` and :meth:`clique_index` both read the layout's own cliques. It
    does match for every model the library ships, and ``tests/graphical.py`` enforces that
    over all of them; a model whose declaration order diverges is a bug in the model, not a
    case this class handles.
    """

    # Contract

    @property
    @abstractmethod
    def placements(self) -> tuple[tuple[tuple[int, ...], LinearClique], ...]:
        """One ``(members, form)`` pair per clique, in storage order."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """The sum of the cliques' dimensions.

        A layout is its cliques: it holds one form per clique and nothing besides, so its
        size is theirs.
        """
        return sum(self.clique_dims)

    @property
    @override
    def cliques(self) -> tuple[tuple[int, ...], ...]:
        """Which nodes each form couples, in storage order.

        The graph, read off the layout, and where each pairing is checked --- see
        :func:`validate_placement`. Every clique fact --- levels, boundary, the level split,
        canonical order --- follows from this and :attr:`root_nodes`.
        """
        out: list[tuple[int, ...]] = []
        for members, form in self.placements:
            validate_placement(members, form.arity)
            out.append(members)
        return tuple(out)

    # Properties

    @property
    def clique_forms(self) -> tuple[LinearClique, ...]:
        """The forms alone, in storage order."""
        return tuple(form for _, form in self.placements)

    @property
    def clique_dims(self) -> tuple[int, ...]:
        """Dimension of each clique's form, in storage order."""
        return tuple(form.dim for _, form in self.placements)

    @property
    def clique_axes(self) -> tuple[tuple[int, ...], ...]:
        """Axis dimensions of each clique's form, parallel to :attr:`clique_dims`."""
        return tuple(form.sub_dims for _, form in self.placements)

    @property
    def node_mans(self) -> tuple[Manifold, ...]:
        """The manifold occupying each node, ascending by label, parallel to :attr:`nodes`.

        Derived, not declared. Every clique states what it expects to find at each node it
        touches --- its axis embeddings' ambient manifolds --- so the graph, the layout and
        what sits on it all come from the one record, and there is no second description to
        drift. The content is the *agreement*: a node touched by several cliques is
        described by each of them, and they have to say the same thing.

        Raises:
            ValueError: if two cliques disagree about what occupies a node, which means one
                of them is coupling a manifold that is not there.
        """
        seen: dict[int, Manifold] = {}
        for members, form in self.placements:
            for node, man in zip(members, form.node_mans, strict=True):
                known = seen.setdefault(node, man)
                if known != man:
                    msg = f"node {node} is {known} in one clique and {man} in {members}"
                    raise ValueError(msg)
        return tuple(seen[node] for node in self.nodes)

    # Methods

    @staticmethod
    def placements_of(
        partition: Manifold,
    ) -> tuple[tuple[tuple[int, ...], LinearClique], ...]:
        """The cliques a partition contributes, in the partition's own frame.

        A partition that is already a clique manifold says what its cliques are; anything else
        occupies one node and contributes one form holding it whole. This is the single
        place the recursion in :class:`LevelCliques` and :class:`CliqueProduct` bottoms out,
        and the only place a partition's structure is inspected.
        """
        if isinstance(partition, LinearCliques):
            return partition.placements
        return (((0,), node_clique(partition)),)

    def clique_offsets(self) -> tuple[int, ...]:
        """Start of each clique's coordinates in the flat parameter vector."""
        offsets: list[int] = []
        running = 0
        for size in self.clique_dims:
            offsets.append(running)
            running += size
        return tuple(offsets)

    def clique_emb(self, members: tuple[int, ...]) -> CliqueEmbedding[Self]:
        """The inclusion of the clique on exactly ``members`` into this manifold.

        The geometric way in: an embedding rather than an offset, so a caller reads or
        writes one clique's coordinates without ever naming an index. This is what
        :meth:`LevelCliques.cross_paths` hands back as a path.
        """
        return CliqueEmbedding(tuple(sorted(members)), self)

    def node_emb(self, node: int) -> CliqueEmbedding[Self]:
        """The inclusion of one node's own coordinates into this manifold.

        The singleton case of :meth:`clique_emb`, and the way a level reaches a node it
        couples. It requires the cover to *have* a singleton clique on that node --- legal
        covers need not, though every layout the library ships does.
        """
        return self.clique_emb((node,))

    def clique_index(self, members: tuple[int, ...]) -> int:
        """Layout position of the clique on exactly ``members``.

        Raises:
            ValueError: if no clique covers exactly those nodes, which is the structural
                condition a coupling into this manifold needs --- there has to be a single
                form holding those members jointly.
        """
        wanted = tuple(sorted(members))
        layout = self.cliques
        for i, clique in enumerate(layout):
            if clique == wanted:
                return i
        msg = f"no clique on {wanted}; this manifold has "
        raise ValueError(f"{msg}{layout}")

    def split_cliques(self, coords: Array) -> tuple[Array, ...]:
        """Split coordinates into one array per clique, in layout order."""
        return split_by_dims(coords, self.clique_dims)

    def join_cliques(self, *parts: Array) -> Array:
        """Concatenate one array per clique back into coordinates.

        Raises:
            ValueError: if the number of arrays or any one's size disagrees with
                :attr:`clique_dims`. Concatenation would otherwise accept wrongly-sized
                arrays and produce a coordinate vector of the wrong length.
        """
        dims = self.clique_dims
        if len(parts) != len(dims):
            raise ValueError(f"expected {len(dims)} cliques, got {len(parts)}")
        for i, (part, size) in enumerate(zip(parts, dims, strict=True)):
            if part.shape != (size,):
                msg = f"clique {i} has shape {part.shape}"
                raise ValueError(f"{msg}, expected ({size},)")
        return jnp.concatenate(parts)

    def cut(self, far_node: int) -> CliqueCut:
        """Re-view the layout with ``far_node`` split off instead of the root nodes.

        Reads the layout, not
        :attr:`~goal.geometry.algebra.clique.Cliques.canonical_cliques`, so the positions
        computed and the dimensions selected come from the same placements. Every condition
        the view requires is checked by :class:`~goal.geometry.manifold.cut.CliqueCut`
        itself.
        """
        return CliqueCut(self.cliques, self.clique_dims, far_node)


@dataclass(frozen=True)
class LevelCliques[Root: Manifold, Cross: Manifold, Deep: Manifold](
    LinearCliques, Tuple, ABC
):
    """Clique manifold laid out as the three partitions of one level ascent.

    Coordinates are stored as ``[root | cross | deep]``: the parameters carried by the root
    nodes, the interactions joining the root nodes to the rest of the graph, and everything
    above. The deep partition is the graph one level up, so the same split applies again there
    --- recursion over the graph is a sequence of these. Levels are distances in the glued
    graph, so how the deep partition roots *itself* is discarded: that is what makes a fork
    depth two rather than depth three.

    Unlike ``Pair`` and ``Triple``, the components are not arbitrary: the graph says what
    each one is. A partition may hold several cliques --- ``deep`` always does past depth two ---
    which is why the three partitions are named rather than the individual cliques.
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
    def cross_placements(self) -> tuple[tuple[tuple[int, ...], LinearClique], ...]:
        """The cliques joining the root nodes to the rest, in this level's frame.

        The cross partition is a bare parameter manifold --- a model supplies its interaction as
        a map --- so which nodes each of its blocks couples is the part only the model
        knows, in this level's own labels: the root partition's, and the deep partition's renumbered
        past them by :attr:`placements`.
        """

    # Overrides

    @property
    def root_placements(self) -> tuple[tuple[tuple[int, ...], LinearClique], ...]:
        """The root partition's cliques, and so how many nodes it occupies.

        By default a structured root partition is *expanded*, contributing one node per node of
        its own: that is what gives a multi-root model like probabilistic CCA its two root
        nodes. Override with a single clique --- ``(((0,), node_clique(self.root_man)),)``
        --- to hold a structured partition as one node instead, which is right whenever this
        level's interaction couples the partition as a unit rather than factoring across its
        nodes: a mixture over a harmonium, say, whose interaction reaches the whole
        parameter vector.

        The choice is not free: :attr:`cross_placements` names nodes in this level's frame,
        so expanding the root partition changes what those names mean.
        """
        return self.placements_of(self.root_man)

    @property
    @override
    def root_nodes(self) -> frozenset[int]:
        """Whichever nodes the root partition occupies, read off :attr:`root_placements`."""
        return frozenset(i for members, _ in self.root_placements for i in members)

    @property
    @override
    def placements(self) -> tuple[tuple[tuple[int, ...], LinearClique], ...]:
        """The partitions' cliques, concatenated in storage order.

        The root partition's are already in this level's frame, and :attr:`cross_placements`
        reports its own. The deep partition's keep the order and the relative labels the deep
        manifold itself gives them, renumbered past the root nodes --- so however the glued
        graph reroots, a deep clique still says which nodes it couples.
        """
        offset = max(self.root_nodes) + 1
        return (
            self.root_placements
            + self.cross_placements
            + shift_placements(self.placements_of(self.deep_man), offset)
        )

    def cross_paths(
        self, members: tuple[int, ...]
    ) -> tuple[LinearEmbedding[Any, Any] | None, LinearEmbedding[Any, Any] | None]:
        """How a crossing clique's two sides are reached: the root path and the deep path.

        A crossing clique couples one root node to a group of deep ones. A *path* says how
        a partition's coordinates reach the nodes in question --- it is the partition's own
        :meth:`~LinearCliques.clique_emb`, and ``None`` exactly when the partition *is* that
        node already, as for a chain's latent or an unexpanded root.

        **Derived, not declared.** Before this, a branch of probabilistic CCA declared
        ``FirstEmbedding(pair)`` and a hierarchical mixture declared
        ``ObservableEmbedding(upper)``; both reproduce ``clique_emb`` of the node they name,
        to the coordinate. Whoever needs a conditional reading of the clique --- a
        harmonium's interaction --- pairs the form with these.

        Raises:
            ValueError: if the clique does not couple exactly one root node, which is what
                makes it a *crossing* clique.
        """
        roots = self.root_nodes
        near = tuple(i for i in members if i in roots)
        if len(near) != 1:
            msg = f"crossing clique {members} touches {len(near)} root nodes"
            raise ValueError(f"{msg}, not exactly one")
        offset = max(roots) + 1
        far = tuple(i - offset for i in members if i not in roots)
        return (
            self._partition_emb(self.root_man, near[0], len(roots) > 1),
            self._partition_emb(self.deep_man, far, len(far) > 0),
        )

    @staticmethod
    def _partition_emb(
        partition: Manifold, nodes: Any, needed: bool
    ) -> LinearEmbedding[Any, Any] | None:
        """How a partition reaches the nodes given, or ``None`` when it *is* them."""
        if not needed or not isinstance(partition, LinearCliques):
            return None
        wanted = nodes if isinstance(nodes, tuple) else (nodes,)
        if len(partition.nodes) == 1:
            return None
        return partition.clique_emb(wanted)

    @override
    def split_coords(self, coords: Array) -> tuple[Array, Array, Array]:
        """Split coordinates into the root, cross, and deep partitions.

        The two offsets come from the *placements*, via
        :meth:`~goal.geometry.algebra.clique.Cliques.level_split` --- not from the partitions'
        own dimensions. Both readings exist and must agree, and taking the split from the
        placements is what makes the agreement structural instead of a coincidence nothing
        checks.
        """
        root_idx, cross_idx, _ = self.level_split()
        dims = self.clique_dims
        root_dim = sum(dims[i] for i in root_idx)
        cross_dim = root_dim + sum(dims[i] for i in cross_idx)
        return coords[:root_dim], coords[root_dim:cross_dim], coords[cross_dim:]

    @override
    def join_coords(self, *components: Array) -> Array:
        """Concatenate the root, cross, and deep partitions."""
        if len(components) != 3:
            raise ValueError(f"expected 3 partitions, got {len(components)}")
        return jnp.concatenate(components)

    # Methods

    def split_level(self, coords: Array) -> tuple[Array, Array, Array]:
        """Split off one level: the root partition, the cross partition, and the deep partition.

        The domain-facing name for :meth:`split_coords`. A graph of depth one has an empty
        cross and deep partition.
        """
        return self.split_coords(coords)

    def join_level(self, root: Array, cross: Array, deep: Array) -> Array:
        """Concatenate the root, cross, and deep partitions."""
        return self.join_coords(root, cross, deep)


@dataclass(frozen=True)
class CliqueProduct[Fst: Manifold, Snd: Manifold](LinearCliques, Pair[Fst, Snd], ABC):
    """Two groups of nodes side by side, with no clique joining them.

    The graph is the disjoint union of the components' graphs, the second component's
    indices shifted past the first, and the parameter layout is the components'
    concatenated. Every node is a root: nothing links the two sides, so a non-root node
    would have no path from the root set at all.

    This is the shape a **multi-root** model's root partition takes --- two root partitions coupled to
    one shared node above, as in probabilistic CCA. A level whose root partition is one of these
    has as many root nodes as the product has components, which is what lets its crossing
    cliques fan out to more than one of them.
    """

    # Overrides

    @property
    @override
    def root_nodes(self) -> frozenset[int]:
        """Every node: nothing links the two sides, so none is above another."""
        return frozenset(self.nodes)

    @property
    @override
    def placements(self) -> tuple[tuple[tuple[int, ...], LinearClique], ...]:
        fst = self.placements_of(self.fst_man)
        offset = max(max(members) for members, _ in fst) + 1
        return fst + shift_placements(self.placements_of(self.snd_man), offset)


### Partition Embeddings ###


@dataclass(frozen=True)
class RootEmbedding[
    Sub: LevelCliques[Any, Any, Any],
    Ambient: LevelCliques[Any, Any, Any],
](LinearEmbedding[Sub, Ambient]):
    """Embeds one clique manifold into another over the same graph, transforming only the root partition.

    Use this when two models share a graph but parameterize the root nodes differently ---
    one restricting its root partition to a submanifold of the other's. The cross and deep partitions
    pass through unchanged, so a difference deeper in
    the graph is expressed by nesting: the deep manifolds are themselves clique manifolds
    related by their own ``RootEmbedding``.

    Mathematically, for partition coordinates $(r, c, d)$: ``embed`` maps $(r, c, d) \\mapsto
    (\\phi(r), c, d)$ and ``project`` maps $(r, c, d) \\mapsto (\\pi(r), c, d)$, where
    $\\phi$ and $\\pi$ are the root embedding's own maps.
    """

    # Fields

    root_emb: LinearEmbedding[Any, Any]
    """Embedding of the restricted root manifold into the full one."""

    _sub_man: Sub
    """The clique manifold with the restricted root partition."""

    _amb_man: Ambient
    """The clique manifold with the full root partition."""

    def __post_init__(self) -> None:
        if not self.sub_man.same_graph(self.amb_man):
            msg = "sub and ambient must share a clique set: "
            sub_roots = sorted(self.sub_man.root_nodes)
            amb_roots = sorted(self.amb_man.root_nodes)
            msg += f"{self.sub_man.cliques} rooted at {sub_roots}"
            raise ValueError(f"{msg} vs {self.amb_man.cliques} rooted at {amb_roots}")
        for name, sub_span, amb_span in (
            ("cross", self.sub_man.cross_man, self.amb_man.cross_man),
            ("deep", self.sub_man.deep_man, self.amb_man.deep_man),
        ):
            if sub_span.dim != amb_span.dim:
                msg = f"{name} partitions differ: {sub_span.dim} vs {amb_span.dim}"
                raise ValueError(f"{msg}; only the root partition may be transformed")
        for name, partition, emb_man in (
            ("sub", self.sub_man.root_man, self.root_emb.sub_man),
            ("ambient", self.amb_man.root_man, self.root_emb.amb_man),
        ):
            if partition.dim != emb_man.dim:
                msg = f"{name} root partition has dimension {partition.dim}"
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
