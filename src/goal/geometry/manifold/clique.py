"""Manifolds whose parameters are laid out over the cliques of a graph.

The clique-indexed sibling of :mod:`goal.geometry.manifold.combinators`. Where ``Pair`` and
``Triple`` concatenate their components' coordinates, a :class:`LinearClique` takes their
*tensor product*, and a :class:`LinearCliques` lays a coordinate vector out over a tuple of
those --- one per clique of a graph, so the manifold and the graph it is defined on are the
same object rather than the manifold holding a reference to one.

A clique is a container of manifolds together with a rep-backed form over their product,
read in a direction it carries: :attr:`LinearClique.out_axes` says which axes are the output
and the rest are contracted. Arity is a degree, not a kind: at arity 1 it is a bias, at arity
2 a matrix, beyond that a higher-order tensor. A clique is still not a *map*, because
applying it needs to know what manifold a caller actually holds, and how that manifold
reaches the nodes the clique couples --- which is a fact about the graph, not about the form.
Whoever wants a map pairs the form with the paths this module derives --- see
:meth:`LevelCliques.cross_paths`.

**Where a clique sits is not part of it.** A partition numbers its own nodes from zero and
can be reused at any depth; the containing layout pairs each form with the nodes it couples,
and renumbers what it receives. :attr:`LinearCliques.placements` is that pairing.
:class:`LevelCliques` is the recursive case, storing its coordinates as the three partitions
of one level ascent, and :class:`CliqueProduct` is the disjoint union, which is what a
multi-root model's root partition is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
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
    """A coupling of nodes read in a fixed direction: one embedding per node, a form over
    what they select, and the split of axes into output and input.

    Each :class:`~goal.geometry.manifold.embedding.LinearEmbedding` goes from the sub-space
    this coupling uses *into that node's own manifold*, and the parameters are the tensor
    over those sub-spaces. A clique says which nodes it couples, how much of each one, and
    which way it is read --- and nothing about where those nodes sit or how anyone reaches
    them.

    One embedding, one **axis**. Axes are positions, ``0`` to ``arity - 1``; which nodes
    they sit at is the layout's business (:attr:`LinearCliques.placements`), and each
    embedding's ``amb_man`` is the manifold the layout must find there
    (:attr:`LinearCliques.node_mans`).

    :attr:`out_axes` names the axes that form the output; the rest are contracted. That
    makes :attr:`matrix_shape` *the* shape of the parameters rather than a convention
    imposed by whoever reads them, and :attr:`dim` a property of the manifold rather than of
    a fold. A form still has $2^n$ conditional readings, but each one is a *different*
    clique over the same embeddings: :meth:`transposed` is the two-group swap, and
    :meth:`to_tensor` is axis-ordered and so view-independent, which is how any other
    reading's parameters are reached.

    Mathematically, for axes of dimensions $(d_0, \\ldots, d_{n-1})$ the parameters are a
    tensor $\\Theta \\in \\mathbb R^{d_0 \\times \\cdots \\times d_{n-1}}$, stored as the
    matrix that groups the output axes as rows and the contracted axes as columns, each
    group flattened in ascending axis order. At arity 2 with ``out_axes = (0,)`` that is the
    ordinary matrix shape; at arity 1 the empty column product is 1 and the form is a
    column, which is exactly a bias.
    """

    # Fields

    rep: MatrixRep
    """The matrix representation strategy for this clique's form."""

    node_embs: tuple[LinearEmbedding[Any, Any], ...]
    """One embedding per axis: the sub-space this coupling uses, inside that node's manifold."""

    out_axes: tuple[int, ...]
    """Which axes form the output; the rest are contracted against the input."""

    def __post_init__(self) -> None:
        axes = self.out_axes
        if any(not 0 <= axis < self.arity for axis in axes):
            msg = f"out_axes {axes} out of range for arity {self.arity}"
            raise ValueError(msg)
        if list(axes) != sorted(set(axes)):
            raise ValueError(f"out_axes {axes} must be ascending and distinct")

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
        """
        return tuple(emb.amb_man for emb in self.node_embs)

    @property
    def sub_dims(self) -> tuple[int, ...]:
        """Dimension of each axis --- the shape of the parameter tensor."""
        return tuple(emb.sub_man.dim for emb in self.node_embs)

    @property
    def arity(self) -> int:
        """Number of axes."""
        return len(self.node_embs)

    @property
    def in_axes(self) -> tuple[int, ...]:
        """The contracted axes: the ascending complement of :attr:`out_axes`."""
        out = set(self.out_axes)
        return tuple(axis for axis in range(self.arity) if axis not in out)

    @property
    def out_dims(self) -> tuple[int, ...]:
        """Selected dimension of each output axis, in ascending axis order."""
        dims = self.sub_dims
        return tuple(dims[axis] for axis in self.out_axes)

    @property
    def in_dims(self) -> tuple[int, ...]:
        """Selected dimension of each contracted axis, in ascending axis order."""
        dims = self.sub_dims
        return tuple(dims[axis] for axis in self.in_axes)

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """Output axes as the rows, contracted axes as the columns, each group flattened."""
        return (prod(self.out_dims), prod(self.in_dims))

    def transposed(self) -> LinearClique:
        """The same coupling read the other way: the two groups exchanged.

        The only re-view a structured ``rep`` can express without densifying, since it
        leaves each group's internal order alone. Its parameters are :meth:`transpose` of
        this clique's.
        """
        return LinearClique(self.rep, self.node_embs, self.in_axes)

    def transpose(self, params: Array) -> Array:
        """Reorder parameters into the layout :meth:`transposed` expects."""
        return self.rep.transpose(self.matrix_shape, params)

    def to_matrix(self, params: Array) -> Array:
        """Unpack flat parameters into a dense (output, input) matrix."""
        return self.rep.to_matrix(self.matrix_shape, params)

    def from_matrix(self, matrix: Array) -> Array:
        """Pack a dense (output, input) matrix into flat parameters."""
        return self.rep.from_matrix(matrix)

    def to_tensor(self, params: Array) -> Array:
        """View flat parameters as a tensor of shape :attr:`sub_dims`, in *axis* order.

        Axis order does not depend on which axes are the output, so this is how one reading's
        parameters become another's: ``other.from_tensor(self.to_tensor(params))``.

        Raises:
            ValueError: unless the representation stores every entry, since a structured
                ``rep`` holds fewer parameters than the tensor has entries.
        """
        self._require_dense()
        grouped = self._grouped_axes
        dims = self.sub_dims
        tensor = params.reshape(tuple(dims[axis] for axis in grouped))
        inverse = [0] * self.arity
        for position, axis in enumerate(grouped):
            inverse[axis] = position
        return jnp.transpose(tensor, tuple(inverse))

    def from_tensor(self, tensor: Array) -> Array:
        """Flatten a tensor of shape :attr:`sub_dims`, in axis order, into parameters."""
        self._require_dense()
        return jnp.transpose(tensor, self._grouped_axes).reshape(-1)

    def contract(self, params: Array, *in_node_coords: Array) -> Array:
        """Contract the input axes, leaving the output group's selected coordinates.

        One node vector per contracted axis, in ascending axis order; each axis's embedding
        restricts it before the contraction. Because the arguments multiply together this
        reads the input group as a product of marginals --- exact when every contracted node
        is observed, and wrong when two or more are latent, where the joint expectation does
        not factorize. Pass the joint through :meth:`project_in` in that case.
        """
        axes = self.in_axes
        if len(in_node_coords) != len(axes):
            msg = f"expected {len(axes)} contracted axes"
            raise ValueError(f"{msg}, got {len(in_node_coords)}")
        selected = _outer(
            tuple(
                self.node_embs[axis].project(coords)
                for axis, coords in zip(axes, in_node_coords, strict=True)
            )
        )
        return self.rep.matvec(self.matrix_shape, params, selected)

    def outer_product(self, out_joint: Array, in_joint: Array) -> Array:
        """Parameters of the outer product of an output joint with an input joint.

        Each argument is a joint over its group's *node* dimensions, flat, in ascending axis
        order --- the two sides' expectations as a layout stores them. It never forms a
        marginal, so it stays correct when the nodes within a group are dependent.
        """
        return self.rep.outer_product(
            self.project_out(out_joint), self.project_in(in_joint)
        )

    def project_out(self, joint: Array) -> Array:
        """Restrict an output-group joint from the nodes' dimensions to the selected ones."""
        return self._project_group(self.out_axes, joint)

    def embed_out(self, selected: Array) -> Array:
        """The adjoint of :meth:`project_out`: back out to the nodes' full dimensions."""
        return self._embed_group(self.out_axes, selected)

    def project_in(self, joint: Array) -> Array:
        """Restrict an input-group joint from the nodes' dimensions to the selected ones.

        An empty input group --- a bias --- has the constant $1$ as its joint.
        """
        return self._project_group(self.in_axes, joint)

    def embed_in(self, selected: Array) -> Array:
        """The adjoint of :meth:`project_in`: back out to the nodes' full dimensions."""
        return self._embed_group(self.in_axes, selected)

    # Private

    @property
    def _grouped_axes(self) -> tuple[int, ...]:
        """Axes in parameter order: the output group, then the contracted group."""
        return self.out_axes + self.in_axes

    def _project_group(self, axes: tuple[int, ...], joint: Array) -> Array:
        embs = self.node_embs
        if not axes:
            return jnp.ones(1)
        if len(axes) == 1:
            # One node: its embedding restricts directly, with no tensor to reshape. Not
            # only an optimization --- an embedding may accept a point it can restrict
            # without its ambient dimension matching exactly, and reshaping would not.
            return embs[axes[0]].project(joint)
        out = joint.reshape(tuple(embs[axis].amb_man.dim for axis in axes))
        for position, axis in enumerate(axes):
            out = map_axis(out, position, embs[axis].project)
        return out.reshape(-1)

    def _embed_group(self, axes: tuple[int, ...], selected: Array) -> Array:
        embs = self.node_embs
        if not axes:
            return jnp.zeros(0)
        if len(axes) == 1:
            return embs[axes[0]].embed(selected)
        dims = self.sub_dims
        out = selected.reshape(tuple(dims[axis] for axis in axes))
        for position, axis in enumerate(axes):
            out = map_axis(out, position, embs[axis].embed)
        return out.reshape(-1)

    def _require_dense(self) -> None:
        if self.rep.num_params(self.matrix_shape) != prod(self.sub_dims):
            msg = f"{type(self.rep).__name__} stores fewer parameters than the tensor"
            raise ValueError(f"{msg} of shape {self.sub_dims} has entries")


def node_clique(node_man: Manifold) -> LinearClique:
    """The clique holding a manifold whole at one node: arity one, a bias.

    Where the recursion in :meth:`LinearCliques.placements_of` bottoms out, and what a level
    supplies when it holds a structured partition as a single node rather than expanding it.
    """
    return LinearClique(Rectangular(), (IdentityEmbedding(node_man),), (0,))


def _outer(coords: tuple[Array, ...]) -> Array:
    """Flat outer product of one vector per axis; the constant $1$ when there are none."""
    out = jnp.ones(1)
    for part in coords:
        out = jnp.tensordot(out, part, axes=0)
    return out.reshape(-1)


def map_axis(tensor: Array, axis: int, fn: Any) -> Array:
    """Apply a vector function along one axis of a tensor, replacing that axis."""
    moved = jnp.moveaxis(tensor, axis, 0)
    trailing = moved.shape[1:]
    columns = moved.reshape(moved.shape[0], -1)
    mapped = jax.vmap(fn, in_axes=1, out_axes=1)(columns)
    return jnp.moveaxis(mapped.reshape((-1, *trailing)), 0, axis)


### Clique Embeddings ###


@dataclass(frozen=True)
class CliqueEmbedding[Ambient: LinearCliques](LinearEmbedding[LinearClique, Ambient]):
    """The inclusion of one clique of a layout into that layout: pure addressing.

    Say which nodes you want and this finds the form holding them *jointly* --- ``project``
    slices its coordinates out, ``embed`` scatters them back into a zero ambient vector.
    Nothing is restricted on the way: the result is the clique's own form, over the *full*
    coordinates of the nodes it couples. Which sub-space of those a coupling actually uses
    is that coupling's :attr:`LinearClique.node_embs`, kept apart so that an embedding's
    ``amb_man`` is always a node's manifold and a layout can check it.

    A form over several nodes is a tensor that need not factorize across them, so reaching
    them together is a slice, not a composition of per-node reaches. That is what this class
    is for: a single node would need only an offset, but a coupling reaching *several* at
    once can only get at them jointly, and only a layout holds them that way.

    Build one with :meth:`LinearCliques.clique_emb` rather than directly. The ambient
    manifold must have a clique on exactly those nodes; :meth:`LinearCliques.clique_index`
    raises when it does not.
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


def _validate_placement(members: tuple[int, ...], arity: int) -> None:
    """The two rules pairing a form with nodes has to satisfy.

    **One axis per node**, since this is the only place a form's arity and a layout's node
    list meet. **Distinct nodes, ascending**, so a clique on a given node set has exactly
    one spelling and :meth:`LinearCliques.clique_index` can find it by naming its nodes.
    Node *labels* are otherwise free; it is only their order within a clique that is fixed,
    because that is what pairs them with the form's axes.

    Raises:
        ValueError: if the number of nodes and the arity disagree, or the nodes repeat or
            descend.
    """
    if len(members) != arity:
        msg = f"clique {members} names {len(members)} nodes"
        raise ValueError(f"{msg} but its form has arity {arity}")
    if tuple(sorted(set(members))) != members:
        raise ValueError(f"clique {members} must name distinct nodes, ascending")


def _shift_placements(
    placements: tuple[tuple[tuple[int, ...], LinearClique], ...], offset: int
) -> tuple[tuple[tuple[int, ...], LinearClique], ...]:
    """Renumber a partition's cliques into an outer frame.

    A partition numbers its own nodes from zero, so whatever contains it moves them past the
    labels already in use. Only the address changes --- the forms are untouched.
    """
    return tuple((tuple(i + offset for i in ms), form) for ms, form in placements)


@dataclass(frozen=True)
class LinearCliques(Cliques, Manifold, ABC):
    """A manifold whose parameters are laid out over the cliques of a graph.

    Each clique carries a linear form over its nodes, which is what the *linear* names. The
    manifold **is** its graph: :attr:`cliques` reads the cover off :attr:`placements`, so
    there is no second description to disagree with the first. :attr:`placements` is the
    primitive --- one ``(members, form)`` pair per clique, in storage order --- so an offset
    and the nodes it belongs to always come from the same record. :class:`LevelCliques` is
    the recursive case and :class:`CliqueProduct` the disjoint union.

    Layout order is *storage* order, the order the forms occupy in the flat coordinate
    vector. Nothing at runtime requires it to match
    :attr:`~goal.geometry.algebra.clique.Cliques.canonical_cliques`, because
    :meth:`clique_offsets` and :meth:`clique_index` both read the layout's own cliques. It
    does match for every model the library ships, and ``tests/graphical.py`` enforces that
    over all of them.
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
        """The sum of the cliques' dimensions: a layout holds one form per clique and nothing besides."""
        return sum(self.clique_dims)

    @property
    @override
    def cliques(self) -> tuple[tuple[int, ...], ...]:
        """Which nodes each form couples, in storage order --- the graph, read off the layout.

        Also where each pairing is checked. Every clique fact --- levels, boundary, the
        level split, canonical order --- follows from this and :attr:`root_nodes`.
        """
        out: list[tuple[int, ...]] = []
        for members, form in self.placements:
            _validate_placement(members, form.arity)
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

        Derived, not declared: every clique states what it expects at each node it touches,
        so there is no second description to drift. The content is the *agreement* --- a
        node touched by several cliques is described by each of them, and they have to say
        the same thing.

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

        A partition that is already a clique manifold says what its cliques are; anything
        else occupies one node and contributes one form holding it whole. The single place
        the recursion in :class:`LevelCliques` and :class:`CliqueProduct` bottoms out.
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
        writes one clique's coordinates without naming an index. This is what
        :meth:`LevelCliques.cross_paths` hands back as a path.
        """
        return CliqueEmbedding(tuple(sorted(members)), self)

    def clique_index(self, members: tuple[int, ...]) -> int:
        """Layout position of the clique on exactly ``members``.

        Raises:
            ValueError: if no clique covers exactly those nodes, which is the structural
                condition a coupling into this manifold needs.
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

        Reads the layout rather than
        :attr:`~goal.geometry.algebra.clique.Cliques.canonical_cliques`, so the positions
        computed and the dimensions selected come from the same placements.
        """
        return CliqueCut(self.cliques, self.clique_dims, far_node)


@dataclass(frozen=True)
class LevelCliques[Root: Manifold, Cross: Manifold, Deep: Manifold](
    LinearCliques, Tuple, ABC
):
    """Clique manifold laid out as the three partitions of one level ascent.

    Coordinates are stored as ``[root | cross | deep]``: the parameters carried by the root
    nodes, the interactions joining the root nodes to the rest of the graph, and everything
    above. The deep partition is the graph one level up, so the same split applies again
    there --- recursion over the graph is a sequence of these. Levels are distances in the
    glued graph, so how the deep partition roots *itself* is discarded: that is what makes a
    fork depth two rather than depth three.

    Unlike ``Pair`` and ``Triple``, the components are not arbitrary: the graph says what
    each one is, and a partition may hold several cliques --- ``deep`` always does past
    depth two --- which is why the three partitions are named rather than the cliques.
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

        The cross partition is a bare parameter manifold --- a model supplies its
        interaction as a map --- so which nodes each of its blocks couples is the part only
        the model knows, in this level's own labels: the root partition's, and the deep
        partition's renumbered past them by :attr:`placements`.
        """

    # Overrides

    @property
    def root_placements(self) -> tuple[tuple[tuple[int, ...], LinearClique], ...]:
        """The root partition's cliques, and so how many nodes it occupies.

        By default a structured root partition is *expanded*, contributing one node per node
        of its own: that is what gives a multi-root model like probabilistic CCA its two
        root nodes. Override with a single clique --- ``(((0,), node_clique(self.root_man)),)``
        --- to hold a structured partition as one node instead, which is right whenever this
        level's interaction couples the partition as a unit rather than factoring across its
        nodes. The choice is not free: :attr:`cross_placements` names nodes in this level's
        frame, so expanding the root partition changes what those names mean.
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

        The root partition's are already in this level's frame, and
        :attr:`cross_placements` reports its own. The deep partition's keep the order and
        the relative labels the deep manifold gives them, renumbered past the root nodes ---
        so however the glued graph reroots, a deep clique still says which nodes it couples.
        """
        offset = max(self.root_nodes) + 1
        return (
            self.root_placements
            + self.cross_placements
            + _shift_placements(self.placements_of(self.deep_man), offset)
        )

    def cross_placement(
        self, rep: MatrixRep, node_embs: Mapping[int, LinearEmbedding[Any, Any]]
    ) -> tuple[tuple[int, ...], LinearClique]:
        """A crossing clique built from the nodes it couples and what it uses at each.

        The geometric statement is all a model has to make: a *set* of nodes, the sub-space
        this coupling uses inside each one, and a representation. Everything positional
        follows. The storage order is the nodes' own ascending order --- the canonical order
        a layout stores its cliques in anyway --- and the output group is the one root node
        among them, since a crossing clique is by definition what joins a root to the
        depths. So no model writes an axis number, and a form cannot be paired with a node
        list that disagrees with it: the node set *is* the arity.

        Raises:
            ValueError: if the nodes do not include exactly one root node, which is what
                makes the clique a *crossing* one.
        """
        members = tuple(sorted(node_embs))
        roots = tuple(node for node in members if node in self.root_nodes)
        if len(roots) != 1:
            msg = f"crossing clique {members} touches {len(roots)} root nodes"
            raise ValueError(f"{msg}, not exactly one")
        form = LinearClique(
            rep,
            tuple(node_embs[node] for node in members),
            (members.index(roots[0]),),
        )
        return members, form

    def cross_paths(
        self, placement: tuple[tuple[int, ...], LinearClique]
    ) -> tuple[LinearEmbedding[Any, Any] | None, LinearEmbedding[Any, Any] | None]:
        """How a crossing clique's two sides are reached: the root path and the deep path.

        A crossing clique couples one root node to a group of deep ones. A *path* says how a
        partition's coordinates reach the nodes in question --- it is the partition's own
        :meth:`~LinearCliques.clique_emb`, and ``None`` exactly when the partition *is* that
        node already, as for a chain's latent or an unexpanded root. Whoever needs a
        conditional reading of the clique --- a harmonium's interaction --- pairs the form
        with these.

        Which axes the form is read out of is **derived here too**, from the same root split
        that yields the paths, and a form declaring anything else is rejected. Otherwise the
        graph and the form would each carry the reading independently, and a disagreement
        between them would silently transpose an interaction rather than fail.

        Raises:
            ValueError: if the clique does not couple exactly one root node, which is what
                makes it a *crossing* clique, or if the form's :attr:`LinearClique.out_axes`
                is not the axis that root node sits at.
        """
        members, form = placement
        roots = self.root_nodes
        near = tuple(i for i in members if i in roots)
        if len(near) != 1:
            msg = f"crossing clique {members} touches {len(near)} root nodes"
            raise ValueError(f"{msg}, not exactly one")
        out_axes = (members.index(near[0]),)
        if form.out_axes != out_axes:
            msg = f"crossing clique {members} is read out of axes {form.out_axes}"
            raise ValueError(f"{msg}, but its root node {near[0]} sits at {out_axes}")
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
        :meth:`~goal.geometry.algebra.clique.Cliques.level_split` --- not from the
        partitions' own dimensions. Both readings exist and must agree, and taking the split
        from the placements is what makes the agreement structural.
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
        """The domain-facing name for :meth:`split_coords`.

        A graph of depth one has an empty cross and deep partition.
        """
        return self.split_coords(coords)

    def join_level(self, root: Array, cross: Array, deep: Array) -> Array:
        """The domain-facing name for :meth:`join_coords`."""
        return self.join_coords(root, cross, deep)


@dataclass(frozen=True)
class CliqueProduct[Fst: Manifold, Snd: Manifold](LinearCliques, Pair[Fst, Snd], ABC):
    """Two groups of nodes side by side, with no clique joining them.

    The graph is the disjoint union of the components' graphs, the second component's
    indices shifted past the first, and the parameter layout is the components'
    concatenated. Every node is a root: nothing links the two sides, so a non-root node
    would have no path from the root set at all.

    This is the shape a **multi-root** model's root partition takes --- two root partitions
    coupled to one shared node above, as in probabilistic CCA. A level whose root partition
    is one of these has as many root nodes as the product has components, which is what lets
    its crossing cliques fan out to more than one of them.
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
        return fst + _shift_placements(self.placements_of(self.snd_man), offset)


### Partition Embeddings ###


@dataclass(frozen=True)
class RootEmbedding[
    Sub: LevelCliques[Any, Any, Any],
    Ambient: LevelCliques[Any, Any, Any],
](LinearEmbedding[Sub, Ambient]):
    """Embeds one clique manifold into another over the same graph, transforming only the root partition.

    Use this when two models share a graph but parameterize the root nodes differently ---
    one restricting its root partition to a submanifold of the other's. The cross and deep
    partitions pass through unchanged, so a difference deeper in the graph is expressed by
    nesting: the deep manifolds are themselves clique manifolds related by their own
    ``RootEmbedding``.

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
