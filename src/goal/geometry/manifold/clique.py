"""Manifolds whose parameters are laid out over the cliques of a graph.

The clique-indexed sibling of :mod:`goal.geometry.manifold.combinators`. Where ``Pair`` and
``Triple`` concatenate their components' coordinates, a :class:`SubspaceMap` takes their
*tensor product* (a :class:`Product`), and a
:class:`LinearCliques` lays a coordinate vector out over a tuple of those --- one per clique
of a graph, so the manifold and the graph it is defined on are the same object rather than
the manifold holding a reference to one.

**This module is the addressing, not the algebra.** A ``SubspaceMap`` knows its factors and
what subspace it uses in each, and nothing about where those factors sit; giving them
*positions* is what happens here. A clique is a ``SubspaceMap`` paired with the nodes its
factors occupy --- :attr:`LinearCliques.placements` is that pairing, member $i$ naming the
node of factor $i$ in ``cod_embs + dom_embs`` order. Reaching those nodes from a whole partition is a *path*,
derived by :meth:`LevelCliques.cross_paths`, and
:class:`~goal.geometry.manifold.interaction.Interaction` is what sums several
path-conjugated maps into one map between partitions.

**Where a clique sits is not part of it.** A partition numbers its own nodes from zero and
can be reused at any depth; the containing layout renumbers what it receives.
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
from .embedding import IdentityEmbedding, LinearEmbedding
from .map import LinearMap
from .util import split_by_dims

### Subspace Maps ###


@dataclass(frozen=True)
class Product(Manifold):
    """Tensor product of manifolds, with coordinates stored as the flattened joint.

    The other way to combine manifolds: where a :class:`~goal.geometry.manifold.combinators.Tuple` concatenates its components'
    coordinates, this takes their outer product, so the dimension is the *product* of the
    factors' rather than the sum. That is why it cannot be a ``Tuple`` --- a joint has no
    decomposition by slicing.

    The empty product has dimension one, its single coordinate being the constant $1$.
    Compare :class:`~goal.geometry.manifold.combinators.Null`, the empty Cartesian product, which has dimension zero.

    Mathematically, $\\mathcal M_1 \\otimes \\cdots \\otimes \\mathcal M_k$ with $\\dim =
    \\prod_i \\dim(\\mathcal M_i)$.
    """

    # Fields

    factors: tuple[Manifold, ...]
    """The manifolds whose joint this is, in factor order."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return prod(factor.dim for factor in self.factors)


@dataclass(frozen=True)
class SubspaceMap(LinearMap[Manifold, Manifold]):
    """A linear map acting on a chosen subspace of each manifold it couples.

    The manifolds are its **factors**, one per
    :class:`~goal.geometry.manifold.embedding.LinearEmbedding`, which names the subspace
    this map uses inside that factor: :attr:`cod_embs` are the codomain factors,
    :attr:`dom_embs` the contracted ones, and :attr:`trn_man` is the two swapped. Applying
    the map uses *both* directions of every embedding --- it projects the incoming point and
    embeds the outgoing one --- and they are not inverses, since ``embed`` takes natural
    parameters and ``project`` takes mean parameters. Arity is a degree, not a kind: one
    factor against no domain factors is a bias, one a side a matrix, beyond that a
    higher-order tensor.

    Mathematically, for codomain subspaces of dimensions $(d_1, \\ldots, d_j)$ and domain
    subspaces $(e_1, \\ldots, e_k)$ the parameters are a tensor $\\Theta \\in \\mathbb
    R^{d_1 \\times \\cdots \\times d_j \\times e_1 \\times \\cdots \\times e_k}$, stored as
    the *matricization* that groups the codomain axes as rows and the domain axes as
    columns. One factor a side is the ordinary matrix shape; with no domain factors the
    column count is the empty product $1$, so the map is a column, which is exactly a bias.
    """

    # Fields

    rep: MatrixRep
    """The matrix representation strategy for this map's parameters."""

    cod_embs: tuple[LinearEmbedding[Any, Any], ...]
    """One embedding per codomain factor: the subspace the output lands in, inside that factor."""

    dom_embs: tuple[LinearEmbedding[Any, Any], ...]
    """One embedding per contracted factor, likewise."""

    # Overrides

    @property
    @override
    def dom_man(self) -> Product:
        """The input group's joint ambient space."""
        return Product(tuple(emb.amb_man for emb in self.dom_embs))

    @property
    @override
    def cod_man(self) -> Product:
        """The output group's joint ambient space."""
        return Product(tuple(emb.amb_man for emb in self.cod_embs))

    @property
    @override
    def dim(self) -> int:
        """What the representation stores, which for a structured ``rep`` is less than the tensor holds."""
        return self.rep.num_params(self.matrix_shape)

    @property
    @override
    def trn_man(self) -> SubspaceMap:
        """The same tensor read the other way: the two groups swapped.

        Its parameters are :meth:`transpose` of this map's. A bias's transpose is a
        functional --- one row over the factor's coordinates.
        """
        return SubspaceMap(self.rep, self.dom_embs, self.cod_embs)

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        """Contract the domain factors' joint ambient coordinates; return the codomain factors'."""
        selected = self.project_dom(v_coords)
        return self.embed_cod(self.rep.matvec(self.matrix_shape, f_coords, selected))

    @override
    def transpose(self, f_coords: Array) -> Array:
        """Reorder parameters into the matricization :attr:`trn_man` expects."""
        return self.rep.transpose(self.matrix_shape, f_coords)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        """Parameters of the outer product of an output joint with an input joint.

        Each argument is a joint over its group's *ambient* dimensions, flat, in factor
        order. It never forms a marginal, so it stays correct when the factors within a
        group are dependent --- which a product of per-factor vectors would not be.
        """
        return self.rep.outer_product(
            self.project_cod(w_coords), self.project_dom(v_coords)
        )

    # Methods

    @property
    def factor_embs(self) -> tuple[LinearEmbedding[Any, Any], ...]:
        """All embeddings in factor order: the codomain factors, then the contracted ones."""
        return self.cod_embs + self.dom_embs

    @property
    def amb_mans(self) -> tuple[Manifold, ...]:
        """The manifold each factor is, in factor order."""
        return tuple(emb.amb_man for emb in self.factor_embs)

    @property
    def sub_dims(self) -> tuple[int, ...]:
        """Dimension of each factor's selected subspace --- the shape of the parameter tensor."""
        return tuple(emb.sub_man.dim for emb in self.factor_embs)

    @property
    def arity(self) -> int:
        """Number of factors."""
        return len(self.cod_embs) + len(self.dom_embs)

    @property
    def cod_dims(self) -> tuple[int, ...]:
        """Selected dimension of each codomain factor."""
        return tuple(emb.sub_man.dim for emb in self.cod_embs)

    @property
    def dom_dims(self) -> tuple[int, ...]:
        """Selected dimension of each contracted factor."""
        return tuple(emb.sub_man.dim for emb in self.dom_embs)

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """The matricization: codomain factors as the rows, contracted factors as the columns."""
        return (prod(self.cod_dims), prod(self.dom_dims))

    @classmethod
    def whole(cls, man: Manifold) -> SubspaceMap:
        """The map holding one manifold whole as its single factor: a bias, with no domain factors.

        No subspace is selected and nothing is contracted, so the parameters are just a
        point of ``man``. This is what a graph node's own potential is, and where the
        recursion in :meth:`~goal.geometry.manifold.clique.LinearCliques.placements_of`
        bottoms out.
        """
        return cls(Rectangular(), (IdentityEmbedding(man),), ())

    def to_matrix(self, params: Array) -> Array:
        """Unpack flat parameters into a dense (output, input) matrix."""
        return self.rep.to_matrix(self.matrix_shape, params)

    def from_matrix(self, matrix: Array) -> Array:
        """Pack a dense (output, input) matrix into flat parameters."""
        return self.rep.from_matrix(matrix)

    def to_tensor(self, params: Array) -> Array:
        """View flat parameters as a tensor of shape :attr:`sub_dims`, one axis per factor.

        Factor order is the storage order --- codomain factors, then contracted ones --- so
        a map and its transpose view one tensor as each other's axis-permuted readings.

        Raises:
            ValueError: unless the representation stores every entry, since a structured
                ``rep`` holds fewer parameters than the tensor has entries.
        """
        self._require_dense()
        return params.reshape(self.sub_dims)

    def from_tensor(self, tensor: Array) -> Array:
        """Flatten a tensor of shape :attr:`sub_dims`, in factor order, into parameters."""
        self._require_dense()
        return tensor.reshape(-1)

    def project_cod(self, joint: Array) -> Array:
        """Restrict a codomain joint from the ambient dimensions to the selected ones."""
        return self._project_group(self.cod_embs, joint)

    def embed_cod(self, selected: Array) -> Array:
        """The other direction on the same factors as :meth:`project_cod`: back out to the ambient dimensions."""
        return self._embed_group(self.cod_embs, selected)

    def project_dom(self, joint: Array) -> Array:
        """Restrict a domain joint from the ambient dimensions to the selected ones.

        With no domain factors --- a bias --- the joint is the constant $1$.
        """
        return self._project_group(self.dom_embs, joint)

    # Private

    @staticmethod
    def _map_axis(tensor: Array, axis: int, fn: Any) -> Array:
        """Apply a vector function along one axis of a tensor, replacing that axis."""
        moved = jnp.moveaxis(tensor, axis, 0)
        trailing = moved.shape[1:]
        columns = moved.reshape(moved.shape[0], -1)
        mapped = jax.vmap(fn, in_axes=1, out_axes=1)(columns)
        return jnp.moveaxis(mapped.reshape((-1, *trailing)), 0, axis)

    @staticmethod
    def _project_group(
        embs: tuple[LinearEmbedding[Any, Any], ...], joint: Array
    ) -> Array:
        if not embs:
            return jnp.ones(1)
        if len(embs) == 1:
            # One factor: its embedding restricts directly, with no tensor to reshape. Not
            # only an optimization --- an embedding may accept a point it can restrict
            # without its ambient dimension matching exactly, and reshaping would not.
            return embs[0].project(joint)
        out = joint.reshape(tuple(emb.amb_man.dim for emb in embs))
        for position, emb in enumerate(embs):
            out = SubspaceMap._map_axis(out, position, emb.project)
        return out.reshape(-1)

    @staticmethod
    def _embed_group(
        embs: tuple[LinearEmbedding[Any, Any], ...], selected: Array
    ) -> Array:
        if not embs:
            # The empty group's space is one-dimensional --- the constant --- so its
            # coordinates pass through, which is what makes a bias's transpose a
            # functional rather than a dead end.
            return selected
        if len(embs) == 1:
            return embs[0].embed(selected)
        out = selected.reshape(tuple(emb.sub_man.dim for emb in embs))
        for position, emb in enumerate(embs):
            out = SubspaceMap._map_axis(out, position, emb.embed)
        return out.reshape(-1)

    def _require_dense(self) -> None:
        if self.rep.num_params(self.matrix_shape) != prod(self.sub_dims):
            msg = f"{type(self.rep).__name__} stores fewer parameters than the tensor"
            raise ValueError(f"{msg} of shape {self.sub_dims} has entries")


### Clique Embeddings ###


@dataclass(frozen=True)
class CliqueEmbedding[Ambient: LinearCliques](LinearEmbedding[SubspaceMap, Ambient]):
    """The inclusion of one clique of a layout into that layout.

    Name the nodes and this addresses the form holding them jointly: :attr:`sub_man` is that
    form, ``project`` slices its coordinates out of a layout vector, and ``embed`` scatters
    them back into a zero one. The nodes are reached together rather than one at a time,
    since a form over several of them need not factorize and only the layout holds it as a
    single block. Nothing is restricted on the way --- the coordinates are the form's own,
    over the *full* coordinates of the nodes it couples; which sub-space of those the
    coupling uses is its :attr:`SubspaceMap.factor_embs`.

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
    def sub_man(self) -> SubspaceMap:
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


@dataclass(frozen=True)
class LinearCliques(Cliques, Manifold, ABC):
    """A manifold whose coordinates are one linear form per clique of a graph.

    A subclass supplies :attr:`placements` --- one ``(members, form)`` pair per clique, in
    storage order --- and the rest follows from it. The forms' parameter blocks are
    concatenated in that order, so :meth:`split_cliques` and :meth:`join_cliques` move
    between a coordinate vector and its per-clique parts, :meth:`clique_emb` reaches a single
    clique by naming its nodes, and :attr:`node_mans` reports what occupies each node. The
    graph is read off the same record --- :attr:`cliques` is the node tuples alone --- so the
    manifold and the graph it is defined on are one object rather than two that can disagree.

    :class:`LevelCliques` is the recursive case, one level of a hierarchy at a time, and
    :class:`CliqueProduct` the disjoint union of two layouts.

    Mathematically, a point is a family $(\\Theta^C)_C$ indexed by the cliques of the cover,
    with $\\dim = \\sum_C \\dim(\\Theta^C)$.
    """

    # Contract

    @property
    @abstractmethod
    def placements(self) -> tuple[tuple[tuple[int, ...], SubspaceMap], ...]:
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

        Validates each pairing on the way through, and rejects a repeated node set since
        :meth:`clique_index` can address only one form per node set.
        """
        out: list[tuple[int, ...]] = []
        for members, form in self.placements:
            self._validate_placement(members, form.arity)
            if members in out:
                msg = f"duplicate clique {members}"
                raise ValueError(f"{msg}: a node set carries exactly one form")
            out.append(members)
        return tuple(out)

    # Properties

    @property
    def clique_forms(self) -> tuple[SubspaceMap, ...]:
        """The forms alone, in storage order."""
        return tuple(form for _, form in self.placements)

    @property
    def clique_dims(self) -> tuple[int, ...]:
        """Dimension of each clique's form, in storage order."""
        return tuple(form.dim for _, form in self.placements)

    @property
    def clique_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Parameter-tensor shape of each clique's form, parallel to :attr:`clique_dims`."""
        return tuple(form.sub_dims for _, form in self.placements)

    @property
    def node_mans(self) -> tuple[Manifold, ...]:
        """The manifold occupying each node, ascending by label, parallel to :attr:`nodes`.

        Every clique states what it expects at each node it touches, so a node touched by
        several is described by several; this reports their agreement.

        Raises:
            ValueError: if two cliques disagree about what occupies a node, which means one
                of them is coupling a manifold that is not there.
        """
        seen: dict[int, Manifold] = {}
        for members, form in self.placements:
            for node, man in zip(members, form.amb_mans, strict=True):
                known = seen.setdefault(node, man)
                if known != man:
                    msg = f"node {node} is {known} in one clique and {man} in {members}"
                    raise ValueError(msg)
        return tuple(seen[node] for node in self.nodes)

    # Methods

    @staticmethod
    def placements_of(
        partition: Manifold,
    ) -> tuple[tuple[tuple[int, ...], SubspaceMap], ...]:
        """The cliques a partition contributes, in the partition's own frame.

        A partition that is already a clique manifold says what its cliques are; anything
        else occupies one node and contributes one form holding it whole. The single place
        the recursion in :class:`LevelCliques` and :class:`CliqueProduct` bottoms out.
        """
        if isinstance(partition, LinearCliques):
            return partition.placements
        return (((0,), SubspaceMap.whole(partition)),)

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

        Reads or writes that clique's coordinates without naming a layout index. This is
        what :meth:`LevelCliques.cross_paths` hands back as a path.
        """
        return CliqueEmbedding(tuple(sorted(members)), self)

    def clique_index(self, members: tuple[int, ...]) -> int:
        """Layout position of the clique on exactly ``members``.

        Raises:
            ValueError: if no clique covers exactly those nodes, which a coupling into this
                manifold requires.
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

    # Private

    @staticmethod
    def _validate_placement(members: tuple[int, ...], arity: int) -> None:
        """The two rules pairing a form with nodes has to satisfy.

        **One factor per node**: a form's arity and the node list it is paired with agree.
        **Distinct nodes, ascending**: a clique on a given node set has exactly one
        spelling, which is how :meth:`clique_index` finds it. Node *labels* are otherwise
        free; only their order within a clique is fixed, since that is what pairs them with
        the form's factors.

        Raises:
            ValueError: if the number of nodes and the arity disagree, or the nodes repeat
                or descend.
        """
        if len(members) != arity:
            msg = f"clique {members} names {len(members)} nodes"
            raise ValueError(f"{msg} but its form has arity {arity}")
        if tuple(sorted(set(members))) != members:
            raise ValueError(f"clique {members} must name distinct nodes, ascending")

    @staticmethod
    def _shift_placements(
        placements: tuple[tuple[tuple[int, ...], SubspaceMap], ...], offset: int
    ) -> tuple[tuple[tuple[int, ...], SubspaceMap], ...]:
        """Renumber a partition's cliques into an outer frame.

        A partition numbers its own nodes from zero, so whatever contains it moves them past
        the labels already in use. Only the address changes --- the forms are untouched.
        """
        return tuple((tuple(i + offset for i in ms), form) for ms, form in placements)


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

    The three components are named, not the cliques, because a partition may hold several
    of them --- ``deep`` always does past depth two --- and the graph, not the caller,
    decides which cliques land where.
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
    def cross_placements(self) -> tuple[tuple[tuple[int, ...], SubspaceMap], ...]:
        """The cliques joining the root nodes to the rest, in this level's frame.

        The cross partition is a bare parameter manifold --- a model supplies its
        interaction as a map --- so which nodes each of its blocks couples is the part only
        the model knows, in this level's own labels: the root partition's, and the deep
        partition's renumbered past them by :attr:`placements`.
        """

    # Overrides

    @property
    def root_placements(self) -> tuple[tuple[tuple[int, ...], SubspaceMap], ...]:
        """The root partition's cliques, and so how many nodes it occupies.

        By default a structured root partition is *expanded*, contributing one node per node
        of its own: that is what gives a multi-root model like probabilistic CCA its two
        root nodes. Override with a single clique --- ``(((0,), SubspaceMap.whole(self.root_man)),)``
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
    def placements(self) -> tuple[tuple[tuple[int, ...], SubspaceMap], ...]:
        """The partitions' cliques, concatenated in storage order.

        The root partition's are already in this level's frame and :attr:`cross_placements`
        reports its own; the deep partition's keep their relative order and labels,
        renumbered past the root nodes. A deep clique therefore names the same nodes however
        the glued graph reroots.
        """
        offset = max(self.root_nodes) + 1
        return (
            self.root_placements
            + self.cross_placements
            + self._shift_placements(self.placements_of(self.deep_man), offset)
        )

    def cross_placement(
        self, rep: MatrixRep, node_embs: Mapping[int, LinearEmbedding[Any, Any]]
    ) -> tuple[tuple[int, ...], SubspaceMap]:
        """A crossing clique built from the nodes it couples and what it uses at each.

        A model gives a *set* of nodes, the sub-space this coupling uses inside each, and a
        representation; everything positional follows from those. Storage order is the
        nodes' own ascending order, which is the order a layout stores its cliques in
        anyway, and the output group is the one root node among them, a crossing clique
        being what joins a root to the depths. The node set fixes the arity, so no model
        writes a factor number.

        Raises:
            ValueError: if the nodes do not include exactly one root node, which is what
                makes the clique a *crossing* one; or if that root node is not the lowest
                label, which the level frame guarantees and the pairing of members with the
                form's ``cod_embs + dom_embs`` order requires.
        """
        members = tuple(sorted(node_embs))
        roots = tuple(node for node in members if node in self.root_nodes)
        if len(roots) != 1:
            msg = f"crossing clique {members} touches {len(roots)} root nodes"
            raise ValueError(f"{msg}, not exactly one")
        if roots[0] != members[0]:
            msg = f"crossing clique {members}: root node {roots[0]} must carry"
            raise ValueError(f"{msg} the lowest label")
        form = SubspaceMap(
            rep,
            (node_embs[roots[0]],),
            tuple(node_embs[node] for node in members[1:]),
        )
        return members, form

    def cross_paths(
        self, placement: tuple[tuple[int, ...], SubspaceMap]
    ) -> tuple[LinearEmbedding[Any, Any] | None, LinearEmbedding[Any, Any] | None]:
        """How a crossing clique's two sides are reached: the root path and the deep path.

        A crossing clique couples one root node to a group of deep ones. A *path* says how a
        partition's coordinates reach the nodes in question --- it is the partition's own
        :meth:`~LinearCliques.clique_emb`, and ``None`` exactly when the partition *is* that
        node already, as for a chain's latent or an unexpanded root. A harmonium's
        interaction pairs the form with these to get a conditional reading.

        The same root split bounds what reading a form may declare: the output group must
        be a single node and the root node must come first, so that ``members`` pairs with
        the form's ``cod_embs + dom_embs`` order. That is weaker than the graph dictating
        direction outright --- a hand-paired form (the one borrowed placement in
        ``models/graphical/mixture.py``) that arrives transposed presents a legal forward
        shape on the flipped pairing, and :attr:`LinearCliques.node_mans` catches it only
        when the two nodes' manifolds differ.

        Raises:
            ValueError: if the clique does not couple exactly one root node, which is what
                makes it a *crossing* clique; if that root node is not the first member; or
                if the form's codomain is not a single factor.
        """
        members, form = placement
        roots = self.root_nodes
        near = tuple(i for i in members if i in roots)
        if len(near) != 1:
            msg = f"crossing clique {members} touches {len(near)} root nodes"
            raise ValueError(f"{msg}, not exactly one")
        if near[0] != members[0]:
            msg = f"crossing clique {members}: root node {near[0]} must carry"
            raise ValueError(f"{msg} the lowest label")
        if len(form.cod_embs) != 1:
            msg = (
                f"crossing clique {members} is read out of {len(form.cod_embs)} factors"
            )
            raise ValueError(f"{msg}, but its root side is a single node")
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

        The two offsets come from :attr:`placements`, each stored clique counted in the
        group :meth:`~goal.geometry.algebra.clique.Cliques.level_split` puts it in, rather
        than from the partitions' own dimensions, so the split always matches the stored
        layout. Storage lists the root, cross, and deep cliques in that order.
        """
        root, cross, _ = self.level_split()
        root_dim = cross_dim = 0
        for members, dim in zip(self.cliques, self.clique_dims, strict=True):
            if members in root:
                root_dim += dim
            elif members in cross:
                cross_dim += dim
        cross_dim += root_dim
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
    is one of these has as many root nodes as the product has components, so its crossing
    cliques may fan out to more than one of them.
    """

    # Overrides

    @property
    @override
    def root_nodes(self) -> frozenset[int]:
        """Every node: nothing links the two sides, so none is above another."""
        return frozenset(i for clique in self.cliques for i in clique)

    @property
    @override
    def placements(self) -> tuple[tuple[tuple[int, ...], SubspaceMap], ...]:
        fst = self.placements_of(self.fst_man)
        offset = max(max(members) for members, _ in fst) + 1
        return fst + self._shift_placements(self.placements_of(self.snd_man), offset)


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
        same_roots = self.sub_man.root_nodes == self.amb_man.root_nodes
        if not same_roots or self.sub_man.cliques != self.amb_man.cliques:
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
