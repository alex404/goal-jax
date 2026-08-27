"""Manifolds whose parameters are laid out over the cliques of a graph.

The clique-indexed sibling of :mod:`goal.geometry.manifold.combinators`. Where ``Pair`` and
``Triple`` split a coordinate vector by arity, a :class:`LinearCliques` splits one by *which
nodes each form couples* --- so the manifold and the graph it is defined on are the same
object, rather than the manifold holding a reference to one.

A :class:`LinearClique` is one such form together with the nodes it couples. Node numbering
is a fact about position in a graph rather than about what sits at a position, so a span
numbers its own nodes from zero, can be reused at any depth, and the containing layout
renumbers what it receives.
:class:`LevelCliques` is the recursive case, storing its coordinates as the three spans of
one level ascent, and :class:`CliqueProduct` is the disjoint union, which is what a
multi-root model's root span is.

Nothing here knows what occupies a node. Attaching sufficient statistics to a form's
factors is :class:`~goal.geometry.exponential_family.clique.EFClique`, one layer up.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from math import prod
from typing import Self, override

import jax.numpy as jnp
from jax import Array

from ..algebra.clique import Cliques
from .base import Manifold
from .combinators import Pair, Tuple
from .cut import CliqueCut
from .util import split_by_dims

### Linear Cliques ###


class LinearClique(Manifold, ABC):
    """A multilinear form together with the nodes it couples.

    What a :class:`LinearCliques` is laid out over, and where its name comes from. The form
    and the clique are one fact --- one factor per node, in member order --- and
    :meth:`validate_clique` is where a subclass says so.

    Parameters are one coefficient per index tuple, stored row-major --- the arity-$n$
    generalization of :class:`~goal.geometry.algebra.matrix.Rectangular`, whose layout it
    reproduces exactly at arity 2. Two operations: build parameters from a tuple of factor
    vectors (:meth:`tensor`), and contract all factors but one (:meth:`contract`).

    Mathematically, for factor spaces of dimensions $(d_1, \\ldots, d_n)$ the parameters
    are a tensor $\\Theta \\in \\mathbb R^{d_1 \\times \\cdots \\times d_n}$, and

    .. math::
        \\mathrm{tensor}(v_1, \\ldots, v_n) = v_1 \\otimes \\cdots \\otimes v_n,
        \\qquad
        \\mathrm{contract}(\\Theta, k, \\ldots)_{i}
            = \\sum_{j_1 \\ldots \\widehat{j_k} \\ldots j_n}
              \\Theta_{j_1 \\ldots i \\ldots j_n} \\prod_{l \\neq k} (v_l)_{j_l}.

    Arity is a degree, not a kind: at arity 1 the form is a bias, at arity 2 a matrix, and
    beyond that a higher-order tensor. The factor dimensions are whatever the subclass
    couples --- for an
    :class:`~goal.geometry.exponential_family.clique.EFClique` the *selected*
    sub-statistic dimensions, not the full node dimensions.

    The clique is in the frame of whatever reports it. A node index is not a property of
    what occupies a node --- a ``Normal`` is the same ``Normal`` wherever it sits --- so a
    span numbers its own nodes from zero and whatever contains it renumbers what it
    receives, through :meth:`shifted`.
    """

    # Contract

    @property
    @abstractmethod
    def sub_dims(self) -> tuple[int, ...]:
        """Dimension of each factor, in index order."""

    @property
    @abstractmethod
    def members(self) -> tuple[int, ...]:
        """Nodes this form couples, ascending and distinct, in its reporter's frame."""

    @abstractmethod
    def shifted(self, offset: int) -> Self:
        """The same clique renumbered into an outer frame."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return prod(self.sub_dims)

    # Methods

    @property
    def arity(self) -> int:
        """Number of factors."""
        return len(self.sub_dims)

    def to_tensor(self, params: Array) -> Array:
        """View flat parameters as a tensor of shape :attr:`sub_dims`."""
        return params.reshape(self.sub_dims)

    def from_tensor(self, tensor: Array) -> Array:
        """Flatten a tensor of shape :attr:`sub_dims` into parameters."""
        return tensor.reshape(-1)

    def tensor(self, *vectors: Array) -> Array:
        """Outer product of one vector per factor, as flat parameters."""
        if len(vectors) != self.arity:
            raise ValueError(f"expected {self.arity} factors, got {len(vectors)}")
        out = vectors[0]
        for vector in vectors[1:]:
            out = jnp.tensordot(out, vector, axes=0)
        return out.reshape(-1)

    def contract(self, params: Array, keep: int, *vectors: Array) -> Array:
        """Contract every factor except ``keep``, leaving a vector on that factor.

        ``vectors`` supplies one vector per contracted factor, in ascending index order.
        At arity 2 this is matrix-vector multiplication (``keep=0``) or its transpose
        (``keep=1``), which is how a conditional distribution reads its parameters off a
        crossing clique.
        """
        if not 0 <= keep < self.arity:
            msg = f"keep must be in 0..{self.arity - 1}, got {keep}"
            raise ValueError(msg)
        return self.partial_contract(params, (keep,), *vectors)

    def partial_contract(
        self, params: Array, keep: tuple[int, ...], *vectors: Array
    ) -> Array:
        """Contract every factor not in ``keep``, leaving a flat tensor over those kept.

        The general form of :meth:`contract`: ``keep`` names the surviving factors in
        ascending index order, and ``vectors`` supplies one vector per contracted factor,
        also in ascending index order. The result is flat, over the kept factors' dimensions
        in index order.
        """
        dropped = [axis for axis in range(self.arity) if axis not in keep]
        if len(vectors) != len(dropped):
            raise ValueError(f"expected {len(dropped)} factors, got {len(vectors)}")
        out = self.to_tensor(params)
        # Descending order so that contracting one axis does not shift the next.
        for axis, vector in sorted(zip(dropped, vectors), key=lambda p: -p[0]):
            out = jnp.tensordot(out, vector, axes=([axis], [0]))
        return out.reshape(-1)

    def validate_clique(self) -> None:
        """The two rules a clique has to satisfy, checked once, at construction.

        **One factor per node.** A subclass whose factors and members come from different
        sources --- an :class:`~goal.geometry.exponential_family.clique.EFClique` reads its
        factors off selectors and is told its members --- has nothing but this to make them
        agree. A subclass where both come from one place, as in :class:`NodeClique`, need
        not call this at all.

        **Distinct members, ascending.** Member order *is* axis order, so a clique on a
        given node set has one spelling and only one. Without that,
        :meth:`LinearCliques.clique_index` could not find a clique by naming its nodes, and
        one node set could sit in the layout twice under two spellings.

        Node *labels* are free --- they need not start at zero, run contiguously, or ascend
        with level. It is only their order **within a clique** that is fixed, and only
        because it indexes the form's axes.

        Raises:
            ValueError: if arity and the number of members disagree, or the members repeat
                or descend.
        """
        if len(self.members) != self.arity:
            msg = f"clique {self.members} names {len(self.members)} nodes"
            raise ValueError(f"{msg} but has arity {self.arity}")
        if tuple(sorted(set(self.members))) != self.members:
            msg = f"clique {self.members} must name distinct nodes, ascending"
            raise ValueError(msg)


@dataclass(frozen=True)
class NodeClique(LinearClique):
    """A manifold held whole at one node: arity one, the manifold's own dimension.

    Where the recursion in :meth:`LinearCliques.forms_of` bottoms out --- what a span with
    no clique structure of its own contributes --- and what a level supplies when it holds
    a structured span as a single node rather than expanding it.

    Both clique rules hold by construction: one node, one factor.
    """

    # Fields

    span: Manifold
    """The manifold occupying the node. Its dimension is the form's one factor."""

    node: int
    """The node it occupies, in its reporter's frame."""

    # Overrides

    @property
    @override
    def sub_dims(self) -> tuple[int, ...]:
        return (self.span.dim,)

    @property
    @override
    def members(self) -> tuple[int, ...]:
        return (self.node,)

    @override
    def shifted(self, offset: int) -> Self:
        return replace(self, node=self.node + offset)


### Clique Layouts ###


class LinearCliques(Cliques, Manifold, ABC):
    """A manifold whose parameters are laid out over the cliques of a graph.

    Each clique carries a multilinear form over its nodes, which is what the *linear*
    names. The manifold **is** its graph: :attr:`cliques` reads the cover off the forms, so
    there is no second description to disagree with the first.

    :attr:`clique_forms` is the primitive: one :class:`LinearClique` per clique, in storage
    order. Sizes and factor shapes are read off the forms and the cover off the cliques,
    so an offset and the nodes it belongs to always come from the same record.
    :class:`LevelCliques` is the recursive case and :class:`CliqueProduct` the disjoint
    union; the recursion terminates wherever a span stops being a clique manifold.

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
    def clique_forms(self) -> tuple[LinearClique, ...]:
        """One clique with its form, in storage order."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """The sum of the cliques' dimensions.

        A layout is its cliques: it holds one form per clique and nothing besides, so its
        size is theirs. Anything with a second reading of its own size --- a span
        decomposition, say --- has to agree with this one, which is what :meth:`validate`
        checks.
        """
        return sum(self.clique_dims)

    @property
    @override
    def cliques(self) -> tuple[tuple[int, ...], ...]:
        """Which nodes each form couples, in storage order.

        The graph, read off the layout. Every clique fact --- levels, boundary, the level
        split, canonical order --- follows from this and :attr:`root_nodes`.
        """
        return tuple(form.members for form in self.clique_forms)

    # Properties

    @property
    def clique_dims(self) -> tuple[int, ...]:
        """Dimension of each clique's form, in storage order."""
        return tuple(form.dim for form in self.clique_forms)

    @property
    def clique_axes(self) -> tuple[tuple[int, ...], ...]:
        """Factor dimensions of each clique's form, parallel to :attr:`clique_dims`."""
        return tuple(form.sub_dims for form in self.clique_forms)

    # Methods

    @staticmethod
    def forms_of(span: Manifold) -> tuple[LinearClique, ...]:
        """The forms a span contributes, in the span's own frame.

        A span that is already a clique manifold says what its forms are; anything else
        occupies one node and contributes one form holding it whole. This is the single
        place the recursion in :class:`LevelCliques` and :class:`CliqueProduct` bottoms
        out, and the only place a span's structure is inspected.
        """
        if isinstance(span, LinearCliques):
            return span.clique_forms
        return (NodeClique(span, 0),)

    def clique_offsets(self) -> tuple[int, ...]:
        """Start of each clique's coordinates in the flat parameter vector."""
        offsets: list[int] = []
        running = 0
        for size in self.clique_dims:
            offsets.append(running)
            running += size
        return tuple(offsets)

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
        computed and the dimensions selected come from the same forms. Every condition the
        view requires is checked by :class:`~goal.geometry.manifold.cut.CliqueCut` itself.
        """
        return CliqueCut(self.cliques, self.clique_dims, far_node)


@dataclass(frozen=True)
class LevelCliques[Root: Manifold, Cross: Manifold, Deep: Manifold](
    LinearCliques, Tuple, ABC
):
    """Clique manifold laid out as the three spans of one level ascent.

    Coordinates are stored as ``[root | cross | deep]``: the parameters carried by the root
    nodes, the interactions joining the root nodes to the rest of the graph, and everything
    above. The deep span is the graph one level up, so the same split applies again there
    --- recursion over the graph is a sequence of these. Levels are distances in the glued
    graph, so how the deep span roots *itself* is discarded: that is what makes a fork
    depth two rather than depth three.

    Unlike ``Pair`` and ``Triple``, the components are not arbitrary: the graph says what
    each one is. A span may hold several cliques --- ``deep`` always does past depth two ---
    which is why the three spans are named rather than the individual cliques.
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
    def cross_forms(self) -> tuple[LinearClique, ...]:
        """The cliques joining the root nodes to the rest, in this level's frame.

        The cross span is a bare parameter manifold --- a model supplies its interaction as
        a map --- so which nodes each of its blocks couples is the part only the model
        knows, in this level's own labels: the root span's, and the deep span's relabelled
        past them by :attr:`clique_forms`.
        """

    # Overrides

    @property
    def root_forms(self) -> tuple[LinearClique, ...]:
        """The root span's forms, and so how many nodes it occupies.

        By default a structured root span is *expanded*, contributing one node per node of
        its own: that is what gives a multi-root model like probabilistic CCA its two root
        nodes. Override with a single form --- ``(NodeClique(self.root_man, 0),)``
        --- to hold a structured span as one node instead, which is right whenever this level's
        interaction couples the span as a unit rather than factoring across its nodes: a
        mixture over a harmonium, say, whose interaction reaches the whole parameter vector.

        The choice is not free: :attr:`cross_forms` names nodes in this level's frame, so
        expanding the root span changes what those names mean.
        """
        return self.forms_of(self.root_man)

    @property
    @override
    def root_nodes(self) -> frozenset[int]:
        """Whichever nodes the root span occupies, read off :attr:`root_forms`."""
        return frozenset(i for form in self.root_forms for i in form.members)

    @property
    @override
    def clique_forms(self) -> tuple[LinearClique, ...]:
        """The spans' forms, concatenated in storage order.

        The root span's forms are already in this level's frame, and
        :attr:`cross_forms` reports its own. The deep span's keep the order and the
        relative labels the deep manifold itself gives them, shifted past the root nodes ---
        so however the glued graph reroots, a deep form still says which nodes it couples.
        """
        offset = max(self.root_nodes) + 1
        return (
            self.root_forms
            + self.cross_forms
            + tuple(form.shifted(offset) for form in self.forms_of(self.deep_man))
        )

    @override
    def split_coords(self, coords: Array) -> tuple[Array, Array, Array]:
        """Split coordinates into the root, cross, and deep spans.

        The two offsets come from the *forms*, via :meth:`~goal.geometry.algebra.clique.Cliques.level_split`
        --- not from the spans' own dimensions. Both readings exist and must agree, and
        taking the split from the forms is what makes the agreement structural instead of
        a coincidence nothing checks. :meth:`validate` is where the two are compared.
        """
        root_idx, cross_idx, _ = self.level_split()
        dims = self.clique_dims
        root_dim = sum(dims[i] for i in root_idx)
        cross_dim = root_dim + sum(dims[i] for i in cross_idx)
        return coords[:root_dim], coords[root_dim:cross_dim], coords[cross_dim:]

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

    This is the shape a **multi-root** model's root span takes --- two root spans coupled to
    one shared node above, as in probabilistic CCA. A level whose root span is one of these
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
    def clique_forms(self) -> tuple[LinearClique, ...]:
        fst_forms = self.forms_of(self.fst_man)
        offset = max(max(form.members) for form in fst_forms) + 1
        return fst_forms + tuple(
            form.shifted(offset) for form in self.forms_of(self.snd_man)
        )
