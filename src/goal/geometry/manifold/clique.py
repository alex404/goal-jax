"""Manifolds whose parameters are laid out over the cliques of a graph.

A :class:`CliqueMap` is a tensor over selected subspaces of the manifolds it couples, and
knows nothing about where they sit. An :class:`EmbeddedCliqueMap` places one between
whole manifolds through a pair of clique embeddings, and a :class:`CrossMap` adds
several into one map. A :class:`LinearCliques` stores one block per clique, and a :class:`RecursiveLinearCliques`
is one laid out as ``[root | cross | deep]`` over a rooted graph, with blocks in the
graph's canonical order, recursing into the deep partition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import Any, override

import jax.numpy as jnp
from jax import Array

from ..algebra.clique import Cliques, RecursiveCliques
from ..algebra.matrix import MatrixRep, Rectangular
from .base import Manifold
from .combinators import Triple
from .embedding import IdentityEmbedding, LinearEmbedding
from .map import LinearMap
from .util import split_by_dims

### Clique Maps ###


@dataclass(frozen=True)
class TensorProduct(Manifold):
    """Tensor product of manifolds, with coordinates stored as the flattened joint.

    Mathematically, $\\mathcal M_1 \\otimes \\cdots \\otimes \\mathcal M_k$ with $\\dim =
    \\prod_i \\dim(\\mathcal M_i)$; the empty product has dimension one.
    """

    mans: tuple[Manifold, ...]

    @property
    @override
    def dim(self) -> int:
        return prod(man.dim for man in self.mans)


@dataclass(frozen=True)
class TensorProductEmbedding(LinearEmbedding[TensorProduct, TensorProduct]):
    """A tensor product of linear embeddings, applied to a joint one axis at a time.

    Mathematically, $\\phi_1 \\otimes \\cdots \\otimes \\phi_k$. Each embedding is
    linear, so the product acts on any joint, not only on products of points.
    """

    embs: tuple[LinearEmbedding[Any, Any], ...]

    @property
    @override
    def amb_man(self) -> TensorProduct:
        return TensorProduct(tuple(emb.amb_man for emb in self.embs))

    @property
    @override
    def sub_man(self) -> TensorProduct:
        return TensorProduct(tuple(emb.sub_man for emb in self.embs))

    @override
    def project(self, coords: Array) -> Array:
        out = coords.reshape(tuple(emb.amb_man.dim for emb in self.embs))
        for axis, emb in enumerate(self.embs):
            out = jnp.apply_along_axis(emb.project, axis, out)
        return out.reshape(-1)

    @override
    def embed(self, coords: Array) -> Array:
        out = coords.reshape(tuple(emb.sub_man.dim for emb in self.embs))
        for axis, emb in enumerate(self.embs):
            out = jnp.apply_along_axis(emb.embed, axis, out)
        return out.reshape(-1)


@dataclass(frozen=True)
class CliqueMap(LinearMap[TensorProduct, TensorProduct]):
    """The linear map of a clique potential, acting on a chosen subspace at each axis.

    Each axis is a manifold reached by one embedding: :attr:`cod_embs` are the output
    axes, :attr:`dom_embs` the contracted ones. A map with no domain axes is a bias.

    Mathematically, the parameters are a tensor $\\Theta \\in \\mathbb R^{d_1 \\times \\cdots
    \\times d_j \\times e_1 \\times \\cdots \\times e_k}$ over the selected subspaces, stored
    as the matricization with the codomain axes as rows and the domain axes as columns.
    The representation acts on that matricization, so a representation with structure
    (diagonal, convolutional) sees the flattened shape $(\\prod_i d_i, \\prod_i e_i)$ when
    a group has more than one axis.
    """

    # Fields

    rep: MatrixRep
    cod_embs: tuple[LinearEmbedding[Any, Any], ...]
    dom_embs: tuple[LinearEmbedding[Any, Any], ...]

    # Overrides

    @property
    @override
    def dom_man(self) -> TensorProduct:
        return self.dom_emb.amb_man

    @property
    @override
    def cod_man(self) -> TensorProduct:
        return self.cod_emb.amb_man

    @property
    @override
    def dim(self) -> int:
        return self.rep.num_params(self.matrix_shape)

    @property
    @override
    def trn_man(self) -> CliqueMap:
        return CliqueMap(self.rep, self.dom_embs, self.cod_embs)

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        selected = self.dom_emb.project(v_coords)
        return self.cod_emb.embed(
            self.rep.matvec(self.matrix_shape, f_coords, selected)
        )

    @override
    def transpose(self, f_coords: Array) -> Array:
        return self.rep.transpose(self.matrix_shape, f_coords)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        """Parameters of the outer product of an output joint with an input joint.

        Each argument is a joint over its group's ambient dimensions, so this stays correct
        when the axes within a group are dependent.
        """
        return self.rep.outer_product(
            self.cod_emb.project(w_coords), self.dom_emb.project(v_coords)
        )

    # Methods

    @property
    def cod_emb(self) -> TensorProductEmbedding:
        return TensorProductEmbedding(self.cod_embs)

    @property
    def dom_emb(self) -> TensorProductEmbedding:
        return TensorProductEmbedding(self.dom_embs)

    @property
    def embs(self) -> tuple[LinearEmbedding[Any, Any], ...]:
        """All embeddings in axis order: codomain, then domain."""
        return self.cod_embs + self.dom_embs

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return (self.cod_emb.sub_man.dim, self.dom_emb.sub_man.dim)

    def to_matrix(self, params: Array) -> Array:
        return self.rep.to_matrix(self.matrix_shape, params)

    def from_matrix(self, matrix: Array) -> Array:
        return self.rep.from_matrix(matrix)


### Cross Maps ###


@dataclass(frozen=True)
class EmbeddedCliqueMap:
    """One clique map of a :class:`CrossMap`, with the clique embeddings that reach it.

    The map's output axes live on nodes of the codomain and its contracted axes on nodes of
    the domain; :attr:`cod_clq_emb` and :attr:`dom_clq_emb` take the whole codomain and
    domain to those nodes. :meth:`RecursiveLinearCliques.crs_embs` supplies them, usually as
    a partition's :class:`CliqueEmbedding`, or an :class:`IdentityEmbedding` when the
    partition is the clique.
    """

    clq_map: CliqueMap
    cod_clq_emb: LinearEmbedding[Any, Any]
    dom_clq_emb: LinearEmbedding[Any, Any]


@dataclass(frozen=True)
class CrossMap[Domain: Manifold, Codomain: Manifold](LinearMap[Domain, Codomain]):
    """Several clique maps summed into one linear map between whole manifolds.

    The domain and codomain are whole manifolds --- a harmonium's posterior and observable,
    say --- rather than the nodes a clique map acts on. Each :class:`EmbeddedCliqueMap`
    bridges that gap with a clique embedding on either side. Parameters are the terms'
    concatenated in order, which :meth:`coord_blocks` splits apart again. A fork, a three-way
    coupling and a plain chain differ only in how many terms the sum has.
    :attr:`RecursiveLinearCliques.crs_man` builds one from the graph's crossing cliques,
    which is where the name comes from.

    Mathematically, writing $\\Theta_t$ for the $t$-th clique map and $\\pi_t, \\phi_t$
    for its domain and codomain clique embeddings, the map is
    $v \\mapsto \\sum_t \\phi_t(\\Theta_t \\cdot \\pi_t(v))$.
    """

    # Fields

    _cod_man: Codomain
    _dom_man: Domain
    terms: tuple[EmbeddedCliqueMap, ...]
    """The terms, in parameter order."""

    # Overrides

    @property
    @override
    def dom_man(self) -> Domain:
        return self._dom_man

    @property
    @override
    def cod_man(self) -> Codomain:
        return self._cod_man

    @property
    @override
    def dim(self) -> int:
        return sum(self.clq_dims)

    @property
    @override
    def trn_man(self) -> CrossMap[Codomain, Domain]:
        """Each clique map transposed, with its two clique embeddings swapped.

        In a harmonium this is what a conditional posterior is, where the forward reading is
        a conditional likelihood.
        """
        terms = tuple(
            EmbeddedCliqueMap(term.clq_map.trn_man, term.dom_clq_emb, term.cod_clq_emb)
            for term in self.terms
        )
        return CrossMap(self._dom_man, self._cod_man, terms)

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        out = self.cod_man.zeros()
        for term, params in zip(self.terms, self.coord_blocks(f_coords), strict=True):
            w_node = term.clq_map(params, term.dom_clq_emb.project(v_coords))
            out = out + term.cod_clq_emb.embed(w_node)
        return out

    @override
    def transpose(self, f_coords: Array) -> Array:
        parts = [
            term.clq_map.transpose(params)
            for term, params in zip(
                self.terms, self.coord_blocks(f_coords), strict=True
            )
        ]
        return jnp.concatenate(parts)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        parts = [
            term.clq_map.outer_product(
                term.cod_clq_emb.project(w_coords), term.dom_clq_emb.project(v_coords)
            )
            for term in self.terms
        ]
        return jnp.concatenate(parts)

    # Methods

    @property
    def clq_dims(self) -> tuple[int, ...]:
        """Parameter dimension of each term, in parameter order."""
        return tuple(term.clq_map.dim for term in self.terms)

    @property
    def clq_map(self) -> CliqueMap:
        """The clique map of a sum with exactly one term.

        Raises:
            ValueError: if there are several terms, where there is no single map.
        """
        if len(self.terms) != 1:
            msg = f"this sum has {len(self.terms)} terms"
            raise ValueError(f"{msg}: there is no single clique map")
        return self.terms[0].clq_map

    def coord_blocks(self, coords: Array) -> tuple[Array, ...]:
        """Split flat parameters into one array per term."""
        return split_by_dims(coords, self.clq_dims)


### Clique Layouts ###


def bias_map(man: Manifold) -> CliqueMap:
    """The map on a node's own clique: the whole node, as one output axis."""
    return CliqueMap(Rectangular(), (IdentityEmbedding(man),), ())


def part_emb(partition: Manifold, part: tuple[int, ...]) -> LinearEmbedding[Any, Any]:
    """How ``partition`` reaches its clique ``part``.

    The clique's block when ``partition`` is a :class:`LinearCliques` holding other cliques
    too, and the identity when it is a single node or that clique alone.
    """
    if isinstance(partition, LinearCliques) and partition.cliques != (part,):
        return partition.clq_emb(part)
    return IdentityEmbedding(partition)


@dataclass(frozen=True)
class LinearCliques(Cliques, Manifold, ABC):
    """A manifold stored as one block per clique, in the order of :attr:`cliques`, with a linear map on each.

    Its dimension, from wherever a subclass takes it, must equal the summed block
    dimensions. It is not a :class:`~goal.geometry.manifold.combinators.Tuple`, because a
    subclass may already be one over coarser components (a :class:`RecursiveLinearCliques`
    is the ``Triple`` of its partitions), so the block split is :meth:`coord_blocks`.
    """

    # Contract

    @abstractmethod
    def clq_map(self, clique: tuple[int, ...]) -> CliqueMap:
        """The map on ``clique``."""

    # Methods

    @property
    def clq_dims(self) -> tuple[int, ...]:
        """Parameter dimension of each clique, in storage order."""
        return tuple(self.clq_map(clique).dim for clique in self.cliques)

    def coord_blocks(self, coords: Array) -> tuple[Array, ...]:
        """Split coordinates into one block per clique, in storage order."""
        return split_by_dims(coords, self.clq_dims)

    def clq_emb(self, clique: tuple[int, ...]) -> CliqueEmbedding:
        """The block of ``clique`` in this manifold's coordinates."""
        return CliqueEmbedding(clique, self)


@dataclass(frozen=True)
class RecursiveLinearCliques[Root: Manifold, Deep: Manifold](
    RecursiveCliques, LinearCliques, Triple[Root, CrossMap[Deep, Root], Deep], ABC
):
    """A manifold stored ``[root | cross | deep]``, with one linear map per clique of its graph.

    A concrete subclass stores its graph (:attr:`raw_cliques`, :attr:`root_nodes`) and states
    the root and deep partitions, and for each crossing clique a matrix representation and
    one subspace per node; the cross partition is derived from these. Blocks are stored in
    the order of :attr:`cliques`, which keeps the partitions contiguous. It is the
    :class:`~goal.geometry.manifold.combinators.Triple` of its partitions, so their
    dimensions must equal the summed dimensions of their clique blocks.

    Labels are global. Each of the root and deep partitions is either a single node or a
    :class:`LinearCliques` holding the cliques this graph assigns it, so cliques pass down
    unrelabelled: the root partition a flat one (CCA's observable pair), the deep partition
    another :class:`RecursiveLinearCliques`.

    A crossing clique splits into its root nodes, which are its output axes, and its deep
    nodes. Each part must be a clique of its partition.

    Mathematically, a point is a family $(\\Theta^C)_C$ indexed by the cliques, with $\\dim =
    \\sum_C \\dim(\\Theta^C)$.
    """

    # Contract

    @property
    @abstractmethod
    def rot_man(self) -> Root:
        """The root partition."""

    @property
    @abstractmethod
    def dep_man(self) -> Deep:
        """The deep partition."""

    @abstractmethod
    def crs_rep(self, clique: tuple[int, ...]) -> MatrixRep:
        """The matrix representation of the map on a crossing clique."""

    @abstractmethod
    def crs_emb_constructors(
        self, clique: tuple[int, ...]
    ) -> tuple[Callable[[Manifold], LinearEmbedding[Any, Any]], ...]:
        """One embedding constructor per node of a crossing clique, in node order."""

    # Overrides

    @property
    @override
    def fst_man(self) -> Root:
        return self.rot_man

    @property
    @override
    def snd_man(self) -> CrossMap[Deep, Root]:
        return self.crs_man

    @property
    @override
    def trd_man(self) -> Deep:
        return self.dep_man

    # Methods

    @property
    def crs_man(self) -> CrossMap[Deep, Root]:
        """The cross partition: the maps on the crossing cliques, reached through :meth:`crs_embs`."""
        terms = tuple(
            EmbeddedCliqueMap(self.clq_map(clique), *self.crs_embs(clique))
            for clique in self.level_split()[1]
        )
        return CrossMap(self.rot_man, self.dep_man, terms)

    def split_clique(
        self, clique: tuple[int, ...]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """A clique's root nodes and its deep nodes."""
        near = tuple(i for i in clique if i in self.root_nodes)
        far = tuple(i for i in clique if i not in self.root_nodes)
        return near, far

    @override
    def clq_map(self, clique: tuple[int, ...]) -> CliqueMap:
        """The map on ``clique``, one axis per node.

        A crossing clique's axes are built on the subspaces used by the root and deep cliques
        it reaches. Any other clique lies in one partition, which gives its map if it is a
        :class:`LinearCliques`, and is otherwise a single node whose bias it is.
        """
        near, far = self.split_clique(clique)
        if near and far:
            cons = dict(zip(clique, self.crs_emb_constructors(clique), strict=True))
            reached = self.clq_map(near).embs + self.clq_map(far).embs
            embs = [
                cons[i](r.sub_man) for i, r in zip(near + far, reached, strict=True)
            ]
            return CliqueMap(
                self.crs_rep(clique), tuple(embs[: len(near)]), tuple(embs[len(near) :])
            )
        partition = self.rot_man if near else self.dep_man
        if isinstance(partition, LinearCliques):
            return partition.clq_map(clique)
        return bias_map(partition)

    def crs_embs(
        self, clique: tuple[int, ...]
    ) -> tuple[LinearEmbedding[Any, Any], LinearEmbedding[Any, Any]]:
        """How :attr:`rot_man` and :attr:`dep_man` reach the two parts of a crossing clique.

        See :func:`part_emb`.
        """
        near, far = self.split_clique(clique)
        return part_emb(self.rot_man, near), part_emb(self.dep_man, far)

    def split_level(self, coords: Array) -> tuple[Array, Array, Array]:
        return self.split_coords(coords)

    def join_level(self, root: Array, cross: Array, deep: Array) -> Array:
        return self.join_coords(root, cross, deep)


@dataclass(frozen=True)
class CliqueEmbedding(LinearEmbedding[CliqueMap, LinearCliques]):
    """The block of one clique in a :class:`LinearCliques`' coordinates.

    ``project`` reads the block and ``embed`` writes it with every other coordinate zero; the
    subspace is the clique's :class:`CliqueMap`. It is the clique embedding of an
    :class:`EmbeddedCliqueMap` when a crossing clique reaches one clique of a partition that holds several (MFA's $(x, y, k)$ reaching the
    mixture's $(y, k)$).

    Mathematically, the coordinates are a direct sum $\\bigoplus_C \\Theta^C$: ``project`` is
    the projection onto one summand and ``embed`` its inclusion, its transpose.
    """

    # Fields

    clique: tuple[int, ...]
    """The clique, as an ascending node tuple."""

    _amb_man: LinearCliques

    def __post_init__(self) -> None:
        # Reject a node set that is not a clique of the ambient.
        _ = self.start

    # Overrides

    @property
    @override
    def sub_man(self) -> CliqueMap:
        return self.amb_man.clq_map(self.clique)

    @property
    @override
    def amb_man(self) -> LinearCliques:
        return self._amb_man

    @override
    def project(self, coords: Array) -> Array:
        return coords[self.start : self.start + self.sub_man.dim]

    @override
    def embed(self, coords: Array) -> Array:
        return (
            self.amb_man.zeros()
            .at[self.start : self.start + coords.shape[0]]
            .set(coords)
        )

    # Methods

    @property
    def start(self) -> int:
        """Where the block begins."""
        amb = self.amb_man
        return sum(amb.clq_dims[: amb.cliques.index(self.clique)])


@dataclass(frozen=True)
class RootEmbedding[
    Sub: RecursiveLinearCliques[Any, Any],
    Ambient: RecursiveLinearCliques[Any, Any],
](LinearEmbedding[Sub, Ambient]):
    """Embeds one layout into another over the same graph, transforming only the root partition.

    Mathematically, ``embed`` maps $(r, c, d) \\mapsto (\\phi(r), c, d)$ and ``project``
    maps $(r, c, d) \\mapsto (\\pi(r), c, d)$, with $\\phi, \\pi$ those of :attr:`rot_emb`.
    """

    # Fields

    rot_emb: LinearEmbedding[Any, Any]
    _sub_man: Sub
    _amb_man: Ambient

    def __post_init__(self) -> None:
        sub, amb = self.sub_man, self.amb_man
        if not sub.same_graph(amb) or (sub.crs_man.dim, sub.dep_man.dim) != (
            amb.crs_man.dim,
            amb.dep_man.dim,
        ):
            raise ValueError("sub and ambient may differ only in the root partition")

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
        return self.sub_man.join_level(self.rot_emb.project(root), cross, deep)

    @override
    def embed(self, coords: Array) -> Array:
        root, cross, deep = self.sub_man.split_level(coords)
        return self.amb_man.join_level(self.rot_emb.embed(root), cross, deep)
