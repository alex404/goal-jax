"""Manifolds whose parameters are laid out over the cliques of a graph.

A :class:`CliqueMap` is a tensor over selected subspaces of the manifolds it couples, and
knows nothing about where they sit. A :class:`LinearCliques` stores one block per clique,
each with its map. A :class:`CrossMap` sums several clique maps into one map between two
manifolds, given the cliques each of them holds. A :class:`RecursiveLinearCliques` is laid
out as ``[root | cross | deep]`` over a rooted graph, with blocks in the graph's canonical
order, recursing into the deep partition; its cross partition is the cross map between
its root and deep partitions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import Any, cast, override

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

    Each axis is a subspace of a manifold, given by one embedding: :attr:`cod_embs` are the
    output axes, :attr:`dom_embs` the contracted ones. A map with no domain axes is a bias.

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


def bias_map(man: Manifold) -> CliqueMap:
    """The map on a node's own clique: the whole node, as one output axis."""
    return CliqueMap(Rectangular(), (IdentityEmbedding(man),), ())


### Linear Cliques ###


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
class CliqueEmbedding(LinearEmbedding[LinearCliques, CliqueMap]):
    """The block of one clique in a :class:`LinearCliques`' coordinates.

    ``project`` reads the block and ``embed`` writes it with every other coordinate zero; the
    subspace is the clique's :class:`CliqueMap`. :meth:`CrossMap.clq_embs` uses one when a
    crossing clique's nodes in a partition form one of several cliques it holds (MFA's
    $(x, y, k)$, whose nodes $(y, k)$ are a clique of the mixture).

    Mathematically, the coordinates are a direct sum $\\bigoplus_C \\Theta^C$: ``project`` is
    the projection onto one summand and ``embed`` its inclusion, its transpose.
    """

    # Fields

    clique: tuple[int, ...]
    """The clique, as an ascending node tuple."""

    _amb_man: LinearCliques

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
        return coords[self._block_location]

    @override
    def embed(self, coords: Array) -> Array:
        return self.amb_man.zeros().at[self._block_location].set(coords)

    # Methods

    @property
    def _block_location(self) -> slice:
        """Where the clique's block sits in the ambient coordinates."""
        amb = self.amb_man
        start = sum(amb.clq_dims[: amb.cliques.index(self.clique)])
        return slice(start, start + self.sub_man.dim)


def part_emb(
    partition: Manifold, group: tuple[tuple[int, ...], ...], clique: tuple[int, ...]
) -> LinearEmbedding[Any, Any]:
    """The block of ``clique``'s nodes in ``partition``, as an embedding into its coordinates.

    ``group`` is the cliques the partition holds, and the part is the clique's nodes among
    theirs. If the part is the partition's only clique, the block is the whole partition and
    the embedding is the identity; otherwise the part is one of its cliques, and the
    embedding is that clique's block (:meth:`LinearCliques.clq_emb`).
    """
    nodes = {i for member in group for i in member}
    part = tuple(i for i in clique if i in nodes)
    if group == (part,):
        return IdentityEmbedding(partition)
    return cast(LinearCliques, partition).clq_emb(part)


### Cross Maps ###


@dataclass(frozen=True)
class CrossMap[Codomain: Manifold, Domain: Manifold](LinearMap[Codomain, Domain]):
    """Several clique maps summed into one linear map between two manifolds.

    Each term is a clique, by its node labels, and the clique map on it. The codomain and
    domain come with the cliques each holds (:attr:`cod_group`, :attr:`dom_group`), so a
    term finds its part of each through :func:`part_emb` (:meth:`clq_embs`). In a
    :class:`RecursiveLinearCliques` these are the root and deep partitions and their clique
    groups. Parameters are the terms' concatenated in order, which :meth:`coord_blocks`
    splits apart again. A fork, a three-way coupling and a plain chain differ only in how
    many terms the sum has.

    Mathematically, writing $\\Theta_t$ for the $t$-th clique map and $\\pi_t, \\phi_t$
    for the blocks of its parts in the domain and codomain, the map is
    $v \\mapsto \\sum_t \\phi_t(\\Theta_t \\cdot \\pi_t(v))$.
    """

    # Fields

    _cod_man: Codomain
    _dom_man: Domain
    cod_group: tuple[tuple[int, ...], ...]
    """The cliques the codomain holds."""

    dom_group: tuple[tuple[int, ...], ...]
    """The cliques the domain holds."""

    terms: tuple[tuple[tuple[int, ...], CliqueMap], ...]
    """Each clique with its map, in parameter order."""

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
    def trn_man(self) -> CrossMap[Domain, Codomain]:
        """Each clique map transposed, with codomain and domain swapped.

        In a harmonium this is what a conditional posterior is, where the forward reading is
        a conditional likelihood.
        """
        terms = tuple((clique, clq_map.trn_man) for clique, clq_map in self.terms)
        return CrossMap(
            self._dom_man, self._cod_man, self.dom_group, self.cod_group, terms
        )

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        out = self.cod_man.zeros()
        for (clique, clq_map), params in zip(
            self.terms, self.coord_blocks(f_coords), strict=True
        ):
            cod_emb, dom_emb = self.clq_embs(clique)
            out = out + cod_emb.embed(clq_map(params, dom_emb.project(v_coords)))
        return out

    @override
    def transpose(self, f_coords: Array) -> Array:
        parts = [
            clq_map.transpose(params)
            for (_, clq_map), params in zip(
                self.terms, self.coord_blocks(f_coords), strict=True
            )
        ]
        return jnp.concatenate(parts)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        parts = []
        for clique, clq_map in self.terms:
            cod_emb, dom_emb = self.clq_embs(clique)
            parts.append(
                clq_map.outer_product(
                    cod_emb.project(w_coords), dom_emb.project(v_coords)
                )
            )
        return jnp.concatenate(parts)

    # Methods

    @property
    def clq_dims(self) -> tuple[int, ...]:
        """Parameter dimension of each term, in parameter order."""
        return tuple(clq_map.dim for _, clq_map in self.terms)

    def clq_embs(
        self, clique: tuple[int, ...]
    ) -> tuple[LinearEmbedding[Any, Any], LinearEmbedding[Any, Any]]:
        """The blocks of ``clique``'s nodes in the codomain and in the domain."""
        return (
            part_emb(self.cod_man, self.cod_group, clique),
            part_emb(self.dom_man, self.dom_group, clique),
        )

    def coord_blocks(self, coords: Array) -> tuple[Array, ...]:
        """Split flat parameters into one array per term."""
        return split_by_dims(coords, self.clq_dims)


### Recursive Layouts ###


@dataclass(frozen=True)
class RecursiveLinearCliques[Root: Manifold, Deep: Manifold](
    RecursiveCliques,
    LinearCliques,
    Triple[Root, CrossMap[Root, Deep], Deep],
    ABC,
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
    def snd_man(self) -> CrossMap[Root, Deep]:
        return self.crs_man

    @property
    @override
    def trd_man(self) -> Deep:
        return self.dep_man

    # Methods

    @property
    def crs_man(self) -> CrossMap[Root, Deep]:
        """The cross partition: the map on each crossing clique, from the deep partition to the root."""
        rot, cross, dep = self.level_split()
        terms = tuple((clique, self.clq_map(clique)) for clique in cross)
        return CrossMap(self.rot_man, self.dep_man, rot, dep, terms)

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

        A crossing clique's axes are built on the subspaces used by its parts: its root nodes
        and its deep nodes, each a clique of its partition. Any other clique lies in one partition: if the partition holds only that
        clique it is that clique, whose map is its bias, and otherwise it gives the map.
        """
        near, far = self.split_clique(clique)
        if near and far:
            cons = dict(zip(clique, self.crs_emb_constructors(clique), strict=True))
            parts = self.clq_map(near).embs + self.clq_map(far).embs
            embs = [cons[i](r.sub_man) for i, r in zip(near + far, parts, strict=True)]
            return CliqueMap(
                self.crs_rep(clique), tuple(embs[: len(near)]), tuple(embs[len(near) :])
            )
        rot, _, dep = self.level_split()
        partition, group = (self.rot_man, rot) if near else (self.dep_man, dep)
        if group == (clique,):
            return bias_map(partition)
        return cast(LinearCliques, partition).clq_map(clique)

    def split_level(self, coords: Array) -> tuple[Array, Array, Array]:
        return self.split_coords(coords)

    def join_level(self, root: Array, cross: Array, deep: Array) -> Array:
        return self.join_coords(root, cross, deep)


@dataclass(frozen=True)
class RootEmbedding[
    Ambient: RecursiveLinearCliques[Any, Any],
    Sub: RecursiveLinearCliques[Any, Any],
](LinearEmbedding[Ambient, Sub]):
    """Embeds one layout into another over the same graph, transforming only the root partition.

    Mathematically, ``embed`` maps $(r, c, d) \\mapsto (\\phi(r), c, d)$ and ``project``
    maps $(r, c, d) \\mapsto (\\pi(r), c, d)$, with $\\phi, \\pi$ those of :attr:`rot_emb`.
    """

    # Fields

    rot_emb: LinearEmbedding[Any, Any]
    _amb_man: Ambient
    _sub_man: Sub

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
