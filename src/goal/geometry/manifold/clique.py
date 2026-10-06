"""Manifolds whose coordinates are laid out over the cliques of a graph.

Each node of the graph has a space, and each clique stores one block of coordinates, a
subspace of the tensor product of its nodes' spaces. A :class:`LinearCliques` stores one
block per clique and numbers its own nodes. A :class:`RecursiveLinearCliques` is stored as
``[root | cross | deep]``: two :class:`LinearCliques`, and a :class:`CrossMap` with one
:class:`CrossTerm` per crossing clique. The block of a crossing clique is a
:class:`CliqueMap`, a linear map from a subspace of a block of the deep partition to a
subspace of a block of the root partition.

Four embeddings relate these layouts. A :class:`CliqueEmbedding` selects the block of one
clique. A :class:`SubCliquesEmbedding` places a smaller layout on some of the cliques of a
larger one. A :class:`RootEmbedding` relates two layouts that differ only in their root
partition. A :class:`SubMapEmbedding` selects a subspace of a block that is itself a map.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
from jax import Array

from ..algebra.clique import Cliques, shift_clique
from ..algebra.matrix import MatrixRep, Rectangular
from .base import Manifold
from .combinators import Triple
from .embedding import LinearComposedEmbedding, LinearEmbedding
from .map import LinearMap
from .util import split_by_dims

### Clique Maps ###


@dataclass(frozen=True)
class CliqueMap(LinearMap[Any, Any]):
    """The block of a crossing clique: a linear map between subspaces of two other blocks.

    :attr:`cod_emb` selects a subspace of the block it writes to, and :attr:`dom_emb` a
    subspace of the block it reads from. The parameters are a matrix between the two
    subspaces in the structure of :attr:`rep`.

    Mathematically, with $\\phi$ the codomain embedding, $\\pi$ the domain projection and
    $\\Theta$ the matrix, the map is $v \\mapsto \\phi(\\Theta \\cdot \\pi(v))$.
    """

    # Fields

    rep: MatrixRep
    cod_emb: LinearEmbedding[Any, Any]
    dom_emb: LinearEmbedding[Any, Any]

    # Overrides

    @property
    @override
    def dom_man(self) -> Any:
        return self.dom_emb.amb_man

    @property
    @override
    def cod_man(self) -> Any:
        return self.cod_emb.amb_man

    @property
    @override
    def dim(self) -> int:
        return self.rep.num_params(self.matrix_shape)

    @property
    @override
    def trn_man(self) -> CliqueMap:
        return CliqueMap(self.rep, self.dom_emb, self.cod_emb)

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
        return self.rep.outer_product(
            self.cod_emb.project(w_coords), self.dom_emb.project(v_coords)
        )

    # Methods

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return (self.cod_emb.sub_man.dim, self.dom_emb.sub_man.dim)

    def to_matrix(self, params: Array) -> Array:
        return self.rep.to_matrix(self.matrix_shape, params)

    def from_matrix(self, matrix: Array) -> Array:
        return self.rep.from_matrix(matrix)


@dataclass(frozen=True)
class SubMapEmbedding(LinearEmbedding[CliqueMap, CliqueMap]):
    """A submatrix of a :class:`CliqueMap`: its rows restricted by :attr:`cod_emb` and its columns by :attr:`dom_emb`.

    The submanifold is again a clique map, with rectangular structure, between the nested
    subspaces. ``project`` restricts the matrix of the ambient map, and ``embed`` places a
    submatrix in it, with every other entry zero.
    """

    # Fields

    _amb_man: CliqueMap

    cod_emb: LinearEmbedding[Any, Any]
    """Embedding into the codomain subspace of the ambient map."""
    dom_emb: LinearEmbedding[Any, Any]
    """Embedding into the domain subspace of the ambient map."""

    # Overrides

    @property
    @override
    def amb_man(self) -> CliqueMap:
        return self._amb_man

    @property
    @override
    def sub_man(self) -> CliqueMap:
        amb = self.amb_man
        return CliqueMap(
            Rectangular(),
            LinearComposedEmbedding(amb.cod_emb, self.cod_emb),
            LinearComposedEmbedding(amb.dom_emb, self.dom_emb),
        )

    @override
    def project(self, coords: Array) -> Array:
        matrix = self.amb_man.to_matrix(coords)
        rows = jax.vmap(self.cod_emb.project, in_axes=1, out_axes=1)(matrix)
        return jax.vmap(self.dom_emb.project)(rows).reshape(-1)

    @override
    def embed(self, coords: Array) -> Array:
        matrix = coords.reshape(self.sub_man.matrix_shape)
        rows = jax.vmap(self.cod_emb.embed, in_axes=1, out_axes=1)(matrix)
        return self.amb_man.from_matrix(jax.vmap(self.dom_emb.embed)(rows))


### Linear Cliques ###


@dataclass(frozen=True)
class LinearCliques(Cliques, Manifold, ABC):
    """A manifold stored as one block of coordinates per clique, in the order of :attr:`cliques`.

    Each node has a space (:attr:`nod_mans`), and the block of a clique is a subspace of the
    tensor product of its nodes' spaces. Cliques that share a node therefore share its space.
    The nodes are numbered $0, \\ldots, n - 1$, and the numbers have no meaning outside this
    manifold; a manifold that contains it translates them (see
    :class:`RecursiveLinearCliques`). The dimension, however a subclass defines it, must
    equal the sum of the block dimensions. The class is not a
    :class:`~goal.geometry.manifold.combinators.Tuple`, because a subclass may already be a
    ``Tuple`` over coarser components (a :class:`RecursiveLinearCliques` is a ``Triple``),
    so the blocks are split by :meth:`coord_blocks`.
    """

    # Contract

    @property
    @abstractmethod
    def nod_mans(self) -> tuple[Manifold, ...]:
        """The space of each node, in node order."""

    @property
    @abstractmethod
    def clq_mans(self) -> tuple[Manifold, ...]:
        """The block of each clique, in storage order.

        The block of a single node is its space, and the block of a crossing clique is its
        :class:`CliqueMap`.
        """

    # Methods

    @property
    def clq_dims(self) -> tuple[int, ...]:
        """Parameter dimension of each clique, in storage order."""
        return tuple(clq_man.dim for clq_man in self.clq_mans)

    def clq_man(self, clique: tuple[int, ...]) -> Manifold:
        """The block of one clique."""
        return self.clq_mans[self.cliques.index(clique)]

    def coord_blocks(self, coords: Array) -> tuple[Array, ...]:
        """Split coordinates into one block per clique, in storage order."""
        return split_by_dims(coords, self.clq_dims)


@dataclass(frozen=True)
class CliqueEmbedding(LinearEmbedding[LinearCliques, Any]):
    """The block of one clique in the coordinates of a :class:`LinearCliques`.

    ``project`` reads the block, and ``embed`` writes it with every other coordinate zero.

    Mathematically, the coordinates are a direct sum $\\bigoplus_C \\Theta^C$. ``project``
    is the projection onto one summand, and ``embed`` is the inclusion, its transpose.
    """

    # Fields

    clique: tuple[int, ...]
    """The clique, as an ascending node tuple."""

    _amb_man: LinearCliques

    # Overrides

    @property
    @override
    def sub_man(self) -> Any:
        return self.amb_man.clq_mans[self._index]

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
    def _index(self) -> int:
        """The clique's position in the ambient storage order."""
        return self.amb_man.cliques.index(self.clique)

    @property
    def _block_location(self) -> slice:
        """Where the clique's block sits in the ambient coordinates."""
        start = sum(self.amb_man.clq_dims[: self._index])
        return slice(start, start + self.sub_man.dim)


@dataclass(frozen=True)
class SubCliquesEmbedding(LinearEmbedding[LinearCliques, LinearCliques]):
    """A :class:`LinearCliques` placed on some of the cliques of a larger one, block for block.

    Node $j$ of the submanifold is node ``nodes[j]`` of the ambient. Each clique of the
    submanifold, relabelled this way, must be a clique of the ambient with the same block. ``embed`` writes each block of the submanifold onto its clique, with every other
    coordinate zero, and ``project`` reads those blocks back.
    """

    # Fields

    nodes: tuple[int, ...]
    """The ambient node of each node of the submanifold."""

    _amb_man: LinearCliques
    _sub_man: LinearCliques

    # Overrides

    @property
    @override
    def amb_man(self) -> LinearCliques:
        return self._amb_man

    @property
    @override
    def sub_man(self) -> LinearCliques:
        return self._sub_man

    @override
    def project(self, coords: Array) -> Array:
        return jnp.concatenate([emb.project(coords) for emb in self._clq_embs])

    @override
    def embed(self, coords: Array) -> Array:
        out = self.amb_man.zeros()
        for emb, block in zip(self._clq_embs, self.sub_man.coord_blocks(coords)):
            out = out + emb.embed(block)
        return out

    # Methods

    @property
    def _clq_embs(self) -> tuple[CliqueEmbedding, ...]:
        """The ambient block of each clique of the submanifold, in its storage order."""
        return tuple(
            CliqueEmbedding(tuple(self.nodes[j] for j in clique), self.amb_man)
            for clique in self.sub_man.cliques
        )


### Cross Maps ###


@dataclass(frozen=True)
class CrossTerm:
    """A clique map pinned to a clique on each side: the codomain clique it writes to and the domain clique it reads from.

    Each clique is in the numbering of its own side.
    """

    # Fields

    cod_clq: tuple[int, ...]
    dom_clq: tuple[int, ...]
    clq_map: CliqueMap


@dataclass(frozen=True)
class CrossMap[Codomain: LinearCliques, Domain: LinearCliques](
    LinearMap[Codomain, Domain]
):
    """Several clique maps summed into one linear map between two :class:`LinearCliques`.

    Each term (:class:`CrossTerm`) is a clique map between the blocks of a clique of the
    codomain and a clique of the domain. The parameters are those of the terms, concatenated
    in order (see :meth:`coord_blocks`).

    Mathematically, with $\\Theta_t$ the $t$-th clique map and $\\pi_t, \\phi_t$ the blocks
    of its cliques in the domain and codomain, the map is
    $v \\mapsto \\sum_t \\phi_t(\\Theta_t \\cdot \\pi_t(v))$.
    """

    # Fields

    _cod_man: Codomain
    _dom_man: Domain
    trms: tuple[CrossTerm, ...]
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
    def trn_man(self) -> CrossMap[Domain, Codomain]:
        """Each clique map transposed, with codomain and domain swapped."""
        trms = tuple(
            CrossTerm(trm.dom_clq, trm.cod_clq, trm.clq_map.trn_man)
            for trm in self.trms
        )
        return CrossMap(self._dom_man, self._cod_man, trms)

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        out = self.cod_man.zeros()
        for trm, params in zip(self.trms, self.coord_blocks(f_coords)):
            selected = CliqueEmbedding(trm.dom_clq, self.dom_man).project(v_coords)
            out = out + CliqueEmbedding(trm.cod_clq, self.cod_man).embed(
                trm.clq_map(params, selected)
            )
        return out

    @override
    def transpose(self, f_coords: Array) -> Array:
        parts = [
            trm.clq_map.transpose(params)
            for trm, params in zip(self.trms, self.coord_blocks(f_coords))
        ]
        return jnp.concatenate(parts)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        parts = [
            trm.clq_map.outer_product(
                CliqueEmbedding(trm.cod_clq, self.cod_man).project(w_coords),
                CliqueEmbedding(trm.dom_clq, self.dom_man).project(v_coords),
            )
            for trm in self.trms
        ]
        return jnp.concatenate(parts)

    # Methods

    @property
    def clq_dims(self) -> tuple[int, ...]:
        """Parameter dimension of each term, in parameter order."""
        return tuple(trm.clq_map.dim for trm in self.trms)

    def coord_blocks(self, coords: Array) -> tuple[Array, ...]:
        """Split parameters into one block per term."""
        return split_by_dims(coords, self.clq_dims)


### Recursive Layouts ###


@dataclass(frozen=True)
class RecursiveLinearCliques[Root: LinearCliques, Deep: LinearCliques](
    LinearCliques,
    Triple[Root, CrossMap[Root, Deep], Deep],
    ABC,
):
    """A manifold stored as ``[root | cross | deep]``, whose graph is composed from its partitions.

    A subclass declares the root and deep partitions and the block of each crossing clique;
    everything else is derived. A crossing clique is a pair of a clique of the root partition
    and a clique of the deep partition, each in its own partition's numbering.

    The composed graph numbers the root partition's nodes first, and the deep partition's
    after them, offset by the root's node count. Its cliques are the root partition's, then
    the crossing cliques, then the deep partition's, which is also the storage order of the
    three blocks. A layer is therefore added at the root, with an existing model as the deep
    partition.

    The block of a crossing clique is a :class:`CliqueMap` from a subspace of the deep
    partition's block on its deep part to a subspace of the root partition's block on its
    root part. Its embeddings must therefore have those blocks as their ambients
    (:meth:`~LinearCliques.clq_man`). This is what keeps the crossing within the blocks the
    partitions store.

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

    @property
    @abstractmethod
    def crs_trms(self) -> tuple[CrossTerm, ...]:
        """Each crossing, in parameter order: a clique of the root partition and a clique of the deep partition, with its block."""

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

    @property
    @override
    def cliques(self) -> tuple[tuple[int, ...], ...]:
        """The root partition's cliques, the crossing cliques, then the deep partition's."""
        n_rot = self.rot_man.n_nodes
        crossing = tuple(near + shift_clique(far, n_rot) for near, far in self.crs_clqs)
        deep = tuple(shift_clique(clique, n_rot) for clique in self.dep_man.cliques)
        return self.rot_man.cliques + crossing + deep

    @property
    @override
    def nod_mans(self) -> tuple[Manifold, ...]:
        """The root partition's node spaces, then the deep partition's."""
        return self.rot_man.nod_mans + self.dep_man.nod_mans

    @property
    @override
    def clq_mans(self) -> tuple[Manifold, ...]:
        """The root partition's blocks, the crossing blocks, then the deep partition's."""
        return self.rot_man.clq_mans + self.crs_maps + self.dep_man.clq_mans

    # Methods

    @property
    def crs_clqs(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        """The crossing cliques of :attr:`crs_trms`."""
        return tuple((trm.cod_clq, trm.dom_clq) for trm in self.crs_trms)

    @property
    def crs_maps(self) -> tuple[CliqueMap, ...]:
        """The blocks of :attr:`crs_trms`."""
        return tuple(trm.clq_map for trm in self.crs_trms)

    @property
    def crs_man(self) -> CrossMap[Root, Deep]:
        """The cross partition: the cross map from the deep partition to the root partition, with one term per crossing clique."""
        return CrossMap(self.rot_man, self.dep_man, self.crs_trms)


@dataclass(frozen=True)
class RootEmbedding[
    Ambient: RecursiveLinearCliques[Any, Any],
    Sub: RecursiveLinearCliques[Any, Any],
](LinearEmbedding[Ambient, Sub]):
    """Embeds one layout into another over the same graph, transforming only the root partition.

    The two layouts have the same cliques and the same cross and deep partitions. Their root
    partitions are related by :attr:`rot_emb`.

    Mathematically, ``embed`` maps $(r, c, d) \\mapsto (\\phi(r), c, d)$ and ``project``
    maps $(r, c, d) \\mapsto (\\pi(r), c, d)$, with $\\phi, \\pi$ those of :attr:`rot_emb`.
    """

    # Fields

    rot_emb: LinearEmbedding[Any, Any]
    """Embedding of the submanifold's root partition into the ambient's."""

    _amb_man: Ambient
    _sub_man: Sub

    def __post_init__(self) -> None:
        sub, amb = self.sub_man, self.amb_man
        if sub.cliques != amb.cliques or (sub.crs_man.dim, sub.dep_man.dim) != (
            amb.crs_man.dim,
            amb.dep_man.dim,
        ):
            raise ValueError("sub and ambient may differ only in the root partition")

    # Overrides

    @property
    @override
    def amb_man(self) -> Ambient:
        return self._amb_man

    @property
    @override
    def sub_man(self) -> Sub:
        return self._sub_man

    @override
    def project(self, coords: Array) -> Array:
        root, cross, deep = self.amb_man.split_coords(coords)
        return self.sub_man.join_coords(self.rot_emb.project(root), cross, deep)

    @override
    def embed(self, coords: Array) -> Array:
        root, cross, deep = self.sub_man.split_coords(coords)
        return self.amb_man.join_coords(self.rot_emb.embed(root), cross, deep)
