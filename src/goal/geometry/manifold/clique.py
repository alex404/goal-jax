"""Manifolds whose parameters are laid out over the cliques of a graph.

A :class:`CliqueMap` is a tensor over selected subspaces of the manifolds it couples, and
knows nothing about where they sit. A :class:`LinearCliques` stores one block per clique,
each with its map, and numbers its own nodes. A :class:`CrossMap` sums several clique maps
into one map between two :class:`LinearCliques`, each term naming a clique of each side in
that side's own numbering. A :class:`RecursiveLinearCliques` is laid out as
``[root | cross | deep]``: its graph is composed from its root and deep partitions and the
crossing cliques it declares between them, and its cross partition is the cross map
between the two.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import Any, override

import jax.numpy as jnp
from jax import Array

from ..algebra.clique import Cliques
from ..algebra.matrix import MatrixRep, Rectangular
from .base import Manifold
from .combinators import Triple
from .embedding import IdentityEmbedding, LinearEmbedding
from .map import LinearMap
from .util import split_by_dims

type Crossing = tuple[tuple[int, ...], tuple[int, ...]]
"""A crossing clique: its root part and its deep part, each in its own partition's numbering."""

type EmbeddingConstructor = Callable[[Manifold], LinearEmbedding[Any, Any]]
"""Builds the embedding of a crossing axis from the subspace its node's part uses."""

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

    The nodes are numbered $0, \\ldots, n - 1$ by the manifold itself, and the numbers mean
    nothing outside it: a manifold that holds this one as a part translates them (see
    :class:`RecursiveLinearCliques`). Its dimension, from wherever a subclass takes it,
    must equal the summed block dimensions. It is not a
    :class:`~goal.geometry.manifold.combinators.Tuple`, because a subclass may already be
    one over coarser components (a :class:`RecursiveLinearCliques` is the ``Triple`` of
    its partitions), so the block split is :meth:`coord_blocks`.
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
    subspace is the clique's :class:`CliqueMap`. A :class:`CrossMap` reads each term's parts
    through one, whether the part is a whole single-node family or one of several cliques of
    a layout (MFA's $(x, y, k)$, whose part $(y, k)$ is a clique of the mixture).

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


### Cross Maps ###


@dataclass(frozen=True)
class CrossMap[Codomain: LinearCliques, Domain: LinearCliques](
    LinearMap[Codomain, Domain]
):
    """Several clique maps summed into one linear map between two :class:`LinearCliques`.

    Each term couples a clique of the codomain with a clique of the domain, each in that
    side's own numbering, through a clique map; it reads and writes its parts through their
    blocks (:meth:`LinearCliques.clq_emb`). Parameters are the terms' concatenated in order,
    which :meth:`coord_blocks` splits apart again. A fork, a three-way coupling and a plain
    chain differ only in which terms the sum has.

    Mathematically, writing $\\Theta_t$ for the $t$-th clique map and $\\pi_t, \\phi_t$
    for the blocks of its parts in the domain and codomain, the map is
    $v \\mapsto \\sum_t \\phi_t(\\Theta_t \\cdot \\pi_t(v))$.
    """

    # Fields

    _cod_man: Codomain
    _dom_man: Domain
    terms: tuple[tuple[tuple[int, ...], tuple[int, ...], CliqueMap], ...]
    """Each term's codomain clique, domain clique and map, in parameter order."""

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
        terms = tuple((dom, cod, clq_map.trn_man) for cod, dom, clq_map in self.terms)
        return CrossMap(self._dom_man, self._cod_man, terms)

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        out = self.cod_man.zeros()
        for (cod, dom, clq_map), params in zip(
            self.terms, self.coord_blocks(f_coords), strict=True
        ):
            selected = self.dom_man.clq_emb(dom).project(v_coords)
            out = out + self.cod_man.clq_emb(cod).embed(clq_map(params, selected))
        return out

    @override
    def transpose(self, f_coords: Array) -> Array:
        parts = [
            clq_map.transpose(params)
            for (_, _, clq_map), params in zip(
                self.terms, self.coord_blocks(f_coords), strict=True
            )
        ]
        return jnp.concatenate(parts)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        parts = [
            clq_map.outer_product(
                self.cod_man.clq_emb(cod).project(w_coords),
                self.dom_man.clq_emb(dom).project(v_coords),
            )
            for cod, dom, clq_map in self.terms
        ]
        return jnp.concatenate(parts)

    # Methods

    @property
    def clq_dims(self) -> tuple[int, ...]:
        """Parameter dimension of each term, in parameter order."""
        return tuple(clq_map.dim for _, _, clq_map in self.terms)

    def coord_blocks(self, coords: Array) -> tuple[Array, ...]:
        """Split flat parameters into one array per term."""
        return split_by_dims(coords, self.clq_dims)


### Recursive Layouts ###


@dataclass(frozen=True)
class RecursiveLinearCliques[Root: LinearCliques, Deep: LinearCliques](
    LinearCliques,
    Triple[Root, CrossMap[Root, Deep], Deep],
    ABC,
):
    """A manifold stored ``[root | cross | deep]``, whose graph is composed from its parts.

    A concrete subclass states the root and deep partitions and the crossing cliques between
    them, and for each crossing clique a matrix representation and one embedding
    constructor per node; everything else is derived. A crossing clique is a pair: a clique
    of the root partition and a clique of the deep partition, each in its own partition's
    numbering, so the partitions are used as they are and nothing is relabelled.

    The composed graph numbers the root partition's nodes first, as the root numbers them,
    then the deep partition's, offset by the root's node count. Its cliques are the root
    partition's, then each crossing clique, then the deep partition's, which is exactly the
    order the ``Triple`` stores them in.

    A crossing clique's output axes are its root nodes. Its axes are built on the subspaces
    its parts use: each part must be a clique of its partition, which is what keeps the
    likelihood and the posterior in their partitions' families, and the part's map gives one
    subspace per node.

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
    def crs_cliques(self) -> tuple[Crossing, ...]:
        """The crossing cliques, in parameter order."""

    @abstractmethod
    def crs_rep(self, crossing: Crossing) -> MatrixRep:
        """The matrix representation of the map on a crossing clique."""

    @abstractmethod
    def crs_emb_constructors(
        self, crossing: Crossing
    ) -> tuple[tuple[EmbeddingConstructor, ...], tuple[EmbeddingConstructor, ...]]:
        """One embedding constructor per node of each part of a crossing clique, root part first."""

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
        crossing = tuple(
            near + tuple(i + n_rot for i in far) for near, far in self.crs_cliques
        )
        deep = tuple(
            tuple(i + n_rot for i in clique) for clique in self.dep_man.cliques
        )
        return self.rot_man.cliques + crossing + deep

    @override
    def clq_map(self, clique: tuple[int, ...]) -> CliqueMap:
        """The map on ``clique``: its partition's if it lies in one, the crossing map otherwise."""
        n_rot = self.rot_man.n_nodes
        near = tuple(i for i in clique if i < n_rot)
        far = tuple(i - n_rot for i in clique if i >= n_rot)
        if not far:
            return self.rot_man.clq_map(near)
        if not near:
            return self.dep_man.clq_map(far)
        return self.crs_map((near, far))

    # Methods

    @property
    def crs_man(self) -> CrossMap[Root, Deep]:
        """The cross partition: the map on each crossing clique, from the deep partition to the root."""
        terms = tuple(
            (near, far, self.crs_map((near, far))) for near, far in self.crs_cliques
        )
        return CrossMap(self.rot_man, self.dep_man, terms)

    def crs_map(self, crossing: Crossing) -> CliqueMap:
        """The map on a crossing clique, one axis per node, root nodes as outputs."""
        near, far = crossing
        cod_cons, dom_cons = self.crs_emb_constructors(crossing)
        cod_parts = self.rot_man.clq_map(near).embs
        dom_parts = self.dep_man.clq_map(far).embs
        return CliqueMap(
            self.crs_rep(crossing),
            tuple(con(p.sub_man) for con, p in zip(cod_cons, cod_parts, strict=True)),
            tuple(con(p.sub_man) for con, p in zip(dom_cons, dom_parts, strict=True)),
        )

    def split_level(self, coords: Array) -> tuple[Array, Array, Array]:
        return self.split_coords(coords)

    def join_level(self, root: Array, cross: Array, deep: Array) -> Array:
        return self.join_coords(root, cross, deep)
