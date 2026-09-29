"""Manifolds whose parameters are laid out over the cliques of a graph.

A :class:`CliqueMap` is a tensor over selected subspaces of the manifolds it couples, and
knows nothing about where they sit. A :class:`LinearCliques` lays a coordinate vector out
over the cliques of a graph: it states the nodes' base spaces, the cliques, and for each
clique a matrix representation and the subspace it uses at each node, and derives the map
on each clique from these. A :class:`RecursiveLinearCliques` stores its cliques as
``[root | cross | deep]``, recursing into the deep partition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain
from math import prod
from typing import Any, Self, override

import jax.numpy as jnp
from jax import Array

from ..algebra.clique import Cliques
from ..algebra.matrix import MatrixRep, Rectangular
from .base import Manifold
from .combinators import Tuple
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

    factors: tuple[Manifold, ...]

    @property
    @override
    def dim(self) -> int:
        return prod(factor.dim for factor in self.factors)


@dataclass(frozen=True)
class TensorProductEmbedding(LinearEmbedding[TensorProduct, TensorProduct]):
    """A tensor product of linear embeddings, applied to a joint one axis at a time.

    Mathematically, $\\phi_1 \\otimes \\cdots \\otimes \\phi_k$. Each factor is
    linear, so it acts on any joint, not only on products of factor points.
    """

    factors: tuple[LinearEmbedding[Any, Any], ...]

    @property
    @override
    def amb_man(self) -> TensorProduct:
        return TensorProduct(tuple(emb.amb_man for emb in self.factors))

    @property
    @override
    def sub_man(self) -> TensorProduct:
        return TensorProduct(tuple(emb.sub_man for emb in self.factors))

    @override
    def project(self, coords: Array) -> Array:
        out = coords.reshape(tuple(emb.amb_man.dim for emb in self.factors))
        for axis, emb in enumerate(self.factors):
            out = jnp.apply_along_axis(emb.project, axis, out)
        return out.reshape(-1)

    @override
    def embed(self, coords: Array) -> Array:
        out = coords.reshape(tuple(emb.sub_man.dim for emb in self.factors))
        for axis, emb in enumerate(self.factors):
            out = jnp.apply_along_axis(emb.embed, axis, out)
        return out.reshape(-1)


@dataclass(frozen=True)
class CliqueMap(LinearMap[TensorProduct, TensorProduct]):
    """The linear map of a clique potential, acting on a chosen subspace of each factor.

    Each factor is a manifold reached by one embedding: :attr:`cod_embs` are the output
    factors, :attr:`dom_embs` the contracted ones. A map with no domain factors is a bias.

    Mathematically, the parameters are a tensor $\\Theta \\in \\mathbb R^{d_1 \\times \\cdots
    \\times d_j \\times e_1 \\times \\cdots \\times e_k}$ over the selected subspaces, stored
    as the matricization with the codomain axes as rows and the domain axes as columns.
    The representation acts on that matricization, so a representation with structure
    (diagonal, convolutional) sees the flattened shape $(\\prod_i d_i, \\prod_i e_i)$ when
    a group has more than one factor.
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
        when the factors within a group are dependent.
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
    def factor_embs(self) -> tuple[LinearEmbedding[Any, Any], ...]:
        """All embeddings in factor order: codomain, then domain."""
        return self.cod_embs + self.dom_embs

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return (self.cod_emb.sub_man.dim, self.dom_emb.sub_man.dim)

    def to_matrix(self, params: Array) -> Array:
        return self.rep.to_matrix(self.matrix_shape, params)

    def from_matrix(self, matrix: Array) -> Array:
        return self.rep.from_matrix(matrix)


### Clique Layouts ###


@dataclass(frozen=True)
class CliqueEmbedding[AmbientCliques: LinearCliques](
    LinearEmbedding[CliqueMap, AmbientCliques]
):
    """The coordinates of one clique of a layout, as a subspace of the whole layout.

    ``project`` reads the block of the clique on ``scope`` out of the layout's coordinates,
    and ``embed`` writes a block back with every other clique zero. The subspace is that
    clique's :class:`CliqueMap`, so a projected block reads as a tensor over the clique's
    nodes. Its main use is as a path in an
    :class:`~goal.geometry.manifold.interaction.Interaction`: when a crossing clique reaches
    a clique inside a partition that has several (MFA's $(x, y, k)$ reaching the mixture's
    $(y, k)$), the path lets the interaction act on the whole partition's coordinates.
    :meth:`RecursiveLinearCliques.cross_paths` builds these. Build one directly with
    :meth:`LinearCliques.clique_emb`, which sorts ``scope``.

    Mathematically, a layout's coordinates are a direct sum $\\bigoplus_C \\Theta^C$, and for
    the clique $C_0$ on ``scope``, ``project`` is the coordinate projection onto the summand
    $\\Theta^{C_0}$ and ``embed`` its inclusion. The two are transposes of each other, so one
    path serves both directions of an interaction.
    """

    scope: tuple[int, ...]
    _amb_man: AmbientCliques

    @property
    @override
    def amb_man(self) -> AmbientCliques:
        return self._amb_man

    @property
    @override
    def sub_man(self) -> CliqueMap:
        return self.amb_man.clique_map(self.scope)

    @override
    def project(self, coords: Array) -> Array:
        return self.amb_man.coord_blocks(coords)[self.index]

    @override
    def embed(self, coords: Array) -> Array:
        blocks = list(self.amb_man.coord_blocks(self.amb_man.zeros()))
        blocks[self.index] = coords
        return self.amb_man.join_blocks(*blocks)

    @property
    def index(self) -> int:
        """The position of the clique on :attr:`scope` in storage order."""
        return self.amb_man.clique_order.index(self.scope)


@dataclass(frozen=True)
class LinearCliques(Cliques, Manifold, ABC):
    """A manifold whose coordinates are one linear map per clique of a graph.

    A subclass states the nodes' base spaces, the cliques as ascending node tuples, and
    for each clique a matrix representation and the subspace it uses at each of its nodes.
    The map on a clique is derived from these. Flat: every node is a root, so the graph has
    one level, and every node of a clique is an output factor. A multi-node clique with a
    structured representation therefore sees a matrix with a single column.

    Mathematically, a point is a family $(\\Theta^C)_C$ indexed by the cliques, with $\\dim =
    \\sum_C \\dim(\\Theta^C)$.
    """

    # Contract

    @property
    @abstractmethod
    def node_mans(self) -> tuple[Manifold, ...]:
        """The base space of each node; node $i$ is ``node_mans[i]``."""

    @abstractmethod
    def clique_rep(self, clique: tuple[int, ...]) -> MatrixRep:
        """The matrix representation of the map on ``clique``."""

    @abstractmethod
    def subspace(
        self, clique: tuple[int, ...], node: int
    ) -> Callable[[Manifold], LinearEmbedding[Any, Any]]:
        """The subspace ``clique`` uses at ``node``, as a constructor applied to its ambient space."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return sum(self.clique_dims)

    @property
    @override
    def root_nodes(self) -> frozenset[int]:
        return frozenset(range(len(self.node_mans)))

    # Methods

    @property
    def clique_order(self) -> tuple[tuple[int, ...], ...]:
        """The cliques in storage order: :attr:`canonical_cliques`, flattened."""
        return tuple(chain.from_iterable(self.canonical_cliques))

    def clique_map(self, clique: tuple[int, ...]) -> CliqueMap:
        """The map on ``clique``, each node's subspace applied to its base space."""
        embs = tuple(self.subspace(clique, i)(self.node_mans[i]) for i in clique)
        return CliqueMap(self.clique_rep(clique), embs, ())

    @property
    def clique_dims(self) -> tuple[int, ...]:
        return tuple(self.clique_map(clique).dim for clique in self.clique_order)

    def coord_blocks(self, coords: Array) -> tuple[Array, ...]:
        """Split coordinates into one block per clique, in storage order."""
        return split_by_dims(coords, self.clique_dims)

    def join_blocks(self, *blocks: Array) -> Array:
        return jnp.concatenate(blocks)

    def clique_emb(self, scope: tuple[int, ...]) -> CliqueEmbedding[Self]:
        return CliqueEmbedding(tuple(sorted(scope)), self)


@dataclass(frozen=True)
class SingletonCliques(LinearCliques):
    """A manifold as a one-node layout: its only clique is the node, using all of it."""

    node_man: Manifold

    @property
    @override
    def node_mans(self) -> tuple[Manifold, ...]:
        return (self.node_man,)

    @property
    @override
    def cliques(self) -> tuple[tuple[int, ...], ...]:
        return ((0,),)

    @override
    def clique_rep(self, clique: tuple[int, ...]) -> MatrixRep:
        return Rectangular()

    @override
    def subspace(
        self, clique: tuple[int, ...], node: int
    ) -> Callable[[Manifold], LinearEmbedding[Any, Any]]:
        return IdentityEmbedding


@dataclass(frozen=True)
class RecursiveLinearCliques[Root: Manifold, Cross: Manifold, Deep: Manifold](
    LinearCliques, Tuple, ABC
):
    """A :class:`LinearCliques` stored ``[root | cross | deep]``, recursing into the deep partition.

    A subclass states the three partitions and the crossing cliques; the rest is read off
    the partitions through :attr:`root_layout` and :attr:`deep_layout`, with deep nodes
    numbered after the root nodes. A crossing clique's output factors are its root nodes.
    At each node, its subspace is applied to the subspace that the partition's clique on
    the same nodes uses there, which is the node's base space when that clique is the
    node alone.
    """

    # Contract

    @property
    @abstractmethod
    def root_man(self) -> Root: ...

    @property
    @abstractmethod
    def cross_man(self) -> Cross: ...

    @property
    @abstractmethod
    def deep_man(self) -> Deep: ...

    @property
    @abstractmethod
    def cross_cliques(self) -> tuple[tuple[int, ...], ...]:
        """The cliques joining root nodes to deep ones, each an ascending node tuple."""

    @abstractmethod
    def cross_rep(self, clique: tuple[int, ...]) -> MatrixRep:
        """The matrix representation of the map on a crossing clique."""

    @abstractmethod
    def cross_subspace(
        self, clique: tuple[int, ...], node: int
    ) -> Callable[[Manifold], LinearEmbedding[Any, Any]]:
        """The subspace a crossing clique uses at ``node``."""

    # Overrides

    @property
    @override
    def node_mans(self) -> tuple[Manifold, ...]:
        return self.root_layout.node_mans + self.deep_layout.node_mans

    @property
    @override
    def root_nodes(self) -> frozenset[int]:
        return frozenset(range(self.offset))

    @property
    @override
    def cliques(self) -> tuple[tuple[int, ...], ...]:
        return (
            self.root_layout.cliques
            + self.cross_cliques
            + self.from_deep(self.deep_layout.cliques)
        )

    @property
    @override
    def clique_order(self) -> tuple[tuple[int, ...], ...]:
        """Root storage, the crossing cliques in canonical order, then deep storage."""
        return (
            self.root_layout.clique_order
            + self.level_split()[1]
            + self.from_deep(self.deep_layout.clique_order)
        )

    @override
    def clique_rep(self, clique: tuple[int, ...]) -> MatrixRep:
        if clique[-1] < self.offset:
            return self.root_layout.clique_rep(clique)
        if clique[0] >= self.offset:
            return self.deep_layout.clique_rep(self.to_deep(clique))
        return self.cross_rep(clique)

    @override
    def subspace(
        self, clique: tuple[int, ...], node: int
    ) -> Callable[[Manifold], LinearEmbedding[Any, Any]]:
        if clique[-1] < self.offset:
            return self.root_layout.subspace(clique, node)
        if clique[0] >= self.offset:
            return self.deep_layout.subspace(self.to_deep(clique), node - self.offset)
        return self.cross_subspace(clique, node)

    @override
    def clique_map(self, clique: tuple[int, ...]) -> CliqueMap:
        if clique[-1] < self.offset:
            return self.root_layout.clique_map(clique)
        if clique[0] >= self.offset:
            return self.deep_layout.clique_map(self.to_deep(clique))
        near, far = self.split_clique(clique)
        near_map = self.root_layout.clique_map(near)
        far_map = self.deep_layout.clique_map(self.to_deep(far))
        near_embs = tuple(
            self.cross_subspace(clique, i)(emb.sub_man)
            for i, emb in zip(near, near_map.factor_embs, strict=True)
        )
        far_embs = tuple(
            self.cross_subspace(clique, i)(emb.sub_man)
            for i, emb in zip(far, far_map.factor_embs, strict=True)
        )
        return CliqueMap(self.cross_rep(clique), near_embs, far_embs)

    @override
    def split_coords(self, coords: Array) -> tuple[Array, Array, Array]:
        root = self.root_man.dim
        cross = root + sum(self.clique_map(c).dim for c in self.cross_cliques)
        return coords[:root], coords[root:cross], coords[cross:]

    @override
    def join_coords(self, *components: Array) -> Array:
        return jnp.concatenate(components)

    # Methods

    @property
    def root_layout(self) -> LinearCliques:
        """The root partition as a layout: itself if flat, otherwise one node.

        A recursive layout spans several levels, so it cannot be the root level.
        """
        man = self.root_man
        if isinstance(man, LinearCliques) and not isinstance(
            man, RecursiveLinearCliques
        ):
            return man
        return SingletonCliques(man)

    @property
    def deep_layout(self) -> LinearCliques:
        """The deep partition as a layout, in its own labels: itself if any layout, otherwise one node."""
        man = self.deep_man
        return man if isinstance(man, LinearCliques) else SingletonCliques(man)

    @property
    def offset(self) -> int:
        """The label of the first deep node."""
        return len(self.root_layout.node_mans)

    def from_deep(
        self, cliques: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]:
        """Deep cliques relabelled from the deep partition's labels to this level's."""
        return tuple(tuple(i + self.offset for i in c) for c in cliques)

    def to_deep(self, clique: tuple[int, ...]) -> tuple[int, ...]:
        """Deep nodes relabelled from this level's labels to the deep partition's."""
        return tuple(i - self.offset for i in clique)

    def split_clique(
        self, clique: tuple[int, ...]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """A crossing clique's root nodes and its deep nodes, in this level's labels."""
        near = tuple(i for i in clique if i < self.offset)
        return near, clique[len(near) :]

    def cross_paths(
        self, clique: tuple[int, ...]
    ) -> tuple[CliqueEmbedding[Any] | None, CliqueEmbedding[Any] | None]:
        """How :attr:`root_man` and :attr:`deep_man` reach a crossing clique's nodes.

        Each side's nodes must be a clique of its partition. A path is ``None`` when its
        partition is that one clique already.
        """
        near, far = self.split_clique(clique)
        root, deep = self.root_layout, self.deep_layout
        return (
            None if len(root.cliques) == 1 else root.clique_emb(near),
            None if len(deep.cliques) == 1 else deep.clique_emb(self.to_deep(far)),
        )

    def split_level(self, coords: Array) -> tuple[Array, Array, Array]:
        return self.split_coords(coords)

    def join_level(self, root: Array, cross: Array, deep: Array) -> Array:
        return self.join_coords(root, cross, deep)


@dataclass(frozen=True)
class RootEmbedding[
    Sub: RecursiveLinearCliques[Any, Any, Any],
    Ambient: RecursiveLinearCliques[Any, Any, Any],
](LinearEmbedding[Sub, Ambient]):
    """Embeds one layout into another over the same graph, transforming only the root partition.

    Mathematically, ``embed`` maps $(r, c, d) \\mapsto (\\phi(r), c, d)$ and ``project``
    maps $(r, c, d) \\mapsto (\\pi(r), c, d)$, with $\\phi, \\pi$ those of :attr:`root_emb`.
    """

    # Fields

    root_emb: LinearEmbedding[Any, Any]
    _sub_man: Sub
    _amb_man: Ambient

    def __post_init__(self) -> None:
        sub, amb = self.sub_man, self.amb_man
        if not sub.same_graph(amb) or (sub.cross_man.dim, sub.deep_man.dim) != (
            amb.cross_man.dim,
            amb.deep_man.dim,
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
        return self.sub_man.join_level(self.root_emb.project(root), cross, deep)

    @override
    def embed(self, coords: Array) -> Array:
        root, cross, deep = self.sub_man.split_level(coords)
        return self.amb_man.join_level(self.root_emb.embed(root), cross, deep)
