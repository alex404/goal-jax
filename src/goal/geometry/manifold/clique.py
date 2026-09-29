"""Manifolds whose parameters are laid out over the cliques of a graph.

A :class:`CliqueMap` is a tensor over selected subspaces of the manifolds it couples, and
knows nothing about where they sit. A :class:`Potential` pairs one with its *scope*, the
nodes its factors occupy. A :class:`LinearCliques` lays a coordinate vector out over its
potentials, and a :class:`RecursiveLinearCliques` stores them as ``[root | cross | deep]``,
recursing into the deep partition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from typing import Any, NamedTuple, Self, override

import jax
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
            out = _map_axis(out, axis, emb.project)
        return out.reshape(-1)

    @override
    def embed(self, coords: Array) -> Array:
        out = coords.reshape(tuple(emb.sub_man.dim for emb in self.factors))
        for axis, emb in enumerate(self.factors):
            out = _map_axis(out, axis, emb.embed)
        return out.reshape(-1)


def _map_axis(tensor: Array, axis: int, fn: Any) -> Array:
    """Apply a vector function along one axis of a tensor."""
    moved = jnp.moveaxis(tensor, axis, 0)
    columns = moved.reshape(moved.shape[0], -1)
    mapped = jax.vmap(fn, in_axes=1, out_axes=1)(columns)
    return jnp.moveaxis(mapped.reshape((-1, *moved.shape[1:])), 0, axis)


@dataclass(frozen=True)
class CliqueMap(LinearMap[TensorProduct, TensorProduct]):
    """The linear map of a clique potential, acting on a chosen subspace of each factor.

    Each factor is a manifold reached by one embedding: :attr:`cod_embs` are the output
    factors, :attr:`dom_embs` the contracted ones. A map with no domain factors is a bias.

    Mathematically, the parameters are a tensor $\\Theta \\in \\mathbb R^{d_1 \\times \\cdots
    \\times d_j \\times e_1 \\times \\cdots \\times e_k}$ over the selected subspaces, stored
    as the matricization with the codomain axes as rows and the domain axes as columns.
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


class Potential(NamedTuple):
    """A :class:`CliqueMap` and its scope: factor $i$ sits at node ``scope[i]``."""

    scope: tuple[int, ...]
    map: CliqueMap


### Clique Layouts ###


@dataclass(frozen=True)
class CliqueEmbedding[AmbientCliques: LinearCliques](
    LinearEmbedding[CliqueMap, AmbientCliques]
):
    """The coordinates of the potential on ``scope`` inside a layout. Build with :meth:`LinearCliques.clique_emb`."""

    scope: tuple[int, ...]
    _amb_man: AmbientCliques

    @property
    @override
    def amb_man(self) -> AmbientCliques:
        return self._amb_man

    @property
    @override
    def sub_man(self) -> CliqueMap:
        return self.amb_man.potentials[self.index].map

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
        """The position of the potential on :attr:`scope` in storage order."""
        return self.amb_man.cliques.index(self.scope)


@dataclass(frozen=True)
class LinearCliques(Cliques, Manifold, ABC):
    """A manifold whose coordinates are one potential per clique of a graph, in storage order.

    Flat: every node is a root, so the graph has one level.

    Mathematically, a point is a family $(\\Theta^C)_C$ indexed by the cliques, with $\\dim =
    \\sum_C \\dim(\\Theta^C)$.
    """

    # Contract

    @property
    @abstractmethod
    def potentials(self) -> tuple[Potential, ...]:
        """Every potential, in storage order."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return sum(self.clique_dims)

    @property
    @override
    def root_nodes(self) -> frozenset[int]:
        return frozenset(i for scope, _ in self.potentials for i in scope)

    @property
    @override
    def cliques(self) -> tuple[tuple[int, ...], ...]:
        """The scopes of :attr:`potentials`, in storage order.

        Raises:
            ValueError: if a scope and its map's arity disagree, a scope is not ascending
                and distinct, or two potentials share a scope.
        """
        scopes = tuple(scope for scope, _ in self.potentials)
        for scope, form in self.potentials:
            if (
                len(scope) != len(form.factor_embs)
                or tuple(sorted(set(scope))) != scope
            ):
                msg = f"clique {scope} must name {len(form.factor_embs)} distinct nodes, ascending"
                raise ValueError(msg)
        if len(set(scopes)) != len(scopes):
            raise ValueError(f"duplicate cliques in {scopes}")
        return scopes

    # Methods

    @property
    def clique_dims(self) -> tuple[int, ...]:
        return tuple(form.dim for _, form in self.potentials)

    def coord_blocks(self, coords: Array) -> tuple[Array, ...]:
        """Split coordinates into one block per potential, in storage order."""
        return split_by_dims(coords, self.clique_dims)

    def join_blocks(self, *blocks: Array) -> Array:
        return jnp.concatenate(blocks)

    def clique_emb(self, scope: tuple[int, ...]) -> CliqueEmbedding[Self]:
        return CliqueEmbedding(tuple(sorted(scope)), self)


@dataclass(frozen=True)
class RecursiveLinearCliques[Root: Manifold, Cross: Manifold, Deep: Manifold](
    LinearCliques, Tuple, ABC
):
    """A :class:`LinearCliques` stored ``[root | cross | deep]``, recursing into the deep partition.

    A subclass states the three partitions and the cross potentials; the rest is read off
    the partitions. The root partition contributes its potentials if it is a flat
    :class:`LinearCliques`, and is otherwise one node --- a recursive layout spans several
    levels, so it cannot be the root level. The deep partition contributes its potentials
    if it is any :class:`LinearCliques`, renumbered past the root nodes, and is otherwise
    one node.
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
    def cross_potentials(self) -> tuple[Potential, ...]:
        """The potentials joining the root nodes to the deep ones, in this level's labels."""

    # Overrides

    @property
    @override
    def root_nodes(self) -> frozenset[int]:
        return frozenset(i for scope, _ in self.root_potentials for i in scope)

    @property
    @override
    def potentials(self) -> tuple[Potential, ...]:
        """Root, cross, then deep renumbered past the root nodes."""
        offset = max(self.root_nodes) + 1
        deep = tuple(
            Potential(tuple(i + offset for i in scope), form)
            for scope, form in self.deep_potentials
        )
        return self.root_potentials + self.cross_potentials + deep

    @override
    def split_coords(self, coords: Array) -> tuple[Array, Array, Array]:
        root = sum(form.dim for _, form in self.root_potentials)
        cross = root + sum(form.dim for _, form in self.cross_potentials)
        return coords[:root], coords[root:cross], coords[cross:]

    @override
    def join_coords(self, *components: Array) -> Array:
        return jnp.concatenate(components)

    # Methods

    @property
    def root_potentials(self) -> tuple[Potential, ...]:
        man = self.root_man
        if isinstance(man, LinearCliques) and not isinstance(
            man, RecursiveLinearCliques
        ):
            return man.potentials
        return (
            Potential((0,), CliqueMap(Rectangular(), (IdentityEmbedding(man),), ())),
        )

    @property
    def deep_potentials(self) -> tuple[Potential, ...]:
        """The deep partition's potentials, in its own labels."""
        man = self.deep_man
        if isinstance(man, LinearCliques):
            return man.potentials
        return (
            Potential((0,), CliqueMap(Rectangular(), (IdentityEmbedding(man),), ())),
        )

    def cross_potential(
        self, rep: MatrixRep, node_embs: Mapping[int, LinearEmbedding[Any, Any]]
    ) -> Potential:
        """A crossing potential on the given nodes, using the given subspace at each.

        Factor order is the nodes ascending; the root nodes among them are the output
        group. Deep labels always follow root labels, so the root nodes come first.
        :meth:`cross_paths` checks the result.
        """
        scope = tuple(sorted(node_embs))
        n_near = sum(i in self.root_nodes for i in scope)
        embs = tuple(node_embs[i] for i in scope)
        return Potential(scope, CliqueMap(rep, embs[:n_near], embs[n_near:]))

    def cross_paths(
        self, potential: Potential
    ) -> tuple[LinearEmbedding[Any, Any] | None, LinearEmbedding[Any, Any] | None]:
        """How :attr:`root_man` and :attr:`deep_man` reach a crossing potential's nodes.

        The root nodes of the scope must be the map's output factors, and the rest its
        input factors; each group must be a clique of its partition. A path is ``None``
        when its partition is that one clique already.

        Raises:
            ValueError: unless the output factors are exactly the scope's root nodes, and
                there is at least one of each.
        """
        scope, form = potential
        roots = self.root_nodes
        n_near = len(form.cod_embs)
        near, far = scope[:n_near], scope[n_near:]
        if not (near and far) or not roots.issuperset(near) or roots & set(far):
            msg = f"crossing clique {scope}: its {n_near} output factors must be"
            raise ValueError(f"{msg} exactly its root nodes, with deep nodes besides")
        offset = max(roots) + 1
        return (
            _path(self.root_man, self.root_potentials, near),
            _path(self.deep_man, self.deep_potentials, tuple(i - offset for i in far)),
        )

    def split_level(self, coords: Array) -> tuple[Array, Array, Array]:
        return self.split_coords(coords)

    def join_level(self, root: Array, cross: Array, deep: Array) -> Array:
        return self.join_coords(root, cross, deep)


def _path(
    man: Manifold, potentials: tuple[Potential, ...], scope: tuple[int, ...]
) -> LinearEmbedding[Any, Any] | None:
    """How a partition reaches its clique on ``scope``, or ``None`` if it is that clique."""
    if len(potentials) == 1 or not isinstance(man, LinearCliques):
        return None
    return man.clique_emb(scope)


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
