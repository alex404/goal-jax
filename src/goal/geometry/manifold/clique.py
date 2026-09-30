"""Manifolds whose parameters are laid out over the cliques of a graph.

A :class:`CliqueMap` is a tensor over selected subspaces of the manifolds it couples, and
knows nothing about where they sit. An :class:`Interaction` sums several of them, each
reached through a pair of embeddings, into one map between whole manifolds. A
:class:`RecursiveLinearCliques` lays a coordinate
vector out over the cliques of a graph, as ``[root | cross | deep]`` with one block per
clique in the graph's canonical order, recursing into the deep partition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain
from math import prod
from typing import Any, override

import jax.numpy as jnp
from jax import Array

from ..algebra.clique import RecursiveCliques
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


### Interactions ###


@dataclass(frozen=True)
class Interaction[Domain: Manifold, Codomain: Manifold](LinearMap[Domain, Codomain]):
    """Several clique forms summed into one linear map between whole manifolds.

    Each form maps its input group's node coordinates to its output group's, and a **path**
    is what carries a domain point down to the nodes one form reads and puts its result back
    in the codomain. Parameters are the forms' concatenated in order, which
    :meth:`coord_blocks` splits apart again. A fork, a three-way coupling and a plain chain
    differ only in how many terms the sum has.

    The domain and codomain are whole manifolds --- a harmonium's observable and posterior,
    say --- rather than individual nodes, which is what the paths bridge. The paths are
    supplied at construction; :attr:`RecursiveLinearCliques.crs_man` derives them from the
    graph through :meth:`RecursiveLinearCliques.crs_embs`.

    Mathematically, writing $\\Theta_t$ for the $t$-th form and $\\pi_t, \\phi_t$ for its
    domain and codomain paths, the map is $v \\mapsto \\sum_t \\phi_t(\\Theta_t \\cdot
    \\pi_t(v))$.
    """

    # Fields

    _cod_man: Codomain
    """What every form's output lands in."""

    _dom_man: Domain
    """What every form's contracted side is read against."""

    cliques: tuple[CliqueMap, ...]
    """The forms, in parameter order."""

    paths: tuple[tuple[LinearEmbedding[Any, Any], LinearEmbedding[Any, Any]], ...]
    """Per form, how the codomain and the domain reach the nodes it couples.

    Transposing an interaction swaps the pair, since the two sides exchange roles.
    """

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
    def trn_man(self) -> Interaction[Codomain, Domain]:
        """The same forms read the other way: each form's :attr:`~goal.geometry.manifold.clique.CliqueMap.trn_man`, paths swapped.

        In a harmonium this is what a conditional posterior is, where the forward reading is
        a conditional likelihood. Its own transpose is this map again, so nothing nests.
        """
        return Interaction(
            self._dom_man,
            self._cod_man,
            tuple(form.trn_man for form in self.cliques),
            tuple((dom_path, cod_path) for cod_path, dom_path in self.paths),
        )

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        out = self.cod_man.zeros()
        for part, clique, (cod_path, dom_path) in zip(
            self.coord_blocks(f_coords), self.cliques, self.paths, strict=True
        ):
            w_node = clique(part, dom_path.project(v_coords))
            out = out + cod_path.embed(w_node)
        return out

    @override
    def transpose(self, f_coords: Array) -> Array:
        parts = [
            clique.transpose(part)
            for clique, part in zip(self.cliques, self.coord_blocks(f_coords))
        ]
        return jnp.concatenate(parts)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        parts = [
            clique.outer_product(cod_path.project(w_coords), dom_path.project(v_coords))
            for clique, (cod_path, dom_path) in zip(
                self.cliques, self.paths, strict=True
            )
        ]
        return jnp.concatenate(parts)

    # Methods

    @property
    def clq_dims(self) -> tuple[int, ...]:
        """Parameter dimension of each form, in parameter order."""
        return tuple(form.dim for form in self.cliques)

    @property
    def clique(self) -> CliqueMap:
        """The single form, for an interaction that has exactly one.

        Raises:
            ValueError: if the model has several, where there is no one form to talk about.
        """
        if len(self.cliques) != 1:
            msg = f"this interaction has {len(self.cliques)} cliques"
            raise ValueError(f"{msg}: there is no single form")
        return self.cliques[0]

    def coord_blocks(self, coords: Array) -> tuple[Array, ...]:
        """Split flat parameters into one array per form."""
        return split_by_dims(coords, self.clq_dims)


### Clique Layouts ###


def bias_map(man: Manifold) -> CliqueMap:
    """The map on a node's own clique: the whole node, as one output axis."""
    return CliqueMap(Rectangular(), (IdentityEmbedding(man),), ())


@dataclass(frozen=True)
class RecursiveLinearCliques[Root: Manifold, Deep: Manifold](
    RecursiveCliques, Triple[Root, Interaction[Deep, Root], Deep], ABC
):
    """A manifold stored ``[root | cross | deep]``, with one linear map per clique of its graph.

    A concrete subclass stores its graph (:attr:`cliques`, :attr:`root_nodes`) and states the
    root and deep partitions, the base space of each root node, and for each crossing clique
    a matrix representation and one subspace per node; the cross partition is derived from
    these. Blocks are stored in canonical order
    (:attr:`canonical_cliques` flattened), which keeps the partitions contiguous. It is the
    :class:`~goal.geometry.manifold.combinators.Triple` of its partitions, so their
    dimensions must equal the summed dimensions of their clique blocks.

    Labels are global. The deep partition is either a single node or another
    :class:`RecursiveLinearCliques` holding the cliques and root set this graph assigns it,
    so cliques pass down unrelabelled.

    A crossing clique splits into its root nodes, which are its output axes, and its deep
    nodes. Each part must be a clique of its partition whose map uses all of each node.

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
    def rot_nod_mans(self) -> tuple[Manifold, ...]:
        """The base space of each root node, in label order."""

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
    def snd_man(self) -> Interaction[Deep, Root]:
        return self.crs_man

    @property
    @override
    def trd_man(self) -> Deep:
        return self.dep_man

    # Methods

    @property
    def crs_man(self) -> Interaction[Deep, Root]:
        """The cross partition: the maps on the crossing cliques, reached through :meth:`crs_embs`."""
        cross = self.level_split()[1]
        maps = tuple(self.clq_map(clique) for clique in cross)
        embs = tuple(self.crs_embs(clique) for clique in cross)
        return Interaction(self.rot_man, self.dep_man, maps, embs)

    @property
    def dep_man_rlc(self) -> RecursiveLinearCliques[Any, Any] | None:
        """The manifold of the deep partition which is either a RecursiveLinearCliques, or ``None`` when it is a single node."""
        dep = self.dep_man
        return dep if isinstance(dep, RecursiveLinearCliques) else None

    @property
    def clq_dims(self) -> tuple[int, ...]:
        """Parameter dimension of each clique, in storage order."""
        order = chain.from_iterable(self.canonical_cliques)
        return tuple(self.clq_map(clique).dim for clique in order)

    def nod_man(self, node: int) -> Manifold:
        """The base space of ``node``."""
        if node in self.root_nodes:
            return self.rot_nod_mans[self.level_sets[0].index(node)]
        dep = self.dep_man_rlc
        return self.dep_man if dep is None else dep.nod_man(node)

    def split_clique(
        self, clique: tuple[int, ...]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """A clique's root nodes and its deep nodes."""
        near = tuple(i for i in clique if i in self.root_nodes)
        far = tuple(i for i in clique if i not in self.root_nodes)
        return near, far

    def clq_map(self, clique: tuple[int, ...]) -> CliqueMap:
        """The map on ``clique``, one axis per node.

        A crossing clique's axes are built on the subspaces used by the root and deep cliques
        it reaches. A clique of a recursive deep partition is that partition's, and any other
        clique is the bias of its node.
        """
        near, far = self.split_clique(clique)
        dep = self.dep_man_rlc
        if near and far:
            cons = dict(zip(clique, self.crs_emb_constructors(clique), strict=True))
            reached = self.clq_map(near).embs + self.clq_map(far).embs
            embs = [
                cons[i](r.sub_man) for i, r in zip(near + far, reached, strict=True)
            ]
            return CliqueMap(
                self.crs_rep(clique), tuple(embs[: len(near)]), tuple(embs[len(near) :])
            )
        if far and dep is not None:
            return dep.clq_map(clique)
        (node,) = clique
        return bias_map(self.nod_man(node))

    def clq_emb(self, clique: tuple[int, ...]) -> CliqueEmbedding:
        """The block of ``clique`` in this manifold's coordinates."""
        return CliqueEmbedding(clique, self)

    def crs_embs(
        self, clique: tuple[int, ...]
    ) -> tuple[LinearEmbedding[Any, Any], LinearEmbedding[Any, Any]]:
        """How :attr:`rot_man` and :attr:`dep_man` reach the two parts of a crossing clique.

        An embedding is the identity when its partition is that part alone.
        """
        near, far = self.split_clique(clique)
        root_part, _, deep_part = self.level_split()
        root = (
            IdentityEmbedding(self.rot_man)
            if root_part == (near,)
            else RootCliqueEmbedding(near, self)
        )
        dep = self.dep_man_rlc
        deep = (
            IdentityEmbedding(self.dep_man)
            if dep is None or deep_part == (far,)
            else dep.clq_emb(far)
        )
        return root, deep

    def coord_blocks(self, coords: Array) -> tuple[Array, ...]:
        """Split coordinates into one block per clique, in storage order."""
        return split_by_dims(coords, self.clq_dims)

    def split_level(self, coords: Array) -> tuple[Array, Array, Array]:
        return self.split_coords(coords)

    def join_level(self, root: Array, cross: Array, deep: Array) -> Array:
        return self.join_coords(root, cross, deep)


@dataclass(frozen=True)
class CliqueEmbedding(LinearEmbedding[CliqueMap, RecursiveLinearCliques[Any, Any]]):
    """The block of one clique in a layout's coordinates.

    ``project`` reads the block and ``embed`` writes it with every other coordinate zero; the
    subspace is the clique's :class:`CliqueMap`. It is the path an
    :class:`Interaction` takes when a crossing clique
    reaches one clique of a partition that holds several (MFA's $(x, y, k)$ reaching the
    mixture's $(y, k)$).

    Mathematically, the coordinates are a direct sum $\\bigoplus_C \\Theta^C$: ``project`` is
    the projection onto one summand and ``embed`` its inclusion, its transpose.
    """

    # Fields

    clique: tuple[int, ...]
    """The clique, as an ascending node tuple."""

    _amb_man: RecursiveLinearCliques[Any, Any]

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
    def amb_man(self) -> RecursiveLinearCliques[Any, Any]:
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
        order = tuple(chain.from_iterable(self.amb_man.canonical_cliques))
        return sum(self.amb_man.clq_dims[: order.index(self.clique)])


@dataclass(frozen=True)
class RootCliqueEmbedding(LinearEmbedding[CliqueMap, Manifold]):
    """The block of one root-partition clique in :attr:`RecursiveLinearCliques.rot_man` coordinates.

    The root partition is stored first, so the block sits where it does in the layout's own
    coordinates. It is the root path when the root partition holds several cliques (CCA's
    observable pair).
    """

    # Fields

    clique: tuple[int, ...]
    """The clique, as an ascending node tuple."""

    layout: RecursiveLinearCliques[Any, Any]
    """The layout whose root partition holds the clique."""

    # Overrides

    @property
    @override
    def sub_man(self) -> CliqueMap:
        return self.layout.clq_map(self.clique)

    @property
    @override
    def amb_man(self) -> Manifold:
        return self.layout.rot_man

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
        return self.layout.clq_emb(self.clique).start


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
