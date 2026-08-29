"""One linear map assembled from several clique forms and the paths that reach them.

A :class:`~goal.geometry.manifold.clique.LinearClique` is a form and nothing else: it has
$2^n$ conditional readings and privileges none. Materializing one as a
:class:`~goal.geometry.manifold.map.LinearMap` needs two facts that are *not* about the
form --- which axis is the output, and what manifold the caller actually holds --- so it
happens here rather than in the module that defines forms.

An :class:`Interaction` picks the **canonical fold** (axis 0 out, the rest contracted) over
one or more forms at once, and sums them. That is what lets a fork, a three-way coupling and
a plain chain all be one object: multiplicity is internal, and the nodes each form couples
travel with it. The *paths* --- how the domain and codomain reach the nodes a form couples
--- are supplied by whoever knows the graph, and derived rather than declared: see
:meth:`~goal.geometry.manifold.clique.LevelCliques.cross_paths`.

:class:`TransposedInteraction` is the same parameters read the other way, which the
canonical fold cannot express in the forward direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import jax.numpy as jnp
from jax import Array

from ..algebra.matrix import MatrixRep
from .base import Manifold
from .clique import LinearClique
from .embedding import LinearEmbedding
from .map import LinearMap
from .util import split_by_dims

### Interactions ###


@dataclass(frozen=True)
class Interaction[Domain: Manifold, Codomain: Manifold](LinearMap[Domain, Codomain]):
    """Several clique forms, read as one linear map under the canonical fold.

    Each form contributes axis 0 to the output and contracts the rest against the domain;
    the results are summed. Parameters are the forms' concatenated, in placement order.
    Summing is what makes multiplicity internal: a fork, a three-way coupling and a plain
    chain are one object with one, two or three placements, and no separate notion of a
    block is needed.

    The manifolds this maps between are the *whole* domain and codomain --- a harmonium's
    observable and posterior, say --- not the individual nodes. Reaching a node from there
    is what a **path** is, and paths are supplied rather than declared: for a harmonium
    they come from :meth:`~goal.geometry.manifold.clique.LevelCliques.cross_paths`, which
    derives them from the graph.
    """

    # Fields

    _cod_man: Codomain
    """What axis 0 of every form lands in."""

    _dom_man: Domain
    """What the remaining axes are contracted against."""

    placements: tuple[tuple[tuple[int, ...], LinearClique], ...]
    """One ``(members, form)`` pair per form, in parameter order."""

    paths: tuple[
        tuple[LinearEmbedding[Any, Any] | None, LinearEmbedding[Any, Any] | None], ...
    ]
    """Per form, how the codomain and the domain reach the nodes it couples.

    ``None`` on either side means that side *is* the node already, which is the common case.
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
        return sum(self.clique_dims)

    @property
    @override
    def trn_man(self) -> TransposedInteraction[Codomain, Domain]:
        """The same parameters contracted the other way."""
        return TransposedInteraction(self)

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        out = self.cod_man.zeros()
        for i, part in enumerate(self.coord_blocks(f_coords)):
            clique = self.cliques[i]
            internal = clique.rep.matvec(
                clique.matrix_shape, part, self._project_domain(i, v_coords)
            )
            out = out + self._embed_axis(i, 0, internal)
        return out

    @override
    def transpose(self, f_coords: Array) -> Array:
        parts = [
            clique.rep.transpose(clique.matrix_shape, part)
            for clique, part in zip(self.cliques, self.coord_blocks(f_coords))
        ]
        return jnp.concatenate(parts)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        parts = [
            self.cliques[i].rep.outer_product(
                self._project_axis(i, 0, w_coords), self._project_domain(i, v_coords)
            )
            for i in range(len(self.placements))
        ]
        return jnp.concatenate(parts)

    # Methods

    @property
    def cliques(self) -> tuple[LinearClique, ...]:
        """The forms alone, in parameter order."""
        return tuple(form for _, form in self.placements)

    @property
    def clique_dims(self) -> tuple[int, ...]:
        """Parameter dimension of each form, in parameter order."""
        return tuple(form.dim for form in self.cliques)

    @property
    def clique(self) -> LinearClique:
        """The single form, for an interaction that has exactly one.

        Raises:
            ValueError: if the model has several, where there is no one form to talk about.
        """
        if len(self.placements) != 1:
            msg = f"this interaction has {len(self.placements)} cliques"
            raise ValueError(f"{msg}: there is no single form")
        return self.cliques[0]

    def coord_blocks(self, coords: Array) -> tuple[Array, ...]:
        """Split flat parameters into one array per form."""
        return split_by_dims(coords, self.clique_dims)

    @property
    def rep(self) -> MatrixRep:
        """The single form's representation."""
        return self.clique.rep

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """The single form's canonical fold."""
        return self.clique.matrix_shape

    def to_matrix(self, params: Array) -> Array:
        """Unpack the single form's parameters into a dense matrix on selected dimensions."""
        return self.clique.to_matrix(params)

    def from_matrix(self, matrix: Array) -> Array:
        """Pack a dense matrix on selected dimensions into the single form's parameters."""
        return self.clique.from_matrix(matrix)

    @property
    def blocks(self) -> tuple[Interaction[Domain, Codomain], ...]:
        """Each form alone, as a map of the same shape.

        A multi-form interaction is a sum, so any one of its terms is an interaction in its
        own right over the same two sides. This is how a model reaches a single branch ---
        probabilistic CCA's, say --- without a separate notion of a block.
        """
        return tuple(
            Interaction(self._cod_man, self._dom_man, (placement,), (path,))
            for placement, path in zip(self.placements, self.paths, strict=True)
        )

    @property
    def cod_path(self) -> LinearEmbedding[Any, Any] | None:
        """How the single form's axis 0 reaches the codomain, if it does not sit there."""
        return self.paths[self._single][0]

    @property
    def dom_path(self) -> LinearEmbedding[Any, Any] | None:
        """How the single form's contracted axes reach the domain, if they are not it."""
        return self.paths[self._single][1]

    def node_coords(self, axis: int, amb_coords: Array) -> Array:
        """A caller's point taken down to one axis's node --- the path alone."""
        return self._node_coords(self._single, axis, amb_coords)

    def amb_coords(self, axis: int, node_coords: Array) -> Array:
        """One axis's node coordinates, placed back where the caller holds them."""
        return self._amb_coords(self._single, axis, node_coords)

    def project_axis(self, axis: int, amb_coords: Array) -> Array:
        """A caller's point restricted to one axis's selected coordinates."""
        return self._project_axis(self._single, axis, amb_coords)

    def embed_axis(self, axis: int, coords: Array) -> Array:
        """One axis's selected coordinates, placed back where they came from."""
        return self._embed_axis(self._single, axis, coords)

    def project_domain(self, v_coords: Array) -> Array:
        """A domain point in the form's contracted coordinates --- the constant $1$ for a bias."""
        return self._project_domain(self._single, v_coords)

    def embed_domain(self, coords: Array) -> Array:
        """The adjoint of :meth:`project_domain`: contracted coordinates back into the domain."""
        return self._embed_domain(self._single, coords)

    # Private

    @property
    def _single(self) -> int:
        """Index of the one clique, for the accessors that only make sense with one."""
        _ = self.clique
        return 0

    def _node_coords(self, index: int, axis: int, amb_coords: Array) -> Array:
        path = self._axis_path(index, axis)
        return amb_coords if path is None else path.project(amb_coords)

    def _amb_coords(self, index: int, axis: int, node_coords: Array) -> Array:
        path = self._axis_path(index, axis)
        return node_coords if path is None else path.embed(node_coords)

    def _project_axis(self, index: int, axis: int, amb_coords: Array) -> Array:
        clique = self.cliques[index]
        return clique.node_embs[axis].project(
            self._node_coords(index, axis, amb_coords)
        )

    def _embed_axis(self, index: int, axis: int, coords: Array) -> Array:
        clique = self.cliques[index]
        return self._amb_coords(index, axis, clique.node_embs[axis].embed(coords))

    def _project_domain(self, index: int, v_coords: Array) -> Array:
        """Two steps, and separating them is the point: the path takes the domain
        coordinates down to the *node* coordinates the form couples, and the form's own
        embeddings restrict from there.
        """
        clique = self.cliques[index]
        contracted = tuple(range(1, clique.arity))
        if not contracted:
            return jnp.ones(1)
        path = self.paths[index][1]
        joint = v_coords if path is None else path.project(v_coords)
        if len(contracted) == 1:
            # One node: its embedding projects directly, with no tensor to reshape. Not
            # only an optimization --- an embedding may accept a point it can restrict
            # without its ambient dimension matching exactly, and reshaping would not.
            return clique.node_embs[1].project(joint)
        return clique.select_joint(contracted, joint)

    def _embed_domain(self, index: int, coords: Array) -> Array:
        clique = self.cliques[index]
        contracted = tuple(range(1, clique.arity))
        if not contracted:
            return jnp.zeros(0)
        if len(contracted) == 1:
            node = clique.node_embs[1].embed(coords)
        else:
            node = clique.embed_joint(contracted, coords)
        path = self.paths[index][1]
        return node if path is None else path.embed(node)

    def _axis_path(self, index: int, axis: int) -> LinearEmbedding[Any, Any] | None:
        """The way in for one axis alone, when there is one.

        Axis 0 always has the codomain path. A contracted axis has the domain path only at
        arity 2, where it reaches exactly that one node; above that the path reaches a
        *group*, and no single axis can be placed through it.
        """
        obs_path, lat_path = self.paths[index]
        if axis == 0:
            return obs_path
        return lat_path if self.cliques[index].arity == 2 else None


@dataclass(frozen=True)
class TransposedInteraction[Domain: Manifold, Codomain: Manifold](
    LinearMap[Domain, Codomain]
):
    """An :class:`Interaction` read backwards: contract axis 0, land on the rest.

    The canonical fold makes axis 0 the output, so transposing puts a *group* of axes there
    instead --- which a forward reading cannot express. Hence a view: the same parameters,
    the same forms, the same paths, read in the other direction. In a harmonium this is what
    a conditional posterior is, where an :class:`Interaction` is a conditional likelihood.

    Its own transpose is the map it came from, so nothing nests.
    """

    # Fields

    fwd: Interaction[Codomain, Domain]
    """The forward reading. Its domain is this map's codomain, and vice versa."""

    # Overrides

    @property
    @override
    def dom_man(self) -> Domain:
        return self.fwd.cod_man

    @property
    @override
    def cod_man(self) -> Codomain:
        return self.fwd.dom_man

    @property
    @override
    def dim(self) -> int:
        return self.fwd.dim

    @property
    @override
    def trn_man(self) -> Interaction[Codomain, Domain]:
        return self.fwd

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        out = self.cod_man.zeros()
        blocks = self.fwd.blocks
        for i, part in enumerate(self.fwd.coord_blocks(f_coords)):
            block = blocks[i]
            rows, cols = block.matrix_shape
            internal = block.rep.matvec(
                (cols, rows), part, block.project_axis(0, v_coords)
            )
            out = out + block.embed_domain(internal)
        return out

    @override
    def transpose(self, f_coords: Array) -> Array:
        parts = []
        for clique, part in zip(self.fwd.cliques, self.fwd.coord_blocks(f_coords)):
            rows, cols = clique.matrix_shape
            parts.append(clique.rep.transpose((cols, rows), part))
        return jnp.concatenate(parts)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        return self.fwd.outer_product(v_coords, w_coords)

    # Methods

    @property
    def rep(self) -> MatrixRep:
        """The single form's representation --- the same numbers, read the other way."""
        return self.fwd.rep

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """The single forward form's fold, flipped."""
        rows, cols = self.fwd.matrix_shape
        return (cols, rows)

    def to_matrix(self, params: Array) -> Array:
        """Unpack flat parameters into a dense matrix on selected dimensions."""
        return self.fwd.clique.rep.to_matrix(self.matrix_shape, params)

    def from_matrix(self, matrix: Array) -> Array:
        """Pack a dense matrix on selected dimensions into flat parameters."""
        return self.fwd.clique.rep.from_matrix(matrix)
