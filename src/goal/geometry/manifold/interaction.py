"""One linear map assembled from several clique forms and the paths that reach them.

A :class:`~goal.geometry.manifold.clique.LinearClique` already knows which of its axes are
the output. What it does not know is what manifold the caller actually holds, and how that
manifold reaches the nodes it couples --- facts about the graph rather than about the form,
which is why materializing a form as a :class:`~goal.geometry.manifold.map.LinearMap`
happens here.

An :class:`Interaction` supplies those paths over one or more forms at once, and sums the
results. That is what lets a fork, a three-way coupling and a plain chain all be one object:
multiplicity is internal, and the nodes each form couples travel with it. The paths
themselves are supplied by whoever knows the graph, and derived rather than declared: see
:meth:`~goal.geometry.manifold.clique.LevelCliques.cross_paths`.

Reading the same parameters backwards is then just another interaction --- over each form's
:meth:`~goal.geometry.manifold.clique.LinearClique.transposed` and the two paths swapped ---
so there is no separate class for it.
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
    """Several clique forms, read as one linear map.

    Each form sends its own output group to the codomain and contracts the rest against the
    domain; the results are summed. Parameters are the forms' concatenated, in placement
    order. Summing is what makes multiplicity internal: a fork, a three-way coupling and a
    plain chain are one object with one, two or three placements, and no separate notion of
    a block is needed.

    The manifolds this maps between are the *whole* domain and codomain --- a harmonium's
    observable and posterior, say --- not the individual nodes. Reaching a node from there
    is what a **path** is, and paths are supplied rather than declared: for a harmonium
    they come from :meth:`~goal.geometry.manifold.clique.LevelCliques.cross_paths`, which
    derives them from the graph.
    """

    # Fields

    _cod_man: Codomain
    """What every form's output group lands in."""

    _dom_man: Domain
    """What every form's contracted axes are read against."""

    placements: tuple[tuple[tuple[int, ...], LinearClique], ...]
    """One ``(members, form)`` pair per form, in parameter order."""

    paths: tuple[
        tuple[LinearEmbedding[Any, Any] | None, LinearEmbedding[Any, Any] | None], ...
    ]
    """Per form, how the codomain and the domain reach the nodes it couples.

    ``None`` on either side means that side *is* the node already, which is the common case.
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
        return sum(self.clique_dims)

    @property
    @override
    def trn_man(self) -> Interaction[Codomain, Domain]:
        """The same forms read the other way: groups exchanged, paths swapped.

        In a harmonium this is what a conditional posterior is, where the forward reading is
        a conditional likelihood. Its own transpose is this map again, so nothing nests.
        """
        return Interaction(
            self._dom_man,
            self._cod_man,
            tuple((members, form.transposed()) for members, form in self.placements),
            tuple((dom_path, cod_path) for cod_path, dom_path in self.paths),
        )

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        out = self.cod_man.zeros()
        for i, part in enumerate(self.coord_blocks(f_coords)):
            clique = self.cliques[i]
            internal = clique.rep.matvec(
                clique.matrix_shape, part, self._project_domain(i, v_coords)
            )
            out = out + self._embed_codomain(i, internal)
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
            self.cliques[i].outer_product(
                self._cod_node_coords(i, w_coords),
                self._dom_node_coords(i, v_coords),
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
        """How the single form's output group reaches the codomain, if it does not sit there."""
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

    def _cod_node_coords(self, index: int, w_coords: Array) -> Array:
        """A codomain point taken down to the node coordinates the output group couples."""
        path = self.paths[index][0]
        return w_coords if path is None else path.project(w_coords)

    def _dom_node_coords(self, index: int, v_coords: Array) -> Array:
        """A domain point taken down to the node coordinates the contracted axes couple."""
        path = self.paths[index][1]
        return v_coords if path is None else path.project(v_coords)

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
        return clique.project_in(self._dom_node_coords(index, v_coords))

    def _embed_domain(self, index: int, coords: Array) -> Array:
        clique = self.cliques[index]
        node = clique.embed_in(coords)
        path = self.paths[index][1]
        return node if path is None else path.embed(node)

    def _embed_codomain(self, index: int, coords: Array) -> Array:
        clique = self.cliques[index]
        node = clique.embed_out(coords)
        path = self.paths[index][0]
        return node if path is None else path.embed(node)

    def _axis_path(self, index: int, axis: int) -> LinearEmbedding[Any, Any] | None:
        """The way in for one axis alone, when there is one.

        An axis has its side's path only when that side's group is exactly this one node.
        When a group spans several axes the path reaches the *group*, and no single axis can
        be placed through it.
        """
        cod_path, dom_path = self.paths[index]
        clique = self.cliques[index]
        if axis in clique.out_axes:
            return cod_path if len(clique.out_axes) == 1 else None
        return dom_path if len(clique.in_axes) == 1 else None
