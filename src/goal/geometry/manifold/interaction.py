"""One linear map assembled from several clique forms and the paths that reach them.

A :class:`~goal.geometry.manifold.map.SubspaceMap` is already a linear map between its
two node groups. What it does not know is what manifold the caller actually holds, and how
that manifold reaches the nodes it couples --- facts about the graph rather than about the
form, which is why a map between whole *partitions* is assembled here.

An :class:`Interaction` supplies those paths over one or more forms at once, and sums the
results. That is what lets a fork, a three-way coupling and a plain chain all be one object:
multiplicity is internal, and the nodes each form couples travel with it. The paths
themselves are supplied by whoever knows the graph, and derived rather than declared: see
:meth:`~goal.geometry.manifold.clique.LevelCliques.cross_paths`.

Reading the same parameters backwards is then just another interaction --- over each form's
:attr:`~goal.geometry.manifold.map.SubspaceMap.trn_man` and the two paths swapped ---
so there is no separate class for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import jax.numpy as jnp
from jax import Array

from ..algebra.util import split_by_dims
from .base import Manifold
from .embedding import LinearEmbedding
from .map import LinearMap, SubspaceMap

### Interactions ###


@dataclass(frozen=True)
class Interaction[Domain: Manifold, Codomain: Manifold](LinearMap[Domain, Codomain]):
    """Several clique forms summed into one linear map between whole manifolds.

    Each form maps its input group's node coordinates to its output group's, and a **path**
    is what carries a domain point down to the nodes one form reads and puts its result back
    in the codomain. Parameters are the forms' concatenated in placement order, which
    :meth:`coord_blocks` splits apart again. A fork, a three-way coupling and a plain chain
    differ only in how many terms the sum has.

    The domain and codomain are whole manifolds --- a harmonium's observable and posterior,
    say --- rather than individual nodes, which is what the paths bridge. They are supplied
    at construction rather than derived here; for a harmonium
    :meth:`~goal.geometry.manifold.clique.LevelCliques.cross_paths` derives them from the
    graph.

    Mathematically, writing $\\Theta_t$ for the $t$-th form and $\\pi_t, \\phi_t$ for its
    domain and codomain paths, the map is $v \\mapsto \\sum_t \\phi_t(\\Theta_t \\cdot
    \\pi_t(v))$.
    """

    # Fields

    _cod_man: Codomain
    """What every form's output lands in."""

    _dom_man: Domain
    """What every form's contracted side is read against."""

    placements: tuple[tuple[tuple[int, ...], SubspaceMap], ...]
    """One ``(members, form)`` pair per form, in parameter order.

    For forward forms, ``members`` ascending pairs positionally with the form's
    ``cod_embs + dom_embs``. In a *transposed* interaction the members are carried verbatim
    while the form's groups have swapped, so that positional pairing holds only for the
    forward reading --- nothing here reads it at runtime.
    """

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
        """The same forms read the other way: each form's :attr:`~goal.geometry.manifold.map.SubspaceMap.trn_man`, paths swapped.

        In a harmonium this is what a conditional posterior is, where the forward reading is
        a conditional likelihood. Its own transpose is this map again, so nothing nests.
        """
        return Interaction(
            self._dom_man,
            self._cod_man,
            tuple((members, form.trn_man) for members, form in self.placements),
            tuple((dom_path, cod_path) for cod_path, dom_path in self.paths),
        )

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        out = self.cod_man.zeros()
        for part, clique, (cod_path, dom_path) in zip(
            self.coord_blocks(f_coords), self.cliques, self.paths, strict=True
        ):
            v_node = v_coords if dom_path is None else dom_path.project(v_coords)
            w_node = clique(part, v_node)
            out = out + (w_node if cod_path is None else cod_path.embed(w_node))
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
            clique.outer_product(
                w_coords if cod_path is None else cod_path.project(w_coords),
                v_coords if dom_path is None else dom_path.project(v_coords),
            )
            for clique, (cod_path, dom_path) in zip(
                self.cliques, self.paths, strict=True
            )
        ]
        return jnp.concatenate(parts)

    # Methods

    @property
    def cliques(self) -> tuple[SubspaceMap, ...]:
        """The forms alone, in parameter order."""
        return tuple(form for _, form in self.placements)

    @property
    def clique_dims(self) -> tuple[int, ...]:
        """Parameter dimension of each form, in parameter order."""
        return tuple(form.dim for form in self.cliques)

    @property
    def clique(self) -> SubspaceMap:
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
