"""Cliques over exponential families: one clique, one inner product, and the layouts over them.

An :class:`EFClique` is a multilinear form on a graph position together with what that form
*pairs against*: the tensor product of its member nodes' sufficient statistics, each
restricted to the sub-statistic this clique actually couples. The selectors carry both
facts at once --- they name the sub-statistics, and their dimensions *are* the form's
factors, so arity is one fact rather than several that can disagree.

:class:`LinearCliques` lays a coordinate vector out over a tuple of such cliques,
:class:`LevelCliques` is the recursive case over one level of a graph, and
:class:`CliqueProduct` the disjoint union a multi-root model's root span needs. The graph's
bare combinatorics, with no parameters attached, are
:class:`~goal.geometry.algebra.clique.Cliques`.

Mathematically, a clique $c$ contributes exactly one term to the log-density,

.. math::
    \\langle \\theta_c, \\bigotimes_{i \\in c} \\pi_i(\\mathbf s_i(x_i)) \\rangle,

with $\\pi_i$ the selector for member $i$. Arity is the only thing that varies: at $|c| = 1$
this is a bias, at $|c| = 2$ a matrix contraction, and beyond that a higher-order tensor
contraction. A bias and a coupling differ in arity, not in kind.

**Why the layouts live here and not below.** A clique's factor dimensions are decided by
which sub-statistics it couples, and a sub-statistic is an exponential-family notion. A
layout layer that could not see selectors would have to recover the factors from something
else, which is exactly the reconstruction this module exists to make unnecessary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.clique import LevelCliques, LinearClique, LinearCliques
from ..manifold.embedding import LinearEmbedding
from ..manifold.map import EmbeddedMap, LinearMap
from .base import ExponentialFamily


@dataclass(frozen=True)
class EFClique(LinearClique):
    """One clique: a multilinear interaction among its member exponential-family nodes.

    Each member contributes a **selector** --- a
    :class:`~goal.geometry.manifold.embedding.LinearEmbedding` from the sub-statistic this
    clique couples into that node's full statistic. One selector per member, most of them
    identities, and nothing else deciding which coordinates the form addresses.

    Mathematically, the clique holds a parameter tensor $\\Theta_c$ contracted against one
    vector per member, $\\langle \\Theta_c, \\bigotimes_{i \\in c} v_i \\rangle$, so its
    parameters inhabit the dense multilinear :attr:`form` over the subspaces its members
    contribute. Arity is a degree, not a kind: a one-member form and a coupling are both
    cliques, and neither needs its own type.

    At arity 2 with selectors ``(cod_emb, dom_emb)`` this reproduces
    :class:`~goal.geometry.manifold.map.EmbeddedMap` exactly, entry for entry ---
    :meth:`tensor` is its ``outer_product`` and :meth:`contract` is its application and
    transposed application. The gain is that arity is no longer capped at 2.

    What this class adds to a bare multilinear form is the selectors, and every operation
    below is the inherited one conjugated by them --- the same relationship
    :class:`~goal.geometry.manifold.map.EmbeddedMap` has to its matrix, generalized past
    arity 2.
    """

    # Fields

    _members: tuple[int, ...]
    """Nodes this clique couples, ascending and distinct, in its reporter's frame."""

    selectors: tuple[LinearEmbedding[Any, ExponentialFamily], ...]
    """One selector per member, in clique order: sub-statistic into full statistic."""

    def __post_init__(self) -> None:
        self.validate_clique()

    # Overrides

    @property
    @override
    def members(self) -> tuple[int, ...]:
        return self._members

    @override
    def shifted(self, offset: int) -> Self:
        return replace(self, _members=tuple(i + offset for i in self.members))

    @property
    @override
    def sub_dims(self) -> tuple[int, ...]:
        """One factor per member: the dimension of the sub-statistic it selects.

        Reading the factors off the selectors is what makes arity a single fact rather
        than several that can disagree --- ``arity == len(selectors)`` holds by
        construction, and :meth:`~goal.geometry.manifold.clique.LinearClique.validate_clique`
        checks it against the members.
        """
        return tuple(sel.sub_man.dim for sel in self.selectors)

    # Methods

    @classmethod
    def from_map(
        cls, int_map: LinearMap[Any, Any], members: tuple[int, ...]
    ) -> EFClique:
        """The clique one interaction already is.

        An interaction states which coordinates it addresses through its two embeddings,
        and those *are* the clique's selectors: the codomain embedding is the first, and
        the domain embedding supplies the rest --- one selector if it addresses a single
        node, several if it addresses a joint form over a group of them. That second case
        is what makes a three-way interaction arity three: its domain is the joint
        statistic of two latent nodes, held by the level above as one form, so it
        contributes one factor per node rather than one for the pair.

        ``members`` has to be passed in: arity is readable from the interaction, but which
        nodes it couples is not --- the pieces of one interaction share a domain and a
        codomain, so only the model knows.

        Raises:
            TypeError: if the interaction carries no embeddings, so there is nothing to
                read selectors off.
        """
        if not isinstance(int_map, EmbeddedMap):
            msg = (
                f"{type(int_map).__name__} carries no embeddings to read selectors off"
            )
            raise TypeError(msg)
        dom = int_map.dom_emb
        tail = dom.selectors if isinstance(dom, CliqueEmbedding) else (dom,)
        return cls(members, selectors=(int_map.cod_emb, *tail))

    @property
    def node_mans(self) -> tuple[ExponentialFamily, ...]:
        """The member nodes' manifolds, in clique order."""
        return tuple(sel.amb_man for sel in self.selectors)

    @override
    def tensor(self, *node_stats: Array) -> Array:
        """Pair one full statistic per member into this clique's parameters.

        Each argument is a member's *full* sufficient statistic; the selectors restrict
        them before the outer product is taken.

        **What the arguments must be.** Each is that member's statistic *on its own*, so
        this builds the parameters as a product of marginals. That is exact when every member
        observed --- one data point, all statistics deterministic --- which is the
        is observed. It is *not* how they look when two or more members are latent: there
        the parameters are $\\mathbb E[\\bigotimes_i \\mathbf s_i]$
        jointly, and expectation does not pass through a tensor product. For that case the
        latent members' joint form comes from the level above and the selectors are
        contracted into it --- see :meth:`partial_contract`.
        """
        selected = [
            sel.project(stats)
            for sel, stats in zip(self.selectors, node_stats, strict=True)
        ]
        return super().tensor(*selected)

    @override
    def contract(self, params: Array, keep: int, *node_stats: Array) -> Array:
        """Contract every member but ``keep``, returning parameters on that node.

        ``node_stats`` supplies one full statistic per contracted member, in ascending
        member order. The result is embedded back into the kept node's full parameter
        space, so it can be added to that node's bias directly. At arity 2 this is the
        likelihood map (``keep`` the codomain) or the posterior map (``keep`` the domain).

        Keeping one member is :meth:`partial_contract` with a one-member ``keep``, so this
        goes through that rather than through the inherited contraction --- the selectors
        must be applied once, not once per level of delegation.
        """
        contracted = self.partial_contract(params, (keep,), *node_stats)
        return self.selectors[keep].embed(contracted)

    @override
    def partial_contract(
        self, params: Array, keep: tuple[int, ...], *node_stats: Array
    ) -> Array:
        """Contract several members at once, leaving a joint tensor over ``keep``.

        The general form of :meth:`contract`, and the one a level split needs: the members
        on the near side of a cut are contracted individually against their own statistics,
        and the members on the far side are left *joined*, because their expectations do
        not factor. ``node_stats`` supplies one full statistic per contracted member, in
        ascending member order; ``keep`` names the surviving members, also in ascending
        order.

        The result is a flat tensor over the kept members' *selected* dimensions, in clique
        order, with no embedding applied --- placing it is the caller's job, since where it
        goes depends on which manifold holds the joint form for those members.
        """
        dropped = [pos for pos in range(self.arity) if pos not in keep]
        selected = [
            self.selectors[pos].project(stats)
            for pos, stats in zip(dropped, node_stats, strict=True)
        ]
        return super().partial_contract(params, keep, *selected)

    def select_joint(self, keep: tuple[int, ...], joint: Array) -> Array:
        """Restrict a joint form over the ``keep`` members to this clique's sub-statistics.

        ``joint`` is $\\mathbb E[\\bigotimes_{i \\in keep} \\mathbf s_i]$ as the level
        above stores it --- a flat tensor over those members' full statistic dimensions,
        in ascending member order. Each of this clique's selectors is contracted into the
        corresponding axis, giving the sub-tensor this clique actually couples.

        This is the operation :meth:`tensor` cannot do: it never forms a marginal, so it
        stays correct when the kept members are dependent.
        """
        out = joint.reshape(tuple(self.selectors[pos].amb_man.dim for pos in keep))
        for axis, pos in enumerate(keep):
            out = _map_axis(out, axis, self.selectors[pos].project)
        return out.reshape(-1)

    def sufficient_statistic(self, *xs: Array) -> Array:
        """This clique's contribution to the joint sufficient statistic.

        Takes one data point per member and returns its parameters, which is what makes
        this an exponential-family clique rather than a bare multilinear form.
        """
        stats = [
            man.sufficient_statistic(x)
            for man, x in zip(self.node_mans, xs, strict=True)
        ]
        return self.tensor(*stats)


### Clique Embeddings ###


def _map_axis(tensor: Array, axis: int, fn: Any) -> Array:
    """Apply a vector function along one axis of a tensor, replacing that axis."""
    moved = jnp.moveaxis(tensor, axis, 0)
    trailing = moved.shape[1:]
    columns = moved.reshape(moved.shape[0], -1)
    mapped = jax.vmap(fn, in_axes=1, out_axes=1)(columns)
    return jnp.moveaxis(mapped.reshape((-1, *trailing)), 0, axis)


@dataclass(frozen=True)
class CliqueEmbedding[Ambient: LinearCliques](
    EFClique, LinearEmbedding[LinearClique, Ambient]
):
    """An :class:`EFClique` that addresses one clique of *another* manifold's layout.

    Same clique, different ownership: an ``EFClique`` on its own describes parameters a
    model holds, while this one names a clique held by ``_amb_man`` and reads or writes it
    in place. Say which nodes you are coupling and how much of each one's coordinates you
    want, and :meth:`project` finds the form holding those members *jointly* and restricts
    it one axis at a time; :meth:`embed` is the adjoint, scattering back into a zero
    ambient vector.

    Restricting a joint form axis by axis is not the same as composing per-member
    restrictions, and the difference is the point: the joint form is a tensor that need
    not factor across its members, so a coupling to several at once must select from that
    tensor rather than combine separate per-member ones. :meth:`EFClique.select_joint` is
    the same walk over a form supplied by the level above rather than read out of an
    ambient layout --- there the axis sizes come from the selectors, here from
    :attr:`~goal.geometry.exponential_family.clique.LinearCliques.clique_axes`, which is why the two
    stay separate methods over one :func:`_map_axis`.

    The ambient manifold must have a clique on exactly these members --- there has to be a
    single form holding them jointly.
    :meth:`~goal.geometry.exponential_family.clique.LinearCliques.clique_index` raises when it does
    not, which is the structural condition higher arity needs.
    """

    # Fields

    _amb_man: Ambient
    """The clique manifold holding the form. Last, so that the clique's own fields lead."""

    # Overrides

    @property
    @override
    def amb_man(self) -> Ambient:
        return self._amb_man

    @property
    @override
    def sub_man(self) -> LinearClique:
        """The selected sub-tensor --- this clique's own multilinear form."""
        return self

    @override
    def project(self, coords: Array) -> Array:
        start, size, axes = self._placement()
        out = coords[start : start + size].reshape(axes)
        for axis, sel in enumerate(self.selectors):
            out = _map_axis(out, axis, sel.project)
        return out.reshape(-1)

    @override
    def embed(self, coords: Array) -> Array:
        start, size, _ = self._placement()
        out = coords.reshape(self.sub_dims)
        for axis, sel in enumerate(self.selectors):
            out = _map_axis(out, axis, sel.embed)
        full = jnp.zeros(self.amb_man.dim)
        return full.at[start : start + size].set(out.reshape(-1))

    # Private

    def _placement(self) -> tuple[int, int, tuple[int, ...]]:
        """Start, size, and factor dimensions of the clique this addresses."""
        index = self.amb_man.clique_index(self.members)
        return (
            self.amb_man.clique_offsets()[index],
            self.amb_man.clique_dims[index],
            self.amb_man.clique_axes[index],
        )


### Span Embeddings ###


@dataclass(frozen=True)
class RootEmbedding[
    Sub: LevelCliques[Any, Any, Any],
    Ambient: LevelCliques[Any, Any, Any],
](LinearEmbedding[Sub, Ambient]):
    """Embeds one clique manifold into another over the same graph, transforming only the root span.

    Use this when two models share a graph but parameterize the root nodes differently ---
    one restricting its root span to a submanifold of the other's. The cross and deep spans
    pass through unchanged, so a difference deeper in
    the graph is expressed by nesting: the deep manifolds are themselves clique manifolds
    related by their own ``RootEmbedding``.

    Mathematically, for span coordinates $(r, c, d)$: ``embed`` maps $(r, c, d) \\mapsto
    (\\phi(r), c, d)$ and ``project`` maps $(r, c, d) \\mapsto (\\pi(r), c, d)$, where
    $\\phi$ and $\\pi$ are the root embedding's own maps.
    """

    # Fields

    root_emb: LinearEmbedding[Any, Any]
    """Embedding of the restricted root manifold into the full one."""

    _sub_man: Sub
    """The clique manifold with the restricted root span."""

    _amb_man: Ambient
    """The clique manifold with the full root span."""

    def __post_init__(self) -> None:
        if not self.sub_man.same_graph(self.amb_man):
            msg = "sub and ambient must share a clique set: "
            sub_roots = sorted(self.sub_man.root_nodes)
            amb_roots = sorted(self.amb_man.root_nodes)
            msg += f"{self.sub_man.cliques} rooted at {sub_roots}"
            raise ValueError(f"{msg} vs {self.amb_man.cliques} rooted at {amb_roots}")
        for name, sub_span, amb_span in (
            ("cross", self.sub_man.cross_man, self.amb_man.cross_man),
            ("deep", self.sub_man.deep_man, self.amb_man.deep_man),
        ):
            if sub_span.dim != amb_span.dim:
                msg = f"{name} spans differ: {sub_span.dim} vs {amb_span.dim}"
                raise ValueError(f"{msg}; only the root span may be transformed")
        for name, span, emb_man in (
            ("sub", self.sub_man.root_man, self.root_emb.sub_man),
            ("ambient", self.amb_man.root_man, self.root_emb.amb_man),
        ):
            if span.dim != emb_man.dim:
                msg = f"{name} root span has dimension {span.dim}"
                raise ValueError(f"{msg}, but root_emb expects {emb_man.dim}")

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
