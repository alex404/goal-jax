"""Cliques over exponential families: one clique, one inner product.

Where a :class:`~goal.geometry.manifold.graphical.LinearClique` is a multilinear form on a graph
position, an :class:`EFClique` says what that form *pairs against*: the tensor product of
its member nodes' sufficient statistics, each restricted to the sub-statistic this clique
actually couples. The selectors are the specialization --- they are what turn abstract
factor dimensions into named sub-statistics of real families.

Mathematically, a clique $c$ contributes exactly one term to the log-density,

.. math::
    \\langle \\theta_c, \\bigotimes_{i \\in c} \\pi_i(\\mathbf s_i(x_i)) \\rangle,

with $\\pi_i$ the selector for member $i$. Arity is the only thing that varies: at $|c| = 1$
this is a bias, at $|c| = 2$ a matrix contraction, and beyond that a higher-order tensor
contraction. A bias and a coupling differ in arity, not in kind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.embedding import LinearEmbedding
from ..manifold.graphical import LinearClique
from .base import ExponentialFamily


@dataclass(frozen=True)
class EFClique(LinearClique):
    """One clique over a tuple of exponential-family nodes.

    Each member contributes a **selector** --- a
    :class:`~goal.geometry.manifold.embedding.LinearEmbedding` from the sub-statistic this
    clique couples into that node's full statistic. This is the "one embedding per clique"
    arrangement: one selector per member, most of them identities, and nothing else
    deciding which coordinates the block addresses.

    At arity 2 with selectors ``(cod_emb, dom_emb)`` this reproduces
    :class:`~goal.geometry.manifold.map.EmbeddedMap` exactly, block for block ---
    :meth:`tensor` is its ``outer_product`` and :meth:`contract` is its application and
    transposed application. The gain is that arity is no longer capped at 2.
    """

    # Fields

    axes: tuple[int, ...] = field(init=False, default=())
    """Derived, never passed: one factor per member, the dimension of the sub-statistic it
    selects. Filling it from the selectors is what makes arity a single fact rather than
    three that can disagree --- ``len(members) == len(selectors) == len(axes)``, the last
    step checked by :class:`~goal.geometry.manifold.graphical.LinearClique` itself."""

    selectors: tuple[LinearEmbedding[Any, ExponentialFamily], ...] = ()
    """One selector per member, in the same order: sub-statistic into full statistic."""

    # Overrides

    @override
    def __post_init__(self) -> None:
        object.__setattr__(
            self, "axes", tuple(sel.sub_man.dim for sel in self.selectors)
        )
        super().__post_init__()

    # Methods

    @property
    def node_mans(self) -> tuple[ExponentialFamily, ...]:
        """The member nodes' manifolds, in clique order."""
        return tuple(sel.amb_man for sel in self.selectors)

    def tensor(self, *node_stats: Array) -> Array:
        """Pair one full statistic per member into this clique's parameter block.

        Each argument is a member's *full* sufficient statistic; the selectors restrict
        them before the outer product is taken.

        **What the arguments must be.** Each is that member's statistic *on its own*, so
        this builds the block as a product of marginals. That is exact when every member is
        observed --- one data point, all statistics deterministic --- which is the
        :meth:`sufficient_statistic` case. It is *not* how the block looks when two or more
        members are latent: there the block is $\\mathbb E[\\bigotimes_i \\mathbf s_i]$
        jointly, and expectation does not pass through a tensor product. For that case the
        latent members' joint block comes from the level above and the selectors are
        contracted into it --- see :meth:`partial_contract`.
        """
        selected = [
            sel.project(stats)
            for sel, stats in zip(self.selectors, node_stats, strict=True)
        ]
        return self.form.tensor(*selected)

    def contract(self, params: Array, keep: int, *node_stats: Array) -> Array:
        """Contract every member but ``keep``, returning parameters on that node.

        ``node_stats`` supplies one full statistic per contracted member, in ascending
        member order. The result is embedded back into the kept node's full parameter
        space, so it can be added to that node's bias directly. At arity 2 this is the
        likelihood map (``keep`` the codomain) or the posterior map (``keep`` the domain).
        """
        others = [pos for pos in range(len(self.members)) if pos != keep]
        selected = [
            self.selectors[pos].project(stats)
            for pos, stats in zip(others, node_stats, strict=True)
        ]
        return self.selectors[keep].embed(self.form.contract(params, keep, *selected))

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
        goes depends on which manifold holds the joint block for those members.
        """
        dropped = [pos for pos in range(len(self.members)) if pos not in keep]
        selected = [
            self.selectors[pos].project(stats)
            for pos, stats in zip(dropped, node_stats, strict=True)
        ]
        out = self.form.to_tensor(params)
        # Descending order so that contracting one axis does not shift the next.
        for axis, vector in sorted(zip(dropped, selected), key=lambda p: -p[0]):
            out = jnp.tensordot(out, vector, axes=([axis], [0]))
        return out.reshape(-1)

    def select_joint(self, keep: tuple[int, ...], joint_block: Array) -> Array:
        """Restrict a joint block over the ``keep`` members to this clique's sub-statistics.

        ``joint_block`` is $\\mathbb E[\\bigotimes_{i \\in keep} \\mathbf s_i]$ as the level
        above stores it --- a flat tensor over those members' full statistic dimensions,
        in ascending member order. Each of this clique's selectors is contracted into the
        corresponding axis, giving the sub-tensor this clique actually couples.

        This is the operation :meth:`tensor` cannot do: it never forms a marginal, so it
        stays correct when the kept members are dependent.
        """
        out = joint_block.reshape(
            tuple(self.selectors[pos].amb_man.dim for pos in keep)
        )
        for axis, pos in enumerate(keep):
            out = jnp.moveaxis(out, axis, 0)
            trailing = out.shape[1:]
            columns = out.reshape(out.shape[0], -1)
            projected = jax.vmap(self.selectors[pos].project, in_axes=1, out_axes=1)(
                columns
            )
            out = jnp.moveaxis(projected.reshape((-1, *trailing)), 0, axis)
        return out.reshape(-1)

    def sufficient_statistic(self, *xs: Array) -> Array:
        """This clique's contribution to the joint sufficient statistic.

        Takes one data point per member and returns the block, which is what makes this an
        exponential-family clique rather than a bare multilinear form.
        """
        stats = [
            man.sufficient_statistic(x)
            for man, x in zip(self.node_mans, xs, strict=True)
        ]
        return self.tensor(*stats)
