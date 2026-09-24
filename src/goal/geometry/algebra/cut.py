"""Re-viewing a clique layout's coordinates across a one-node cut.

A :class:`~goal.geometry.manifold.clique.LinearCliques` stores its coordinates grouped
around its root set. A ``CliqueCut`` regroups the *same* coordinates around a different
node: everything not touching that node, the cliques that couple to it gathered into one
matrix, and the node's own clique. That regrouping is what lets a hierarchical model be read
as a mixture --- see
:meth:`~goal.models.graphical.mixture.CompleteMixtureOfHarmoniums.to_mixture_coords`.

The cut is a block permutation and nothing else, so one applies to natural and mean
coordinates alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from .util import split_by_dims


@dataclass(frozen=True)
class CliqueCut:
    """A clique layout regrouped as ``(near | cross | far)`` around one node.

    ``far_node`` is split off in place of the root set: the *near* side is every clique
    that does not touch it, the *far* side is its own clique, and the *crossing* cliques
    become one matrix whose rows follow the near side's order and whose column axis is the
    far side. Mathematically this is the re-rooting isomorphism --- the same coordinate
    vector read against a different bipartition of the graph.

    :meth:`project` takes a layout's coordinate vector to the three parts, :meth:`join` puts
    them back, and :attr:`near_idx`, :attr:`cross_idx`, :attr:`far_idx` give the layout
    positions of each group. Construction checks that the regrouping is well posed, so the
    three parts always agree with the layout's dimensions.

    Raises:
        ValueError: if ``far_node`` is not a node of the cover, or is its only node; if it
            carries no clique of its own, so the cut has no columns; if some crossing
            clique's near part is not itself a clique, so it has no row to sit in; or if a
            crossing clique does not couple the whole far side, so the crossing cliques do
            not share one column axis.
    """

    # Fields

    cliques: tuple[tuple[int, ...], ...]
    """Which nodes each clique couples, in the layout's storage order."""

    clique_dims: tuple[int, ...]
    """Clique sizes, parallel to :attr:`cliques`."""

    far_node: int
    """The node being split off."""

    def __post_init__(self) -> None:
        nodes = sorted({i for clique in self.cliques for i in clique})
        if self.far_node not in nodes:
            msg = f"far_node {self.far_node} is not one of the graph's nodes {nodes}"
            raise ValueError(msg)
        if len(nodes) == 1:
            raise ValueError("cutting the only node off leaves no near side")

        near_idx, cross_idx, far_idx = self._group()
        if not far_idx:
            msg = f"far_node {self.far_node} has no clique of its own"
            raise ValueError(f"{msg}: the cut would have no columns")

        n_cols = sum(self.clique_dims[i] for i in far_idx)
        cross_rows = self._rows(near_idx, cross_idx)
        for pos, i in enumerate(cross_idx):
            height = self.clique_dims[near_idx[cross_rows[pos]]]
            if self.clique_dims[i] != height * n_cols:
                msg = f"clique {self.cliques[i]} has dimension {self.clique_dims[i]}"
                msg += f", not {height} x {n_cols}"
                raise ValueError(f"{msg}: it does not couple the whole far side")

    # Properties

    @property
    def near_idx(self) -> tuple[int, ...]:
        """Positions of the cliques that do not touch :attr:`far_node`."""
        return self._group()[0]

    @property
    def cross_idx(self) -> tuple[int, ...]:
        """Positions of the cliques that couple :attr:`far_node` to the near side."""
        return self._group()[1]

    @property
    def far_idx(self) -> tuple[int, ...]:
        """Positions of the cliques lying wholly on :attr:`far_node`."""
        return self._group()[2]

    @property
    def cross_rows(self) -> tuple[int, ...]:
        """For each crossing clique, the position *within* :attr:`near_idx` of its near part."""
        near_idx, cross_idx, _ = self._group()
        return self._rows(near_idx, cross_idx)

    @property
    def n_cols(self) -> int:
        """Width of the crossing matrix: the far side's total dimension."""
        return sum(self.clique_dims[i] for i in self.far_idx)

    # Methods

    def project(self, coords: Array) -> tuple[Array, Array, Array]:
        """Split coordinates into ``(near, cross, far)`` under this cut.

        ``cross`` is the crossing matrix ravelled row-major. A near clique with no
        crossing clique above it still gets a row band, zero filled, so the matrix has one
        row band per near clique and the far side is one shared column axis.
        """
        near_idx, cross_idx, far_idx = self._group()
        n_cols = sum(self.clique_dims[i] for i in far_idx)
        parts = split_by_dims(coords, self.clique_dims)
        near = jnp.concatenate([parts[i] for i in near_idx])
        far = jnp.concatenate([parts[i] for i in far_idx])
        cross_rows = self._rows(near_idx, cross_idx)
        row_source = {row: pos for pos, row in enumerate(cross_rows)}
        rows: list[Array] = []
        for pos, i in enumerate(near_idx):
            height = self.clique_dims[i]
            if pos in row_source:
                part = parts[cross_idx[row_source[pos]]]
                rows.append(part.reshape(height, n_cols))
            else:
                rows.append(jnp.zeros((height, n_cols)))
        return near, jnp.vstack(rows).ravel(), far

    def join(self, near: Array, cross: Array, far: Array) -> Array:
        """Reassemble coordinates from a ``(near, cross, far)`` view.

        The inverse of :meth:`project` on the cliques this cut names. Row bands that
        :meth:`project` zero filled are dropped rather than written back, so the round trip
        is the identity on coordinates but not on an arbitrary crossing matrix.
        """
        near_idx, cross_idx, far_idx = self._group()
        near_dims = tuple(self.clique_dims[i] for i in near_idx)
        near_parts = split_by_dims(near, near_dims)
        far_parts = split_by_dims(far, tuple(self.clique_dims[i] for i in far_idx))
        n_cols = sum(self.clique_dims[i] for i in far_idx)
        cross_matrix = cross.reshape(sum(near_dims), n_cols)

        offsets: list[int] = []
        running = 0
        for size in near_dims:
            offsets.append(running)
            running += size

        parts: list[Array | None] = [None] * len(self.clique_dims)
        for pos, i in enumerate(near_idx):
            parts[i] = near_parts[pos]
        for pos, i in enumerate(far_idx):
            parts[i] = far_parts[pos]
        cross_rows = self._rows(near_idx, cross_idx)
        for pos, i in enumerate(cross_idx):
            row = cross_rows[pos]
            start = offsets[row]
            parts[i] = cross_matrix[start : start + near_dims[row], :].ravel()

        return jnp.concatenate([p for p in parts if p is not None])

    # Private

    def _group(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Positions of the near, crossing, and far cliques, in storage order."""
        near: list[int] = []
        crossing: list[int] = []
        far: list[int] = []
        for pos, clique in enumerate(self.cliques):
            if self.far_node not in clique:
                near.append(pos)
            elif len(clique) == 1:
                far.append(pos)
            else:
                crossing.append(pos)
        return tuple(near), tuple(crossing), tuple(far)

    def _rows(
        self, near_idx: tuple[int, ...], cross_idx: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Which near clique each crossing clique sits above.

        A crossing clique couples ``far_node`` to the rest of its members, so deleting
        ``far_node`` from it leaves a near part; that near part must itself be a clique,
        and the crossing clique becomes the row band above it. Cutting **one** node is what
        makes this assignment injective: two crossing cliques with the same near part $P$
        would both be $P \\cup \\{far\\}$, hence the same clique.
        """
        near_at = {self.cliques[i]: pos for pos, i in enumerate(near_idx)}
        rows: list[int] = []
        for i in cross_idx:
            near_part = tuple(j for j in self.cliques[i] if j != self.far_node)
            if near_part not in near_at:
                msg = f"clique {self.cliques[i]} crosses the cut, near part {near_part}"
                raise ValueError(f"{msg} is not itself a clique, so it has no row band")
            rows.append(near_at[near_part])
        return tuple(rows)
