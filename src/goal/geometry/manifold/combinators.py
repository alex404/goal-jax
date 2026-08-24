"""Combinators for building complex manifolds from simpler ones.

Provides product manifolds of fixed arity (`Pair`, `Triple`, `Quadruple`), the graph-indexed `CliqueManifold` family (`Node`, `Clique`, `CompositeClique`), homogeneous products (`Replicated`), and the zero-dimensional `Null`. Each combinator stores coordinates as a flat concatenation and provides ``split_coords`` / ``join_coords`` for component access.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ..algebra.clique import CliqueSet
from .base import Manifold


@dataclass(frozen=True)
class Null(Manifold):
    """A zero-dimensional manifold with no coordinates."""

    @property
    @override
    def dim(self) -> int:
        return 0


@dataclass(frozen=True)
class Tuple(Manifold, ABC):
    """Abstract Cartesian product of manifolds, with coordinates stored as a flat concatenation.

    Mathematically, the Cartesian product $\\mathcal M_1 \\times \\cdots \\times \\mathcal M_k$ has $\\dim = \\sum_i \\dim(\\mathcal M_i)$. Subclasses (``Pair``, ``Triple``) fix the arity and provide typed ``split_coords`` / ``join_coords``.
    """

    @abstractmethod
    def split_coords(self, coords: Array) -> tuple[Array, ...]:
        """Split coordinates into tuple components."""

    @abstractmethod
    def join_coords(self, *components: Array) -> Array:
        """Join tuple components into a single array.

        Note: Subclasses implement this with specific numbers of coordinates
        (e.g., Pair takes exactly 2, Triple takes exactly 3).
        """


@dataclass(frozen=True)
class Pair[First: Manifold, Second: Manifold](Tuple, ABC):
    """Binary Cartesian product, with coordinates stored as ``[fst | snd]``."""

    # Contract

    @property
    @abstractmethod
    def fst_man(self) -> First:
        """First component manifold."""

    @property
    @abstractmethod
    def snd_man(self) -> Second:
        """Second component manifold."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return self.fst_man.dim + self.snd_man.dim

    @override
    def split_coords(self, coords: Array) -> tuple[Array, Array]:
        """Split into ``(fst, snd)`` components."""
        first_coords = coords[: self.fst_man.dim]
        second_coords = coords[self.fst_man.dim :]
        return first_coords, second_coords

    @override
    def join_coords(self, fst_coords: Array, snd_coords: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Concatenate component coordinates."""
        return jnp.concatenate([fst_coords, snd_coords])


@dataclass(frozen=True)
class Triple[First: Manifold, Second: Manifold, Third: Manifold](Tuple, ABC):
    """Product of three manifolds, with coordinates stored as ``[fst | snd | trd]``."""

    # Contract

    @property
    @abstractmethod
    def fst_man(self) -> First:
        """First component manifold."""

    @property
    @abstractmethod
    def snd_man(self) -> Second:
        """Second component manifold."""

    @property
    @abstractmethod
    def trd_man(self) -> Third:
        """Third component manifold."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """Total dimension is the sum of component dimensions."""
        return self.fst_man.dim + self.snd_man.dim + self.trd_man.dim

    @override
    def split_coords(self, coords: Array) -> tuple[Array, Array, Array]:
        """Split into ``(fst, snd, trd)`` components."""
        first_dim = self.fst_man.dim
        second_dim = self.snd_man.dim

        fst_coords = coords[:first_dim]
        snd_coords = coords[first_dim : first_dim + second_dim]
        trd_coords = coords[first_dim + second_dim :]

        return (fst_coords, snd_coords, trd_coords)

    @override
    def join_coords(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, fst_coords: Array, snd_coords: Array, trd_coords: Array
    ) -> Array:
        """Concatenate component coordinates."""
        return jnp.concatenate([fst_coords, snd_coords, trd_coords])


@dataclass(frozen=True)
class Quadruple[First: Manifold, Second: Manifold, Third: Manifold, Fourth: Manifold](
    Tuple, ABC
):
    """Product of four manifolds, with coordinates stored as ``[fst | snd | trd | fth]``."""

    # Contract

    @property
    @abstractmethod
    def fst_man(self) -> First:
        """First component manifold."""

    @property
    @abstractmethod
    def snd_man(self) -> Second:
        """Second component manifold."""

    @property
    @abstractmethod
    def trd_man(self) -> Third:
        """Third component manifold."""

    @property
    @abstractmethod
    def fth_man(self) -> Fourth:
        """Fourth component manifold."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """Total dimension is the sum of component dimensions."""
        return self.fst_man.dim + self.snd_man.dim + self.trd_man.dim + self.fth_man.dim

    @override
    def split_coords(self, coords: Array) -> tuple[Array, Array, Array, Array]:
        """Split into ``(fst, snd, trd, fth)`` components."""
        d1 = self.fst_man.dim
        d2 = self.snd_man.dim
        d3 = self.trd_man.dim

        fst_coords = coords[:d1]
        snd_coords = coords[d1 : d1 + d2]
        trd_coords = coords[d1 + d2 : d1 + d2 + d3]
        fth_coords = coords[d1 + d2 + d3 :]

        return (fst_coords, snd_coords, trd_coords, fth_coords)

    @override
    def join_coords(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        fst_coords: Array,
        snd_coords: Array,
        trd_coords: Array,
        fth_coords: Array,
    ) -> Array:
        """Concatenate component coordinates."""
        return jnp.concatenate([fst_coords, snd_coords, trd_coords, fth_coords])


### Clique Manifolds ###


@dataclass(frozen=True)
class CliqueManifold(Manifold, ABC):
    """A manifold whose parameters are laid out over the cliques of a graph.

    The graph is :attr:`clq_set`; the layout order is its
    :attr:`~goal.geometry.algebra.clique.CliqueSet.canonical_cliques`. Two forms implement
    this: :class:`Clique`, whose parameters are a single indivisible block, and
    :class:`CompositeClique`, which decomposes into the three spans of one level ascent and
    whose spans are themselves clique manifolds. The recursion terminates at cliques.
    """

    # Contract

    @property
    @abstractmethod
    def clq_set(self) -> CliqueSet:
        """The graph this manifold is defined on."""

    @property
    @abstractmethod
    def clique_dims(self) -> tuple[int, ...]:
        """Dimension of each clique's parameter block, in layout order.

        Parallel to :attr:`clq_set`'s
        :attr:`~goal.geometry.algebra.clique.CliqueSet.canonical_cliques`: entry $i$ is the
        size of the block belonging to clique $i$. Together with that ordering this is the
        full parameter layout, which is what makes an arbitrary re-view of the graph
        (:meth:`cut`) a block permutation rather than index arithmetic.
        """

    # Methods

    def split_cliques(self, coords: Array) -> tuple[Array, ...]:
        """Split coordinates into per-clique blocks, in layout order."""
        return _split_by_dims(coords, self.clique_dims)

    def join_cliques(self, *blocks: Array) -> Array:
        """Concatenate per-clique blocks back into coordinates."""
        if len(blocks) != len(self.clique_dims):
            raise ValueError(
                f"expected {len(self.clique_dims)} clique blocks, got {len(blocks)}"
            )
        return jnp.concatenate(blocks)

    def cut(self, far_nodes: frozenset[int]) -> CliqueCut:
        """Re-view the layout as bipartite across ``far_nodes`` and the rest.

        Every clique lands on one side or crosses; the crossing ones are laid out as a
        single matrix whose rows follow the *near* side's clique order and whose columns
        are the far side.

        The view exists only when **every crossing clique couples the far side in full**,
        so that they share one column axis. That holds when the far side is coupled through
        its whole sufficient statistic --- a categorical node in a mixture, for instance ---
        and fails when a coupling reaches only a sub-statistic, which is the usual case for
        Gaussian interactions. So this is not a generalization of
        :meth:`~CompositeClique.split_level`: a level split's cross span couples the root
        to the *boundary* rather than to everything above it, and is free to couple a
        sub-statistic of it.

        Raises:
            ValueError: if the declared graph and the parameter layout disagree on the
                number of blocks, or if a crossing clique does not couple the whole far
                side, so that no single matrix view exists.
        """
        cliques = self.clq_set.canonical_cliques
        dims = self.clique_dims
        if len(cliques) != len(dims):
            msg = f"{len(cliques)} cliques but {len(dims)} blocks"
            raise ValueError(f"{msg}: graph and parameter layout disagree")

        near_idx = tuple(i for i, c in enumerate(cliques) if far_nodes.isdisjoint(c))
        far_idx = tuple(i for i, c in enumerate(cliques) if set(c) <= far_nodes)
        cross_idx = tuple(
            i for i in range(len(cliques)) if i not in near_idx and i not in far_idx
        )

        n_cols = sum(dims[i] for i in far_idx)
        near_at = {cliques[i]: pos for pos, i in enumerate(near_idx)}

        cross_rows: list[int] = []
        for i in cross_idx:
            near_part = tuple(sorted(set(cliques[i]) - far_nodes))
            if near_part not in near_at:
                msg = f"clique {cliques[i]} crosses the cut, near part {near_part}"
                raise ValueError(
                    f"{msg} is not itself a clique, so it has no row block"
                )
            pos = near_at[near_part]
            if dims[i] != dims[near_idx[pos]] * n_cols:
                want = f"{dims[near_idx[pos]]} x {n_cols}"
                msg = f"clique {cliques[i]} has dimension {dims[i]}, not {want}"
                raise ValueError(f"{msg}: it does not couple the whole far side")
            cross_rows.append(pos)

        return CliqueCut(
            clique_dims=dims,
            near_idx=near_idx,
            cross_idx=cross_idx,
            far_idx=far_idx,
            cross_rows=tuple(cross_rows),
            n_cols=n_cols,
        )


@dataclass(frozen=True)
class Clique(CliqueManifold, ABC):
    """The parameters of a *single* clique, held as one indivisible block.

    Mathematically, a clique $c$ contributes exactly one inner product to the log-density,

    .. math::
        \\langle \\theta_c, \\bigotimes_{i \\in c} \\mathbf s_i(x_i) \\rangle,

    pairing its parameter block against the tensor product of its members' sufficient
    statistics. Arity is the only thing that varies: at $|c| = 1$ this is a bias
    $\\langle \\theta_i, \\mathbf s_i(x_i) \\rangle$ (:class:`Node`), at $|c| = 2$ a matrix
    contraction $\\langle \\Theta_{ij}, \\mathbf s_i \\otimes \\mathbf s_j \\rangle$, and beyond
    that a higher-order tensor contraction. A bias and a coupling differ in arity, not in
    kind --- which is why both are cliques and neither needs its own type.

    Terminal form of the recursion: :attr:`dim` comes from the manifold itself rather than
    from summing spans, and :attr:`clq_set` is a cover of exactly one clique.

    Deliberately *not* a :class:`Tuple`: a clique has no component split, and giving it one
    would invite callers to split parameters that have no such structure.
    """

    # Overrides

    @property
    @override
    def clique_dims(self) -> tuple[int, ...]:
        """One clique, so one block: the whole manifold."""
        return (self.dim,)

    # Methods

    @staticmethod
    def single_cover(n_members: int) -> CliqueSet:
        """The cover consisting of one clique on ``n_members`` nodes.

        What every :class:`Clique` reports as its :attr:`clq_set`, with arity the only
        thing that varies. Node ``0`` is the root, so an arity-2 cover orients itself the
        way a harmonium orients its interaction.
        """
        return CliqueSet(
            n_nodes=n_members, n_roots=1, cliques=(tuple(range(n_members)),)
        )


@dataclass(frozen=True)
class Node(Clique, ABC):
    """A clique of one node, carrying a bias and no coupling.

    The base case for leaf families. The cover is the same for every such family, so this
    is concrete rather than abstract: how many *variables* a family is over has nothing to
    do with how many graph nodes it occupies. A population of $n$ Poisson neurons is one
    node with an $n$-dimensional statistic; a Boltzmann machine over a 10-variable junction
    tree is one node whose internal graph is its own affair; a Normal is one node despite
    its location/shape pair.
    """

    # Overrides

    @property
    @override
    def clq_set(self) -> CliqueSet:
        return self.single_cover(1)


@dataclass(frozen=True)
class CompositeClique[Root: Manifold, Cross: Manifold, Deep: Manifold](
    CliqueManifold, Tuple, ABC
):
    """Clique manifold laid out as the three spans of one level ascent.

    Coordinates are stored as ``[root | cross | deep]``: the parameters carried by the root
    nodes, the interactions joining the root nodes to the rest of the graph, and everything
    above. The deep span is laid out exactly as the manifold on ``clq_set.ascend_level()``
    lays itself out, so the same split applies again one level up --- recursion over the
    graph is a sequence of these.

    Unlike ``Pair`` and ``Triple``, the components are not arbitrary: the graph says what
    each one is. A span may hold several cliques --- ``deep`` always does past depth two ---
    which is why the three spans are named rather than the individual blocks.
    """

    # Contract

    @property
    @abstractmethod
    def root_man(self) -> Root:
        """Manifold of the parameters carried by the root nodes."""

    @property
    @abstractmethod
    def cross_man(self) -> Cross:
        """Manifold of the interactions joining the root nodes to the rest of the graph."""

    @property
    @abstractmethod
    def deep_man(self) -> Deep:
        """Manifold of everything above the root level."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """Total dimension is the sum of the three span dimensions."""
        return self.root_man.dim + self.cross_man.dim + self.deep_man.dim

    @property
    @override
    def clq_set(self) -> CliqueSet:
        """The graph glued from the spans' graphs.

        The root span supplies the root nodes, the deep span supplies the rest with its
        indices shifted past them, and the cross span supplies the clique joining the two.
        Levels are then recomputed by breadth-first search from the root set, so how the
        deep span roots *itself* is discarded --- which is what makes a fork
        (:class:`~goal.models.graphical.mixture.CompleteMixtureOfHarmoniums`) depth two
        rather than depth three.

        Override this when the cross span holds more than one clique: a
        :class:`~goal.geometry.manifold.map.BlockMap`'s blocks share a domain and codomain
        and so cannot say which nodes each couples, and only the model knows.
        """
        root = span_cover(self.root_man)
        cross = span_cover(self.cross_man)
        deep = span_cover(self.deep_man)
        if root.n_nodes != 1:
            msg = f"root span spans {root.n_nodes} nodes, not 1"
            raise ValueError(f"{msg}: declare clq_set on the model instead")
        if cross.n_nodes != 2 or len(cross.cliques) != 1:
            msg = f"cross span is {len(cross.cliques)} cliques on {cross.n_nodes} nodes"
            raise ValueError(f"{msg}: declare clq_set on the model instead")
        offset = root.n_nodes
        cliques = list(root.cliques)
        cliques += [tuple(i + offset for i in c) for c in deep.cliques]
        cliques.append((0, offset))
        if len(cliques) != len(self.clique_dims):
            msg = f"glued {len(cliques)} cliques for {len(self.clique_dims)} blocks"
            raise ValueError(f"{msg}: declare clq_set on the model instead")
        return CliqueSet(offset + deep.n_nodes, offset, tuple(cliques))

    @property
    @override
    def clique_dims(self) -> tuple[int, ...]:
        """The spans' blocks, concatenated.

        Matches :attr:`clq_set`'s canonical order because that order is itself root
        cliques, then cross cliques, then the deep graph's own order --- the same
        three-way split, one level down.
        """
        return (
            span_blocks(self.root_man)
            + span_blocks(self.cross_man)
            + span_blocks(self.deep_man)
        )

    @override
    def split_coords(self, coords: Array) -> tuple[Array, Array, Array]:
        """Split coordinates into the root, cross, and deep spans."""
        root_dim = self.root_man.dim
        cross_dim = self.cross_man.dim
        return (
            coords[:root_dim],
            coords[root_dim : root_dim + cross_dim],
            coords[root_dim + cross_dim :],
        )

    @override
    def join_coords(self, *components: Array) -> Array:
        """Concatenate the root, cross, and deep spans."""
        if len(components) != 3:
            raise ValueError(f"expected 3 spans, got {len(components)}")
        return jnp.concatenate(components)

    # Methods

    def split_level(self, coords: Array) -> tuple[Array, Array, Array]:
        """Split off one level: the root span, the cross span, and the deep span.

        The domain-facing name for :meth:`split_coords`. A graph of depth one has an empty
        cross and deep span.
        """
        return self.split_coords(coords)

    def join_level(self, root: Array, cross: Array, deep: Array) -> Array:
        """Concatenate the root, cross, and deep spans."""
        return self.join_coords(root, cross, deep)


@dataclass(frozen=True)
class CliqueProduct[Fst: Manifold, Snd: Manifold](CliqueManifold, Pair[Fst, Snd], ABC):
    """Two groups of nodes side by side, with no clique joining them.

    The graph is the disjoint union of the components' graphs, the second component's
    indices shifted past the first, and the parameter layout is the components'
    concatenated. Every node is a root: nothing links the two sides, so a non-root node
    would have no path from the root set at all.

    This is the shape a **multi-root** model's root span takes --- two observables sharing
    one latent, say --- which no other combinator here provides. A
    :class:`CompositeClique` cannot: its root span is a single node by construction.
    """

    # Overrides

    @property
    @override
    def clq_set(self) -> CliqueSet:
        fst = span_cover(self.fst_man)
        snd = span_cover(self.snd_man)
        for cover in (fst, snd):
            if cover.n_roots != cover.n_nodes:
                msg = f"component has {cover.n_roots} roots over {cover.n_nodes} nodes"
                raise ValueError(f"{msg}: a product's components must be all roots")
        offset = fst.n_nodes
        cliques = fst.cliques + tuple(tuple(i + offset for i in c) for c in snd.cliques)
        n_nodes = offset + snd.n_nodes
        return CliqueSet(n_nodes, n_nodes, cliques)

    @property
    @override
    def clique_dims(self) -> tuple[int, ...]:
        return span_blocks(self.fst_man) + span_blocks(self.snd_man)


### Cut Views ###


def span_cover(span: Manifold) -> CliqueSet:
    """The graph one span of a :class:`CompositeClique` contributes.

    A span that declares a clique structure contributes its own graph; anything else is a
    single node, matching what :func:`span_blocks` treats it as.
    """
    if isinstance(span, CliqueManifold):
        return span.clq_set
    return Clique.single_cover(1)


def span_blocks(span: Manifold) -> tuple[int, ...]:
    """Block sizes contributed by one span of a :class:`CompositeClique`.

    A span that declares a clique structure contributes its own blocks; anything else is
    one opaque block. The fallback is what lets a model carry a span whose parameters are
    not a clique decomposition --- a learned variational correction, for instance --- and
    it is why the span bounds stay at ``Manifold`` rather than ``CliqueManifold``.
    """
    if isinstance(span, CliqueManifold):
        return span.clique_dims
    return (span.dim,)


@dataclass(frozen=True)
class CliqueCut:
    """A clique manifold's layout, re-viewed as bipartite across a node cut.

    Produced by :meth:`CliqueManifold.cut`. Holds only integers, so the re-view is a block
    permutation and nothing else --- which is why it applies identically to natural and to
    mean parameters.
    """

    # Fields

    clique_dims: tuple[int, ...]
    """Block sizes of the manifold being re-viewed, in its layout order."""

    near_idx: tuple[int, ...]
    """Layout positions of the cliques lying wholly on the near side."""

    cross_idx: tuple[int, ...]
    """Layout positions of the cliques crossing the cut."""

    far_idx: tuple[int, ...]
    """Layout positions of the cliques lying wholly on the far side."""

    cross_rows: tuple[int, ...]
    """For each crossing clique, the position *within* ``near_idx`` of its near part."""

    n_cols: int
    """Width of the crossing matrix: the far side's total dimension."""

    # Methods

    def project(self, coords: Array) -> tuple[Array, Array, Array]:
        """Split coordinates into ``(near, cross, far)`` under this cut.

        ``cross`` is the crossing matrix ravelled row-major, its rows following the near
        side's clique order.
        """
        blocks = _split_by_dims(coords, self.clique_dims)
        near = jnp.concatenate([blocks[i] for i in self.near_idx])
        far = jnp.concatenate([blocks[i] for i in self.far_idx])
        row_source = {row: pos for pos, row in enumerate(self.cross_rows)}
        rows: list[Array] = []
        for pos, i in enumerate(self.near_idx):
            height = self.clique_dims[i]
            if pos in row_source:
                block = blocks[self.cross_idx[row_source[pos]]]
                rows.append(block.reshape(height, self.n_cols))
            else:
                rows.append(jnp.zeros((height, self.n_cols)))
        return near, jnp.vstack(rows).ravel(), far

    def join(self, near: Array, cross: Array, far: Array) -> Array:
        """Reassemble coordinates from a ``(near, cross, far)`` view."""
        near_dims = tuple(self.clique_dims[i] for i in self.near_idx)
        near_blocks = _split_by_dims(near, near_dims)
        far_blocks = _split_by_dims(
            far, tuple(self.clique_dims[i] for i in self.far_idx)
        )
        cross_matrix = cross.reshape(sum(near_dims), self.n_cols)

        offsets: list[int] = []
        running = 0
        for block_dim in near_dims:
            offsets.append(running)
            running += block_dim

        blocks: list[Array | None] = [None] * len(self.clique_dims)
        for pos, i in enumerate(self.near_idx):
            blocks[i] = near_blocks[pos]
        for pos, i in enumerate(self.far_idx):
            blocks[i] = far_blocks[pos]
        for pos, i in enumerate(self.cross_idx):
            row = self.cross_rows[pos]
            start = offsets[row]
            blocks[i] = cross_matrix[start : start + near_dims[row], :].ravel()

        return jnp.concatenate([b for b in blocks if b is not None])


def _split_by_dims(coords: Array, dims: tuple[int, ...]) -> tuple[Array, ...]:
    """Split a flat array into consecutive slices of the given sizes."""
    out: list[Array] = []
    offset = 0
    for block_dim in dims:
        out.append(coords[offset : offset + block_dim])
        offset += block_dim
    return tuple(out)


@dataclass(frozen=True)
class Replicated[M: Manifold](Manifold, ABC):
    """Homogeneous product of $n$ copies of the same manifold, stored flat as ``[n_reps * rep_man.dim]``.

    Used for collections where every element lives on the same manifold (e.g. mixture components, time series states). The ``map`` method applies a function across copies via ``vmap``.

    Mathematically, the $n$-fold product $\\mathcal M^n = \\mathcal M \\times \\cdots \\times \\mathcal M$ with $\\dim = n \\cdot \\dim(\\mathcal M)$. Unlike ``Tuple``, the homogeneity allows ``vmap``-based operations over the copies.
    """

    # Contract

    @property
    @abstractmethod
    def rep_man(self) -> M:
        """The base manifold being replicated."""

    @property
    @abstractmethod
    def n_reps(self) -> int:
        """Number of copies of the base manifold."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        """Total dimension is product of base dimension and number of copies."""
        return self.rep_man.dim * self.n_reps

    # Methods

    def get_replicate(self, coords: Array, idx: int) -> Array:
        """Extract the ``idx``-th replicate from flat coordinates."""
        start = idx * self.rep_man.dim
        end = start + self.rep_man.dim
        return coords[start:end]

    def to_2d(self, coords: Array) -> Array:
        """Convert flat coordinates to 2D array with shape ``[n_reps, rep_man.dim]``."""
        return coords.reshape([self.n_reps, self.rep_man.dim])

    def to_1d(self, array: Array) -> Array:
        """Convert 2D array back to flat coordinates."""
        return array.ravel()

    def map(
        self,
        f: Callable[[Array], Array],
        coords: Array,
        flatten: bool = False,
    ) -> Array:
        """Map a function across replicates.

        By default, returns stacked 2D results for easier indexing and inspection.
        Use ``flatten=True`` when the result should be flat coordinates on another manifold.

        Args:
            f: Function that takes coordinates for one replicate (shape ``[rep_man.dim]``)
            coords: Flat array of replicated coordinates (shape ``[n_reps * rep_man.dim]``)
            flatten: If True, return flat array ``[n_reps * f_result_dim]``.
                     If False (default), return stacked array ``[n_reps, *f_result_shape]``
                     for easier indexing and inspection.

        Returns:
            Stacked 2D array by default, or flat 1D array if ``flatten=True``
        """
        shaped = coords.reshape([self.n_reps, self.rep_man.dim])
        result = jax.vmap(f)(shaped)
        return result.ravel() if flatten else result
