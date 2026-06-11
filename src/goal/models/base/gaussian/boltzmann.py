"""Boltzmann machines: exponential families over binary vectors with pairwise couplings.

The family is organized by coupling sparsity, mirroring the ``Normal`` covariance
representations: :class:`FullBoltzmann` (dense couplings, enumeration + Gibbs),
:class:`DiagonalBoltzmann` (independent units, closed form),
:class:`ChordalBoltzmann` (bounded-treewidth couplings, exact junction-tree
inference), and :class:`ChainBoltzmann` (path-shaped clique trees, exact
inference at logarithmic depth). The abstract :class:`Boltzmann` base captures
everything that follows from binariness alone; subclasses supply the coupling
manifold and its parameter layout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ....geometry import (
    Analytic,
    Differentiable,
    SquareMap,
    Symmetric,
)
from ..categorical import Bernoullis
from .coupling import (
    ChainTree,
    JunctionTree,
    chain_log_partition,
    chain_sample,
    dense_gibbs_step,
    dense_log_partition,
    dense_sample,
    dense_states,
    dense_unit_conditional_prob,
    jt_log_partition,
    jt_sample,
)
from .generalized import Euclidean, GeneralizedGaussian


class Boltzmann[Shape: Differentiable](
    GeneralizedGaussian[Bernoullis, Shape],
    ABC,
):
    """ABC for Boltzmann machines over binary vectors $x \\in \\{0, 1\\}^n$.

    Mathematically, $x_i^2 = x_i$ for binary variables, which collapses the
    Gaussian-like structure: the bias is absorbed into the diagonal of the
    coupling matrix (so the location component is identically zero in natural
    coordinates), $\\mathbb{E}[x_i^2] = \\mathbb{E}[x_i]$ (so the first moment
    is the diagonal of the second-moment matrix), and the density is

    $$p(x) \\propto \\exp(\\mathbf s_{\\text{shp}}(x) \\cdot \\theta),$$

    entirely in terms of the shape (coupling) manifold. Subclasses fix the
    coupling sparsity through :attr:`shp_man` and its parameter layout through
    :meth:`split_couplings` / :meth:`join_couplings`; the exponential-family
    structure and the GeneralizedGaussian split/join conversions all follow.

    The precision convention matches the energy $x^T \\Theta x$: natural
    parameters map to the precision slot via $\\theta_{ii} \\mapsto -2
    \\theta_{ii}$ on the diagonal and $\\theta_{ij} \\mapsto -\\theta_{ij}$ off
    the diagonal.
    """

    # Contract

    @abstractmethod
    def split_couplings(self, coords: Array) -> tuple[Array, Array]:
        """Split a shape-layout coordinate vector into (diagonal, off-diagonal) components.

        Pure layout knowledge --- applies to natural, mean, or precision
        coordinates alike. The off-diagonal component is empty for diagonal
        couplings.
        """

    @abstractmethod
    def join_couplings(self, diag: Array, off_diag: Array) -> Array:
        """Inverse of :meth:`split_couplings`."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return self.shp_man.dim

    @property
    @override
    def loc_man(self) -> Bernoullis:
        return Bernoullis(self.data_dim)

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        return self.shp_man.sufficient_statistic(x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)

    @override
    def log_partition_function(self, params: Array) -> Array:
        return self.shp_man.log_partition_function(params)

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        return self.shp_man.sample(key, params, n)

    @override
    def split_location_precision(self, params: Array) -> tuple[Array, Array]:
        """Location is identically zero (absorbed); precision applies the $-2$/$-1$ scaling."""
        diag, off_diag = self.split_couplings(params)
        return jnp.zeros(self.data_dim), self.join_couplings(-2.0 * diag, -off_diag)

    @override
    def join_location_precision(self, location: Array, precision: Array) -> Array:
        """Inverse of :meth:`split_location_precision`; the location adds onto the diagonal."""
        diag, off_diag = self.split_couplings(precision)
        return self.join_couplings(-0.5 * diag + location, -off_diag)

    @override
    def split_mean_second_moment(self, means: Array) -> tuple[Array, Array]:
        """The first moment is the diagonal of the second moment (binary $x_i^2 = x_i$)."""
        diag, _ = self.split_couplings(means)
        return diag, means

    @override
    def join_mean_second_moment(self, mean: Array, second_moment: Array) -> Array:
        """The second moment already carries the first on its diagonal."""
        return second_moment


@dataclass(frozen=True)
class DiagonalBoltzmann(
    Boltzmann[Bernoullis],
    Analytic,
):
    """Boltzmann machine with no couplings: $n$ independent binary units.

    Distributionally this is just :class:`Bernoullis` --- with $x^2 = x$ the
    diagonal couplings are ordinary biases. The value of the class is the
    packaging: it presents the product-Bernoulli family in Boltzmann
    coordinates, so it slots into ``Boltzmann[Shape]`` APIs as the
    independence baseline for the coupled variants (and as the ansatz family
    a mean-field approximation of those variants optimizes over). Both
    location and shape manifolds are :class:`Bernoullis` --- for independent
    binary variables the first and second order statistics are completely
    redundant, so the coupling layout has no off-diagonal part. Closed-form
    everywhere (:class:`Analytic`).
    """

    n_neurons: int

    # Overrides

    @property
    @override
    def data_dim(self) -> int:
        return self.n_neurons

    @property
    @override
    def shp_man(self) -> Bernoullis:
        return Bernoullis(self.n_neurons)

    @override
    def negative_entropy(self, means: Array) -> Array:
        return self.shp_man.negative_entropy(means)

    @override
    def split_couplings(self, coords: Array) -> tuple[Array, Array]:
        return coords, coords[:0]

    @override
    def join_couplings(self, diag: Array, off_diag: Array) -> Array:
        return diag


class CouplingMatrix(SquareMap[Euclidean], Differentiable):
    """Exponential family over moment matrices $x \\otimes x$ with Symmetric storage.

    Distribution: $p(x) \\propto \\exp(\\tr(\\Theta^T(x \\otimes x)))$. The
    algorithms live in :mod:`~goal.models.base.gaussian.coupling.dense`: exact
    $\\log Z$ by enumeration, sampling by Gibbs.
    """

    def __init__(self, n_neurons: int):
        super().__init__(Symmetric(), Euclidean(n_neurons))

    # Overrides

    @property
    @override
    def data_dim(self) -> int:
        return self.n_neurons

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """$x \\otimes x$ stored as upper triangular."""
        return self.outer_product(x, x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)

    @override
    def log_partition_function(self, params: Array) -> Array:
        return dense_log_partition(self.n_neurons, params)

    @override
    def sample(
        self,
        key: Array,
        params: Array,
        n: int = 1,
        n_burnin: int = 1000,
        n_thin: int = 10,
    ) -> Array:
        """Generate samples using Gibbs sampling."""
        return dense_sample(self.n_neurons, params, key, n, n_burnin, n_thin)

    @override
    def gibbs_step(self, key: Array, params: Array, state: Array) -> Array:
        """Gibbs step updating all units via conditional distributions."""
        return dense_gibbs_step(self.n_neurons, params, state, key)

    # Methods

    @property
    def n_neurons(self) -> int:
        """Number of neurons (dimensions)."""
        return self.matrix_shape[0]

    @property
    def states(self) -> Array:
        """All possible binary states."""
        return dense_states(self.n_neurons)

    def unit_conditional_prob(
        self, state: Array, unit_idx: Array | int, params: Array
    ) -> Array:
        """Compute P(x_unit = 1 | x_other) for a single unit."""
        return dense_unit_conditional_prob(self.n_neurons, params, state, unit_idx)


@dataclass(frozen=True)
class FullBoltzmann(
    Boltzmann[CouplingMatrix],
    Differentiable,
):
    """Boltzmann machine with dense pairwise couplings.

    Exact computations enumerate all $2^n$ states; sampling is Gibbs. The
    coupling layout is the packed upper triangle of :class:`Symmetric` storage.
    """

    # Fields

    n_neurons: int

    # Overrides

    @property
    @override
    def data_dim(self) -> int:
        return self.n_neurons

    @property
    @override
    def shp_man(self) -> CouplingMatrix:
        return CouplingMatrix(self.n_neurons)

    @override
    def sample(
        self,
        key: Array,
        params: Array,
        n: int = 1,
        n_burnin: int = 1000,
        n_thin: int = 10,
    ) -> Array:
        return self.shp_man.sample(key, params, n, n_burnin, n_thin)

    @override
    def gibbs_step(self, key: Array, params: Array, state: Array) -> Array:
        return self.shp_man.gibbs_step(key, params, state)

    @override
    def split_couplings(self, coords: Array) -> tuple[Array, Array]:
        rows, cols = np.triu_indices(self.n_neurons)
        return coords[np.flatnonzero(rows == cols)], coords[np.flatnonzero(rows != cols)]

    @override
    def join_couplings(self, diag: Array, off_diag: Array) -> Array:
        rows, cols = np.triu_indices(self.n_neurons)
        out = jnp.zeros(rows.size, dtype=diag.dtype)
        out = out.at[np.flatnonzero(rows == cols)].set(diag)
        return out.at[np.flatnonzero(rows != cols)].set(off_diag)

    # Methods

    @property
    def states(self) -> Array:
        """All possible binary states."""
        return self.shp_man.states


@dataclass(frozen=True)
class ChordalCouplingMatrix(Differentiable):
    """Exponential family over chordal-pattern moment matrices $x \\otimes x$.

    Analog of ``CouplingMatrix`` parametrised only on the chordal sparsity pattern of
    a ``JunctionTree``: the parameter vector is $[\\theta_{ii}, \\theta_{ij}]$ with
    diagonal entries first ($n$ values), then off-diagonal couplings for chordal
    edges in canonical order ($n_\\text{chordal\\_edges}$ values).

    Distribution: $p(x) \\propto \\exp(\\sum_i \\theta_{ii} x_i + \\sum_{(i,j) \\in E^\\star} \\theta_{ij} x_i x_j)$
    where $E^\\star$ is the chordal completion edge set. Exact $\\log Z$ and exact
    ancestral sampling via junction-tree sum-product in $O(n \\cdot 2^{w+1})$ time.
    """

    # Fields

    junction_tree: JunctionTree

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return self.junction_tree.n_params

    @property
    @override
    def data_dim(self) -> int:
        return self.junction_tree.n_nodes

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Diagonal $x_i$ entries followed by $x_i x_j$ for each chordal edge."""
        edges = self.junction_tree.chordal_edges_arr
        if edges.shape[0] == 0:
            return x
        off_diag = x[edges[:, 0]] * x[edges[:, 1]]
        return jnp.concatenate([x, off_diag])

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)

    @override
    def log_partition_function(self, params: Array) -> Array:
        diag, off_diag = self._split(params)
        return jt_log_partition(self.junction_tree, diag, off_diag)

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Draw $n$ exact i.i.d. samples via junction-tree ancestral sampling.

        The collect pass is key-independent, so ``vmap`` batching semantics
        compute it once across all $n$ draws; only the walk-down repeats.
        """
        diag, off_diag = self._split(params)
        keys = jax.random.split(key, n)

        def sample_one(k: Array) -> Array:
            return jt_sample(self.junction_tree, diag, off_diag, k)

        return jax.vmap(sample_one)(keys)

    # Methods

    @property
    def n_neurons(self) -> int:
        return self.junction_tree.n_nodes

    def _split(self, params: Array) -> tuple[Array, Array]:
        n = self.junction_tree.n_nodes
        return params[:n], params[n:]

    def to_matrix(self, params: Array) -> Array:
        """Scatter chordal-pattern parameters into a dense symmetric $n \\times n$ matrix."""
        n = self.junction_tree.n_nodes
        diag, off_diag = self._split(params)
        mat = jnp.zeros((n, n))
        mat = mat.at[jnp.arange(n), jnp.arange(n)].set(diag)
        edges = self.junction_tree.chordal_edges_arr
        if edges.shape[0] > 0:
            i = jnp.asarray(edges[:, 0])
            j = jnp.asarray(edges[:, 1])
            mat = mat.at[i, j].set(off_diag)
            mat = mat.at[j, i].set(off_diag)
        return mat

    def from_matrix(self, matrix: Array) -> Array:
        """Gather chordal-pattern parameters from a dense matrix; off-pattern entries are dropped."""
        diag = jnp.diag(matrix)
        edges = self.junction_tree.chordal_edges_arr
        if edges.shape[0] == 0:
            return diag
        i = jnp.asarray(edges[:, 0])
        j = jnp.asarray(edges[:, 1])
        off_diag = 0.5 * (matrix[i, j] + matrix[j, i])
        return jnp.concatenate([diag, off_diag])


@dataclass(frozen=True)
class ChainCouplingMatrix(ChordalCouplingMatrix):
    """Chordal couplings whose clique tree is a path (a :class:`ChainTree`).

    Same parameter layout and distribution as :class:`ChordalCouplingMatrix`;
    the log-partition runs the $O(\\log n)$-depth transfer-matrix kernel
    unconditionally --- the topology guarantee is carried by the
    :class:`ChainTree` field, not checked at evaluation time.
    """

    junction_tree: ChainTree

    @override
    def log_partition_function(self, params: Array) -> Array:
        diag, off_diag = self._split(params)
        return chain_log_partition(self.junction_tree, diag, off_diag)

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Draw $n$ exact i.i.d. samples via parallel backward sampling."""
        diag, off_diag = self._split(params)
        keys = jax.random.split(key, n)

        def sample_one(k: Array) -> Array:
            return chain_sample(self.junction_tree, diag, off_diag, k)

        return jax.vmap(sample_one)(keys)


@dataclass(frozen=True)
class ChordalBoltzmann(
    Boltzmann[ChordalCouplingMatrix],
    Differentiable,
):
    """Boltzmann machine with couplings restricted to a chordal sparsity pattern.

    Exact $\\log Z$ and exact i.i.d. sampling via junction-tree sum-product in
    $O(n \\cdot 2^{w+1})$ time, where $w$ is the treewidth of the chordal
    completion. The coupling layout is $[\\theta_{ii}, \\theta_{ij}]$ ---
    diagonal entries first, then chordal edges in canonical order. Within the
    sparsity constraint this preserves the pairwise expressiveness of
    :class:`FullBoltzmann` with the exactness of :class:`DiagonalBoltzmann` ---
    no Gibbs burn-in, no MC variance.
    """

    # Fields

    junction_tree: JunctionTree

    # Constructors

    @staticmethod
    def from_edges(
        n_neurons: int,
        edges: Sequence[tuple[int, int]],
        max_treewidth: int | None = None,
    ) -> ChordalBoltzmann:
        """Build a ChordalBoltzmann from a graph on ``n_neurons`` nodes with the given edge list.

        The edge list seeds the topology: it is completed to a chordal graph,
        and any fill-in edges added by triangulation become genuine couplings
        of the model --- ``dim`` reflects the completion, not the input list.
        Already-chordal seeds (chains, trees, bands) incur no fill-in.
        """
        return ChordalBoltzmann(JunctionTree.from_edges(n_neurons, edges, max_treewidth))

    # Overrides

    @property
    @override
    def data_dim(self) -> int:
        return self.junction_tree.n_nodes

    @property
    @override
    def shp_man(self) -> ChordalCouplingMatrix:
        return ChordalCouplingMatrix(self.junction_tree)

    @override
    def split_couplings(self, coords: Array) -> tuple[Array, Array]:
        n = self.junction_tree.n_nodes
        return coords[:n], coords[n:]

    @override
    def join_couplings(self, diag: Array, off_diag: Array) -> Array:
        return jnp.concatenate([diag, off_diag])

    # Methods

    @property
    def n_neurons(self) -> int:
        return self.junction_tree.n_nodes


@dataclass(frozen=True)
class ChainBoltzmann(ChordalBoltzmann):
    """Chordal Boltzmann machine whose clique tree is a path.

    Same distribution family and parameter layout as :class:`ChordalBoltzmann`,
    restricted to topologies whose triangulation yields a path-shaped clique
    tree --- chains, band graphs, triangulated cycles (interval graphs of
    bounded pathwidth). The constraint is validated at construction (via
    :class:`ChainTree`), in exchange for which the log-partition and its
    gradients (hence ``to_mean``) run the $O(\\log n)$-depth
    ``associative_scan`` kernel --- orders of magnitude faster on accelerators
    than the sequential collect at large $n$.
    """

    # Fields

    junction_tree: ChainTree

    # Constructors

    @staticmethod
    @override
    def from_edges(
        n_neurons: int,
        edges: Sequence[tuple[int, int]],
        max_treewidth: int | None = None,
    ) -> ChainBoltzmann:
        """Build a ChainBoltzmann from a graph on ``n_neurons`` nodes; raises
        ``ValueError`` if the clique tree is not a path."""
        return ChainBoltzmann(ChainTree.from_edges(n_neurons, edges, max_treewidth))

    # Overrides

    @property
    @override
    def shp_man(self) -> ChainCouplingMatrix:
        return ChainCouplingMatrix(self.junction_tree)
