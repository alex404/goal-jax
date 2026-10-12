"""Boltzmann machines: exponential families over binary vectors with pairwise couplings.

The family is organized by coupling sparsity, mirroring the ``Normal`` covariance
representations: :class:`DiagonalBoltzmann` (independent units, closed form),
:class:`FullBoltzmann` (dense couplings, exact by enumeration of the $2^n$ states) and
:class:`ChordalBoltzmann` (couplings on a chordal graph, exact by junction-tree sum-product,
at logarithmic depth when the junction tree is a :class:`~goal.geometry.ChainTree`). The abstract
:class:`Boltzmann` base captures everything that follows from binariness alone. Each family
runs its own algorithms; its shape manifold (:class:`CouplingMatrix`,
:class:`ChordalCouplingMatrix`) is the layout of its couplings, as the covariance manifold is
for a ``Normal``. A :class:`ChordalBoltzmann` builds the clique log-potentials of its
couplings and runs sum-product on its :class:`~goal.geometry.JunctionTree`, as a ``Normal``
runs the matrix operations of its representation.
"""

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
    ExponentialFamily,
    JunctionTree,
    SquareMap,
    Symmetric,
)
from ....geometry.algebra.junction_tree import binary_states
from ..categorical import Bernoulli, Bernoullis
from .generalized import Euclidean, GeneralizedGaussian


class Boltzmann[Shape: ExponentialFamily](
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
    coupling sparsity through :attr:`shp_man`, its parameter layout through
    :meth:`split_couplings` / :meth:`join_couplings`, and the log-partition function
    and sampling of that sparsity; the GeneralizedGaussian split/join conversions
    follow.

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

    @property
    @abstractmethod
    def coupled_pairs(self) -> tuple[tuple[int, int], ...]:
        """The coupled pairs $(i, j)$, $i < j$, in the order of the off-diagonal component of :meth:`split_couplings`.

        Every Boltzmann machine stores the coupling of a pair as the coefficient of $x_i x_j$,
        so two machines on the same neurons differ only in which pairs they couple.
        """

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
    def log_partition_function(self, params: Array) -> Array:
        return self.shp_man.log_partition_function(params)

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        return self.shp_man.sample(key, params, n)

    @override
    def negative_entropy(self, means: Array) -> Array:
        return self.shp_man.negative_entropy(means)

    @override
    def split_couplings(self, coords: Array) -> tuple[Array, Array]:
        return coords, coords[:0]

    @override
    def join_couplings(self, diag: Array, off_diag: Array) -> Array:
        return diag

    @property
    @override
    def coupled_pairs(self) -> tuple[tuple[int, int], ...]:
        return ()


class CouplingMatrix(SquareMap[Euclidean], ExponentialFamily):
    """Shape component of a :class:`FullBoltzmann`: the couplings as a symmetric matrix over $x \\otimes x$.

    The coordinates are the packed upper triangle of :class:`Symmetric` storage, the layout of
    the shape of a full-covariance ``Normal``.
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

    # Methods

    @property
    def n_neurons(self) -> int:
        """Number of neurons (dimensions)."""
        return self.matrix_shape[0]


@dataclass(frozen=True)
class FullBoltzmann(
    Boltzmann[CouplingMatrix],
    Differentiable,
):
    """Boltzmann machine with dense pairwise couplings.

    Exact computations, sampling included, enumerate all $2^n$ states, so the number of
    neurons must be small. :meth:`gibbs_step` gives the Gibbs sampler for running a chain.
    The coupling layout is the packed upper triangle of :class:`Symmetric` storage.
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
    def log_partition_function(self, params: Array) -> Array:
        """Exact $\\log Z$ by enumeration over all $2^n$ states."""
        return jax.scipy.special.logsumexp(self.energies(params))

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Exact samples, drawn from the probabilities of all $2^n$ states.

        Inverse-CDF sampling: one cumulative sum over the state probabilities and a binary
        search per sample, rather than a categorical draw per sample over all states.
        """
        cdf = jnp.cumsum(jax.nn.softmax(self.energies(params)))
        us = jax.random.uniform(key, (n,)) * cdf[-1]
        return self.states[jnp.searchsorted(cdf, us)]

    @override
    def gibbs_step(self, key: Array, params: Array, state: Array) -> Array:
        """One Gibbs sweep, updating all units in random order from their conditional distributions."""
        perm_key, update_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, self.n_neurons)

        def update_unit(state: Array, unit_idx: Array) -> tuple[Array, None]:
            prob = self.unit_conditional_prob(state, unit_idx, params)
            subkey = jax.random.fold_in(update_key, unit_idx)
            new_val = jax.random.bernoulli(subkey, prob).astype(state.dtype)
            return state.at[unit_idx].set(new_val), None

        final_state, _ = jax.lax.scan(update_unit, state, perm)
        return final_state

    @override
    def split_couplings(self, coords: Array) -> tuple[Array, Array]:
        rows, cols = np.triu_indices(self.n_neurons)
        return coords[np.flatnonzero(rows == cols)], coords[
            np.flatnonzero(rows != cols)
        ]

    @override
    def join_couplings(self, diag: Array, off_diag: Array) -> Array:
        rows, cols = np.triu_indices(self.n_neurons)
        out = jnp.zeros(rows.size, dtype=diag.dtype)
        out = out.at[np.flatnonzero(rows == cols)].set(diag)
        return out.at[np.flatnonzero(rows != cols)].set(off_diag)

    @property
    @override
    def coupled_pairs(self) -> tuple[tuple[int, int], ...]:
        rows, cols = np.triu_indices(self.n_neurons, k=1)
        return tuple(zip(rows.tolist(), cols.tolist(), strict=True))

    # Methods

    @property
    def states(self) -> Array:
        """All $2^n$ binary states, least significant unit first."""
        return jnp.asarray(binary_states(self.n_neurons), dtype=float)

    def energies(self, params: Array) -> Array:
        """The energy $\\theta \\cdot \\mathbf s(x)$ of each state, in the order of :attr:`states`.

        Mathematically, scattering the packed parameters into a symmetric matrix $M$ with
        $M_{ii} = \\theta_{ii}$ and $M_{ij} = M_{ji} = \\theta_{ij} / 2$ gives $x^T M x =
        \\sum_i \\theta_{ii} x_i + \\sum_{i<j} \\theta_{ij} x_i x_j$ on binary states.
        """
        n = self.n_neurons
        rows, cols = np.triu_indices(n)
        m = (
            jnp.zeros((n, n)).at[rows, cols].set(params).at[cols, rows].add(params)
            / 2.0
        )
        states = self.states
        return jnp.einsum("si,ij,sj->s", states, m, states)

    def unit_conditional_prob(
        self, state: Array, unit_idx: Array | int, params: Array
    ) -> Array:
        """$P(x_k = 1 \\mid x_{\\setminus k})$ for a single unit, in $O(n)$.

        The conditional logit is $\\theta_{kk} + \\sum_{j \\neq k} \\theta_{kj} x_j$, read from the
        packed upper-triangular storage without building the full sufficient statistics.
        """
        n = self.n_neurons
        k = jnp.asarray(unit_idx)

        # Flat index for (i, j) with i <= j in upper-triangular storage:
        # i * n - i * (i + 1) / 2 + j.
        def get_triu_idx(i: Array, j: Array) -> Array:
            return i * n - i * (i + 1) // 2 + j

        # For each j, the parameter \theta_kj (symmetric: \theta_kj = \theta_jk):
        # element (k, j) lives in row k when j >= k, in row j otherwise.
        j = jnp.arange(n)
        idx = jnp.where(j >= k, get_triu_idx(k, j), get_triu_idx(j, k))
        theta_k = params[idx]

        # Logit = \theta_kk + sum_{j != k} \theta_kj x_j. The dot term includes
        # the j = k contribution \theta_kk x_k, and \theta_kk (1 - x_k) tops it
        # up to the full \theta_kk --- so the result is independent of the
        # current x_k without masking the state array.
        energy_diff = theta_k[k] * (1 - state[k]) + jnp.dot(theta_k, state)
        return Bernoulli().to_mean(jnp.atleast_1d(energy_diff))[0]


@dataclass(frozen=True)
class ChordalCouplingMatrix(ExponentialFamily):
    """Shape component of a :class:`ChordalBoltzmann`: the couplings on the edges of a chordal graph, given by its junction tree.

    Analog of :class:`CouplingMatrix`: the parameter vector is $[\\theta_{ii}, \\theta_{ij}]$
    with diagonal entries first ($n$ values), then the couplings of the edges of the junction
    tree, in the order of its ``edges``.
    """

    # Fields

    junction_tree: JunctionTree

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return self.junction_tree.n_nodes + len(self.junction_tree.edges)

    @property
    @override
    def data_dim(self) -> int:
        return self.junction_tree.n_nodes

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Diagonal $x_i$ entries followed by $x_i x_j$ for each edge."""
        edges = self._edge_array
        if edges.shape[0] == 0:
            return x
        off_diag = x[edges[:, 0]] * x[edges[:, 1]]
        return jnp.concatenate([x, off_diag])

    @override
    def log_base_measure(self, x: Array) -> Array:
        return jnp.array(0.0)

    # Methods

    @property
    def _edge_array(self) -> np.ndarray:
        """The edges as an ``(n_edges, 2)`` int array, for vectorized indexing."""
        edges = self.junction_tree.edges
        if not edges:
            return np.zeros((0, 2), dtype=np.int32)
        return np.asarray(edges, dtype=np.int32)

    def _split(self, params: Array) -> tuple[Array, Array]:
        n = self.junction_tree.n_nodes
        return params[:n], params[n:]

    def clique_potentials(self, params: Array) -> Array:
        """The clique log-potentials of the couplings, in the layout that :class:`~goal.geometry.JunctionTree` runs sum-product on.

        Each bias and each coupling is counted in exactly one clique, the first that contains
        it, so the potentials sum to the energy $\\theta \\cdot \\mathbf s(x)$.
        """
        jt = self.junction_tree
        diag, off_diag = self._split(params)
        nc, m = jt.n_cliques, jt.max_clique_size
        local = jt.local_positions
        node_owner = np.asarray(jt.node_owner, dtype=np.int64)
        edge_owner = np.asarray(jt.edge_owner, dtype=np.int64)
        i, j = self._edge_array.T
        # Each clique's own biases and couplings, at the positions of their nodes in it.
        biases = (
            jnp.zeros((nc, m))
            .at[node_owner, local[node_owner, np.arange(len(diag))]]
            .set(diag)
        )
        couplings = (
            jnp.zeros((nc, m, m))
            .at[edge_owner, local[edge_owner, i], local[edge_owner, j]]
            .set(off_diag)
        )
        bits = jnp.asarray(binary_states(m), dtype=biases.dtype)
        return biases @ bits.T + jnp.einsum("sp,cpq,sq->cs", bits, couplings, bits)

    def to_matrix(self, params: Array) -> Array:
        """Scatter the parameters into a dense symmetric $n \\times n$ matrix."""
        n = self.junction_tree.n_nodes
        diag, off_diag = self._split(params)
        mat = jnp.zeros((n, n))
        mat = mat.at[jnp.arange(n), jnp.arange(n)].set(diag)
        edges = self._edge_array
        if edges.shape[0] > 0:
            i = jnp.asarray(edges[:, 0])
            j = jnp.asarray(edges[:, 1])
            mat = mat.at[i, j].set(off_diag)
            mat = mat.at[j, i].set(off_diag)
        return mat

    def from_matrix(self, matrix: Array) -> Array:
        """Gather the parameters from a dense matrix; entries off the edges are dropped."""
        diag = jnp.diag(matrix)
        edges = self._edge_array
        if edges.shape[0] == 0:
            return diag
        i = jnp.asarray(edges[:, 0])
        j = jnp.asarray(edges[:, 1])
        off_diag = 0.5 * (matrix[i, j] + matrix[j, i])
        return jnp.concatenate([diag, off_diag])


@dataclass(frozen=True)
class ChordalBoltzmann(
    Boltzmann[ChordalCouplingMatrix],
    Differentiable,
):
    """Boltzmann machine with couplings restricted to a chordal sparsity pattern.

    Exact $\\log Z$ and exact i.i.d. sampling via junction-tree sum-product in
    $O(n \\cdot 2^{w+1})$ time, where $w$ is the treewidth of the chordal
    completion. The coupling layout is $[\\theta_{ii}, \\theta_{ij}]$ ---
    diagonal entries first, then the edges of the junction tree in its order.

    When :attr:`junction_tree` is a :class:`~goal.geometry.ChainTree`, whose cliques form a path
    (chains, band graphs, triangulated cycles), the log-partition function and sampling run the
    $O(\\log n)$-depth ``associative_scan`` kernels instead of the sequential collect: faster on
    accelerators at large $n$, for separators of up to about 6 nodes. Build one with
    ``ChordalBoltzmann(ChainTree.from_edges(n, edges))``.
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
        return ChordalBoltzmann(
            JunctionTree.from_edges(n_neurons, edges, max_treewidth)
        )

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

    @property
    @override
    def coupled_pairs(self) -> tuple[tuple[int, int], ...]:
        return self.junction_tree.edges

    @override
    def log_partition_function(self, params: Array) -> Array:
        return self.junction_tree.log_partition(self.shp_man.clique_potentials(params))

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Draw $n$ exact i.i.d. samples, by ancestral sampling over the junction tree.

        The collect pass is key-independent, so ``vmap`` batching semantics compute it once
        across all $n$ draws; only the walk-down repeats.
        """
        phi = self.shp_man.clique_potentials(params)
        return jax.vmap(lambda k: self.junction_tree.sample(k, phi))(
            jax.random.split(key, n)
        )

    # Methods

    @property
    def n_neurons(self) -> int:
        return self.junction_tree.n_nodes
