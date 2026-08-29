"""Maps between manifolds: a generic ``Map`` and the linear/affine specializations.

A ``Map`` is itself a ``Manifold`` whose points are the parameters of a function from a domain manifold to a codomain manifold. Subclasses define the structure of the function: ``LinearMap`` for linear transformations (with matrix-rep specializations like ``MatrixMap`` and ``SquareMap``), ``AffineMap`` for affine maps, and beyond that any parameterized differentiable map (e.g. an MLP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ..algebra.matrix import MatrixRep, Square
from .base import Manifold
from .combinators import Pair

### Maps ###


@dataclass(frozen=True)
class Map[Domain: Manifold, Codomain: Manifold](Manifold, ABC):
    """A parameterized function between manifolds, itself a ``Manifold`` whose points are the function's parameters.

    The contract is just enough to apply the function: a domain manifold, a codomain manifold, and ``__call__`` that takes parameters and a domain point and returns a codomain point. ``dim`` (inherited from ``Manifold``) is the number of parameters.

    Subclasses specialize the structure: ``LinearMap`` for linear transformations with matrix representations, and ``MultilayerPerceptron`` for feedforward networks.
    """

    # Contract

    @property
    @abstractmethod
    def dom_man(self) -> Domain:
        """The domain manifold."""

    @property
    @abstractmethod
    def cod_man(self) -> Codomain:
        """The codomain manifold."""

    @abstractmethod
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        """Apply the map: takes map parameters and a domain point, returns a codomain point."""


### Linear Maps ###


@dataclass(frozen=True)
class LinearMap[Domain: Manifold, Codomain: Manifold](Map[Domain, Codomain], ABC):
    """A linear transformation between manifolds.

    Adds linear-specific operations to ``Map``: transpose and outer product. Concrete
    implementations choose how to store and execute the matrix, and whether the matrix acts
    on the full domain and codomain (:class:`MatrixMap`) or on selected sub-spaces of them
    (:class:`~goal.geometry.manifold.clique.LinearClique`).

    Mathematically, a linear map $L: V \\to W$ satisfies $L(\\alpha x + \\beta y) = \\alpha L(x) + \\beta L(y)$.
    """

    # Contract

    @property
    @abstractmethod
    def trn_man(self) -> LinearMap[Codomain, Domain]:
        """Manifold of transposed linear maps."""

    @abstractmethod
    def transpose(self, f_coords: Array) -> Array:
        """Return parameters of the transposed map."""

    @abstractmethod
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        """Outer product $w \\otimes v$, returned as map parameters."""

    # Methods

    def transpose_apply(self, f_coords: Array, w_coords: Array) -> Array:
        """Apply the transpose: takes map parameters and a codomain point, returns a domain point."""
        f_trn_coords = self.transpose(f_coords)
        return self.trn_man(f_trn_coords, w_coords)


@dataclass(frozen=True)
class MatrixMap[Domain: Manifold, Codomain: Manifold](LinearMap[Domain, Codomain]):
    """A linear map backed by a ``MatrixRep``, acting on the full domain and codomain.

    The matrix has shape $(\\dim(codomain), \\dim(domain))$ and is stored flat according to
    ``rep``, so a map's parameters mean exactly what its domain and codomain say they do.
    A map that should address only *part* of its domain or codomain says so by being a
    :class:`~goal.geometry.manifold.clique.LinearClique`, which carries the sub_embs that
    restrict it.
    """

    # Fields

    rep: MatrixRep
    """The matrix representation strategy for this linear map."""

    _dom_man: Domain
    """The domain manifold."""

    _cod_man: Codomain
    """The codomain manifold."""

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
        return self.rep.num_params(self.matrix_shape)

    @property
    @override
    def trn_man(self) -> MatrixMap[Codomain, Domain]:
        return MatrixMap(self.rep, self.cod_man, self.dom_man)

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        return self.rep.matvec(self.matrix_shape, f_coords, v_coords)

    @override
    def transpose(self, f_coords: Array) -> Array:
        return self.rep.transpose(self.matrix_shape, f_coords)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        return self.rep.outer_product(w_coords, v_coords)

    # Methods

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """Shape $(\\dim(codomain), \\dim(domain))$ of the underlying matrix."""
        return (self.cod_man.dim, self.dom_man.dim)

    def from_matrix(self, matrix: Array) -> Array:
        """Pack a dense 2D matrix into flat parameters."""
        return self.rep.from_matrix(matrix)

    def to_matrix(self, f_coords: Array) -> Array:
        """Unpack flat parameters into a dense 2D matrix."""
        return self.rep.to_matrix(self.matrix_shape, f_coords)

    def get_diagonal(self, f_coords: Array) -> Array:
        """Extract diagonal elements from the matrix."""
        return self.rep.get_diagonal(self.matrix_shape, f_coords)

    def map_diagonal(
        self, f_coords: Array, diagonal_f: Callable[[Array], Array]
    ) -> Array:
        """Apply a function to the diagonal elements, preserving matrix structure."""
        return self.rep.map_diagonal(self.matrix_shape, f_coords, diagonal_f)

    def embed_rep(
        self, f_coords: Array, target_rep: MatrixRep
    ) -> tuple[MatrixMap[Domain, Codomain], Array]:
        """Embed into a more general representation (e.g. Diagonal -> Symmetric)."""
        target_man = MatrixMap(target_rep, self.dom_man, self.cod_man)
        coords = self.rep.embed_params(self.matrix_shape, f_coords, target_rep)
        return target_man, coords

    def project_rep(
        self, f_coords: Array, target_rep: MatrixRep
    ) -> tuple[MatrixMap[Domain, Codomain], Array]:
        """Project to a more constrained representation (e.g. Symmetric -> Diagonal)."""
        target_man = MatrixMap(target_rep, self.dom_man, self.cod_man)
        coords = self.rep.project_params(self.matrix_shape, f_coords, target_rep)
        return target_man, coords


@dataclass(frozen=True)
class SquareMap[M: Manifold](MatrixMap[M, M]):
    """Square ``MatrixMap`` (domain = codomain), exposing inverse, log-determinant, and positive-definiteness checks.

    Domain and codomain are the same manifold, so this is a self-interaction *inside* one
    node rather than a coupling between two --- which is why square maps are never wrapped
    as cliques. ``Covariance`` and ``CouplingMatrix`` are both of this shape: a node's
    second moment, and the one place structured matrix representations live.
    """

    # Fields

    rep: Square

    # Methods

    def __init__(self, rep: MatrixRep, dom_man: M):
        # Check that the representation is square
        if not issubclass(type(rep), Square):
            raise TypeError("SquareMap requires a square matrix representation.")

        super().__init__(rep, dom_man, dom_man)

    def inverse(self, f_coords: Array) -> Array:
        """Parameters of the inverse matrix."""
        return self.rep.inverse(self.matrix_shape, f_coords)

    def logdet(self, f_coords: Array) -> Array:
        """Log determinant."""
        return self.rep.logdet(self.matrix_shape, f_coords)

    def is_positive_definite(self, f_coords: Array) -> Array:
        """Check positive definiteness."""
        return self.rep.is_positive_definite(self.matrix_shape, f_coords)


### Affine Maps ###


@dataclass(frozen=True)
class AffineMap[
    Domain: Manifold,
    Codomain: Manifold,
](
    Pair[Codomain, LinearMap[Domain, Codomain]],
):
    """A linear map plus a bias: $A(x) = L(x) + b$.

    Stored as a ``Pair`` of the bias $b$ (on the codomain) and the linear map $L$. This is the natural parameter space for exponential family likelihoods (bias = observable natural parameters, linear part = interaction).
    """

    # Fields

    map_man: LinearMap[Domain, Codomain]
    """The linear transformation for this affine map."""

    dom_man: Domain
    """The domain of the affine map."""

    # Overrides

    @property
    @override
    def fst_man(self) -> Codomain:
        return self.map_man.cod_man

    @property
    @override
    def snd_man(self) -> LinearMap[Domain, Codomain]:
        return self.map_man

    # Methods

    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        """Apply the affine map: $L(v) + b$."""
        bias, linear = self.split_coords(f_coords)
        return bias + self.snd_man(linear, v_coords)


### Multi-Layer Perceptron ###


@dataclass(frozen=True)
class MultilayerPerceptron[Domain: Manifold, Codomain: Manifold](Map[Domain, Codomain]):
    """A feedforward MLP between manifolds.

    Outputs ``cod_man.dim`` raw values via configurable hidden layers and a fixed activation. For codomains with parameter constraints (e.g. positive-definite precision matrices), constraint-respecting outputs are the responsibility of a wrapping layer or a specialized subclass — this base class emits raw vectors.

    Parameters are flattened concatenations of per-layer ``(W, b)`` pairs, with ``W`` stored in row-major (codomain-first) order.
    """

    # Fields

    _dom_man: Domain
    """The domain manifold."""

    _cod_man: Codomain
    """The codomain manifold."""

    hidden_dims: tuple[int, ...]
    """Widths of hidden layers; empty for a linear MLP (single weight matrix)."""

    activation: Callable[[Array], Array]
    """Activation applied after each hidden layer; not applied to the output."""

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
        return sum(
            in_d * out_d + out_d
            for in_d, out_d in zip(self.layer_dims[:-1], self.layer_dims[1:])
        )

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        h = v_coords
        offset = 0
        layers = list(zip(self.layer_dims[:-1], self.layer_dims[1:]))
        last_idx = len(layers) - 1
        for i, (in_d, out_d) in enumerate(layers):
            w_size = in_d * out_d
            w = f_coords[offset : offset + w_size].reshape(out_d, in_d)
            offset += w_size
            b = f_coords[offset : offset + out_d]
            offset += out_d
            h = w @ h + b
            if i < last_idx:
                h = self.activation(h)
        return h

    # Methods

    @property
    def layer_dims(self) -> tuple[int, ...]:
        """Sequence of layer widths --- domain, hidden layers, codomain."""
        return (self.dom_man.dim, *self.hidden_dims, self.cod_man.dim)

    def glorot_initialize(self, key: Array) -> Array:
        """Glorot uniform initialization: weights from $U(-\\sqrt{6/(d_{in} + d_{out})}, +\\sqrt{6/(d_{in} + d_{out})})$, biases zero."""
        keys = jax.random.split(key, len(self.layer_dims) - 1)
        chunks: list[Array] = []
        for k, in_d, out_d in zip(keys, self.layer_dims[:-1], self.layer_dims[1:]):
            bound = jnp.sqrt(6.0 / (in_d + out_d))
            w = jax.random.uniform(k, (out_d, in_d), minval=-bound, maxval=bound)
            b = jnp.zeros(out_d)
            chunks.append(w.reshape(-1))
            chunks.append(b)
        return jnp.concatenate(chunks)
