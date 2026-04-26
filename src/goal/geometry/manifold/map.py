"""Maps between manifolds: a generic ``Map`` and the linear/affine specializations.

A ``Map`` is itself a ``Manifold`` whose points are the parameters of a function from a domain manifold to a codomain manifold. Subclasses define the structure of the function: ``LinearMap`` for linear transformations (with matrix-rep specializations like ``EmbeddedMap``, ``BlockMap``, ``SquareMap``), ``AffineMap`` for affine maps, and beyond that any parameterized differentiable map (e.g. an MLP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from .base import Manifold
from .combinators import Pair
from .embedding import IdentityEmbedding, LinearComposedEmbedding, LinearEmbedding
from .matrix import MatrixRep, Square

### Maps ###


@dataclass(frozen=True)
class Map[Domain: Manifold, Codomain: Manifold](Manifold, ABC):
    """A parameterized function between manifolds, itself a ``Manifold`` whose points are the function's parameters.

    The contract is just enough to apply the function: a domain manifold, a codomain manifold, and ``__call__`` that takes parameters and a domain point and returns a codomain point. ``dim`` (inherited from ``Manifold``) is the number of parameters.

    Subclasses specialize the structure: ``LinearMap`` for linear transformations with matrix representations, and (in later phases) ``MLPMap`` for feedforward networks. ``StatefulMap`` is a sibling abstraction with a different application signature for stateful maps (RNNs).
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

    Adds linear-specific operations to ``Map``: transpose, outer product, and embedding manipulation. Concrete implementations choose how to store and execute the matrix.

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

    @abstractmethod
    def map_domain_embedding[NewDomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Manifold, Domain]], LinearEmbedding[Manifold, NewDomain]
        ],
    ) -> LinearMap[NewDomain, Codomain]:
        """Transform the domain embedding(s) using the given function.

        This enables operations like tensoring with a new factor by wrapping
        the domain embedding in a more complex structure.
        """

    @abstractmethod
    def map_codomain_embedding[NewCodomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Manifold, Codomain]],
            LinearEmbedding[Manifold, NewCodomain],
        ],
    ) -> LinearMap[Domain, NewCodomain]:
        """Transform the codomain embedding(s) using the given function.

        This enables operations like tensoring with a new factor by wrapping
        the codomain embedding in a more complex structure.
        """

    # Methods

    def transpose_apply(self, f_coords: Array, w_coords: Array) -> Array:
        """Apply the transpose: takes map parameters and a codomain point, returns a domain point."""
        f_trn_coords = self.transpose(f_coords)
        return self.trn_man(f_trn_coords, w_coords)

    def prepend_embedding[NewDomain: Manifold](
        self,
        emb: LinearEmbedding[Domain, NewDomain],
    ) -> LinearMap[NewDomain, Codomain]:
        """Prepend an embedding to the domain."""
        return self.map_domain_embedding(
            lambda dom_emb: LinearComposedEmbedding(dom_emb, emb)
        )

    def append_embedding[NewCodomain: Manifold](
        self,
        emb: LinearEmbedding[Codomain, NewCodomain],
    ) -> LinearMap[Domain, NewCodomain]:
        """Append an embedding to the codomain."""
        return self.map_codomain_embedding(
            lambda cod_emb: LinearComposedEmbedding(cod_emb, emb)
        )


@dataclass(frozen=True)
class EmbeddedMap[Domain: Manifold, Codomain: Manifold](LinearMap[Domain, Codomain]):
    """A linear map backed by a ``MatrixRep`` and domain/codomain embeddings.

    Application follows the pipeline: project input to internal domain, multiply by the matrix, embed result into external codomain. This lets the matrix operate on a lower-dimensional internal space while the map's external interface matches the full manifold dimensions.
    """

    # Fields

    rep: MatrixRep
    """The matrix representation strategy for this linear map."""

    dom_emb: LinearEmbedding[Manifold, Domain]
    """Embedding from internal domain to external domain manifold."""

    cod_emb: LinearEmbedding[Manifold, Codomain]
    """Embedding from internal codomain to external codomain manifold."""

    # Overrides

    @property
    @override
    def dom_man(self) -> Domain:
        return self.dom_emb.amb_man

    @property
    @override
    def cod_man(self) -> Codomain:
        return self.cod_emb.amb_man

    @property
    @override
    def dim(self) -> int:
        return self.rep.num_params(self.matrix_shape)

    @property
    @override
    def trn_man(self) -> EmbeddedMap[Codomain, Domain]:
        """Manifold of transposed linear maps."""
        return EmbeddedMap(self.rep, self.cod_emb, self.dom_emb)

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        internal_v = self.dom_emb.project(v_coords)
        internal_result = self.rep.matvec(self.matrix_shape, f_coords, internal_v)
        return self.cod_emb.embed(internal_result)

    @override
    def transpose(self, f_coords: Array) -> Array:
        return self.rep.transpose(self.matrix_shape, f_coords)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        internal_w = self.cod_emb.project(w_coords)
        internal_v = self.dom_emb.project(v_coords)
        return self.rep.outer_product(internal_w, internal_v)

    @override
    def map_domain_embedding[NewDomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Manifold, Domain]], LinearEmbedding[Manifold, NewDomain]
        ],
    ) -> EmbeddedMap[NewDomain, Codomain]:
        new_dom_emb = f(self.dom_emb)
        return EmbeddedMap(self.rep, new_dom_emb, self.cod_emb)

    @override
    def map_codomain_embedding[NewCodomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Manifold, Codomain]],
            LinearEmbedding[Manifold, NewCodomain],
        ],
    ) -> EmbeddedMap[Domain, NewCodomain]:
        new_cod_emb = f(self.cod_emb)
        return EmbeddedMap(self.rep, self.dom_emb, new_cod_emb)

    @override
    def prepend_embedding[NewDomain: Manifold](
        self,
        emb: LinearEmbedding[Domain, NewDomain],
    ) -> EmbeddedMap[NewDomain, Codomain]:
        """Prepend an embedding to the domain."""
        return self.map_domain_embedding(
            lambda dom_emb: LinearComposedEmbedding(dom_emb, emb)
        )

    @override
    def append_embedding[NewCodomain: Manifold](
        self,
        emb: LinearEmbedding[Codomain, NewCodomain],
    ) -> EmbeddedMap[Domain, NewCodomain]:
        """Append an embedding to the codomain."""
        return self.map_codomain_embedding(
            lambda cod_emb: LinearComposedEmbedding(cod_emb, emb)
        )

    # Methods

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """Shape of the matrix operating on internal dimensions.

        Returns the shape $(\\dim(internal\\_codomain), \\dim(internal\\_domain))$
        of the underlying matrix in the internal coordinate system.
        """
        return (self.cod_emb.sub_man.dim, self.dom_emb.sub_man.dim)

    def from_matrix(self, matrix: Array) -> Array:
        """Pack a dense 2D matrix (in internal dimensions) into flat parameters."""
        return self.rep.from_matrix(matrix)

    def to_matrix(self, f_coords: Array) -> Array:
        """Unpack flat parameters into a dense 2D matrix (in internal dimensions)."""
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
    ) -> tuple[EmbeddedMap[Domain, Codomain], Array]:
        """Embed into a more general representation (e.g. Diagonal -> Symmetric)."""
        target_man = EmbeddedMap(target_rep, self.dom_emb, self.cod_emb)
        coords = self.rep.embed_params(self.matrix_shape, f_coords, target_rep)
        return target_man, coords

    def project_rep(
        self, f_coords: Array, target_rep: MatrixRep
    ) -> tuple[EmbeddedMap[Domain, Codomain], Array]:
        """Project to a more constrained representation (e.g. Symmetric -> Diagonal)."""
        target_man = EmbeddedMap(target_rep, self.dom_emb, self.cod_emb)
        coords = self.rep.project_params(self.matrix_shape, f_coords, target_rep)
        return target_man, coords


@dataclass(frozen=True)
class BlockMap[Domain: Manifold, Codomain: Manifold](LinearMap[Domain, Codomain]):
    """Sum of independent linear maps (blocks), each potentially with a different ``MatrixRep``.

    Used when different parts of a transformation have different structure --- e.g. a harmonium interaction matrix with one dense block and one diagonal block. Parameters are the concatenation of each block's parameters.
    """

    # Fields

    blocks: list[LinearMap[Domain, Codomain]]
    """The embedded linear maps that compose this block map."""

    # Overrides

    @property
    @override
    def dom_man(self) -> Domain:
        return self.blocks[0].dom_man

    @property
    @override
    def cod_man(self) -> Codomain:
        return self.blocks[0].cod_man

    @property
    @override
    def dim(self) -> int:
        return sum(block.dim for block in self.blocks)

    @property
    @override
    def trn_man(self) -> BlockMap[Codomain, Domain]:
        """Manifold of transposed linear maps."""
        transposed_blocks = [block.trn_man for block in self.blocks]
        return BlockMap(transposed_blocks)

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        result = self.cod_man.zeros()
        for block, block_coords in zip(self.blocks, self.coord_blocks(f_coords)):
            result = result + block(block_coords, v_coords)
        return result

    @override
    def transpose(self, f_coords: Array) -> Array:
        transposed_coords = [
            block.transpose(block_coords)
            for block, block_coords in zip(self.blocks, self.coord_blocks(f_coords))
        ]
        return jnp.concatenate(transposed_coords)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        outer_params = [
            block.outer_product(w_coords, v_coords) for block in self.blocks
        ]
        return jnp.concatenate(outer_params)

    @override
    def map_domain_embedding[NewDomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Manifold, Domain]], LinearEmbedding[Manifold, NewDomain]
        ],
    ) -> BlockMap[NewDomain, Codomain]:
        return BlockMap([block.map_domain_embedding(f) for block in self.blocks])

    @override
    def map_codomain_embedding[NewCodomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Manifold, Codomain]],
            LinearEmbedding[Manifold, NewCodomain],
        ],
    ) -> BlockMap[Domain, NewCodomain]:
        return BlockMap([block.map_codomain_embedding(f) for block in self.blocks])

    # Methods

    def coord_blocks(self, coords: Array) -> list[Array]:
        """Split flat parameters into per-block slices."""
        sections = []
        offset = 0
        for block in self.blocks:
            dim = block.dim
            sections.append(coords[offset : offset + dim])
            offset += dim
        return sections


class AmbientMap[Domain: Manifold, Codomain: Manifold](EmbeddedMap[Domain, Codomain]):
    """Convenience wrapper: an ``EmbeddedMap`` with identity embeddings on both sides.

    Use this when the map operates on the full domain and codomain without restriction.
    """

    def __init__(self, rep: MatrixRep, dom_man: Domain, cod_man: Codomain):
        super().__init__(rep, IdentityEmbedding(dom_man), IdentityEmbedding(cod_man))


@dataclass(frozen=True)
class SquareMap[M: Manifold](AmbientMap[M, M]):
    """Square ``AmbientMap`` (domain = codomain), exposing inverse, log-determinant, and positive-definiteness checks."""

    # Fields

    rep: Square

    def __init__(self, rep: MatrixRep, dom_man: M):
        # Check that the representation is square
        if not issubclass(type(rep), Square):
            raise TypeError("SquareMap requires a square matrix representation.")

        super().__init__(rep, dom_man, dom_man)

    # Methods

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


### Stateful Maps ###


@dataclass(frozen=True)
class StatefulMap[
    Domain: Manifold,
    Codomain: Manifold,
    State: Manifold,
](Manifold, ABC):
    """A parameterized map that carries a hidden state across applications.

    Sibling to ``Map`` with a different application signature: each call takes a state in addition to the input and produces a new state along with the output. Used for recurrent dynamics (RNNs, GRUs, LSTMs) where the map's output depends on a running summary of past inputs.

    Like ``Map``, ``StatefulMap`` is itself a ``Manifold`` whose points are the function's parameters; the state is a separate array that flows between calls.
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

    @property
    @abstractmethod
    def state_man(self) -> State:
        """The state manifold."""

    @abstractmethod
    def init_state(self, key: Array) -> Array:
        """Initialize a state array."""

    @abstractmethod
    def __call__(
        self, f_coords: Array, v_coords: Array, state: Array
    ) -> tuple[Array, Array]:
        """Apply the map: takes parameters, a domain point, and a state; returns (codomain point, new state)."""


### Multi-Layer Perceptron ###


@dataclass(frozen=True)
class MLPMap[Domain: Manifold, Codomain: Manifold](Map[Domain, Codomain]):
    """A feedforward MLP between manifolds.

    Outputs ``cod_man.dim`` raw values via configurable hidden layers and a fixed activation. For codomains with parameter constraints (e.g. positive-definite precision matrices), constraint-respecting outputs are the responsibility of a wrapping layer or a specialized subclass — this base class emits raw vectors.

    Parameters are flattened concatenations of per-layer ``(W, b)`` pairs, with ``W`` stored in row-major (codomain-first) order.
    """

    # Fields

    _dom_man: Domain
    """The domain manifold."""

    _cod_man: Codomain
    """The codomain manifold."""

    hidden_dims: tuple[int, ...] = ()
    """Widths of hidden layers; empty for a linear MLP (single weight matrix)."""

    activation: Callable[[Array], Array] = jax.nn.relu
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
        """Sequence of layer widths: domain, hidden layers, codomain."""
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
