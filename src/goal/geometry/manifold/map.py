"""Maps between manifolds: a generic ``Map`` and the linear/affine specializations.

A ``Map`` is itself a ``Manifold`` whose points are the parameters of a function from a domain manifold to a codomain manifold. Subclasses define the structure of the function: ``LinearMap`` for linear transformations (with matrix-rep specializations like ``MatrixMap`` and ``SquareMap``), ``AffineMap`` for affine maps, and beyond that any parameterized differentiable map (e.g. an MLP).

``SubspaceMap`` is the arity-$n$ linear leaf: a tensor over *selected subspaces* of the manifolds it is multilinear in --- its **factors** --- read in one matricization, which is what ``MatrixMap`` becomes once a map may address part of a space and have more than two factors. It is address-free --- where its factors sit inside anything larger is the business of whatever holds several of them (:mod:`goal.geometry.manifold.clique` places them over the nodes of a graph, :mod:`goal.geometry.manifold.interaction` sums them into a map between whole manifolds).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import Any, override

import jax
import jax.numpy as jnp
from jax import Array

from ..algebra.matrix import MatrixRep, Rectangular, Square
from .base import Manifold
from .combinators import Pair, Product
from .embedding import IdentityEmbedding, LinearEmbedding

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
    (:class:`~goal.geometry.manifold.map.SubspaceMap`).

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
    :class:`SubspaceMap`, which carries the per-factor embeddings that restrict it.
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


### Subspace Maps ###


@dataclass(frozen=True)
class SubspaceMap(LinearMap[Manifold, Manifold]):
    """A linear map acting on a chosen subspace of each manifold it couples.

    The manifolds are its **factors**, one per
    :class:`~goal.geometry.manifold.embedding.LinearEmbedding`, which names the subspace
    this map uses inside that factor: :attr:`cod_embs` are the codomain factors,
    :attr:`dom_embs` the contracted ones, and :attr:`trn_man` is the two swapped. Applying
    the map uses *both* directions of every embedding --- it projects the incoming point and
    embeds the outgoing one --- and they are not inverses, since ``embed`` takes natural
    parameters and ``project`` takes mean parameters. Arity is a degree, not a kind: one
    factor against no domain factors is a bias, one a side a matrix, beyond that a
    higher-order tensor.

    Mathematically, for codomain subspaces of dimensions $(d_1, \\ldots, d_j)$ and domain
    subspaces $(e_1, \\ldots, e_k)$ the parameters are a tensor $\\Theta \\in \\mathbb
    R^{d_1 \\times \\cdots \\times d_j \\times e_1 \\times \\cdots \\times e_k}$, stored as
    the *matricization* that groups the codomain axes as rows and the domain axes as
    columns. One factor a side is the ordinary matrix shape; with no domain factors the
    column count is the empty product $1$, so the map is a column, which is exactly a bias.
    """

    # Fields

    rep: MatrixRep
    """The matrix representation strategy for this map's parameters."""

    cod_embs: tuple[LinearEmbedding[Any, Any], ...]
    """One embedding per codomain factor: the subspace the output lands in, inside that factor."""

    dom_embs: tuple[LinearEmbedding[Any, Any], ...]
    """One embedding per contracted factor, likewise."""

    # Overrides

    @property
    @override
    def dom_man(self) -> Product:
        """The input group's joint ambient space."""
        return Product(tuple(emb.amb_man for emb in self.dom_embs))

    @property
    @override
    def cod_man(self) -> Product:
        """The output group's joint ambient space."""
        return Product(tuple(emb.amb_man for emb in self.cod_embs))

    @property
    @override
    def dim(self) -> int:
        """What the representation stores, which for a structured ``rep`` is less than the tensor holds."""
        return self.rep.num_params(self.matrix_shape)

    @property
    @override
    def trn_man(self) -> SubspaceMap:
        """The same tensor read the other way: the two groups swapped.

        Its parameters are :meth:`transpose` of this map's. A bias's transpose is a
        functional --- one row over the factor's coordinates.
        """
        return SubspaceMap(self.rep, self.dom_embs, self.cod_embs)

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        """Contract the domain factors' joint ambient coordinates; return the codomain factors'."""
        selected = self.project_dom(v_coords)
        return self.embed_cod(self.rep.matvec(self.matrix_shape, f_coords, selected))

    @override
    def transpose(self, f_coords: Array) -> Array:
        """Reorder parameters into the matricization :attr:`trn_man` expects."""
        return self.rep.transpose(self.matrix_shape, f_coords)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        """Parameters of the outer product of an output joint with an input joint.

        Each argument is a joint over its group's *ambient* dimensions, flat, in factor
        order. It never forms a marginal, so it stays correct when the factors within a
        group are dependent --- which a product of per-factor vectors would not be.
        """
        return self.rep.outer_product(
            self.project_cod(w_coords), self.project_dom(v_coords)
        )

    # Methods

    @property
    def factor_embs(self) -> tuple[LinearEmbedding[Any, Any], ...]:
        """All embeddings in factor order: the codomain factors, then the contracted ones."""
        return self.cod_embs + self.dom_embs

    @property
    def amb_mans(self) -> tuple[Manifold, ...]:
        """The manifold each factor is, in factor order."""
        return tuple(emb.amb_man for emb in self.factor_embs)

    @property
    def sub_dims(self) -> tuple[int, ...]:
        """Dimension of each factor's selected subspace --- the shape of the parameter tensor."""
        return tuple(emb.sub_man.dim for emb in self.factor_embs)

    @property
    def arity(self) -> int:
        """Number of factors."""
        return len(self.cod_embs) + len(self.dom_embs)

    @property
    def cod_dims(self) -> tuple[int, ...]:
        """Selected dimension of each codomain factor."""
        return tuple(emb.sub_man.dim for emb in self.cod_embs)

    @property
    def dom_dims(self) -> tuple[int, ...]:
        """Selected dimension of each contracted factor."""
        return tuple(emb.sub_man.dim for emb in self.dom_embs)

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """The matricization: codomain factors as the rows, contracted factors as the columns."""
        return (prod(self.cod_dims), prod(self.dom_dims))

    @classmethod
    def whole(cls, man: Manifold) -> SubspaceMap:
        """The map holding one manifold whole as its single factor: a bias, with no domain factors.

        No subspace is selected and nothing is contracted, so the parameters are just a
        point of ``man``. This is what a graph node's own potential is, and where the
        recursion in :meth:`~goal.geometry.manifold.clique.LinearCliques.placements_of`
        bottoms out.
        """
        return cls(Rectangular(), (IdentityEmbedding(man),), ())

    def to_matrix(self, params: Array) -> Array:
        """Unpack flat parameters into a dense (output, input) matrix."""
        return self.rep.to_matrix(self.matrix_shape, params)

    def from_matrix(self, matrix: Array) -> Array:
        """Pack a dense (output, input) matrix into flat parameters."""
        return self.rep.from_matrix(matrix)

    def to_tensor(self, params: Array) -> Array:
        """View flat parameters as a tensor of shape :attr:`sub_dims`, one axis per factor.

        Factor order is the storage order --- codomain factors, then contracted ones --- so
        a map and its transpose view one tensor as each other's axis-permuted readings.

        Raises:
            ValueError: unless the representation stores every entry, since a structured
                ``rep`` holds fewer parameters than the tensor has entries.
        """
        self._require_dense()
        return params.reshape(self.sub_dims)

    def from_tensor(self, tensor: Array) -> Array:
        """Flatten a tensor of shape :attr:`sub_dims`, in factor order, into parameters."""
        self._require_dense()
        return tensor.reshape(-1)

    def project_cod(self, joint: Array) -> Array:
        """Restrict a codomain joint from the ambient dimensions to the selected ones."""
        return self._project_group(self.cod_embs, joint)

    def embed_cod(self, selected: Array) -> Array:
        """The other direction on the same factors as :meth:`project_cod`: back out to the ambient dimensions."""
        return self._embed_group(self.cod_embs, selected)

    def project_dom(self, joint: Array) -> Array:
        """Restrict a domain joint from the ambient dimensions to the selected ones.

        With no domain factors --- a bias --- the joint is the constant $1$.
        """
        return self._project_group(self.dom_embs, joint)

    # Private

    @staticmethod
    def _map_axis(tensor: Array, axis: int, fn: Any) -> Array:
        """Apply a vector function along one axis of a tensor, replacing that axis."""
        moved = jnp.moveaxis(tensor, axis, 0)
        trailing = moved.shape[1:]
        columns = moved.reshape(moved.shape[0], -1)
        mapped = jax.vmap(fn, in_axes=1, out_axes=1)(columns)
        return jnp.moveaxis(mapped.reshape((-1, *trailing)), 0, axis)

    @staticmethod
    def _project_group(
        embs: tuple[LinearEmbedding[Any, Any], ...], joint: Array
    ) -> Array:
        if not embs:
            return jnp.ones(1)
        if len(embs) == 1:
            # One factor: its embedding restricts directly, with no tensor to reshape. Not
            # only an optimization --- an embedding may accept a point it can restrict
            # without its ambient dimension matching exactly, and reshaping would not.
            return embs[0].project(joint)
        out = joint.reshape(tuple(emb.amb_man.dim for emb in embs))
        for position, emb in enumerate(embs):
            out = SubspaceMap._map_axis(out, position, emb.project)
        return out.reshape(-1)

    @staticmethod
    def _embed_group(
        embs: tuple[LinearEmbedding[Any, Any], ...], selected: Array
    ) -> Array:
        if not embs:
            # The empty group's space is one-dimensional --- the constant --- so its
            # coordinates pass through, which is what makes a bias's transpose a
            # functional rather than a dead end.
            return selected
        if len(embs) == 1:
            return embs[0].embed(selected)
        out = selected.reshape(tuple(emb.sub_man.dim for emb in embs))
        for position, emb in enumerate(embs):
            out = SubspaceMap._map_axis(out, position, emb.embed)
        return out.reshape(-1)

    def _require_dense(self) -> None:
        if self.rep.num_params(self.matrix_shape) != prod(self.sub_dims):
            msg = f"{type(self.rep).__name__} stores fewer parameters than the tensor"
            raise ValueError(f"{msg} of shape {self.sub_dims} has entries")


### Affine Maps ###


@dataclass(frozen=True)
class AffineMap[
    Domain: Manifold,
    Codomain: Manifold,
](
    Pair[Codomain, LinearMap[Domain, Codomain]],
    Map[Domain, Codomain],
):
    """A linear map plus a bias: $A(x) = L(x) + b$.

    Stored as a ``Pair`` of the bias $b$ (on the codomain) and the linear map $L$, so it is
    both a ``Map`` --- ``__call__`` applies it --- and a ``Tuple``, whose ``split_coords``
    separates the bias from the linear part. This is the natural parameter space for
    exponential family likelihoods (bias = observable natural parameters, linear part =
    interaction).
    """

    # Fields

    map_man: LinearMap[Domain, Codomain]
    """The linear transformation for this affine map."""

    _dom_man: Domain
    """The domain of the affine map."""

    # Overrides

    @property
    @override
    def dom_man(self) -> Domain:
        return self._dom_man

    @property
    @override
    def cod_man(self) -> Codomain:
        return self.map_man.cod_man

    @property
    @override
    def fst_man(self) -> Codomain:
        return self.cod_man

    @property
    @override
    def snd_man(self) -> LinearMap[Domain, Codomain]:
        return self.map_man

    @override
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
