"""``LatticeConvolution`` --- a (strided, multi-channel) convolution as a ``LinearMap``.

This is a *library-candidate* abstraction, written to slot in beside
``EmbeddedMap`` / ``BlockMap`` in ``goal/geometry/manifold/map.py``. It lives in
``examples/`` until the surrounding hierarchical model is reviewed; nothing here
depends on the playground code.

What it is
----------
A stride-``s`` transposed convolution over a lattice, viewed as a linear map from a
(coarse) input feature map to a (finer) output feature map:

    input:  ``in_lattice`` cells x ``in_channels``   (e.g. a 7x7 spike grid)
    output: ``out_lattice`` cells x ``out_channels``  (e.g. a 28x28 image),
            with ``out_lattice = in_lattice * stride``.

Input cell ``q`` writes ``kernel[:, ci, co]`` into the output patch centered at
``q * stride`` (for each channel pair). The free parameters are the shared
``kernel`` of shape ``(prod(kernel_shape), in_channels, out_channels)`` -- small --
so the dense matrix ``W`` it represents is block-Toeplitz and each input cell only
writes to a bounded output neighborhood.

Why it belongs here (the conjugation motivation)
------------------------------------------------
For a linear-Gaussian likelihood ``x | y ~ N(mu + W s_Y(y), Sigma)`` with fixed
``Sigma``, the LGM conjugation identity (:mod:`goal.models.harmonium.lgm`, ``LGM``
docstring) gives the latent-side second-order conjugation parameter
``P^sigma = 1/2 W^T Sigma W``, added onto the latent precision / Boltzmann
couplings. With ``W`` a strided conv, ``P^sigma`` couples two input cells only when
their output footprints overlap, i.e. within Chebyshev distance
``floor((kernel-1)/stride)`` on the *input* lattice (all channel pairs at
overlapping cells couple). So restricting ``Theta_XY`` to this convolutional
subparameterization keeps the induced latent couplings on a bounded-neighborhood
graph *analytically*, for every kernel; on a coarse/thin input lattice that graph
is bounded-treewidth (chordal) and junction-tree inference stays cheap. See
:meth:`induced_coupling_graph`.

Index convention
----------------
Feature maps are flattened as ``(cell, channel)`` with **channel fastest**:
``unit = cell_index * n_channels + channel``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from goal.geometry.manifold.base import Manifold
from goal.geometry.manifold.embedding import LinearEmbedding
from goal.geometry.manifold.map import LinearMap


def _row_major_strides(shape: tuple[int, ...]) -> np.ndarray:
    dims = np.array(shape)
    return np.array([int(prod(dims[i + 1 :])) for i in range(len(dims))])


def _strided_spatial_basis(
    in_lattice: tuple[int, ...],
    stride: tuple[int, ...],
    kernel_shape: tuple[int, ...],
) -> np.ndarray:
    """Per-tap 0/1 spatial matrices ``S[k][p, q] = 1`` iff output cell ``p`` reads
    input cell ``q`` through kernel tap ``k`` (``p == q*stride + k - center``).

    Geometry only (no kernel values) -> a traced constant under ``jax.jit``. Shape
    ``(prod(kernel_shape), prod(out_lattice), prod(in_lattice))`` with out lattice
    ``in_lattice * stride`` and taps in ``np.ndindex`` (row-major) order.
    """
    in_dims = np.array(in_lattice)
    out_dims = in_dims * np.array(stride)
    center = np.array(kernel_shape) // 2
    in_n, out_n = int(prod(in_lattice)), int(np.prod(out_dims))
    in_strides, out_strides = (
        _row_major_strides(in_lattice),
        _row_major_strides(tuple(out_dims)),
    )
    axes = [np.arange(d) for d in in_lattice]
    in_grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        in_n, len(in_dims)
    )
    flat_q = in_grid @ in_strides
    taps = []
    for ki in np.ndindex(*kernel_shape):
        p = in_grid * np.array(stride) + (np.array(ki) - center)
        valid = np.all((p >= 0) & (p < out_dims), axis=1)
        flat_p = p @ out_strides
        s = np.zeros((out_n, in_n))
        s[flat_p[valid], flat_q[valid]] = 1.0
        taps.append(s)
    return np.stack(taps)


@dataclass(frozen=True)
class LatticeConvolution[Codomain: Manifold, Domain: Manifold](
    LinearMap[Codomain, Domain]
):
    """A strided, multi-channel transposed convolution as a linear map.

    ``dom_man`` / ``cod_man`` are arbitrary manifolds whose ``dim`` equals the input
    / output feature-map size (``prod(in_lattice) * in_channels`` and
    ``prod(out_lattice) * out_channels``); the map cares only about their sizes.
    Construct via :meth:`create`, which fills the output lattice and manifolds.
    """

    # Fields

    _dom_man: Domain
    _cod_man: Codomain
    in_lattice: tuple[int, ...]
    """Input (coarse) lattice extent, e.g. ``(7, 7)``."""
    stride: tuple[int, ...]
    """Upsampling factor per axis; output lattice is ``in_lattice * stride``."""
    kernel_shape: tuple[int, ...]
    """Kernel extent per axis; same rank as the lattice."""
    in_channels: int
    out_channels: int
    transposed: bool = False
    """Internal: if set, this instance represents the adjoint map (see :attr:`trn_man`)."""

    @classmethod
    def create(
        cls,
        dom_man: Domain,
        cod_man: Codomain,
        in_lattice: tuple[int, ...],
        stride: tuple[int, ...],
        kernel_shape: tuple[int, ...],
        in_channels: int = 1,
        out_channels: int = 1,
    ) -> LatticeConvolution[Codomain, Domain]:
        conv = cls(
            dom_man,
            cod_man,
            in_lattice,
            stride,
            kernel_shape,
            in_channels,
            out_channels,
        )
        in_n = prod(in_lattice) * in_channels
        out_n = prod(tuple(np.array(in_lattice) * np.array(stride))) * out_channels
        if dom_man.dim != in_n or cod_man.dim != out_n:
            raise ValueError(
                f"dom/cod dim must be {in_n}/{out_n} "
                + f"(in x Cin / out x Cout); got {dom_man.dim}/{cod_man.dim}"
            )
        if not (len(stride) == len(kernel_shape) == len(in_lattice)):
            raise ValueError("in_lattice, stride, kernel_shape must share rank")
        return conv

    @property
    def out_lattice(self) -> tuple[int, ...]:
        return tuple(int(x) for x in np.array(self.in_lattice) * np.array(self.stride))

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
        return prod(self.kernel_shape) * self.in_channels * self.out_channels

    @property
    @override
    def trn_man(self) -> LatticeConvolution[Domain, Codomain]:
        # Same geometry/params; only the manifolds swap and the adjoint flag flips.
        # transpose(f) returns f unchanged and to_dense yields W^T when transposed.
        return LatticeConvolution(
            self._cod_man,
            self._dom_man,
            self.in_lattice,
            self.stride,
            self.kernel_shape,
            self.in_channels,
            self.out_channels,
            transposed=not self.transposed,
        )

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        return self.to_dense(f_coords) @ v_coords

    @override
    def transpose(self, f_coords: Array) -> Array:
        return f_coords  # the adjoint reuses the same kernel; trn_man transposes W

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        # Kernel-subspace projection of w v^T, i.e. d/dkernel <w, W(kernel) v>.
        # __call__ is linear in the kernel, so the gradient at 0 gives it exactly.
        zero = jnp.zeros(self.dim)
        return jax.grad(lambda k: jnp.dot(w_coords, self(k, v_coords)))(zero)

    @override
    def map_domain_embedding[NewDomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Domain, Manifold]], LinearEmbedding[NewDomain, Manifold]
        ],
    ) -> LinearMap[Codomain, NewDomain]:
        raise NotImplementedError(
            "Embedding composition is the harmonium-wiring seam; deferred (see module docstring)."
        )

    @override
    def map_codomain_embedding[NewCodomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Codomain, Manifold]],
            LinearEmbedding[NewCodomain, Manifold],
        ],
    ) -> LinearMap[NewCodomain, Domain]:
        raise NotImplementedError(
            "Embedding composition is the harmonium-wiring seam; deferred (see module docstring)."
        )

    # Methods

    def _spatial_basis(self) -> np.ndarray:
        """Per-tap spatial shift matrices (geometry only; constant-folded under jit)."""
        return _strided_spatial_basis(self.in_lattice, self.stride, self.kernel_shape)

    def _forward_dense(self, f_coords: Array) -> Array:
        """The (out_dim x in_dim) upsampling matrix ``W`` before any transpose."""
        s = jnp.asarray(self._spatial_basis())  # (taps, out_spatial, in_spatial)
        kernel = f_coords.reshape(
            prod(self.kernel_shape), self.in_channels, self.out_channels
        )
        # W[(p,co),(q,ci)] = sum_k S[k,p,q] kernel[k,ci,co]
        w4 = jnp.einsum(
            "kpq,kIO->pOqI", s, kernel
        )  # (out_spatial, Cout, in_spatial, Cin)
        out_dim = s.shape[1] * self.out_channels
        in_dim = s.shape[2] * self.in_channels
        return w4.reshape(out_dim, in_dim)

    def to_dense(self, f_coords: Array) -> Array:
        """Dense matrix of this map: ``W`` (or ``W^T`` for the adjoint instance)."""
        w = self._forward_dense(f_coords)
        return w.T if self.transposed else w

    def induced_coupling_graph(self) -> Array:
        """0/1 ``(in_dim, in_dim)`` support that ``P^sigma = W^T Sigma W`` can occupy.

        Two input units couple iff their output footprints overlap (independent of
        kernel values and of diagonal ``Sigma``): a spatial footprint-overlap graph
        tensored with the all-ones ``in_channels`` block. Build the Boltzmann prior
        graph to contain this and the conjugation stays on it.
        """
        s = self._spatial_basis()
        footprint = (s.sum(0) > 0).astype(float)  # (out_spatial, in_spatial)
        spatial = (footprint.T @ footprint > 0).astype(
            float
        )  # (in_spatial, in_spatial)
        block = np.ones((self.in_channels, self.in_channels))
        return jnp.asarray(np.kron(spatial, block))

    def induced_edges(self) -> list[tuple[int, int]]:
        """Undirected off-diagonal pairs of :meth:`induced_coupling_graph`.

        The edge set a Boltzmann prior must contain so that ``P^sigma`` stays
        representable on its couplings (feed to ``ChordalBoltzmann.from_edges``).
        """
        g = np.asarray(self.induced_coupling_graph())
        return [
            (i, j)
            for i in range(g.shape[0])
            for j in range(i + 1, g.shape[0])
            if g[i, j] > 0
        ]


@dataclass(frozen=True)
class EmbeddedLinearMap[Codomain: Manifold, Domain: Manifold](
    LinearMap[Codomain, Domain]
):
    """A linear map backed by an arbitrary inner ``LinearMap`` plus embeddings.

    Exactly :class:`goal.geometry.EmbeddedMap`, but the inner operator is any
    ``LinearMap`` (e.g. a :class:`LatticeConvolution`) rather than a ``MatrixRep``.
    Application projects the input into the inner domain, applies the inner map,
    and embeds the result into the external codomain -- so a structured inner map
    (conv) can drive a harmonium interaction between the full latent/observable
    manifolds. The free parameters are exactly the inner map's.
    """

    # Fields

    inner: LinearMap[Manifold, Manifold]
    dom_emb: LinearEmbedding[Domain, Manifold]
    cod_emb: LinearEmbedding[Codomain, Manifold]

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
        return self.inner.dim

    @property
    @override
    def trn_man(self) -> EmbeddedLinearMap[Domain, Codomain]:
        return EmbeddedLinearMap(self.inner.trn_man, self.cod_emb, self.dom_emb)

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        internal_v = self.dom_emb.project(v_coords)
        internal_result = self.inner(f_coords, internal_v)
        return self.cod_emb.embed(internal_result)

    @override
    def transpose(self, f_coords: Array) -> Array:
        return self.inner.transpose(f_coords)

    @override
    def outer_product(self, w_coords: Array, v_coords: Array) -> Array:
        internal_w = self.cod_emb.project(w_coords)
        internal_v = self.dom_emb.project(v_coords)
        return self.inner.outer_product(internal_w, internal_v)

    @override
    def map_domain_embedding[NewDomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Domain, Manifold]], LinearEmbedding[NewDomain, Manifold]
        ],
    ) -> EmbeddedLinearMap[Codomain, NewDomain]:
        return EmbeddedLinearMap(self.inner, f(self.dom_emb), self.cod_emb)

    @override
    def map_codomain_embedding[NewCodomain: Manifold](
        self,
        f: Callable[
            [LinearEmbedding[Codomain, Manifold]],
            LinearEmbedding[NewCodomain, Manifold],
        ],
    ) -> EmbeddedLinearMap[NewCodomain, Domain]:
        return EmbeddedLinearMap(self.inner, self.dom_emb, f(self.cod_emb))
