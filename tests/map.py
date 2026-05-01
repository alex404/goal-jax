"""Tests for geometry/manifold/map.py.

Confirms LinearMap and matrix-rep machinery still work after the rename, and exercises MultilayerPerceptron shape, parameter count, and ``__call__`` correctness.
"""

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
import pytest

from goal.geometry import (
    AmbientMap,
    MultilayerPerceptron,
    PositiveDefinite,
    Rectangular,
    SquareMap,
)
from goal.geometry.manifold.base import Manifold

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-5
ATOL = 1e-7


@dataclass(frozen=True)
class _DimManifold(Manifold):
    """Test fixture: a minimal Manifold of a given dimension."""

    _dim: int

    @property
    @override
    def dim(self) -> int:
        return self._dim


class TestLinearMapRegression:
    """Confirms matrix-rep machinery still works after the linear -> map rename."""

    def test_ambient_rectangular_matvec(self) -> None:
        """Rectangular AmbientMap applies as a plain matrix-vector product."""
        dom = _DimManifold(3)
        cod = _DimManifold(2)
        m = AmbientMap(Rectangular(), dom, cod)
        assert m.dim == 6  # 2x3 dense matrix
        # Row-major params for matrix [[0, 1, 2], [3, 4, 5]]
        params = jnp.arange(6, dtype=jnp.float64)
        x = jnp.array([1.0, 2.0, 3.0])
        y = m(params, x)
        assert y.shape == (2,)
        expected = jnp.array([0 + 2 + 6, 3 + 8 + 15], dtype=jnp.float64)
        assert jnp.allclose(y, expected, rtol=RTOL, atol=ATOL)

    def test_square_pd_identity(self) -> None:
        """PositiveDefinite SquareMap with identity params acts as identity."""
        dom = _DimManifold(2)
        m = SquareMap(PositiveDefinite(), dom)
        assert m.dim == 3  # 2*(2+1)/2 unique entries
        params = m.from_matrix(jnp.eye(2))
        x = jnp.array([1.0, 2.0])
        y = m(params, x)
        assert jnp.allclose(y, x, rtol=RTOL, atol=ATOL)


class TestMultilayerPerceptron:
    """MultilayerPerceptron dim, __call__, and initialization."""

    @pytest.mark.parametrize(
        "dom_dim,cod_dim,hidden",
        [
            (3, 2, ()),
            (3, 2, (4,)),
            (5, 4, (8, 6)),
        ],
    )
    def test_dim(self, dom_dim: int, cod_dim: int, hidden: tuple[int, ...]) -> None:
        """``dim`` matches sum of per-layer (in*out + out)."""
        m = MultilayerPerceptron(
            _DimManifold(dom_dim),
            _DimManifold(cod_dim),
            hidden_dims=hidden,
            activation=jax.nn.relu,
        )
        layer_dims = (dom_dim, *hidden, cod_dim)
        expected = sum(
            in_d * out_d + out_d for in_d, out_d in zip(layer_dims[:-1], layer_dims[1:])
        )
        assert m.dim == expected

    @pytest.mark.parametrize(
        "dom_dim,cod_dim,hidden",
        [
            (3, 2, ()),
            (3, 2, (4,)),
            (5, 4, (8, 6)),
        ],
    )
    def test_call_shape(
        self, dom_dim: int, cod_dim: int, hidden: tuple[int, ...]
    ) -> None:
        """``__call__`` produces a codomain-shaped output."""
        m = MultilayerPerceptron(
            _DimManifold(dom_dim),
            _DimManifold(cod_dim),
            hidden_dims=hidden,
            activation=jax.nn.relu,
        )
        params = m.glorot_initialize(jax.random.PRNGKey(0))
        y = m(params, jnp.zeros(dom_dim))
        assert y.shape == (cod_dim,)

    def test_glorot_initialize_param_count(self) -> None:
        m = MultilayerPerceptron(
            _DimManifold(3), _DimManifold(2), hidden_dims=(4,), activation=jax.nn.relu
        )
        params = m.glorot_initialize(jax.random.PRNGKey(0))
        assert params.shape == (m.dim,)

    def test_no_hidden_matches_matmul(self) -> None:
        """With no hidden layers and zero biases, MultilayerPerceptron reduces to ``W @ x``."""
        m = MultilayerPerceptron(
            _DimManifold(3), _DimManifold(2), hidden_dims=(), activation=jax.nn.relu
        )
        w = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = jnp.zeros(2)
        params = jnp.concatenate([w.reshape(-1), b])
        x = jnp.array([1.0, 1.0, 1.0])
        y = m(params, x)
        assert jnp.allclose(y, w @ x, rtol=RTOL, atol=ATOL)

    def test_no_activation_on_output(self) -> None:
        """Activation is not applied to the output layer."""
        m = MultilayerPerceptron(
            _DimManifold(2), _DimManifold(2), hidden_dims=(), activation=jax.nn.relu
        )
        w = jnp.eye(2)
        b = jnp.zeros(2)
        params = jnp.concatenate([w.reshape(-1), b])
        x = jnp.array([-1.0, -2.0])
        y = m(params, x)
        assert jnp.allclose(y, x, rtol=RTOL, atol=ATOL)

    def test_glorot_within_bounds(self) -> None:
        """Glorot weights respect $\\sqrt{6/(d_{in} + d_{out})}$ bounds; biases zero."""
        m = MultilayerPerceptron(
            _DimManifold(10), _DimManifold(10), hidden_dims=(), activation=jax.nn.relu
        )
        params = m.glorot_initialize(jax.random.PRNGKey(0))
        weights = params[:100]
        biases = params[100:]
        bound = jnp.sqrt(6.0 / 20.0)
        assert jnp.all(jnp.abs(weights) <= bound)
        assert jnp.allclose(biases, 0.0, rtol=RTOL, atol=ATOL)
