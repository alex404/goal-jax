"""Tests for ``MultilinearMap`` in geometry/manifold/map.py.

A multilinear map is the arity-$n$ generalization of ``Rectangular``. The decisive tests
are the arity-2 ones: they pin it against the ``EmbeddedMap`` machinery that every
interaction in the library already runs on, so the generalization is verified against
working code rather than against a fresh derivation. The arity-3 tests then check the two
properties that make higher arity usable --- that contraction order does not matter, and
that partial contraction composes.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import (
    AmbientMap,
    MultilinearMap,
    Rectangular,
)
from goal.models import Euclidean

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def _key(seed: int) -> Array:
    return jax.random.PRNGKey(seed)


class TestArityTwoMatchesEmbeddedMap:
    """At arity 2 the new primitive must reproduce the existing matrix machinery."""

    @staticmethod
    def _pair(cod_dim: int, dom_dim: int):
        emb_map = AmbientMap(Rectangular(), Euclidean(dom_dim), Euclidean(cod_dim))
        return emb_map, MultilinearMap((cod_dim, dom_dim))

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (1, 6), (6, 1)])
    def test_dim_matches(self, cod_dim: int, dom_dim: int) -> None:
        emb_map, form = self._pair(cod_dim, dom_dim)
        assert form.dim == emb_map.dim

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (1, 6)])
    def test_tensor_matches_outer_product(self, cod_dim: int, dom_dim: int) -> None:
        emb_map, form = self._pair(cod_dim, dom_dim)
        w = jax.random.normal(_key(0), (cod_dim,))
        v = jax.random.normal(_key(1), (dom_dim,))
        assert jnp.array_equal(form.tensor(w, v), emb_map.outer_product(w, v))

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (6, 1)])
    def test_contract_domain_matches_application(
        self, cod_dim: int, dom_dim: int
    ) -> None:
        """``keep=0`` is matrix-vector multiplication."""
        emb_map, form = self._pair(cod_dim, dom_dim)
        params = jax.random.normal(_key(2), (form.dim,))
        v = jax.random.normal(_key(3), (dom_dim,))
        assert jnp.allclose(form.contract(params, 0, v), emb_map(params, v))

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (1, 6)])
    def test_contract_codomain_matches_transpose(
        self, cod_dim: int, dom_dim: int
    ) -> None:
        """``keep=1`` is the transposed application."""
        emb_map, form = self._pair(cod_dim, dom_dim)
        params = jax.random.normal(_key(4), (form.dim,))
        w = jax.random.normal(_key(5), (cod_dim,))
        assert jnp.allclose(
            form.contract(params, 1, w), emb_map.transpose_apply(params, w)
        )

    def test_storage_order_is_row_major_over_codomain_then_domain(self) -> None:
        """The convention every ``int_man`` in the library already stores in."""
        emb_map, form = self._pair(2, 3)
        params = jnp.arange(6.0)
        assert jnp.array_equal(form.to_tensor(params), emb_map.to_matrix(params))


class TestHigherArity:
    """Properties that make arity 3 usable, which arity 2 cannot distinguish."""

    form: MultilinearMap = MultilinearMap((2, 3, 4))

    def test_dim_is_the_product(self) -> None:
        assert self.form.dim == 24
        assert self.form.arity == 3

    def test_tensor_round_trips_through_contraction(self) -> None:
        """Contracting a rank-one tensor against its own factors rescales the third."""
        u = jax.random.normal(_key(6), (2,))
        v = jax.random.normal(_key(7), (3,))
        w = jax.random.normal(_key(8), (4,))
        params = self.form.tensor(u, v, w)
        assert jnp.allclose(self.form.contract(params, 0, v, w), u * (v @ v) * (w @ w))
        assert jnp.allclose(self.form.contract(params, 1, u, w), v * (u @ u) * (w @ w))
        assert jnp.allclose(self.form.contract(params, 2, u, v), w * (u @ u) * (v @ v))

    def test_partial_contraction_composes(self) -> None:
        """Contracting axes one at a time equals contracting them together.

        This is what lets a level split contract the root axes and leave the deep ones:
        the result must not depend on the order the root axes are taken in.
        """
        params = jax.random.normal(_key(9), (self.form.dim,))
        u = jax.random.normal(_key(10), (2,))
        v = jax.random.normal(_key(11), (3,))
        both = self.form.contract(params, 2, u, v)

        # axis 0 first, leaving a (3, 4) form; then axis 0 of that (the old axis 1)
        step = MultilinearMap((3, 4))
        after_u = jnp.tensordot(self.form.to_tensor(params), u, axes=([0], [0]))
        stepwise = step.contract(step.from_tensor(after_u), 1, v)
        assert jnp.allclose(both, stepwise)

    def test_to_from_tensor_round_trip(self) -> None:
        params = jax.random.normal(_key(12), (self.form.dim,))
        assert jnp.array_equal(
            self.form.from_tensor(self.form.to_tensor(params)), params
        )

    def test_wrong_factor_count_is_rejected(self) -> None:
        u = jnp.ones(2)
        with pytest.raises(ValueError, match="expected 3 factors, got 2"):
            self.form.tensor(u, jnp.ones(3))
        params = jnp.zeros(self.form.dim)
        with pytest.raises(ValueError, match="expected 2 factors, got 1"):
            self.form.contract(params, 0, jnp.ones(3))


class TestArityOne:
    """A bias is a multilinear map of one factor; nothing should special-case it."""

    def test_tensor_and_contract_are_identity(self) -> None:
        form = MultilinearMap((5,))
        v = jax.random.normal(_key(13), (5,))
        assert form.dim == 5
        assert jnp.array_equal(form.tensor(v), v)
        assert jnp.array_equal(form.contract(v, 0), v)


class TestContractGuards:
    """``keep`` names an axis, so it must be one."""

    @pytest.mark.parametrize("keep", [-1, 3])
    def test_rejects_an_out_of_range_axis(self, keep: int) -> None:
        # Without the bounds check these contract every axis and return a scalar, and
        # keep=-1 additionally resolves to the last factor by Python indexing.
        man = MultilinearMap((2, 3, 4))
        params = jnp.arange(float(man.dim))
        vectors = [jnp.ones(2), jnp.ones(3), jnp.ones(4)]
        with pytest.raises(ValueError, match=r"keep must be in 0\.\.2"):
            man.contract(params, keep, *vectors)
