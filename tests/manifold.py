"""Tests for geometry/manifold: clique layouts (clique.py), maps (map.py) and embeddings (embedding.py, clique.py).

Every case is a shipped model or a part of one. The layout of each model's coordinates is
checked against its graph, each embedding against the laws of a coordinate inclusion, each
model's cross map against its dense matrix, and the multilayer perceptron against a forward
pass written out from its documented parameter layout.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import (
    CliqueEmbedding,
    CliqueMap,
    Diagonal,
    DifferentiableTuple,
    IdentityEmbedding,
    LinearEmbedding,
    MatrixMap,
    MultilayerPerceptron,
    ObservableEmbedding,
    PositiveDefinite,
    Rectangular,
    RecursiveLinearCliques,
    RootEmbedding,
    Scale,
    SubCliquesEmbedding,
    SubMapEmbedding,
)
from goal.models import (
    BoltzmannLGM,
    BoltzmannNormalHarmonium,
    CanonicalCorrelationAnalysis,
    CompleteMixture,
    DiagonalBoltzmann,
    Euclidean,
    MixtureOfFactorAnalyzers,
    Normal,
    PoissonVonMisesHarmonium,
    analytic_hmog,
    differentiable_hmog,
    factor_analysis,
    poisson_mixture,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-5
ATOL = 1e-7

type Layout = RecursiveLinearCliques[Any, Any]


### Shipped models ###


def _cca() -> CanonicalCorrelationAnalysis[Any, Any, Any]:
    """Asymmetric branches, so a swapped layout shows in the dimensions."""
    return CanonicalCorrelationAnalysis(
        3, PositiveDefinite(), 2, Diagonal(), 2, PositiveDefinite()
    )


def _mfa() -> MixtureOfFactorAnalyzers:
    return MixtureOfFactorAnalyzers(
        n_categories=3, bas_hrm=factor_analysis(obs_dim=4, lat_dim=2)
    )


def _hmog_scale() -> Any:
    """An HMoG whose posterior latent is isotropic, so its ``pst_prr_emb`` is a ``RootEmbedding`` that is not a coordinate inclusion."""
    return differentiable_hmog(
        obs_dim=4, obs_rep=Diagonal(), lat_dim=2, pst_rep=Scale(), n_components=3
    )


def _hmog_diagonal() -> Any:
    return differentiable_hmog(
        obs_dim=4, obs_rep=Diagonal(), lat_dim=2, pst_rep=Diagonal(), n_components=3
    )


MODELS: dict[str, Callable[[], Layout]] = {
    "factor_analysis": lambda: factor_analysis(obs_dim=4, lat_dim=2),
    "analytic_hmog": lambda: analytic_hmog(
        obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4
    ),
    "differentiable_hmog": _hmog_scale,
    "cca": _cca,
    "mfa": _mfa,
    "mfa_mixture_view": lambda: _mfa().mix_man,
    "poisson_mixture": lambda: poisson_mixture(n_neurons=4, n_components=3),
    "mixture_of_cca": lambda: CompleteMixture(_cca(), 3),
    "boltzmann_lgm": lambda: BoltzmannLGM(3, PositiveDefinite(), 2),
    "boltzmann_normal": lambda: BoltzmannNormalHarmonium(DiagonalBoltzmann(3), 2),
    "poisson_von_mises": lambda: PoissonVonMisesHarmonium(5, 1),
}
"""Every shipped model shape with crossings."""


### Layouts ###


def layout_problems(man: Layout) -> list[str]:
    """Every way a model's composed graph and its parameter layout can disagree.

    The coordinate blocks must tile the coordinates in the partitions' own sizes, the
    partitions must be where ``split_coords`` puts them, and each clique of the root and
    deep partitions must hold that partition's block for the same clique in its own
    numbering. None of this is checked at construction.
    """
    if sum(man.clq_dims) != man.dim:
        return [f"cliques sum to {sum(man.clq_dims)}, but dim is {man.dim}"]
    parts = man.rot_man.clq_dims + man.crs_man.clq_dims + man.dep_man.clq_dims
    if man.clq_dims != parts:
        return [f"block sizes {man.clq_dims} are not the partitions' {parts}"]
    coords = jnp.arange(float(man.dim))
    root, cross, deep = man.split_coords(coords)
    sizes = (root.size, cross.size, deep.size)
    if sizes != (man.rot_man.dim, man.crs_man.dim, man.dep_man.dim):
        return [f"split_coords cuts at {sizes}, not at the partition dimensions"]
    if not jnp.array_equal(man.join_coords(root, cross, deep), coords):
        return ["join_coords does not invert split_coords"]
    n_rot = man.rot_man.n_nodes
    for clique in man.rot_man.cliques:
        if not jnp.array_equal(
            CliqueEmbedding(clique, man).project(coords),
            CliqueEmbedding(clique, man.rot_man).project(root),
        ):
            return [f"root clique {clique} is misplaced"]
    for clique in man.dep_man.cliques:
        lifted = tuple(i + n_rot for i in clique)
        if not jnp.array_equal(
            CliqueEmbedding(lifted, man).project(coords),
            CliqueEmbedding(clique, man.dep_man).project(deep),
        ):
            return [f"deep clique {clique} is misplaced as {lifted}"]
    return []


class TestLayout:
    """A model's graph agrees with its parameter layout."""

    @pytest.mark.parametrize("name", MODELS)
    def test_layout_agrees_with_graph(self, name: str) -> None:
        assert layout_problems(MODELS[name]()) == []

    @pytest.mark.parametrize("name", MODELS)
    def test_crossings_read_and_write_their_blocks(self, name: str) -> None:
        """A crossing maps from a subspace of the deep block it touches to one of the root block it touches."""
        man = MODELS[name]()
        for trm in man.crs_trms:
            assert trm.clq_map.cod_man == man.rot_man.clq_man(trm.cod_clq), trm
            assert trm.clq_map.dom_man == man.dep_man.clq_man(trm.dom_clq), trm

    @pytest.mark.parametrize("name", MODELS)
    def test_a_single_node_block_is_the_node_space(self, name: str) -> None:
        man = MODELS[name]()
        assert len(man.nod_mans) == man.n_nodes
        for clique, clq_man in zip(man.cliques, man.clq_mans):
            if len(clique) == 1:
                assert clq_man == man.nod_mans[clique[0]]

    def test_declaration_order_is_storage_order(self) -> None:
        """A mixture of CCA declares its crossings in the CCA's storage order, which is not sorted, and each crossing clique holds that crossing's parameters."""
        man = CompleteMixture(_cca(), 3)
        crossings = ((0, 3), (1, 3), (0, 2, 3), (1, 2, 3), (2, 3))
        assert man.cliques[5:10] == crossings
        assert list(crossings) != sorted(crossings)
        params = jnp.arange(float(man.dim))
        crs_params = man.crs_man.clq_coords(man.split_coords(params)[1])
        for clique, clq_params in zip(crossings, crs_params):
            assert jnp.array_equal(
                CliqueEmbedding(clique, man).project(params), clq_params
            )

    def test_a_crossing_past_the_deep_root_composes(self) -> None:
        """MFA's crossing $x - k$ touches only clique $(k)$ of the mixture, not its root $y$; the mixture is still stored in its own order."""
        man = _mfa()
        assert ((0,), (1,)) in man.crs_clqs
        assert man.cliques == ((0,), (0, 1), (0, 1, 2), (0, 2), (1,), (1, 2), (2,))
        assert layout_problems(man) == []

    def test_tuple_offsets_multi_clique_elements(self) -> None:
        """Two multi-node elements flatten into one graph, each offset by the nodes before it, with their blocks where the elements store them."""
        fst = _cca()
        snd = CanonicalCorrelationAnalysis(
            1, Diagonal(), 2, PositiveDefinite(), 1, PositiveDefinite()
        )
        tup = DifferentiableTuple((fst, Normal(2, PositiveDefinite()), snd))
        n = fst.n_nodes
        assert tup.elm_nod_offsets == (0, n, n + 1)
        assert tup.cliques == (
            *fst.cliques,
            (n,),
            *(tuple(i + n + 1 for i in clique) for clique in snd.cliques),
        )
        assert tup.clq_mans == (*fst.clq_mans, tup.elm_mans[1], *snd.clq_mans)
        assert sum(tup.clq_dims) == tup.dim
        params = jnp.arange(float(tup.dim))
        fst_params, _, snd_params = tup.split_coords(params)
        for elm, offset, elm_params in ((fst, 0, fst_params), (snd, n + 1, snd_params)):
            for clique in elm.cliques:
                shifted = tuple(i + offset for i in clique)
                assert jnp.array_equal(
                    CliqueEmbedding(shifted, tup).project(params),
                    CliqueEmbedding(clique, elm).project(elm_params),
                )

    def test_layout_is_jit_static(self) -> None:
        """Models hash, since they are passed as static jit arguments."""
        man = MODELS["analytic_hmog"]()

        def root_sum(model: Layout, coords: Array) -> Array:
            return jnp.sum(model.split_coords(coords)[0])

        coords = jnp.arange(float(man.dim))
        total = jax.jit(root_sum, static_argnums=0)(man, coords)
        assert jnp.allclose(total, jnp.sum(coords[: man.rot_man.dim]))


### Embeddings ###


EMBEDDINGS: dict[str, Callable[[], LinearEmbedding[Any, Any]]] = {
    "clique_mfa_xyk": lambda: CliqueEmbedding((0, 1, 2), _mfa()),
    "sub_map_mfa_xyk": lambda: _mfa().crs_maps[1].dom_emb,
    "sub_cliques_deep_in_hmog": lambda: SubCliquesEmbedding(
        (1, 2), _hmog_diagonal(), _hmog_diagonal().pst_man
    ),
    "sub_cliques_root_in_mixture": lambda: SubCliquesEmbedding(
        (0,), _hmog_diagonal().pst_man, _hmog_diagonal().lwr_hrm.pst_man
    ),
    "root_hmog_diagonal": lambda: _hmog_diagonal().pst_prr_emb,
}
"""Coordinate inclusions found on shipped models."""

ADJOINT_ONLY: dict[str, Callable[[], LinearEmbedding[Any, Any]]] = {
    "root_hmog_scale": lambda: _hmog_scale().pst_prr_emb,
}
"""Embeddings whose projection is not a left inverse: an isotropic covariance round trip scales the variance by $1/d$."""


class TestEmbeddingLaws:
    """``embed`` is an inclusion and ``project`` its transpose."""

    @pytest.mark.parametrize("name", EMBEDDINGS)
    def test_project_inverts_embed(self, name: str) -> None:
        emb = EMBEDDINGS[name]()
        sub = jax.random.normal(jax.random.PRNGKey(0), (emb.sub_man.dim,))
        assert jnp.allclose(emb.project(emb.embed(sub)), sub)

    @pytest.mark.parametrize("name", [*EMBEDDINGS, *ADJOINT_ONLY])
    def test_project_is_the_adjoint_of_embed(self, name: str) -> None:
        """$\\langle \\phi(a), b \\rangle = \\langle a, \\pi(b) \\rangle$."""
        emb = {**EMBEDDINGS, **ADJOINT_ONLY}[name]()
        k_a, k_b = jax.random.split(jax.random.PRNGKey(1))
        a = jax.random.normal(k_a, (emb.sub_man.dim,))
        b = jax.random.normal(k_b, (emb.amb_man.dim,))
        assert jnp.allclose(jnp.dot(emb.embed(a), b), jnp.dot(a, emb.project(b)))


class TestSubMapEmbedding:
    """MFA's crossing $(x, y, k)$ reads a sub-block of the mixture's $(y, k)$ block: the location rows of $y$, all of $k$."""

    def test_the_sub_block_acts_as_its_embedding(self) -> None:
        """Applying the sub-block, forward or transposed, equals applying the ambient block at its embedding."""
        emb = _mfa().crs_maps[1].dom_emb
        assert isinstance(emb, SubMapEmbedding)
        sub, amb = emb.sub_man, emb.amb_man
        keys = jax.random.split(jax.random.PRNGKey(2), 3)
        params = jax.random.normal(keys[0], (sub.dim,))
        v = jax.random.normal(keys[1], (amb.dom_man.dim,))
        w = jax.random.normal(keys[2], (amb.cod_man.dim,))
        amb_params = emb.embed(params)
        assert jnp.allclose(sub(params, v), amb(amb_params, v))
        assert jnp.allclose(
            sub.transpose_apply(params, w), amb.transpose_apply(amb_params, w)
        )


class TestSubCliquesEmbedding:
    """``embed`` writes the submanifold onto its cliques and zeros everything else."""

    def test_deep_partition_of_hmog(self) -> None:
        model = _hmog_diagonal()
        emb = SubCliquesEmbedding((1, 2), model, model.pst_man)
        v = jax.random.normal(jax.random.PRNGKey(3), (model.pst_man.dim,))
        expected = model.join_coords(model.obs_man.zeros(), model.int_man.zeros(), v)
        assert jnp.array_equal(emb.embed(v), expected)

    def test_root_node_of_the_mixture(self) -> None:
        model = _hmog_diagonal()
        emb = SubCliquesEmbedding((0,), model.pst_man, model.lwr_hrm.pst_man)
        v = jax.random.normal(jax.random.PRNGKey(4), (model.lwr_hrm.pst_man.dim,))
        assert jnp.array_equal(
            emb.embed(v), ObservableEmbedding(model.pst_man).embed(v)
        )


class TestRootEmbedding:
    """HMoG's ``pst_prr_emb``: the root partition goes through the lower embedding, the cross and deep partitions pass through."""

    def test_root_is_the_lower_embedding(self) -> None:
        model = _hmog_scale()
        emb = model.pst_prr_emb
        v = jax.random.normal(jax.random.PRNGKey(5), (emb.sub_man.dim,))
        pst_root, _, _ = emb.sub_man.split_coords(v)
        prr_root, _, _ = emb.amb_man.split_coords(emb.embed(v))
        assert jnp.array_equal(prr_root, model.lwr_hrm.pst_prr_emb.embed(pst_root))

    def test_cross_and_deep_pass_through(self) -> None:
        emb = _hmog_scale().pst_prr_emb
        k_v, k_w = jax.random.split(jax.random.PRNGKey(6))
        v = jax.random.normal(k_v, (emb.sub_man.dim,))
        w = jax.random.normal(k_w, (emb.amb_man.dim,))
        for src, out, src_man, out_man in (
            (v, emb.embed(v), emb.sub_man, emb.amb_man),
            (w, emb.project(w), emb.amb_man, emb.sub_man),
        ):
            _, src_cross, src_deep = src_man.split_coords(src)
            _, out_cross, out_deep = out_man.split_coords(out)
            assert jnp.array_equal(out_cross, src_cross)
            assert jnp.array_equal(out_deep, src_deep)

    def test_rejects_mismatched_graphs(self) -> None:
        """The whole three-node model as ambient of the two-node upper harmonium: same root, different graph."""
        model = _hmog_scale()
        with pytest.raises(ValueError, match="differ only in the root partition"):
            RootEmbedding(model.lwr_hrm.pst_prr_emb, model, model.pst_upr_hrm)


### Maps ###


class TestCliqueMap:
    """A clique map over whole blocks is a ``MatrixMap``."""

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (1, 6), (6, 1)])
    def test_matches_matrix_map(self, cod_dim: int, dom_dim: int) -> None:
        mat_map = MatrixMap(Rectangular(), Euclidean(cod_dim), Euclidean(dom_dim))
        clq_map = CliqueMap(
            Rectangular(),
            IdentityEmbedding(Euclidean(cod_dim)),
            IdentityEmbedding(Euclidean(dom_dim)),
        )
        keys = jax.random.split(jax.random.PRNGKey(7), 3)
        params = jax.random.normal(keys[0], (mat_map.dim,))
        v = jax.random.normal(keys[1], (dom_dim,))
        w = jax.random.normal(keys[2], (cod_dim,))
        assert clq_map.dim == mat_map.dim
        assert jnp.array_equal(clq_map.to_matrix(params), mat_map.to_matrix(params))
        assert jnp.allclose(clq_map.outer_product(w, v), mat_map.outer_product(w, v))
        assert jnp.allclose(clq_map(params, v), mat_map(params, v))
        assert jnp.allclose(
            clq_map.transpose_apply(params, w), mat_map.transpose_apply(params, w)
        )
        assert clq_map.trn_man.trn_man == clq_map
        assert jnp.allclose(
            clq_map.trn_man.transpose(clq_map.transpose(params)), params
        )


class TestCrossMap:
    """Each model's cross map against its dense matrix $M(p)$, built column by column from basis vectors."""

    @pytest.mark.parametrize("name", MODELS)
    def test_matches_dense_matrix(self, name: str) -> None:
        """$M(p) v$, $M(p)^T w$, and $\\langle p, w \\otimes v \\rangle = w^T M(p) v$."""
        crs_man = MODELS[name]().crs_man
        keys = jax.random.split(jax.random.PRNGKey(8), 3)
        params = jax.random.normal(keys[0], (crs_man.dim,))
        v = jax.random.normal(keys[1], (crs_man.dom_man.dim,))
        w = jax.random.normal(keys[2], (crs_man.cod_man.dim,))
        basis = jnp.eye(crs_man.dom_man.dim)
        matrix = jax.vmap(lambda e: crs_man(params, e), out_axes=1)(basis)

        assert jnp.allclose(crs_man(params, v), matrix @ v, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(
            crs_man.transpose_apply(params, w), matrix.T @ w, rtol=RTOL, atol=ATOL
        )
        assert jnp.allclose(
            jnp.dot(params, crs_man.outer_product(w, v)), w @ matrix @ v, rtol=RTOL
        )


class TestMultilayerPerceptron:
    """Parameters are per-layer ``(W, b)``, ``W`` row-major with shape (out, in); the activation follows every layer but the last."""

    def test_no_hidden_layers_is_affine(self) -> None:
        mlp = MultilayerPerceptron(Euclidean(2), Euclidean(3), (), jax.nn.relu)
        keys = jax.random.split(jax.random.PRNGKey(9), 3)
        weights = jax.random.normal(keys[0], (2, 3))
        bias = jax.random.normal(keys[1], (2,))
        x = jax.random.normal(keys[2], (3,))
        params = jnp.concatenate([weights.reshape(-1), bias])
        assert jnp.allclose(mlp(params, x), weights @ x + bias)

    def test_no_activation_on_output(self) -> None:
        """Negative outputs survive a relu network."""
        mlp = MultilayerPerceptron(Euclidean(2), Euclidean(2), (), jax.nn.relu)
        params = jnp.concatenate([jnp.eye(2).reshape(-1), jnp.zeros(2)])
        x = jnp.array([-1.0, -2.0])
        assert jnp.allclose(mlp(params, x), x)

    def test_one_hidden_layer(self) -> None:
        mlp = MultilayerPerceptron(Euclidean(2), Euclidean(3), (4,), jnp.tanh)
        keys = jax.random.split(jax.random.PRNGKey(10), 5)
        w1 = jax.random.normal(keys[0], (4, 3))
        b1 = jax.random.normal(keys[1], (4,))
        w2 = jax.random.normal(keys[2], (2, 4))
        b2 = jax.random.normal(keys[3], (2,))
        x = jax.random.normal(keys[4], (3,))
        params = jnp.concatenate([w1.reshape(-1), b1, w2.reshape(-1), b2])
        assert params.size == mlp.dim
        assert jnp.allclose(mlp(params, x), w2 @ jnp.tanh(w1 @ x + b1) + b2)
