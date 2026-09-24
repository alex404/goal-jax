# Clique container: state of the branch

Branch `clique-container`, last commit `9285556`. Everything below is **uncommitted**
(33 files, +845 / −1101 against that commit). Full suite on the current tree: 553 passed
(2026-09-06). ruff, basedpyright and Sphinx clean.

## Where the design sits

- **`SubspaceMap`** (`geometry/manifold/map.py`) is the arity-n linear leaf: a `MatrixRep`
  plus two groups of per-factor embeddings, `cod_embs` and `dom_embs`. Direction is which
  group a factor is in; `trn_man` swaps the groups. It is address-free: it knows its factors
  and the subspace it uses in each, not which nodes they occupy. `SubspaceMap.whole(man)`
  builds a root form (a bias with no domain factors).
- **`Product`** (`geometry/manifold/combinators.py`) is the tensor product of manifolds
  (dim is a product; `Product(())` has dim 1). It is the `dom_man`/`cod_man` of a
  multi-factor `SubspaceMap`. `Null` is the unit of the Cartesian product, `Product(())` the
  unit of the tensor product.
- **`LinearCliques`** (`geometry/manifold/clique.py`) is addressing only: `placements` is one
  `(members, form)` pair per clique, member i naming the node of factor i in
  `cod_embs + dom_embs` order. `LevelCliques.cross_placement` derives all positional
  information from a node set; `cross_paths` derives the paths `Interaction` uses.
- **`Interaction`** (`geometry/manifold/interaction.py`) sums path-conjugated forms:
  v ↦ Σₜ φₜ(Θₜ · πₜ(v)).
- **`CliqueCut`** (`geometry/algebra/cut.py`) regroups a layout's coordinates as
  (near | cross | far) around one node. Its one production use is
  `CompleteMixtureOfHarmoniums.mix_cut`, which reads the hierarchical layout as a
  `CompleteMixture` over the base harmonium. `split_by_dims` moved with it to
  `geometry/algebra/util.py`.

Vocabulary: **factor** for a manifold a map is multilinear in, **axis** only for array
axes. Main's embedding API on maps (`EmbeddedMap`, `AmbientMap`, `BlockMap`,
`prepend_embedding`, …) no longer exists on this branch; embeddings live per factor in
`SubspaceMap`.

## Next

1. Walk through `manifold/clique.py` method by method (the stated next step).

## Open questions (recorded, not started)

1. **`Interaction` as triples.** Replace `placements` + `paths` with one
   `(form, cod_path, dom_path)` per term. This deletes the unused `members` field and the
   caveat that `members` does not pair with the form's factors inside a transposed
   interaction.
2. **Downward closure is assumed but not stated.** `CliqueCut` (a crossing clique's near
   part must be a clique) and `LevelCliques.cross_paths` both rely on the cover being closed
   under taking sub-cliques in the relevant cases. Either state it on `LinearCliques` or
   check it once at construction.
3. **Field order.** `SubspaceMap` stores codomain first (`rep, cod_embs, dom_embs`),
   `MatrixMap` stores domain first. Pick one convention.
4. **Utilities.** `batched_mean` stays in `manifold/util.py`, `split_by_dims` is in
   `algebra/util.py`. Decide whether that split is right or whether one `geometry/util.py`
   is simpler.
5. **`CliqueEmbedding.sub_man`** returns the `SubspaceMap` form itself, not a `Product` of
   the nodes it addresses. Decide whether the sub-manifold should be the form or the
   node space.

## Later (out of scope for this branch)

- A convolutional `MatrixRep` subclass — the reason `rep` stays on `SubspaceMap`.
- A generic conjugation cascade over `LevelCliques`, and the variational `rho` slot.

## Known limitations accepted

- A hand-paired form borrowed across a level (`graphical/mixture.py`'s θ_XY) that arrives
  transposed is detected by `node_mans` disagreement only when the two nodes' manifolds
  differ.
- `canonical_cliques` has no production callers. It is kept because it encodes the nesting
  property the `LevelCliques` recursion rests on; the docstring no longer advertises it.
