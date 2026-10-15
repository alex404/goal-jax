"""Hidden Markov Model example with gradient-based and EM learning.

Discrete analogue of the Kalman filter example: sample sequences from an HMM with sticky transitions and emissions that favour one symbol per state, run filtering and smoothing under the true parameters, fit a fresh model via gradient ascent on $\\log p(x_{1:T})$ and via expectation maximization, and compare.
"""

import itertools
from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.geometry import AnalyticConjugated
from goal.models import Categorical, HiddenMarkovModel

from ..shared import example_paths, jax_cli
from .types import HMMResults

jax.config.update("jax_enable_x64", True)

N_OBS = 4
N_STATES = 3
N_STEPS = 50
N_SEQUENCES = 50
TRANSITION = jnp.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
EMISSION = jnp.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.4, 0.4]])
N_GRAD_STEPS = 1000
N_EM_STEPS = 500
N_EM_RESTARTS = 4


def categorical_likelihood(
    hrm: AnalyticConjugated[Categorical, Categorical], probs: Array
) -> Array:
    """Likelihood natural parameters of a categorical mixture with rows ``probs[z]`` $= p(x \\mid z)$."""
    logits = jnp.log(probs[:, 1:]) - jnp.log(probs[:, :1])
    (xz_map,) = hrm.crs_maps
    return hrm.lkl_fun_man.join_coords(
        logits[0], xz_map.from_matrix((logits[1:] - logits[0]).T)
    )


def avg_log_lik(
    model: HiddenMarkovModel, params: Array, observations_batch: Array
) -> Array:
    return jnp.mean(
        jax.vmap(lambda obs: model.log_observable_density(params, obs))(
            observations_batch
        )
    )


def fit_gradient(
    key: Array,
    model: HiddenMarkovModel,
    observations_batch: Array,
    n_steps: int,
) -> tuple[Array, Array, Array]:
    init_params = model.initialize(key, shape=0.2)

    def obj(p: Array) -> Array:
        return avg_log_lik(model, p, observations_batch)

    optimizer = optax.adam(learning_rate=2e-2)

    def grad_step(carry: tuple[Array, Any], _: None) -> tuple[tuple[Array, Any], Array]:
        params, opt_state = carry
        ll = obj(params)
        grads = jax.grad(obj)(params)
        updates, new_opt_state = optimizer.update(
            jax.tree.map(jnp.negative, grads), opt_state
        )
        new_params = jnp.asarray(optax.apply_updates(params, updates))
        return (new_params, new_opt_state), ll

    opt_state = optimizer.init(init_params)
    (final_params, _), lls = jax.lax.scan(
        grad_step, (init_params, opt_state), None, length=n_steps
    )
    return lls.ravel(), init_params, final_params


fit_gradient = jax.jit(fit_gradient, static_argnames=["model", "n_steps"])


def fit_em(
    key: Array,
    model: HiddenMarkovModel,
    observations_batch: Array,
    n_steps: int,
) -> tuple[Array, Array, Array]:
    """EM from ``N_EM_RESTARTS`` random initializations, keeping the run with the highest final log-likelihood.

    EM from a near-uniform initialization stays near the symmetric point where all states are equal, and
    from a single random one it can end in a local optimum.
    """

    def run(init_params: Array) -> tuple[Array, Array]:
        def em_step(params: Array, _: Any) -> tuple[Array, Array]:
            ll = avg_log_lik(model, params, observations_batch)
            return model.expectation_maximization(params, observations_batch), ll

        return jax.lax.scan(em_step, init_params, None, length=n_steps)

    inits = jax.vmap(lambda k: model.initialize(k, shape=1.0))(
        jax.random.split(key, N_EM_RESTARTS)
    )
    finals, lls = jax.vmap(run)(inits)
    best = jnp.argmax(lls[:, -1])
    return lls[best], inits[best], finals[best]


fit_em = jax.jit(fit_em, static_argnames=["model", "n_steps"])


def state_alignment(probs: Array, reference: Array) -> Array:
    """The permutation of the state columns of ``probs`` that best matches ``reference``: learned state labels are arbitrary."""
    perms = jnp.array(list(itertools.permutations(range(probs.shape[1]))))
    best = jnp.argmax(jax.vmap(lambda p: jnp.sum(probs[:, p] * reference))(perms))
    return perms[best]


def extract_state_probs(
    model: HiddenMarkovModel, params: Array, observations: Array
) -> tuple[Array, Array]:
    """Filter and smooth, then convert per-step natural params to state probabilities."""
    filtered, _ = model.filter(params, observations)
    _, smoothed_seq, _ = model.smooth(params, observations)

    def to_probs(nat: Array) -> Array:
        return model.lat_man.to_probs(model.lat_man.to_mean(nat))

    filt_probs = jax.vmap(to_probs)(filtered)
    smth_probs = jax.vmap(to_probs)(smoothed_seq)
    return filt_probs, smth_probs


def main() -> None:
    jax_cli()
    paths = example_paths(__file__)

    key = jax.random.PRNGKey(0)
    key_sample, key_train = jax.random.split(key)

    model = HiddenMarkovModel(n_obs=N_OBS, n_states=N_STATES)
    true_params = model.join_coords(
        model.lat_man.from_probs(jnp.ones(N_STATES) / N_STATES),
        categorical_likelihood(model.ems_hrm, EMISSION),
        categorical_likelihood(model.trn_map.kernel, TRANSITION),
    )

    # Sample training trajectories
    keys_sample = jax.random.split(key_sample, N_SEQUENCES)
    obs_batch, state_batch = jax.vmap(lambda k: model.sample(k, true_params, N_STEPS))(
        keys_sample
    )

    observations = obs_batch[0]
    true_states = state_batch[0][
        1:
    ]  # z_1, ..., z_T: the states that emit x_1, ..., x_T

    # Oracle inference under true params
    true_filt_probs, true_smth_probs = extract_state_probs(
        model, true_params, observations
    )

    true_avg_ll = float(avg_log_lik(model, true_params, obs_batch))
    init_params = model.initialize(key_train, shape=0.2)
    initial_ll = float(avg_log_lik(model, init_params, obs_batch))

    # Training
    grad_lls, _, grad_final = fit_gradient(key_train, model, obs_batch, N_GRAD_STEPS)
    em_lls, _, em_final = fit_em(key_train, model, obs_batch, N_EM_STEPS)

    learned: dict[str, tuple[Array, Array]] = {}
    for name, params in (("Gradient", grad_final), ("EM", em_final)):
        filt, smth = extract_state_probs(model, params, observations)
        perm = state_alignment(smth, true_smth_probs)
        learned[name] = (filt[:, perm], smth[:, perm])

    # Convert observations and states from (T, 1) integer arrays to flat lists.
    obs_int = observations.reshape(-1).astype(jnp.int32).tolist()
    state_int = true_states.reshape(-1).astype(jnp.int32).tolist()

    results = HMMResults(
        n_obs=N_OBS,
        n_states=N_STATES,
        n_steps=N_STEPS,
        n_sequences=N_SEQUENCES,
        observations=obs_int,
        true_states=state_int,
        filtered_probs=true_filt_probs.tolist(),
        smoothed_probs=true_smth_probs.tolist(),
        log_likelihoods={
            "Gradient": grad_lls.tolist(),
            "EM": em_lls.tolist(),
        },
        learned_filtered_probs={k: v[0].tolist() for k, v in learned.items()},
        learned_smoothed_probs={k: v[1].tolist() for k, v in learned.items()},
        true_log_lik=true_avg_ll,
        initial_log_lik=initial_ll,
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
