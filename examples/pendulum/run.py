"""Training a Sokoloski-style variational Bayes filter for a stochastic pendulum.

Reimplements (a focused version of) experiment 3 from Sokoloski 2017
"Implementing a Bayes Filter in a Neural Circuit": filter the state
$(\\theta, \\dot\\theta)$ of a stochastic pendulum from Poisson population
responses with joint von Mises x Normal tuning curves. The conjugation
condition holds only approximately, so the variational machinery
(``VariationalLatentProcess`` + learned $\\rho$) is essential.

The model is :class:`~goal.models.PopulationCodeProcess`: a
:class:`~goal.models.PoissonPopulationCode` with one population tuned to both the angle and
the velocity, and an MLP transition on beliefs. The example adds the pendulum's latent family,
a transition that keeps the velocity precision positive, and the initialization from tuning
curves.

The velocity tuning curves tile $[-8, 8]$, the range the simulated velocities visit (their
s.d. is about 2.2). With a tiling of $[-4, 4]$ the total rate of the population fell off at
large $|v|$, the fitted $\\rho$ carried a velocity precision of about 2, and the update
$\\theta_{pred} - \\rho + \\Theta^\\top \\mathbf s(x)$ became improper for a broad prediction
and a silent observation (ELBO = NaN from the first steps).

Usage::

    uv run python -m examples.pendulum.run
"""

from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.geometry import (
    DifferentiableTuple,
    MultilayerPerceptron,
    PositiveDefinite,
)
from goal.geometry.exponential_family.variational import (
    conjugation_metrics,
    regress_conjugation_parameters,
)
from goal.models import (
    Normal,
    PoissonPopulationCode,
    PoissonPopulationHarmonium,
    PopulationCodeProcess,
    VonMisesProduct,
)

from ..shared import example_paths, jax_cli
from .types import (
    ModeResults,
    PendulumConfig,
    Results,
    TrainingConfig,
    TrainingHistory,
    TuningParams,
)

# Required for numerical stability of the variational filter (KL,
# log_partition_function on Normal[PositiveDefinite], REINFORCE gradients).
jax.config.update("jax_enable_x64", True)

# =====================================================================
# Local manifold / harmonium / variational / latent-process classes
# =====================================================================


type VonMisesNormalPair = DifferentiableTuple[
    VonMisesProduct | Normal[PositiveDefinite]
]


def von_mises_normal_pair() -> VonMisesNormalPair:
    """Heterogeneous exponential family: 1D circular variable (VonMises) x 1D real variable (Normal).

    Sufficient statistic is $(\\cos\\theta, \\sin\\theta, x, x^2)$; ``data_dim = 2``, ``dim = 4``.
    """
    return DifferentiableTuple((VonMisesProduct(1), Normal(1, PositiveDefinite())))


@dataclass(frozen=True)
class PendulumTransition(MultilayerPerceptron[VonMisesNormalPair, VonMisesNormalPair]):
    """MLP transition with a soft clamp on the Normal precision parameter.

    The Normal portion of the latent natural parameters has the constraint $\\theta_2 < 0$ (precision must be positive-definite). An unconstrained MLP can produce arbitrary signs, which makes the downstream ``log_partition_function`` and KL divergence return NaN. We apply $\\theta_2 \\mapsto -\\mathrm{softplus}(-\\theta_2) - \\varepsilon$ which is smooth, near-identity for $\\theta_2 \\ll 0$, and saturates strictly below zero for $\\theta_2 \\geq 0$.
    """

    precision_epsilon: float = 1e-2

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        raw = super().__call__(f_coords, v_coords)
        return jnp.concatenate(
            [
                raw[:2],
                raw[2:3],
                -jax.nn.softplus(-raw[3:4]) - self.precision_epsilon,
            ]
        )


type PendulumFilter = PopulationCodeProcess[VonMisesNormalPair, PendulumTransition]


def pendulum_filter(
    n_neurons: int, mlp_hidden_dims: tuple[int, ...], precision_epsilon: float
) -> PendulumFilter:
    """One population of ``n_neurons`` tuned to both the angle and the velocity, and the MLP transition."""
    lat = von_mises_normal_pair()
    hrm = PoissonPopulationHarmonium((n_neurons,), lat, ((0, 0), (0, 1)))
    transition = PendulumTransition(
        lat,
        lat,
        hidden_dims=mlp_hidden_dims,
        activation=jax.nn.tanh,
        precision_epsilon=precision_epsilon,
    )
    return PopulationCodeProcess(PoissonPopulationCode(hrm), transition)


def initialize_from_tuning_curves(
    code: PoissonPopulationCode[VonMisesNormalPair],
    key: Array,
    preferred_angles: Array,
    preferred_velocities: Array,
    angle_concentrations: Array,
    velocity_variances: Array,
    log_gains: Array,
    prior_velocity_variance: float,
    n_regression_samples: int,
) -> Array:
    """Construct likelihood + prior parameters from per-neuron tuning curves, then fit $\\rho$.

    Each neuron's log-rate is $\\log\\gamma_i + \\kappa_i\\cos(\\theta - \\theta_i^0) - (v - v_i^0)^2 / (2\\sigma_{v,i}^2)$, expanded into linear coefficients against $(\\cos\\theta, \\sin\\theta, v, v^2)$.
    """
    obs_params = log_gains - preferred_velocities**2 / (2.0 * velocity_variances)

    col_cos = angle_concentrations * jnp.cos(preferred_angles)
    col_sin = angle_concentrations * jnp.sin(preferred_angles)
    col_v_lin = preferred_velocities / velocity_variances
    col_v_quad = -1.0 / (2.0 * velocity_variances)
    # The interaction is stored term by term: the von Mises block, then the
    # normal block, each a neurons x 2 matrix
    int_params = jnp.concatenate(
        [
            jnp.stack([col_cos, col_sin], axis=1).ravel(),
            jnp.stack([col_v_lin, col_v_quad], axis=1).ravel(),
        ]
    )
    prior_vm = jnp.zeros(2)  # uniform on circle
    prior_normal = jnp.array([0.0, -1.0 / (2.0 * prior_velocity_variance)])
    prior_nat = jnp.concatenate([prior_vm, prior_normal])

    # Fit rho at zero rho, where the prior is prior_nat, then keep the prior
    def params_at(lat_params: Array, rho: Array) -> Array:
        hrm_params = code.gen_hrm.join_coords(obs_params, int_params, lat_params)
        return code.join_coords(hrm_params, rho)

    zero_rho = jnp.zeros(code.cnj_fun_man.dim)
    rho, _, _, _ = regress_conjugation_parameters(
        code, key, params_at(prior_nat, zero_rho), n_regression_samples
    )
    return params_at(prior_nat - rho, rho)


# =====================================================================
# Pendulum SDE + observation model
# =====================================================================


def simulate_pendulum_trajectory(
    key: Array,
    init_angle: Array,
    init_velocity: Array,
    n_steps: int,
    dt: float,
    omega: float,
    damping: float,
    sigma: float,
) -> Array:
    """Euler-Maruyama simulation of a stochastic pendulum.

    Returns ``(n_steps, 2)`` array of ``[angle, velocity]``; angles wrapped to $(-\\pi, \\pi]$.
    """

    def step(
        carry: tuple[Array, Array], noise: Array
    ) -> tuple[tuple[Array, Array], Array]:
        theta, vel = carry
        theta_new = (theta + vel * dt + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        vel_new = (
            vel
            + (-(omega**2) * jnp.sin(theta) - damping * vel) * dt
            + sigma * jnp.sqrt(dt) * noise
        )
        return (theta_new, vel_new), jnp.stack([theta_new, vel_new])

    noise = jax.random.normal(key, shape=(n_steps,))
    _, traj = jax.lax.scan(step, (init_angle, init_velocity), noise)
    return traj


def generate_poisson_observations(
    key: Array, latent_trajectory: Array, tuning: TuningParams
) -> Array:
    """Simulate Poisson population responses to a latent trajectory.

    Each neuron's rate at $(\\theta, v)$ is $\\exp(\\log\\gamma + \\kappa\\cos(\\theta - \\theta^0) - (v - v^0)^2 / (2\\sigma_v^2))$. Returns ``(T, n_neurons)`` counts.
    """
    pref_a = jnp.array(tuning["preferred_angles"])
    pref_v = jnp.array(tuning["preferred_velocities"])
    kappas = jnp.array(tuning["angle_concentrations"])
    sigmas = jnp.array(tuning["velocity_variances"])
    log_gains = jnp.array(tuning["log_gains"])

    def rate_at(state: Array) -> Array:
        theta, v = state[0], state[1]
        return jnp.exp(
            log_gains
            + kappas * jnp.cos(theta - pref_a)
            - (v - pref_v) ** 2 / (2.0 * sigmas)
        )

    rates = jax.vmap(rate_at)(latent_trajectory)
    return jax.random.poisson(key, rates).astype(jnp.float64)


# =====================================================================
# Ground-truth tuning + extraction
# =====================================================================


def make_ground_truth_tuning(
    n_angle_bins: int,
    n_velocity_bins: int,
    velocity_range: float,
    angle_kappa: float,
    velocity_variance: float,
    log_gain: float,
) -> TuningParams:
    """Tile $(\\theta, v)$ space with ``n_angle_bins * n_velocity_bins`` neurons."""
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, n_angle_bins, endpoint=False)
    velocities = jnp.linspace(-velocity_range, velocity_range, n_velocity_bins)

    pref_a, pref_v = jnp.meshgrid(angles, velocities, indexing="xy")
    pref_a = pref_a.ravel()
    pref_v = pref_v.ravel()

    n_neurons = pref_a.shape[0]
    return TuningParams(
        preferred_angles=pref_a.tolist(),
        preferred_velocities=pref_v.tolist(),
        angle_concentrations=jnp.full((n_neurons,), angle_kappa).tolist(),
        velocity_variances=jnp.full((n_neurons,), velocity_variance).tolist(),
        log_gains=jnp.full((n_neurons,), log_gain).tolist(),
    )


def extract_learned_tuning(model: PendulumFilter, params: Array) -> TuningParams:
    """Read tuning parameters back out of trained likelihood parameters."""
    _, ems_lkl, _, _ = model.split_coords(params)
    obs_params, int_params = model.ems_hrm.gen_hrm.lkl_fun_man.split_coords(ems_lkl)

    # The interaction is stored term by term: the von Mises block, then the normal
    # block, each a neurons x 2 matrix
    vm_mat, n_mat = int_params.reshape(2, -1, 2)
    col_cos, col_sin = vm_mat[:, 0], vm_mat[:, 1]
    col_v_lin, col_v_quad = n_mat[:, 0], n_mat[:, 1]

    kappas = jnp.sqrt(col_cos**2 + col_sin**2)
    pref_a = jnp.arctan2(col_sin, col_cos)
    sigmas = -1.0 / (2.0 * col_v_quad)
    pref_v = col_v_lin * sigmas
    log_gains = obs_params + pref_v**2 / (2.0 * sigmas)

    return TuningParams(
        preferred_angles=pref_a.tolist(),
        preferred_velocities=pref_v.tolist(),
        angle_concentrations=kappas.tolist(),
        velocity_variances=sigmas.tolist(),
        log_gains=log_gains.tolist(),
    )


def belief_to_state(model: PendulumFilter, beliefs: Array) -> tuple[Array, Array]:
    """Convert natural-parameter beliefs to $(\\hat\\theta, \\hat{\\dot\\theta})$ point estimates."""
    lat = model.lat_man

    def one(b: Array) -> Array:
        means = lat.to_mean(b)
        vm_means, n_means = lat.split_coords(means)
        theta_hat = jnp.arctan2(vm_means[1], vm_means[0])
        v_hat = n_means[0]
        return jnp.stack([theta_hat, v_hat])

    states = jax.vmap(one)(beliefs)
    return states[:, 0], states[:, 1]


# =====================================================================
# Training
# =====================================================================


def train_mode(
    key: Array,
    model: PendulumFilter,
    init_params: Array,
    train_trajectories: Array,
    mode: str,
    n_steps: int,
    learning_rate: float,
    batch_size: int,
    n_mc_samples: int,
    n_conj_samples: int,
    conj_weight: float,
    conj_warmup_steps: int,
    log_interval: int,
) -> tuple[Array, TrainingHistory]:
    """Train one mode: ``"free"`` (vanilla ELBO) or ``"regularized"`` (ELBO + ``weight_t * Var[r]`` penalty).

    In regularized mode the per-step weight ramps linearly from 0 to ``conj_weight`` over ``conj_warmup_steps``, then holds. Lets the model find good likelihood parameters before the conjugation penalty starts squeezing $\\rho$ toward a sub-optimal interior solution.
    """
    use_conj_penalty = mode == "regularized"
    n_trajectories = train_trajectories.shape[0]

    print(f"\n{'=' * 60}")
    mode_label = (
        f"REGULARIZED (conj_weight={conj_weight}, warmup={conj_warmup_steps})"
        if use_conj_penalty
        else "LEARNABLE rho"
    )
    print(f"Training [{mode_label}]")
    print(f"{'=' * 60}")

    # Global-norm clipping keeps the transition MLP and likelihood interaction
    # from overshooting into regions where the recognition Normal becomes
    # non-PD (\\theta_2 \\geq 0) and downstream log_partition_function NaNs.
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    opt_state = optimizer.init(init_params)

    def loss_fn_free(p: Array, lkey: Array, batch: Array) -> tuple[Array, Array]:
        elbo = model.mean_elbo(lkey, p, batch, n_mc_samples)
        return -elbo, elbo

    def loss_fn_reg(
        p: Array, lkey: Array, batch: Array, weight: Array
    ) -> tuple[Array, tuple[Array, Array]]:
        elbo_key, conj_key = jax.random.split(lkey)
        elbo = model.mean_elbo(elbo_key, p, batch, n_mc_samples)

        prior_params, ems_lkl, rho, _ = model.split_coords(p)
        ems_hrm = model.ems_hrm
        ems_full = model.emission_params(ems_lkl, rho, prior_params)
        z_sg = jax.lax.stop_gradient(
            ems_hrm.prr_man.sample(conj_key, prior_params, n_conj_samples)
        )
        r_vals = jax.vmap(lambda z: ems_hrm.conjugation_residual(ems_full, z))(z_sg)
        conj_var = jnp.var(r_vals)
        return -elbo + weight * conj_var, (elbo, conj_var)

    def step_free(carry: Any, _: None) -> tuple[Any, Array]:
        p, opt_st, k = carry
        k, next_k, batch_k = jax.random.split(k, 3)
        batch = train_trajectories[
            jax.random.choice(batch_k, n_trajectories, shape=(batch_size,))
        ]
        (_, elbo), grads = jax.value_and_grad(loss_fn_free, has_aux=True)(
            p, next_k, batch
        )
        updates, opt_st = optimizer.update(grads, opt_st, p)
        p = optax.apply_updates(p, updates)
        return (p, opt_st, k), elbo

    def step_reg(carry: Any, _: None) -> tuple[Any, Array]:
        p, opt_st, k, weight = carry
        k, next_k, batch_k = jax.random.split(k, 3)
        batch = train_trajectories[
            jax.random.choice(batch_k, n_trajectories, shape=(batch_size,))
        ]
        (_, (elbo, conj_var)), grads = jax.value_and_grad(loss_fn_reg, has_aux=True)(
            p, next_k, batch, weight
        )
        updates, opt_st = optimizer.update(grads, opt_st, p)
        p = optax.apply_updates(p, updates)
        return (p, opt_st, k, weight), jnp.stack([elbo, conj_var])

    n_chunks = max(1, n_steps // log_interval)
    chunk_len = n_steps // n_chunks

    elbos_hist: list[float] = []
    var_cr_hist: list[float] = []
    r_sq_hist: list[float] = []
    rho_norm_hist: list[float] = []

    params = init_params
    train_key = key

    for chunk in range(n_chunks):
        train_key, k = jax.random.split(train_key)
        if use_conj_penalty:
            # Ramp: weight at the midpoint of this chunk. Cheap per-chunk
            # piecewise-constant approximation to a per-step linear ramp;
            # avoids threading a step counter through the scan.
            chunk_mid_step = (chunk + 0.5) * chunk_len
            ramp_frac = min(1.0, chunk_mid_step / max(conj_warmup_steps, 1))
            weight_t = jnp.array(conj_weight * ramp_frac)
            (params, opt_state, _, _), out_reg = jax.lax.scan(
                step_reg, (params, opt_state, k, weight_t), None, length=chunk_len
            )
            chunk_elbos = jnp.take(out_reg, jnp.array(0), axis=1)
            elbos_hist.extend(chunk_elbos.tolist())
        else:
            (params, opt_state, _), out_free = jax.lax.scan(
                step_free, (params, opt_state, k), None, length=chunk_len
            )
            elbos_hist.extend(out_free.tolist())

        train_key, metrics_key = jax.random.split(train_key)
        prior_params, ems_lkl, rho, _ = model.split_coords(params)
        ems_full = model.emission_params(ems_lkl, rho, prior_params)
        var_f, _, r_sq = conjugation_metrics(
            model.ems_hrm, metrics_key, ems_full, n_samples=n_conj_samples
        )
        var_cr_hist.append(float(var_f))
        r_sq_hist.append(float(r_sq))
        rho_norm_hist.append(float(jnp.linalg.norm(rho)))

        weight_disp = (
            f" w_cr={conj_weight * min(1.0, ((chunk + 1) * chunk_len) / max(conj_warmup_steps, 1)):.2f}"
            if use_conj_penalty
            else ""
        )
        msg = (
            f"  chunk {chunk + 1}/{n_chunks}: "
            + f"ELBO={elbos_hist[-1]:.2f} "
            + f"Var[CR]={var_cr_hist[-1]:.3f} "
            + f"R^2={r_sq_hist[-1]:.3f} "
            + f"||rho||={rho_norm_hist[-1]:.3f}"
            + weight_disp
        )
        print(msg)

    history = TrainingHistory(
        elbos=elbos_hist,
        var_cr=var_cr_hist,
        r_squared=r_sq_hist,
        rho_norms=rho_norm_hist,
    )
    return params, history


# =====================================================================
# Main
# =====================================================================


def main(**overrides: Any) -> None:
    """Run pendulum training. ``**overrides`` patch any default ``TrainingConfig`` field, useful for sweep drivers (``main(learning_rate=1e-3, conj_weight=50.0)``)."""
    jax_cli()
    paths = example_paths(__file__)

    pendulum_cfg = PendulumConfig(
        omega=2.0,
        damping=0.3,
        sigma=1.5,
        dt=0.05,
        init_angle=0.5,
        init_velocity=0.0,
    )

    train_cfg = TrainingConfig(
        n_neurons=48,  # 8 angle x 6 velocity bins
        n_train_steps=3000,
        n_test_steps=120,
        trajectory_length=80,
        batch_size=16,
        n_trajectories=128,
        learning_rate=5e-4,
        n_mc_samples=16,
        n_conj_samples=512,
        conj_weight=30.0,
        conj_warmup_steps=500,
        mlp_hidden_dims=[64, 64],
        seed=0,
    )

    # Apply any sweep-driver overrides
    for k, v in overrides.items():
        if k not in train_cfg:
            raise KeyError(f"Unknown TrainingConfig field: {k}")
        train_cfg[k] = v

    key = jax.random.PRNGKey(train_cfg["seed"])
    keys = jax.random.split(key, 8)

    # ---- Ground-truth tuning ----
    n_angle_bins = 8
    n_velocity_bins = train_cfg["n_neurons"] // n_angle_bins
    gt_tuning = make_ground_truth_tuning(
        n_angle_bins=n_angle_bins,
        n_velocity_bins=n_velocity_bins,
        velocity_range=8.0,
        angle_kappa=1.5,
        velocity_variance=4.0,
        log_gain=float(jnp.log(2.0)),
    )

    # ---- Simulate training trajectories ----
    print("Simulating training trajectories...")
    init_thetas = jax.random.uniform(
        keys[0],
        shape=(train_cfg["n_trajectories"],),
        minval=-jnp.pi,
        maxval=jnp.pi,
    )
    traj_keys = jax.random.split(keys[1], train_cfg["n_trajectories"])

    def simulate_traj(k: Array, init_theta: Array) -> Array:
        return simulate_pendulum_trajectory(
            k,
            init_angle=init_theta,
            init_velocity=jnp.array(pendulum_cfg["init_velocity"]),
            n_steps=train_cfg["trajectory_length"],
            dt=pendulum_cfg["dt"],
            omega=pendulum_cfg["omega"],
            damping=pendulum_cfg["damping"],
            sigma=pendulum_cfg["sigma"],
        )

    train_latents = jax.vmap(simulate_traj)(traj_keys, init_thetas)

    obs_keys = jax.random.split(keys[2], train_cfg["n_trajectories"])
    train_observations = jax.vmap(
        lambda k, lat: generate_poisson_observations(k, lat, gt_tuning)
    )(obs_keys, train_latents)

    # ---- Test trajectory for visualization ----
    print("Simulating test trajectory...")
    test_init_theta = jax.random.uniform(keys[3], minval=-jnp.pi, maxval=jnp.pi)
    test_latents = simulate_pendulum_trajectory(
        keys[4],
        init_angle=test_init_theta,
        init_velocity=jnp.array(pendulum_cfg["init_velocity"]),
        n_steps=train_cfg["n_test_steps"],
        dt=pendulum_cfg["dt"],
        omega=pendulum_cfg["omega"],
        damping=pendulum_cfg["damping"],
        sigma=pendulum_cfg["sigma"],
    )
    test_observations = generate_poisson_observations(keys[5], test_latents, gt_tuning)

    # ---- Initialize model from ground-truth tuning curves ----
    print("Initializing PendulumFilter from ground-truth tuning curves...")
    model = pendulum_filter(
        train_cfg["n_neurons"],
        tuple(train_cfg["mlp_hidden_dims"]),
        precision_epsilon=1e-2,
    )

    pref_a = jnp.array(gt_tuning["preferred_angles"])
    pref_v = jnp.array(gt_tuning["preferred_velocities"])
    kappas = jnp.array(gt_tuning["angle_concentrations"])
    sigmas = jnp.array(gt_tuning["velocity_variances"])
    log_gains = jnp.array(gt_tuning["log_gains"])

    ems_full = initialize_from_tuning_curves(
        model.ems_hrm,
        keys[6],
        preferred_angles=pref_a,
        preferred_velocities=pref_v,
        angle_concentrations=kappas,
        velocity_variances=sigmas,
        log_gains=log_gains,
        prior_velocity_variance=5.0,
        n_regression_samples=4000,
    )
    ems_lkl = model.ems_hrm.likelihood_function(ems_full)
    _, rho = model.ems_hrm.split_coords(ems_full)
    prior_part = model.ems_hrm.conjugated_prior_params(ems_full)
    trns_params = model.trn_map.glorot_initialize(keys[7])

    init_params = model.join_coords(prior_part, ems_lkl, rho, trns_params)
    dim_msg = (
        f"Model dim: {model.dim}"
        + f" (prior={model.lat_man.dim},"
        + f" ems_lkl={model.ems_hrm.gen_hrm.lkl_fun_man.dim},"
        + f" rho={model.ems_hrm.cnj_fun_man.dim},"
        + f" trns={model.trn_map.dim})"
    )
    print(dim_msg)

    # ---- Train both modes and filter the test trajectory ----
    mode_results: dict[str, ModeResults] = {}
    train_key = keys[7]

    for mode in ("free", "regularized"):
        train_key, k = jax.random.split(train_key)
        trained_params, history = train_mode(
            k,
            model=model,
            init_params=init_params,
            train_trajectories=train_observations,
            mode=mode,
            n_steps=train_cfg["n_train_steps"],
            learning_rate=train_cfg["learning_rate"],
            batch_size=train_cfg["batch_size"],
            n_mc_samples=train_cfg["n_mc_samples"],
            n_conj_samples=train_cfg["n_conj_samples"],
            conj_weight=train_cfg["conj_weight"],
            conj_warmup_steps=train_cfg["conj_warmup_steps"],
            log_interval=max(1, train_cfg["n_train_steps"] // 15),
        )

        train_key, filter_key = jax.random.split(train_key)
        beliefs, _ = model.filter(
            filter_key,
            trained_params,
            test_observations,
            n_mc_samples=train_cfg["n_mc_samples"],
        )
        filt_angles, filt_velocities = belief_to_state(model, beliefs)

        mode_results[mode] = ModeResults(
            history=history,
            final_elbo=history["elbos"][-1],
            final_var_cr=history["var_cr"][-1],
            final_r_squared=history["r_squared"][-1],
            final_rho_norm=history["rho_norms"][-1],
            learned_tuning=extract_learned_tuning(model, trained_params),
            filtered_angles=filt_angles.tolist(),
            filtered_velocities=filt_velocities.tolist(),
        )

    out = Results(
        models=mode_results,
        ground_truth_tuning=gt_tuning,
        pendulum_config=pendulum_cfg,
        training_config=train_cfg,
        true_angles=test_latents[:, 0].tolist(),
        true_velocities=test_latents[:, 1].tolist(),
        n_test_steps=train_cfg["n_test_steps"],
    )

    paths.save_analysis(out)
    print(f"\nSaved analysis to {paths.analysis_path}")


if __name__ == "__main__":
    main()
