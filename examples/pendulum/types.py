"""Type definitions for the pendulum example."""

from typing import TypedDict


class TuningParams(TypedDict):
    """Per-neuron tuning curve parameters."""

    preferred_angles: list[float]
    preferred_velocities: list[float]
    angle_concentrations: list[float]
    velocity_variances: list[float]
    log_gains: list[float]


class TrainingHistory(TypedDict):
    """Training metrics tracked during optimization."""

    elbos: list[float]
    var_cr: list[float]
    r_squared: list[float]
    rho_norms: list[float]


class ModeResults(TypedDict):
    """Results for a single training mode."""

    history: TrainingHistory
    final_elbo: float
    final_var_cr: float
    final_r_squared: float
    final_rho_norm: float
    learned_tuning: TuningParams
    filtered_angles: list[float]
    filtered_velocities: list[float]


class PendulumConfig(TypedDict):
    """Configuration for pendulum simulation."""

    omega: float
    damping: float
    sigma: float
    dt: float
    init_angle: float
    init_velocity: float


class TrainingConfig(TypedDict):
    """Configuration for training."""

    n_neurons: int
    n_train_steps: int
    n_test_steps: int
    trajectory_length: int
    batch_size: int
    n_trajectories: int
    learning_rate: float
    n_mc_samples: int
    n_conj_samples: int
    conj_weight: float
    conj_warmup_steps: int
    mlp_hidden_dims: list[int]
    seed: int


class Results(TypedDict):
    """Complete results from training."""

    models: dict[str, ModeResults]
    ground_truth_tuning: TuningParams
    pendulum_config: PendulumConfig
    training_config: TrainingConfig
    true_angles: list[float]
    true_velocities: list[float]
    n_test_steps: int
