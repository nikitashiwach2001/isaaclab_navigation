from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoActorCriticRecurrentCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Default config — kept for Template-v0 and Stage3-v0 (small cartpole-style nets)."""
    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "cartpole_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Stage1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO-LSTM warm-up on Stage 1: open arena, four outer walls, no obstacles.

    Uses the same LSTM architecture as Stage 4 so the Stage 1 checkpoint can be
    loaded directly as a Stage 4 warm-start via --resume --checkpoint.
    """

    num_steps_per_env = 64
    max_iterations = 300
    save_interval = 50
    experiment_name = "stage1_ppo_lstm"

    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Stage4PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO-LSTM for Stage 4: 5×5 m maze + 3 inner walls + 2 moving obstacles.

    LSTM hidden state lets the policy track moving obstacle trajectories across
    timesteps using only the existing 102-dim LiDAR observation — no extra sensors.

    Batch per update = num_steps_per_env × num_envs = 64 × 256 = 16 384 transitions.
    Use task IsaaclabScene-Stage4-PPO-v0 (not Stage4-v0) so terminal rewards are in the env.
    """

    num_steps_per_env = 64
    max_iterations = 3000
    save_interval = 50
    experiment_name = "stage4_ppo"

    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.3,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Stage5PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO fine-tune for Stage 5: same maze + 4 inner walls + 3 moving obstacles.

    Intended to resume from a Stage 4 PPO checkpoint:
      python scripts/rsl_rl/train.py \\
        --task IsaaclabScene-Stage5-PPO-v0 \\
        --num_envs 256 \\
        --resume \\
        --checkpoint logs/rsl_rl/stage4_ppo/<run>/model_<iter>.pt \\
        --headless
    """

    num_steps_per_env = 64
    max_iterations = 2000
    save_interval = 50
    experiment_name = "stage5_ppo"

    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
