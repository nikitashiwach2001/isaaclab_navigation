import gymnasium as gym

from . import agents

# Stage 1 — open arena, four outer walls only, no obstacles (PPO warm-up)
gym.register(
    id="IsaaclabScene-Stage1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage1_env_cfg:Stage1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage1PPORunnerCfg",
    },
)

# Stage 1 PPO — same arena, but env rewards include terminal bonus/penalty for RSL-RL
# Uses LSTM architecture matching Stage 4 so the checkpoint can be resumed directly
gym.register(
    id="IsaaclabScene-Stage1-PPO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage1_env_cfg:Stage1PPOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage1PPORunnerCfg",
    },
)

gym.register(
    id="Template-Isaaclab-Scene-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaaclab_scene_env_cfg:BaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# Stage 3 — pinwheel rotating obstacles (currently active stage)
gym.register(
    id="IsaaclabScene-Stage3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage3_env_cfg:Stage3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# Stage 3.1 — Stage4 maze (inner walls + static obstacles), no movement (curriculum step)
gym.register(
    id="IsaaclabScene-Stage31-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage4_env_cfg:Stage31EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage4PPORunnerCfg",
    },
)

# Stage 4 — two independently moving obstacles (TD3 path: no terminal rewards in env)
gym.register(
    id="IsaaclabScene-Stage4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage4_env_cfg:Stage4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage4PPORunnerCfg",
    },
)

# Stage 4 PPO — same scene, but env rewards include terminal bonus/penalty for RSL-RL
gym.register(
    id="IsaaclabScene-Stage4-PPO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage4_env_cfg:Stage4PPOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage4PPORunnerCfg",
    },
)

# Stage 5 — three independently moving obstacles + one additional inner wall (TD3 path)
gym.register(
    id="IsaaclabScene-Stage5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage5_env_cfg:Stage5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage5PPORunnerCfg",
    },
)

# Stage 5 PPO — same scene, but env rewards include terminal bonus/penalty for RSL-RL
gym.register(
    id="IsaaclabScene-Stage5-PPO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage5_env_cfg:Stage5PPOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage5PPORunnerCfg",
    },
)
