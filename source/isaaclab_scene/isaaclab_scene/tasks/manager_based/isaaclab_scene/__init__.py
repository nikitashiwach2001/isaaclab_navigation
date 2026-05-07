import gymnasium as gym

from . import agents

# Both IDs point to the same config so existing training commands still work
# stage1 and stag3 
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

# Stage 4 — two independently moving obstacles (A: X axis, B: Y axis)
gym.register(
    id="IsaaclabScene-Stage4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage4_env_cfg:Stage4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
