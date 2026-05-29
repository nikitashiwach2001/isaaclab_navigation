import gymnasium as gym

from . import agents

# Stage 1 — open arena, four outer walls only, no obstacles (PPO warm-up)
gym.register(
    id="IsaaclabScene-Stage1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage1_env_cfg:Stage1EnvCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage1PPORunnerCfg",
    },
)

# Stage 1 PPO — same arena, but env rewards include terminal bonus/penalty for RSL-RL
# Uses LSTM architecture matching Stage 4 so the checkpoint can be resumed directly
# gym.register(
#     id="IsaaclabScene-Stage1-PPO-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.stage1_env_cfg:Stage1PPOEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage1PPORunnerCfg",
#     },
# )

gym.register(
    id="Template-Isaaclab-Scene-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaaclab_scene_env_cfg:BaseEnvCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# Stage 3 — pinwheel rotating obstacles (currently active stage)
gym.register(
    id="IsaaclabScene-Stage3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage3_env_cfg:Stage3EnvCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# Stage 3.1 — Stage4 maze (inner walls + static obstacles), no movement (curriculum step)
gym.register(
    id="IsaaclabScene-Stage31-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage4_env_cfg:Stage31EnvCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage4PPORunnerCfg",
    },
)

# Stage 4 — two independently moving obstacles (TD3 path: no terminal rewards in env)
gym.register(
    id="IsaaclabScene-Stage4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage4_env_cfg:Stage4EnvCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage4PPORunnerCfg",
    },
)

# Stage 4 random-spawn — freeze-fix variant: robot starts at a random arena
# position each episode. Same scene/reward as Stage4-v0; only the spawn differs.
gym.register(
    id="IsaaclabScene-Stage4-RandSpawn-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage4_env_cfg:Stage4RandSpawnEnvCfg",
    },
)

# Stage 4 mixed-spawn — ~50% fixed-origin, ~50% random spawn per episode.
# Combines fixed-spawn cylinder-timing sharpness with random-spawn robustness.
gym.register(
    id="IsaaclabScene-Stage4-MixedSpawn-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage4_env_cfg:Stage4MixedSpawnEnvCfg",
    },
)

# Stage 4 three-pillar — Stage 4 + a third moving pillar (simple slow motion).
# Generalization test: does the policy avoid an obstacle it never trained on?
gym.register(
    id="IsaaclabScene-Stage4-ThreePillar-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage4_env_cfg:Stage4ThreePillarEnvCfg",
    },
)

# Stage 4 PPO — same scene, but env rewards include terminal bonus/penalty for RSL-RL
# gym.register(
#     id="IsaaclabScene-Stage4-PPO-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.stage4_env_cfg:Stage4PPOEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage4PPORunnerCfg",
#     },
# )

# Stage 5 — three independently moving obstacles + one additional inner wall (TD3 path)
gym.register(
    id="IsaaclabScene-Stage5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage5_env_cfg:Stage5EnvCfg",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage5PPORunnerCfg",
    },
)

# Stage 5 mixed-spawn — ~50% fixed-origin, ~50% random spawn per episode.
# Finetune target for the random-spawn Stage 4 checkpoint.
gym.register(
    id="IsaaclabScene-Stage5-MixedSpawn-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage5_env_cfg:Stage5MixedSpawnEnvCfg",
    },
)

# Stage 5 mixed-spawn + orbit-penalty reward — same scene/events, only the
# reward is shaped to discourage spin-in-place dodging. Finetune target for
# fixing a policy whose dodge collapses into orbiting nearby obstacles.
gym.register(
    id="IsaaclabScene-Stage5-MixedSpawn-Orbit-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage5_env_cfg:Stage5MixedSpawnOrbitEnvCfg",
    },
)

# Stage 5 mixed-spawn + orbit penalty + gap-through bonus — stacks both
# shaping terms. Finetune target for producing the "best behavior" deployed
# policy: no orbiting, prefers side-weaving past obstacles.
gym.register(
    id="IsaaclabScene-Stage5-MixedSpawn-Smooth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage5_env_cfg:Stage5MixedSpawnSmoothEnvCfg",
    },
)

# Stage 6 — Stage 5 arena + five obstacles (four moving + one static blocker).
# Planner stress-test; a Stage 5 policy runs on it unchanged (same obs/reward).
gym.register(
    id="IsaaclabScene-Stage6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage6_env_cfg:Stage6EnvCfg",
    },
)

# Stage 6a — open arena (no outer walls): robot starts inside a left-side
# corridor and must exit east to reach a goal in the Stage 6 obstacle
# field. Planner eval scene only; uses the same Stage 6 obstacle motions.
gym.register(
    id="IsaaclabScene-Stage6a-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stage6_env_cfg:Stage6aEnvCfg",
    },
)

# Stage 5 PPO — same scene, but env rewards include terminal bonus/penalty for RSL-RL
# gym.register(
#     id="IsaaclabScene-Stage5-PPO-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.stage5_env_cfg:Stage5PPOEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage5PPORunnerCfg",
#     },
# )
