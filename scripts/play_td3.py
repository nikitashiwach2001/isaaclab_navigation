# scripts/play_td3.py

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher


# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser(description="Play/evaluate trained TD3 policy in Isaac Lab.")

parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--eval_episodes", type=int, default=100)

# Isaac Lab launcher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# -------------------------
# Launch Isaac Sim first
# -------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -------------------------
# Imports after SimulationApp
# -------------------------
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_scene.tasks  # noqa: F401

from td3.td3_agent import TD3Agent


def get_env_buffer(env, name: str, num_envs: int, device) -> torch.Tensor:
    return getattr(
        env.unwrapped,
        name,
        torch.zeros(num_envs, dtype=torch.bool, device=device),
    )


def print_eval_summary(
    checkpoint: str,
    episode_count: int,
    counts: dict,
):
    total = max(episode_count, 1)
    pct = lambda n: f"{100.0 * n / total:.2f}%"

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Checkpoint: {checkpoint}")
    print(f"Total episodes: {episode_count}")
    print(f"SUCCESS:        {counts['success']:<8} ({pct(counts['success'])})")
    print(f"COLL_DYNAMIC:   {counts['coll_dynamic']:<8} ({pct(counts['coll_dynamic'])})")
    print(f"COLL_STATIC:    {counts['coll_static']:<8} ({pct(counts['coll_static'])})")
    print(f"COLL_BOUNDARY:  {counts['coll_boundary']:<8} ({pct(counts['coll_boundary'])})")
    print(f"TIMEOUT:        {counts['timeout']:<8} ({pct(counts['timeout'])})")
    print(f"TUMBLE:         {counts['tumble']:<8} ({pct(counts['tumble'])})")
    print(f"TERMINATED:     {counts['terminated']:<8} ({pct(counts['terminated'])})")
    print("========================================\n")


def main():
    device = args_cli.device

    # -------------------------
    # Env config
    # -------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)

    env.action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=env.action_space.shape,
        dtype=np.float32,
    )

    obs, _ = env.reset()
    state = obs["policy"]

    num_envs = env.unwrapped.num_envs
    state_dim = state.shape[1]
    action_dim = env.action_space.shape[1]

    # Observation layout (103 dims):
    #   0  – 89  : LiDAR scan          (90)
    #   90 – 97  : temporal sector diff (8)
    #   98       : goal distance        (1)
    #   99       : goal angle           (1)
    #   100– 101 : previous actions     (2)
    GOAL_DIST_IDX  = 98
    GOAL_ANGLE_IDX = 99
    MAX_GOAL_DIST  = 7.07106781187   # sqrt(5² + 5²)

    # -------------------------
    # TD3 Agent
    # -------------------------
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        hidden_dim=args_cli.hidden_dim,
    )

    agent.load(args_cli.checkpoint)
    agent.actor.eval()

    print("[INFO] Loaded checkpoint:", args_cli.checkpoint)
    print("[INFO] state_dim:", state_dim)
    print("[INFO] action_dim:", action_dim)
    print("[INFO] hidden_dim:", args_cli.hidden_dim)
    print("[INFO] num_envs:", num_envs)
    print("[INFO] eval_episodes:", args_cli.eval_episodes)

    # -------------------------
    # Evaluation buffers
    # -------------------------
    episode_reward_sum = torch.zeros(num_envs, device=device)
    episode_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    episode_count = 0
    total_steps   = 0

    counts = dict(
        success=0,
        coll_dynamic=0,
        coll_static=0,
        coll_boundary=0,
        timeout=0,
        tumble=0,
        terminated=0,
    )

    # -------------------------
    # Play / Evaluate loop
    # -------------------------
    while simulation_app.is_running():
        with torch.inference_mode():
            action = agent.select_action(state)

            obs, reward, terminated, truncated, _ = env.step(action)
            state = obs["policy"]

            done = terminated | truncated

            episode_reward_sum += reward
            episode_step_count += 1
            total_steps += num_envs

            if done.any():
                done_env_ids = torch.where(done)[0]

                goal_reached_buf       = get_env_buffer(env, "goal_reached_buf",       num_envs, device)
                collision_buf          = get_env_buffer(env, "collision_buf",           num_envs, device)
                collision_dynamic_buf  = get_env_buffer(env, "collision_dynamic_buf",   num_envs, device)
                collision_static_buf   = get_env_buffer(env, "collision_static_buf",    num_envs, device)
                collision_boundary_buf = get_env_buffer(env, "collision_boundary_buf",  num_envs, device)
                tumble_buf             = get_env_buffer(env, "tumble_buf",              num_envs, device)

                for env_id in done_env_ids.tolist():
                    episode_count += 1

                    epi_reward = episode_reward_sum[env_id].item()
                    epi_steps  = episode_step_count[env_id].item()

                    goal_dist_real  = state[env_id, GOAL_DIST_IDX].item()  * MAX_GOAL_DIST
                    goal_angle_norm = state[env_id, GOAL_ANGLE_IDX].item()

                    # Priority: goal_reached > collision > tumble > timeout
                    if goal_reached_buf[env_id]:
                        outcome = "SUCCESS"
                        counts["success"] += 1
                    elif collision_dynamic_buf[env_id]:
                        outcome = "COLL_DYNAMIC"
                        counts["coll_dynamic"] += 1
                    elif collision_boundary_buf[env_id]:
                        outcome = "COLL_BOUNDARY"
                        counts["coll_boundary"] += 1
                    elif collision_static_buf[env_id]:
                        outcome = "COLL_STATIC"
                        counts["coll_static"] += 1
                    elif collision_buf[env_id]:
                        outcome = "COLL_STATIC"
                        counts["coll_static"] += 1
                    elif tumble_buf[env_id]:
                        outcome = "TUMBLE"
                        counts["tumble"] += 1
                    elif truncated[env_id]:
                        outcome = "TIMEOUT"
                        counts["timeout"] += 1
                    else:
                        outcome = "TERMINATED"
                        counts["terminated"] += 1

                    print(
                        f"Epi: {episode_count:<5} "
                        f"env: {env_id:<3} "
                        f"R: {epi_reward:<8.0f} "
                        f"outcome: {outcome:<14} "
                        f"steps: {epi_steps:<6} "
                        f"goal_dist: {goal_dist_real:<6.3f} "
                        f"goal_angle: {goal_angle_norm:<6.3f}"
                    )

                    episode_reward_sum[env_id] = 0.0
                    episode_step_count[env_id] = 0

                    if episode_count >= args_cli.eval_episodes:
                        print_eval_summary(
                            checkpoint=args_cli.checkpoint,
                            episode_count=episode_count,
                            counts=counts,
                        )
                        env.close()
                        return

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
