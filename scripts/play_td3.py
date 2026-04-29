# scripts/play_td3.py

import argparse
import os
import sys
import time

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


def print_eval_summary(
    checkpoint: str,
    episode_count: int,
    success_count: int,
    collision_count: int,
    timeout_count: int,
    tumble_count: int,
    terminated_count: int,
):
    total_finished = max(episode_count, 1)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Checkpoint: {checkpoint}")
    print(f"Total episodes: {episode_count}")
    print(f"SUCCESS:    {success_count:<8} ({100.0 * success_count / total_finished:.2f}%)")
    print(f"COLL_WALL:  {collision_count:<8} ({100.0 * collision_count / total_finished:.2f}%)")
    print(f"TIMEOUT:    {timeout_count:<8} ({100.0 * timeout_count / total_finished:.2f}%)")
    print(f"TUMBLE:     {tumble_count:<8} ({100.0 * tumble_count / total_finished:.2f}%)")
    print(f"TERMINATED: {terminated_count:<8} ({100.0 * terminated_count / total_finished:.2f}%)")
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

    # Patch action space to match TD3 normalized action
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

    if hasattr(env.unwrapped, "goal_pos_w"):
        print("[INFO] goal_pos_w:", env.unwrapped.goal_pos_w)

    # -------------------------
    # Evaluation buffers
    # -------------------------
    episode_reward_sum = torch.zeros(num_envs, device=device)
    episode_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_start_time = [time.time() for _ in range(num_envs)]

    episode_count = 0
    total_steps = 0

    success_count = 0
    collision_count = 0
    timeout_count = 0
    tumble_count = 0
    terminated_count = 0

    # -------------------------
    # Play / Evaluate loop
    # -------------------------
    while simulation_app.is_running():
        with torch.inference_mode():
            # Deterministic policy action, no exploration noise during evaluation
            action = agent.select_action(state)

            obs, reward, terminated, truncated, info = env.step(action)
            state = obs["policy"]

            done = terminated | truncated

            episode_reward_sum += reward
            episode_step_count += 1
            total_steps += num_envs

            if done.any():
                done_env_ids = torch.where(done)[0]

                goal_reached_buf = getattr(
                    env.unwrapped,
                    "goal_reached_buf",
                    torch.zeros(num_envs, dtype=torch.bool, device=device),
                )

                collision_buf = getattr(
                    env.unwrapped,
                    "collision_buf",
                    torch.zeros(num_envs, dtype=torch.bool, device=device),
                )

                tumble_buf = getattr(
                    env.unwrapped,
                    "tumble_buf",
                    torch.zeros(num_envs, dtype=torch.bool, device=device),
                )

                for env_id in done_env_ids.tolist():
                    episode_count += 1

                    epi_reward = episode_reward_sum[env_id].item()
                    epi_steps = episode_step_count[env_id].item()
                    epi_time = time.time() - episode_start_time[env_id]

                    if truncated[env_id]:
                        outcome = "TIMEOUT"
                        timeout_count += 1
                    elif goal_reached_buf[env_id]:
                        outcome = "SUCCESS"
                        success_count += 1
                    elif collision_buf[env_id]:
                        outcome = "COLL_WALL"
                        collision_count += 1
                    elif tumble_buf[env_id]:
                        outcome = "TUMBLE"
                        tumble_count += 1
                    else:
                        outcome = "TERMINATED"
                        terminated_count += 1

                    # These indices match your current observation layout:
                    # 0:40 lidar, 40 goal_dist_obs, 41 goal_angle_obs, 42:44 previous action
                    goal_dist_obs = state[env_id, 40].item()
                    goal_dist_real = goal_dist_obs * 7.07106781187
                    goal_angle_obs = state[env_id, 41].item()

                    print(
                        f"Epi: {episode_count:<5} "
                        f"env: {env_id:<3} "
                        f"R: {epi_reward:<8.0f} "
                        f"outcome: {outcome:<12} "
                        f"steps: {epi_steps:<6} "
                        f"steps_total: {total_steps:<8} "
                        f"time: {epi_time:<6.2f} "
                        f"goal_dist: {goal_dist_real:<6.3f} "
                        f"goal_angle: {goal_angle_obs:<6.3f}"
                    )

                    episode_reward_sum[env_id] = 0.0
                    episode_step_count[env_id] = 0
                    episode_start_time[env_id] = time.time()

                    if episode_count >= args_cli.eval_episodes:
                        print_eval_summary(
                            checkpoint=args_cli.checkpoint,
                            episode_count=episode_count,
                            success_count=success_count,
                            collision_count=collision_count,
                            timeout_count=timeout_count,
                            tumble_count=tumble_count,
                            terminated_count=terminated_count,
                        )
                        env.close()
                        return

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()