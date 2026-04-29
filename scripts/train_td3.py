# scripts/train_td3.py

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
parser = argparse.ArgumentParser(description="Train TD3 on Isaac Lab TurtleBot navigation task.")

parser.add_argument("--task", type=str, required=True, help="Isaac Lab task name.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs.")
parser.add_argument("--disable_fabric", action="store_true", default=False)

parser.add_argument("--total_steps", type=int, default=200_000)
parser.add_argument("--start_steps", type=int, default=5_000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--buffer_size", type=int, default=500_000)

parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--actor_lr", type=float, default=1e-4)
parser.add_argument("--critic_lr", type=float, default=1e-3)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)

parser.add_argument("--policy_noise", type=float, default=0.2)
parser.add_argument("--noise_clip", type=float, default=0.5)
parser.add_argument("--policy_delay", type=int, default=2)
parser.add_argument("--expl_noise", type=float, default=0.1)

parser.add_argument("--save_interval", type=int, default=25_000)

parser.add_argument("--run_name", type=str, default="td3_turtlebot_nav")
parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to TD3 checkpoint to load for finetuning.")

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

from td3.replay_buffer import ReplayBuffer
from td3.td3_agent import TD3Agent
from td3.noise import OUNoise


def get_env_buffer(env, name: str, num_envs: int, device) -> torch.Tensor:
    """Safely get a bool buffer from env.unwrapped."""
    return getattr(
        env.unwrapped,
        name,
        torch.zeros(num_envs, dtype=torch.bool, device=device),
    )


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

    # Patch action space for TD3 normalized action [-1, 1]
    env.action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=env.action_space.shape,
        dtype=np.float32,
    )

    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")

    obs, _ = env.reset()
    state = obs["policy"]

    num_envs = env.unwrapped.num_envs
    state_dim = state.shape[1]
    action_dim = env.action_space.shape[1]

    print(f"[INFO] num_envs: {num_envs}")
    print(f"[INFO] state_dim: {state_dim}")
    print(f"[INFO] action_dim: {action_dim}")

    # -------------------------
    # TD3
    # -------------------------
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        hidden_dim=args_cli.hidden_dim,
        actor_lr=args_cli.actor_lr,
        critic_lr=args_cli.critic_lr,
        gamma=args_cli.gamma,
        tau=args_cli.tau,
        policy_noise=args_cli.policy_noise,
        noise_clip=args_cli.noise_clip,
        policy_delay=args_cli.policy_delay,
    )

    if args_cli.load_checkpoint is not None:
        agent.load(args_cli.load_checkpoint)
        print(f"[INFO] Loaded checkpoint for finetuning: {args_cli.load_checkpoint}")

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=args_cli.buffer_size,
        device=device,
    )

    # OU noise for smoother robot exploration.
    # IMPORTANT: sample/reset is kept outside torch.inference_mode().
    ou_noise = OUNoise(
        num_envs=num_envs,
        action_dim=action_dim,
        device=device,
        sigma=args_cli.expl_noise,
    )

    # -------------------------
    # Save directory
    # -------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("logs", args_cli.run_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Saving checkpoints to: {save_dir}")

    # -------------------------
    # Episode tracking
    # -------------------------
    global_step = 0

    episode_reward_sum = torch.zeros(num_envs, device=device)
    episode_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    episode_count = 0

    success_count = 0
    collision_count = 0
    timeout_count = 0
    tumble_count = 0
    terminated_count = 0

    # Rolling/window summary counters
    window_success_count = 0
    window_collision_count = 0
    window_timeout_count = 0
    window_tumble_count = 0
    window_terminated_count = 0
    window_reward_sum = 0.0
    window_step_sum = 0
    window_start_episode = 0

    # -------------------------
    # Training loop
    # -------------------------
    while simulation_app.is_running() and global_step < args_cli.total_steps:
        # -------------------------
        # Action selection
        # -------------------------
        with torch.inference_mode():
            if global_step < args_cli.start_steps:
                # Random observe phase
                action = torch.empty((num_envs, action_dim), device=device).uniform_(-1.0, 1.0)
            else:
                # Actor action only
                action = agent.select_action(state)

        # OU noise has internal state, so it must stay outside inference_mode.
        if global_step >= args_cli.start_steps:
            noise = ou_noise.sample()
            action = torch.clamp(action + noise, -1.0, 1.0)

        # -------------------------
        # Environment step
        # -------------------------
        with torch.inference_mode():
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = next_obs["policy"]
            done = terminated | truncated

        # -------------------------
        # Terminal reward injection
        # -------------------------
        goal_reached_buf = get_env_buffer(env, "goal_reached_buf", num_envs, device)
        collision_buf = get_env_buffer(env, "collision_buf", num_envs, device)
        tumble_buf = get_env_buffer(env, "tumble_buf", num_envs, device)

        reward = reward.clone()

        reward = torch.where(
            done & goal_reached_buf,
            reward + 2500.0,
            reward,
        )

        reward = torch.where(
            done & collision_buf,
            reward - 2000.0,
            reward,
        )

        reward = torch.where(
            done & tumble_buf,
            reward - 2000.0,
            reward,
        )

        reward = torch.where(
            done & truncated,
            reward - 500.0,
            reward,
        )

        # -------------------------
        # Store transition
        # -------------------------
        replay_buffer.add(
            states=state,
            actions=action,
            rewards=reward,
            next_states=next_state,
            dones=done,
        )

        episode_reward_sum += reward
        episode_step_count += 1

        # -------------------------
        # Episode outcome accounting
        # -------------------------
        if done.any():
            done_env_ids = torch.where(done)[0]

            # Reset OU noise only for completed envs.
            # IMPORTANT: this is outside inference_mode.
            ou_noise.reset(done_env_ids)

            goal_reached_buf = get_env_buffer(env, "goal_reached_buf", num_envs, device)
            collision_buf = get_env_buffer(env, "collision_buf", num_envs, device)
            tumble_buf = get_env_buffer(env, "tumble_buf", num_envs, device)

            for env_id in done_env_ids.tolist():
                episode_count += 1

                epi_reward = episode_reward_sum[env_id].item()
                epi_steps = episode_step_count[env_id].item()

                if truncated[env_id]:
                    timeout_count += 1
                    window_timeout_count += 1

                elif goal_reached_buf[env_id]:
                    success_count += 1
                    window_success_count += 1

                elif collision_buf[env_id]:
                    collision_count += 1
                    window_collision_count += 1

                elif tumble_buf[env_id]:
                    tumble_count += 1
                    window_tumble_count += 1

                else:
                    terminated_count += 1
                    window_terminated_count += 1

                window_reward_sum += epi_reward
                window_step_sum += epi_steps

                episode_reward_sum[env_id] = 0.0
                episode_step_count[env_id] = 0

        state = next_state
        global_step += num_envs

        # -------------------------
        # Train TD3
        # -------------------------
        if len(replay_buffer) >= args_cli.batch_size and global_step >= args_cli.start_steps:
            agent.train(replay_buffer, args_cli.batch_size)

        # -------------------------
        # Save checkpoint
        # -------------------------
        if global_step % args_cli.save_interval < num_envs:
            ckpt_path = os.path.join(save_dir, f"td3_step_{global_step}.pt")
            agent.save(ckpt_path)

            window_total = (
                window_success_count + window_collision_count
                + window_timeout_count + window_tumble_count
                + window_terminated_count
            )

            if window_total > 0:
                avg_reward = window_reward_sum / window_total
                avg_steps  = window_step_sum  / window_total

                print(f"\n========== CHECKPOINT SUMMARY | step {global_step} ==========")
                print(f"Checkpoint: {ckpt_path}")
                print(f"Episodes in window: {window_total}  ({window_start_episode + 1} - {episode_count})")
                print(f"Avg reward: {avg_reward:.2f}   Avg steps: {avg_steps:.1f}")
                print(f"SUCCESS:    {window_success_count:<8} ({100.0 * window_success_count / window_total:.2f}%)")
                print(f"COLL_WALL:  {window_collision_count:<8} ({100.0 * window_collision_count / window_total:.2f}%)")
                print(f"TIMEOUT:    {window_timeout_count:<8} ({100.0 * window_timeout_count / window_total:.2f}%)")
                print(f"TUMBLE:     {window_tumble_count:<8} ({100.0 * window_tumble_count / window_total:.2f}%)")
                print(f"TERMINATED: {window_terminated_count:<8} ({100.0 * window_terminated_count / window_total:.2f}%)")
                print("=" * 58 + "\n")

                window_success_count    = 0
                window_collision_count  = 0
                window_timeout_count    = 0
                window_tumble_count     = 0
                window_terminated_count = 0
                window_reward_sum       = 0.0
                window_step_sum         = 0
                window_start_episode    = episode_count

    # -------------------------
    # Final summary and save
    # -------------------------
    total_finished = max(episode_count, 1)

    print("\n========== TRAINING SUMMARY ==========")
    print(f"Total episodes: {episode_count}")
    print(f"SUCCESS:    {success_count:<8} ({100.0 * success_count / total_finished:.2f}%)")
    print(f"COLL_WALL:  {collision_count:<8} ({100.0 * collision_count / total_finished:.2f}%)")
    print(f"TIMEOUT:    {timeout_count:<8} ({100.0 * timeout_count / total_finished:.2f}%)")
    print(f"TUMBLE:     {tumble_count:<8} ({100.0 * tumble_count / total_finished:.2f}%)")
    print(f"TERMINATED: {terminated_count:<8} ({100.0 * terminated_count / total_finished:.2f}%)")
    print("======================================\n")

    final_path = os.path.join(save_dir, "td3_final.pt")
    agent.save(final_path)
    print(f"[INFO] Saved final checkpoint: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()