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

# -------------------------
# Stage mode
# -------------------------
parser.add_argument(
    "--stage_mode",
    type=str,
    default="goal",
    choices=["goal", "survival"],
    help=(
        "goal = normal goal-reaching training. "
        "survival = Stage A1 local obstacle avoidance, where clean timeout means survived."
    ),
)

# -------------------------
# Terminal rewards
# -------------------------
parser.add_argument("--success_reward", type=float, default=2500.0)
parser.add_argument("--collision_penalty", type=float, default=2000.0)
parser.add_argument("--tumble_penalty", type=float, default=2000.0)
parser.add_argument("--timeout_penalty", type=float, default=300.0)

# Used only when --stage_mode survival
parser.add_argument("--survival_reward", type=float, default=300.0)

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


def pct(count: int, total: int) -> float:
    return 100.0 * count / max(total, 1)


def make_count_dict():
    """
    Keep all keys available for both training modes.

    In goal mode:
        SUCCESS and TIMEOUT are meaningful.

    In survival mode:
        SURVIVED is meaningful.
        TIMEOUT is not failure anymore.
    """
    return {
        "success": 0,
        "survived": 0,
        "coll_dynamic": 0,
        "coll_static": 0,
        "coll_boundary": 0,
        "coll_unknown": 0,
        "timeout": 0,
        "tumble": 0,
        "terminated": 0,
    }


def print_outcome_counts(counts, total):
    print(f"SUCCESS:       {counts['success']:<8} ({pct(counts['success'], total):.2f}%)")
    print(f"SURVIVED:      {counts['survived']:<8} ({pct(counts['survived'], total):.2f}%)")
    print(f"COLL_DYNAMIC:  {counts['coll_dynamic']:<8} ({pct(counts['coll_dynamic'], total):.2f}%)")
    print(f"COLL_STATIC:   {counts['coll_static']:<8} ({pct(counts['coll_static'], total):.2f}%)")
    print(f"COLL_BOUNDARY: {counts['coll_boundary']:<8} ({pct(counts['coll_boundary'], total):.2f}%)")
    print(f"COLL_UNKNOWN:  {counts['coll_unknown']:<8} ({pct(counts['coll_unknown'], total):.2f}%)")
    print(f"TIMEOUT:       {counts['timeout']:<8} ({pct(counts['timeout'], total):.2f}%)")
    print(f"TUMBLE:        {counts['tumble']:<8} ({pct(counts['tumble'], total):.2f}%)")
    print(f"TERMINATED:    {counts['terminated']:<8} ({pct(counts['terminated'], total):.2f}%)")


def print_checkpoint_summary(
    *,
    global_step,
    ckpt_path,
    window_start_episode,
    episode_count,
    window_counts,
    window_reward_sum,
    window_step_sum,
    window_env_steps,
    window_action_linear_sum,
    window_action_angular_abs_sum,
    window_reward_step_sum,
    window_done_sum,
    last_losses,
    replay_buffer,
    window_start_time,
):
    window_total = sum(window_counts.values())
    elapsed = max(time.time() - window_start_time, 1e-6)
    fps = window_env_steps / elapsed

    print(f"\n========== CHECKPOINT SUMMARY | step {global_step} ==========")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Replay buffer: {len(replay_buffer)}")
    print(f"Window env steps: {window_env_steps}")
    print(f"Training FPS: {fps:.1f}")

    if window_env_steps > 0:
        print(f"Mean reward / env-step: {window_reward_step_sum / window_env_steps:.4f}")
        print(f"Mean done / env-step: {window_done_sum / window_env_steps:.4f}")
        print(f"Mean action linear: {window_action_linear_sum / window_env_steps:.4f}")
        print(f"Mean |action angular|: {window_action_angular_abs_sum / window_env_steps:.4f}")

    print(f"Critic loss: {last_losses.get('critic_loss', 0.0):.6f}")
    print(f"Actor loss:  {last_losses.get('actor_loss', 0.0):.6f}")

    if window_total > 0:
        avg_reward = window_reward_sum / window_total
        avg_steps = window_step_sum / window_total

        print("-" * 58)
        print(f"Episodes in window: {window_total}  ({window_start_episode + 1} - {episode_count})")
        print(f"Avg episode reward: {avg_reward:.2f}")
        print(f"Avg episode steps:  {avg_steps:.1f}")

        print_outcome_counts(window_counts, window_total)
    else:
        print("-" * 58)
        print("No completed episodes in this checkpoint window.")

    print("=" * 58 + "\n")


def classify_collision_outcome(
    *,
    env_id: int,
    collision_dynamic_buf,
    collision_static_buf,
    collision_boundary_buf,
) -> str:
    if collision_dynamic_buf[env_id]:
        return "coll_dynamic"
    if collision_static_buf[env_id]:
        return "coll_static"
    if collision_boundary_buf[env_id]:
        return "coll_boundary"
    return "coll_unknown"


def main():
    device = args_cli.device

    print("\n========== TRAINING MODE ==========")
    print(f"stage_mode:        {args_cli.stage_mode}")
    print(f"success_reward:    {args_cli.success_reward}")
    print(f"survival_reward:   {args_cli.survival_reward}")
    print(f"collision_penalty: {args_cli.collision_penalty}")
    print(f"tumble_penalty:    {args_cli.tumble_penalty}")
    print(f"timeout_penalty:   {args_cli.timeout_penalty}")
    print("===================================\n")

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

    ou_noise = OUNoise(
        num_envs=num_envs,
        action_dim=action_dim,
        device=device,
        sigma=args_cli.expl_noise,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("logs", args_cli.run_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Saving checkpoints to: {save_dir}")

    global_step = 0
    episode_count = 0

    episode_reward_sum = torch.zeros(num_envs, device=device)
    episode_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    total_counts = make_count_dict()
    window_counts = make_count_dict()

    window_reward_sum = 0.0
    window_step_sum = 0
    window_start_episode = 0

    window_env_steps = 0
    window_action_linear_sum = 0.0
    window_action_angular_abs_sum = 0.0
    window_reward_step_sum = 0.0
    window_done_sum = 0.0
    window_start_time = time.time()

    last_losses = {
        "critic_loss": 0.0,
        "actor_loss": 0.0,
    }

    while simulation_app.is_running() and global_step < args_cli.total_steps:
        with torch.inference_mode():
            if global_step < args_cli.start_steps:
                action = torch.empty((num_envs, action_dim), device=device).uniform_(-1.0, 1.0)
            else:
                action = agent.select_action(state)

        if global_step >= args_cli.start_steps:
            noise = ou_noise.sample()
            action = torch.clamp(action + noise, -1.0, 1.0)

        with torch.inference_mode():
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = next_obs["policy"]
            done = terminated | truncated

        goal_reached_buf = get_env_buffer(env, "goal_reached_buf", num_envs, device)
        collision_buf = get_env_buffer(env, "collision_buf", num_envs, device)
        tumble_buf = get_env_buffer(env, "tumble_buf", num_envs, device)

        collision_dynamic_buf = get_env_buffer(env, "collision_dynamic_buf", num_envs, device)
        collision_static_buf = get_env_buffer(env, "collision_static_buf", num_envs, device)
        collision_boundary_buf = get_env_buffer(env, "collision_boundary_buf", num_envs, device)

        reward = reward.clone()

        # -------------------------------------------------
        # Terminal reward handling
        # -------------------------------------------------
        if args_cli.stage_mode == "goal":
            # Original goal-reaching behavior.
            reward = torch.where(done & goal_reached_buf, reward + args_cli.success_reward, reward)
            reward = torch.where(done & collision_buf, reward - args_cli.collision_penalty, reward)
            reward = torch.where(done & tumble_buf, reward - args_cli.tumble_penalty, reward)

            if args_cli.timeout_penalty != 0.0:
                clean_timeout = truncated & ~goal_reached_buf & ~collision_buf & ~tumble_buf
                reward = torch.where(clean_timeout, reward - args_cli.timeout_penalty, reward)

        elif args_cli.stage_mode == "survival":
            # Stage A1 behavior.
            # A clean 50-second timeout is success/survival, not failure.
            clean_survival = truncated & ~collision_buf & ~tumble_buf

            reward = torch.where(clean_survival, reward + args_cli.survival_reward, reward)
            reward = torch.where(done & collision_buf, reward - args_cli.collision_penalty, reward)
            reward = torch.where(done & tumble_buf, reward - args_cli.tumble_penalty, reward)

        else:
            raise ValueError(f"Unknown stage_mode: {args_cli.stage_mode}")

        replay_buffer.add(
            states=state,
            actions=action,
            rewards=reward,
            next_states=next_state,
            dones=done,
        )

        episode_reward_sum += reward
        episode_step_count += 1

        # Window step stats only; no per-step printing.
        window_env_steps += num_envs
        window_reward_step_sum += reward.sum().item()
        window_done_sum += done.sum().item()

        # Normalized action stats. action[:, 0] is linear command, action[:, 1] angular command.
        window_action_linear_sum += action[:, 0].sum().item()
        if action_dim > 1:
            window_action_angular_abs_sum += torch.abs(action[:, 1]).sum().item()

        if done.any():
            done_env_ids = torch.where(done)[0]
            ou_noise.reset(done_env_ids)

            goal_reached_buf = get_env_buffer(env, "goal_reached_buf", num_envs, device)
            collision_buf = get_env_buffer(env, "collision_buf", num_envs, device)
            tumble_buf = get_env_buffer(env, "tumble_buf", num_envs, device)

            collision_dynamic_buf = get_env_buffer(env, "collision_dynamic_buf", num_envs, device)
            collision_static_buf = get_env_buffer(env, "collision_static_buf", num_envs, device)
            collision_boundary_buf = get_env_buffer(env, "collision_boundary_buf", num_envs, device)

            for env_id in done_env_ids.tolist():
                episode_count += 1

                epi_reward = episode_reward_sum[env_id].item()
                epi_steps = episode_step_count[env_id].item()

                # -------------------------------------------------
                # Outcome classification
                # -------------------------------------------------
                if args_cli.stage_mode == "goal":
                    if goal_reached_buf[env_id]:
                        outcome = "success"

                    elif collision_buf[env_id]:
                        outcome = classify_collision_outcome(
                            env_id=env_id,
                            collision_dynamic_buf=collision_dynamic_buf,
                            collision_static_buf=collision_static_buf,
                            collision_boundary_buf=collision_boundary_buf,
                        )

                    elif tumble_buf[env_id]:
                        outcome = "tumble"

                    elif truncated[env_id]:
                        outcome = "timeout"

                    else:
                        outcome = "terminated"

                elif args_cli.stage_mode == "survival":
                    if collision_buf[env_id]:
                        outcome = classify_collision_outcome(
                            env_id=env_id,
                            collision_dynamic_buf=collision_dynamic_buf,
                            collision_static_buf=collision_static_buf,
                            collision_boundary_buf=collision_boundary_buf,
                        )

                    elif tumble_buf[env_id]:
                        outcome = "tumble"

                    elif truncated[env_id]:
                        # Clean 50-second episode end.
                        outcome = "survived"

                    else:
                        outcome = "terminated"

                else:
                    raise ValueError(f"Unknown stage_mode: {args_cli.stage_mode}")

                total_counts[outcome] += 1
                window_counts[outcome] += 1

                window_reward_sum += epi_reward
                window_step_sum += epi_steps

                episode_reward_sum[env_id] = 0.0
                episode_step_count[env_id] = 0

        state = next_state
        global_step += num_envs

        if len(replay_buffer) >= args_cli.batch_size and global_step >= args_cli.start_steps:
            losses = agent.train(replay_buffer, args_cli.batch_size)
            if losses is not None:
                last_losses = losses

        if global_step % args_cli.save_interval < num_envs:
            ckpt_path = os.path.join(save_dir, f"td3_step_{global_step}.pt")
            agent.save(ckpt_path)

            print_checkpoint_summary(
                global_step=global_step,
                ckpt_path=ckpt_path,
                window_start_episode=window_start_episode,
                episode_count=episode_count,
                window_counts=window_counts,
                window_reward_sum=window_reward_sum,
                window_step_sum=window_step_sum,
                window_env_steps=window_env_steps,
                window_action_linear_sum=window_action_linear_sum,
                window_action_angular_abs_sum=window_action_angular_abs_sum,
                window_reward_step_sum=window_reward_step_sum,
                window_done_sum=window_done_sum,
                last_losses=last_losses,
                replay_buffer=replay_buffer,
                window_start_time=window_start_time,
            )

            window_counts = make_count_dict()
            window_reward_sum = 0.0
            window_step_sum = 0
            window_start_episode = episode_count

            window_env_steps = 0
            window_action_linear_sum = 0.0
            window_action_angular_abs_sum = 0.0
            window_reward_step_sum = 0.0
            window_done_sum = 0.0
            window_start_time = time.time()

    total_finished = max(episode_count, 1)

    print("\n========== TRAINING SUMMARY ==========")
    print(f"Total episodes: {episode_count}")
    print_outcome_counts(total_counts, total_finished)
    print("======================================\n")

    final_path = os.path.join(save_dir, "td3_final.pt")
    agent.save(final_path)
    print(f"[INFO] Saved final checkpoint: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()