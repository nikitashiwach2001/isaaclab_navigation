# scripts/play_td3_watchdog.py
#
# Evaluation with a stuck-recovery WATCHDOG layered on top of the policy.
# This is an engineering band-aid, not a policy fix: when the robot's speed
# stays near-zero for too long (a freeze), the watchdog overrides the policy
# with an escape burst — turn toward the most-open lidar direction and drive
# forward — for a fixed number of steps, then hands control back. The escape
# aims at OPEN SPACE, never at the goal (the goal may be behind a wall).
#
# play_td3.py is left untouched; this is a separate eval path.

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a TD3 policy with a stuck-recovery watchdog.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--eval_episodes", type=int, default=1500)
parser.add_argument("--use_gru", action="store_true", default=False)
parser.add_argument("--use_conv", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stuck_speed", type=float, default=0.05,
                    help="Robot speed (m/s) below which a step counts as stuck.")
parser.add_argument("--stuck_steps", type=int, default=15,
                    help="Consecutive stuck steps that trigger watchdog recovery.")
parser.add_argument("--recover_steps", type=int, default=20,
                    help="Length of the escape burst once recovery triggers.")
parser.add_argument("--escape_forward", type=float, default=0.3,
                    help="Forward action[0] during recovery (-1..1; maps through the env action scale).")
parser.add_argument("--escape_turn", type=float, default=0.5,
                    help="Turn magnitude action[1] during recovery, signed toward the open side.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_scene.tasks  # noqa: F401

from td3.td3_agent import TD3Agent

LIDAR_CAP = 3.5


def get_env_buffer(env, name, num_envs, device):
    return getattr(env.unwrapped, name, torch.zeros(num_envs, dtype=torch.bool, device=device))


def lidar_open_side(env, num_envs, device):
    """Return per-env signed turn direction toward the more-open forward side.
    +1 = turn left (left-forward is clearer), -1 = turn right."""
    lidar = env.unwrapped.scene["lidar"]
    ranges = torch.norm(lidar.data.ray_hits_w - lidar.data.pos_w.unsqueeze(1), dim=-1)
    ranges = torch.nan_to_num(ranges, nan=LIDAR_CAP, posinf=LIDAR_CAP, neginf=LIDAR_CAP)
    ranges = torch.clamp(ranges, 0.0, LIDAR_CAP).reshape(num_envs, -1)
    n = ranges.shape[1]
    # forward-left and forward-right quadrants (skip the exact front and the rear)
    fl = ranges[:, n // 8: 3 * n // 8].mean(dim=1)
    fr = ranges[:, 5 * n // 8: 7 * n // 8].mean(dim=1)
    return torch.where(fl >= fr,
                       torch.ones(num_envs, device=device),
                       -torch.ones(num_envs, device=device))


def print_eval_summary(checkpoint, episode_count, counts, watchdog_episodes):
    total = max(episode_count, 1)
    pct = lambda n: f"{100.0 * n / total:.2f}%"
    print("\n========== EVALUATION SUMMARY (watchdog) ==========")
    print(f"Checkpoint: {checkpoint}")
    print(f"Total episodes: {episode_count}")
    print(f"SUCCESS:        {counts['success']:<8} ({pct(counts['success'])})")
    print(f"COLL_DYNAMIC:   {counts['coll_dynamic']:<8} ({pct(counts['coll_dynamic'])})")
    print(f"COLL_STATIC:    {counts['coll_static']:<8} ({pct(counts['coll_static'])})")
    print(f"COLL_BOUNDARY:  {counts['coll_boundary']:<8} ({pct(counts['coll_boundary'])})")
    print(f"TIMEOUT:        {counts['timeout']:<8} ({pct(counts['timeout'])})")
    print(f"TUMBLE:         {counts['tumble']:<8} ({pct(counts['tumble'])})")
    print(f"TERMINATED:     {counts['terminated']:<8} ({pct(counts['terminated'])})")
    print(f"WATCHDOG fired in: {watchdog_episodes} episodes ({pct(watchdog_episodes)})")
    print("===================================================\n")


def main():
    device = args_cli.device
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task, device=device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.action_space = spaces.Box(low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32)

    obs, _ = env.reset()
    state = obs["policy"]
    num_envs = env.unwrapped.num_envs

    agent = TD3Agent(
        state_dim=state.shape[1], action_dim=env.action_space.shape[1], device=device,
        hidden_dim=args_cli.hidden_dim, use_gru=args_cli.use_gru, use_conv=args_cli.use_conv,
    )
    agent.load_actor_only(args_cli.checkpoint)
    agent.actor.eval()
    print(f"[INFO] Loaded {args_cli.checkpoint}  seed={args_cli.seed}")
    print(f"[INFO] watchdog: stuck_speed={args_cli.stuck_speed} stuck_steps={args_cli.stuck_steps} "
          f"recover_steps={args_cli.recover_steps} escape_forward={args_cli.escape_forward} "
          f"escape_turn={args_cli.escape_turn}")

    episode_count = 0
    counts = dict(success=0, coll_dynamic=0, coll_static=0, coll_boundary=0, timeout=0, tumble=0, terminated=0)

    stuck_count   = torch.zeros(num_envs, dtype=torch.long, device=device)
    recovering    = torch.zeros(num_envs, dtype=torch.bool, device=device)
    recover_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_used_watchdog = torch.zeros(num_envs, dtype=torch.bool, device=device)
    watchdog_episodes = 0

    from collections import defaultdict
    per_goal_counts = defaultdict(lambda: defaultdict(int))

    while simulation_app.is_running():
        with torch.inference_mode():
            action = agent.select_action(state)

            # Watchdog override for any env currently in recovery.
            if recovering.any():
                turn = lidar_open_side(env, num_envs, device)
                escape = torch.stack([
                    torch.full((num_envs,), args_cli.escape_forward, device=device),
                    turn * args_cli.escape_turn,
                ], dim=-1)
                action = torch.where(recovering.unsqueeze(-1), escape, action)

            obs, reward, terminated, truncated, _ = env.step(action)
            state = obs["policy"]
            done = terminated | truncated

            robot = env.unwrapped.scene["robot"]
            robot_speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
            is_slow = robot_speed < args_cli.stuck_speed

            # Stuck tracking (only when NOT already recovering).
            stuck_count = torch.where(is_slow & ~recovering, stuck_count + 1,
                                      torch.where(recovering, stuck_count, torch.zeros_like(stuck_count)))
            newly_triggered = (stuck_count >= args_cli.stuck_steps) & ~recovering
            recovering = recovering | newly_triggered
            recover_count = torch.where(newly_triggered, torch.zeros_like(recover_count), recover_count)
            episode_used_watchdog = episode_used_watchdog | newly_triggered

            # Advance / end recovery.
            recover_count = torch.where(recovering, recover_count + 1, recover_count)
            recovery_done = recovering & (recover_count >= args_cli.recover_steps)
            recovering = recovering & ~recovery_done
            stuck_count = torch.where(recovery_done, torch.zeros_like(stuck_count), stuck_count)

            if done.any():
                done_ids = torch.where(done)[0]
                agent.reset_hidden(done_ids)

                goal_reached_buf       = get_env_buffer(env, "goal_reached_buf", num_envs, device)
                collision_buf          = get_env_buffer(env, "collision_buf", num_envs, device)
                collision_dynamic_buf  = get_env_buffer(env, "collision_dynamic_buf", num_envs, device)
                collision_static_buf   = get_env_buffer(env, "collision_static_buf", num_envs, device)
                collision_boundary_buf = get_env_buffer(env, "collision_boundary_buf", num_envs, device)
                tumble_buf             = get_env_buffer(env, "tumble_buf", num_envs, device)

                env_origins = env.unwrapped.scene.env_origins[:, :2]
                goal_local  = env.unwrapped.goal_pos_w - env_origins

                for env_id in done_ids.tolist():
                    episode_count += 1
                    if goal_reached_buf[env_id]:
                        outcome = "SUCCESS"; counts["success"] += 1
                    elif collision_dynamic_buf[env_id]:
                        outcome = "COLL_DYNAMIC"; counts["coll_dynamic"] += 1
                    elif collision_boundary_buf[env_id]:
                        outcome = "COLL_BOUNDARY"; counts["coll_boundary"] += 1
                    elif collision_static_buf[env_id] or collision_buf[env_id]:
                        outcome = "COLL_STATIC"; counts["coll_static"] += 1
                    elif tumble_buf[env_id]:
                        outcome = "TUMBLE"; counts["tumble"] += 1
                    elif truncated[env_id]:
                        outcome = "TIMEOUT"; counts["timeout"] += 1
                    else:
                        outcome = "TERMINATED"; counts["terminated"] += 1

                    if episode_used_watchdog[env_id]:
                        watchdog_episodes += 1

                    gx = round(goal_local[env_id, 0].item(), 1)
                    gy = round(goal_local[env_id, 1].item(), 1)
                    per_goal_counts[(gx, gy)]["total"] += 1
                    per_goal_counts[(gx, gy)][outcome] += 1

                    # reset per-env watchdog state for the new episode
                    stuck_count[env_id] = 0
                    recovering[env_id] = False
                    recover_count[env_id] = 0
                    episode_used_watchdog[env_id] = False

                    if episode_count >= args_cli.eval_episodes:
                        print_eval_summary(args_cli.checkpoint, episode_count, counts, watchdog_episodes)
                        print("\n========== PER-GOAL BREAKDOWN ==========")
                        print(f"{'goal (x,y)':<14} {'N':>4} {'SR':>7} {'STATIC':>7} {'DYN':>5} {'TIMEOUT':>8}")
                        for g, cs in sorted(per_goal_counts.items(),
                                            key=lambda kv: kv[1].get("SUCCESS", 0) / max(kv[1]["total"], 1)):
                            n = cs["total"]
                            sr = 100.0 * cs.get("SUCCESS", 0) / max(n, 1)
                            cs_static = 100.0 * cs.get("COLL_STATIC", 0) / max(n, 1)
                            cs_dyn = 100.0 * cs.get("COLL_DYNAMIC", 0) / max(n, 1)
                            cs_to = 100.0 * cs.get("TIMEOUT", 0) / max(n, 1)
                            print(f"({g[0]:>4.1f},{g[1]:>4.1f})  {n:>4} {sr:>6.1f}% {cs_static:>6.1f}% {cs_dyn:>4.1f}% {cs_to:>7.1f}%")
                        print("========================================\n")
                        env.close()
                        return

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
