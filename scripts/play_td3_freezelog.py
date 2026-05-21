# scripts/play_td3_freezelog.py
#
# Freeze-event diagnostic. Runs a normal eval but, whenever the robot's speed
# stays below a threshold for N consecutive steps, logs the exact state:
# goal distance, nearest wall distance, nearest cylinder distance + whether
# that cylinder is approaching or receding, robot speed, episode step.
# Each freeze is tagged with the episode's eventual outcome so we can see
# whether freezes recover or end in timeout.
#
# Output: <checkpoint_dir>/freeze_events.csv  +  a printed summary.

import argparse
import csv
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Freeze-event diagnostic eval for a TD3 policy.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--eval_episodes", type=int, default=1500)
parser.add_argument("--use_gru", action="store_true", default=False)
parser.add_argument("--use_conv", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--freeze_speed", type=float, default=0.05,
                    help="Robot speed (m/s) below which a step counts as 'frozen'.")
parser.add_argument("--freeze_steps", type=int, default=15,
                    help="Consecutive slow steps to register a freeze event (~0.5s at 30Hz).")
parser.add_argument("--spin_thresh", type=float, default=0.30,
                    help="Mean yaw rate (rad/s) over the freeze window above which the "
                         "freeze is classed 'spinning' (reorienting) rather than 'stopped'.")

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


def get_env_buffer(env, name, num_envs, device):
    return getattr(env.unwrapped, name, torch.zeros(num_envs, dtype=torch.bool, device=device))


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
    # Actor-only load — eval runs the lidar-only actor; works for both normal
    # and asymmetric-critic checkpoints (the 562-dim privileged critic is skipped).
    agent.load_actor_only(args_cli.checkpoint)
    agent.actor.eval()
    print(f"[INFO] Loaded {args_cli.checkpoint}  seed={args_cli.seed}")
    print(f"[INFO] freeze_speed={args_cli.freeze_speed}  freeze_steps={args_cli.freeze_steps}")

    scene = env.unwrapped.scene
    wall_keys = [k for k in scene.keys() if k.startswith("inner_wall_")]
    obs_keys  = [k for k in scene.keys() if k.startswith("obstacle_")]

    episode_step  = torch.zeros(num_envs, dtype=torch.long, device=device)
    low_speed_run = torch.zeros(num_envs, dtype=torch.long, device=device)
    low_speed_ang_sum = torch.zeros(num_envs, device=device)   # accumulates |yaw rate| over the low-speed run
    freeze_logged = torch.zeros(num_envs, dtype=torch.bool, device=device)

    pending = {i: [] for i in range(num_envs)}   # freeze dicts for the in-progress episode
    freeze_events = []                            # finalized (with outcome)
    episode_count = 0

    counts = dict(success=0, coll_dynamic=0, coll_static=0, coll_boundary=0, timeout=0, tumble=0, terminated=0)

    while simulation_app.is_running():
        with torch.inference_mode():
            action = agent.select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            state = obs["policy"]
            done = terminated | truncated
            episode_step += 1

            robot = env.unwrapped.scene["robot"]
            robot_xy  = robot.data.root_pos_w[:, :2]
            robot_vel = robot.data.root_lin_vel_w[:, :2]
            robot_speed = torch.norm(robot_vel, dim=-1)
            robot_ang = torch.abs(robot.data.root_ang_vel_w[:, 2])   # yaw rate magnitude

            is_slow = robot_speed < args_cli.freeze_speed
            low_speed_run = torch.where(is_slow, low_speed_run + 1, torch.zeros_like(low_speed_run))
            low_speed_ang_sum = torch.where(is_slow, low_speed_ang_sum + robot_ang, torch.zeros_like(low_speed_ang_sum))
            freeze_logged = torch.where(~is_slow, torch.zeros_like(freeze_logged), freeze_logged)

            newly_frozen = (low_speed_run == args_cli.freeze_steps) & ~freeze_logged
            for env_id in torch.where(newly_frozen)[0].tolist():
                rxy = robot_xy[env_id]
                goal_xy = env.unwrapped.goal_pos_w[env_id]
                goal_dist = torch.norm(goal_xy - rxy).item()

                nearest_wall = min(
                    (torch.norm(scene[k].data.root_pos_w[env_id, :2] - rxy).item() for k in wall_keys),
                    default=float("nan"))

                nearest_cyl = float("inf")
                cyl_approaching = False
                for okey in obs_keys:
                    cxy = scene[okey].data.root_pos_w[env_id, :2]
                    cvel = scene[okey].data.root_lin_vel_w[env_id, :2]
                    d = torch.norm(cxy - rxy).item()
                    if d < nearest_cyl:
                        nearest_cyl = d
                        direction = (cxy - rxy) / max(d, 1e-6)
                        rel_vel = robot_vel[env_id] - cvel
                        cyl_approaching = bool((rel_vel * direction).sum().item() > 0.0)

                mean_ang = (low_speed_ang_sum[env_id] / args_cli.freeze_steps).item()
                freeze_type = "spinning" if mean_ang > args_cli.spin_thresh else "stopped"

                pending[env_id].append(dict(
                    epi=episode_count, env=env_id, step=int(episode_step[env_id].item()),
                    goal_dist=round(goal_dist, 3),
                    nearest_wall=round(nearest_wall, 3),
                    nearest_cyl=round(nearest_cyl, 3) if nearest_cyl != float("inf") else -1.0,
                    cyl_approaching=int(cyl_approaching),
                    robot_speed=round(robot_speed[env_id].item(), 4),
                    mean_ang=round(mean_ang, 4),
                    freeze_type=freeze_type,
                ))
            freeze_logged = freeze_logged | newly_frozen

            if done.any():
                done_ids = torch.where(done)[0]
                agent.reset_hidden(done_ids)

                goal_reached_buf       = get_env_buffer(env, "goal_reached_buf",       num_envs, device)
                collision_dynamic_buf  = get_env_buffer(env, "collision_dynamic_buf",  num_envs, device)
                collision_static_buf   = get_env_buffer(env, "collision_static_buf",   num_envs, device)
                collision_boundary_buf = get_env_buffer(env, "collision_boundary_buf", num_envs, device)
                collision_buf          = get_env_buffer(env, "collision_buf",          num_envs, device)
                tumble_buf             = get_env_buffer(env, "tumble_buf",             num_envs, device)

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

                    for ev in pending[env_id]:
                        ev["outcome"] = outcome
                        freeze_events.append(ev)
                    pending[env_id] = []

                    episode_step[env_id]      = 0
                    low_speed_run[env_id]     = 0
                    low_speed_ang_sum[env_id] = 0.0
                    freeze_logged[env_id]     = False

                    if episode_count >= args_cli.eval_episodes:
                        _report(args_cli.checkpoint, episode_count, counts, freeze_events)
                        env.close()
                        return

    env.close()


def _report(checkpoint, episode_count, counts, freeze_events):
    out_dir = os.path.dirname(checkpoint) or "."
    csv_path = os.path.join(out_dir, "freeze_events.csv")
    fields = ["epi", "env", "step", "goal_dist", "nearest_wall", "nearest_cyl", "cyl_approaching", "robot_speed", "mean_ang", "freeze_type", "outcome"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ev in freeze_events:
            w.writerow(ev)

    n = len(freeze_events)
    print("\n========== FREEZE DIAGNOSTIC ==========")
    print(f"Checkpoint: {checkpoint}")
    print(f"Episodes: {episode_count}   Freeze events: {n}")
    print(f"CSV written: {csv_path}")
    if n:
        near_wall   = sum(1 for e in freeze_events if e["nearest_wall"] >= 0 and e["nearest_wall"] < 0.6)
        near_cyl    = sum(1 for e in freeze_events if e["nearest_cyl"]  >= 0 and e["nearest_cyl"]  < 0.8)
        cyl_recede  = sum(1 for e in freeze_events if e["nearest_cyl"] >= 0 and e["nearest_cyl"] < 1.2 and e["cyl_approaching"] == 0)
        open_space  = sum(1 for e in freeze_events
                          if (e["nearest_wall"] < 0 or e["nearest_wall"] >= 0.6)
                          and (e["nearest_cyl"] < 0 or e["nearest_cyl"] >= 0.8))
        near_goal   = sum(1 for e in freeze_events if e["goal_dist"] < 0.5)
        spinning    = sum(1 for e in freeze_events if e["freeze_type"] == "spinning")
        stopped     = sum(1 for e in freeze_events if e["freeze_type"] == "stopped")
        from collections import Counter
        oc = Counter(e["outcome"] for e in freeze_events)
        print(f"  froze near a wall (<0.6m):        {near_wall:4d}  ({100.0*near_wall/n:.1f}%)")
        print(f"  froze near a cylinder (<0.8m):    {near_cyl:4d}  ({100.0*near_cyl/n:.1f}%)")
        print(f"   ...of which cylinder RECEDING:   {cyl_recede:4d}  (path clearing — should NOT freeze)")
        print(f"  froze in open space:              {open_space:4d}  ({100.0*open_space/n:.1f}%)")
        print(f"  froze near goal (<0.5m):          {near_goal:4d}  ({100.0*near_goal/n:.1f}%)")
        print(f"  --- freeze type (linear stalled) ---")
        print(f"  SPINNING (turning in place):      {spinning:4d}  ({100.0*spinning/n:.1f}%)  — reorienting, cosmetically slow")
        print(f"  STOPPED  (no turn, no move):      {stopped:4d}  ({100.0*stopped/n:.1f}%)  — genuine freeze")
        print(f"  freeze episode outcomes: {dict(oc)}")
    print("=======================================\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
