# scripts/play_td3_colllog.py
#
# Collision diagnostic. Runs a normal eval; on every episode that ends in a
# DYNAMIC (cylinder) or STATIC (wall) collision, logs the geometry at the
# moment of impact, captured from a pre-step snapshot (the env auto-resets the
# instant an episode ends, so the collided state must be grabbed beforehand):
#   - robot speed, last action
#   - nearest cylinder: distance, closing speed, approach zone (front/side/behind)
#   - nearest wall: distance, zone
#   - goal distance
#
# Output: <checkpoint_dir>/collision_events.csv  +  a printed summary.

import argparse
import csv
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collision (dynamic + static) diagnostic for a TD3 policy.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--eval_episodes", type=int, default=1500)
parser.add_argument("--use_gru", action="store_true", default=False)
parser.add_argument("--use_conv", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--fast_speed", type=float, default=0.12,
                    help="Robot speed (m/s) above which it counts as 'moving fast' at impact.")

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


def yaw_from_quat(quat):
    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def zone(bearing_deg):
    """front: obstacle ahead of the robot's heading; side: ~abeam; behind: to the rear."""
    if bearing_deg < 60.0:
        return "front"
    if bearing_deg < 120.0:
        return "side"
    return "behind"


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

    scene = env.unwrapped.scene
    wall_keys = [k for k in scene.keys() if k.startswith("inner_wall_")]
    obs_keys  = [k for k in scene.keys() if k.startswith("obstacle_")]
    wall_xy = {k: scene[k].data.root_pos_w[:, :2].clone() for k in wall_keys}   # walls are static

    episode_count = 0
    counts = dict(success=0, coll_dynamic=0, coll_static=0, coll_boundary=0, timeout=0, tumble=0, terminated=0)
    events = []

    while simulation_app.is_running():
        with torch.inference_mode():
            action = agent.select_action(state)

            robot = env.unwrapped.scene["robot"]
            # Pre-step snapshot — collisions happen inside env.step, after which
            # the collided envs are auto-reset. Grab the impact state now.
            pre_robot_xy  = robot.data.root_pos_w[:, :2].clone()
            pre_robot_vel = robot.data.root_lin_vel_w[:, :2].clone()
            pre_robot_yaw = yaw_from_quat(robot.data.root_quat_w).clone()
            pre_robot_spd = torch.norm(pre_robot_vel, dim=-1)
            pre_cyl = {k: (scene[k].data.root_pos_w[:, :2].clone(),
                           scene[k].data.root_lin_vel_w[:, :2].clone()) for k in obs_keys}
            pre_goal = env.unwrapped.goal_pos_w.clone()

            obs, reward, terminated, truncated, _ = env.step(action)
            state = obs["policy"]
            done = terminated | truncated

            if done.any():
                done_ids = torch.where(done)[0]
                agent.reset_hidden(done_ids)

                goal_reached_buf       = get_env_buffer(env, "goal_reached_buf", num_envs, device)
                collision_dynamic_buf  = get_env_buffer(env, "collision_dynamic_buf", num_envs, device)
                collision_static_buf   = get_env_buffer(env, "collision_static_buf", num_envs, device)
                collision_boundary_buf = get_env_buffer(env, "collision_boundary_buf", num_envs, device)
                collision_buf          = get_env_buffer(env, "collision_buf", num_envs, device)
                tumble_buf             = get_env_buffer(env, "tumble_buf", num_envs, device)

                for env_id in done_ids.tolist():
                    episode_count += 1

                    is_dyn   = bool(collision_dynamic_buf[env_id])
                    is_bound = bool(collision_boundary_buf[env_id])
                    is_stat  = (not is_dyn) and (not is_bound) and \
                               bool(collision_static_buf[env_id] or collision_buf[env_id])

                    if goal_reached_buf[env_id]:
                        counts["success"] += 1
                    elif is_dyn:
                        counts["coll_dynamic"] += 1
                    elif is_bound:
                        counts["coll_boundary"] += 1
                    elif is_stat:
                        counts["coll_static"] += 1
                    elif tumble_buf[env_id]:
                        counts["tumble"] += 1
                    elif truncated[env_id]:
                        counts["timeout"] += 1
                    else:
                        counts["terminated"] += 1

                    if is_dyn or is_stat:
                        rxy  = pre_robot_xy[env_id]
                        ryaw = pre_robot_yaw[env_id]

                        ncyl_d, ncyl_closing, ncyl_bear = float("inf"), 0.0, 180.0
                        for k in obs_keys:
                            cxy, cvel = pre_cyl[k][0][env_id], pre_cyl[k][1][env_id]
                            d = torch.norm(cxy - rxy).item()
                            if d < ncyl_d:
                                ncyl_d = d
                                dirv = (cxy - rxy) / max(d, 1e-6)
                                relv = pre_robot_vel[env_id] - cvel
                                ncyl_closing = (relv * dirv).sum().item()
                                to = torch.atan2(cxy[1] - rxy[1], cxy[0] - rxy[0])
                                b = torch.atan2(torch.sin(to - ryaw), torch.cos(to - ryaw))
                                ncyl_bear = abs(b.item()) * 180.0 / math.pi

                        nwall_d, nwall_bear = float("inf"), 180.0
                        for k in wall_keys:
                            wxy = wall_xy[k][env_id]
                            d = torch.norm(wxy - rxy).item()
                            if d < nwall_d:
                                nwall_d = d
                                to = torch.atan2(wxy[1] - rxy[1], wxy[0] - rxy[0])
                                b = torch.atan2(torch.sin(to - ryaw), torch.cos(to - ryaw))
                                nwall_bear = abs(b.item()) * 180.0 / math.pi

                        goal_d = torch.norm(pre_goal[env_id] - rxy).item()
                        act = action[env_id]
                        events.append(dict(
                            epi=episode_count, env=env_id,
                            coll_type="DYNAMIC" if is_dyn else "STATIC",
                            robot_speed=round(pre_robot_spd[env_id].item(), 4),
                            action_lin=round(act[0].item(), 3),
                            action_ang=round(act[1].item(), 3),
                            goal_dist=round(goal_d, 3),
                            nearest_cyl=round(ncyl_d, 3) if ncyl_d != float("inf") else -1.0,
                            cyl_closing=round(ncyl_closing, 4),
                            cyl_zone=zone(ncyl_bear),
                            nearest_wall=round(nwall_d, 3) if nwall_d != float("inf") else -1.0,
                            wall_zone=zone(nwall_bear),
                        ))

                    if episode_count >= args_cli.eval_episodes:
                        _report(args_cli.checkpoint, episode_count, counts, events, args_cli.fast_speed)
                        env.close()
                        return

    env.close()


def _report(checkpoint, episode_count, counts, events, fast_speed):
    out_dir = os.path.dirname(checkpoint) or "."
    csv_path = os.path.join(out_dir, "collision_events.csv")
    fields = ["epi", "env", "coll_type", "robot_speed", "action_lin", "action_ang",
              "goal_dist", "nearest_cyl", "cyl_closing", "cyl_zone", "nearest_wall", "wall_zone"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in events:
            w.writerow(e)

    dyn  = [e for e in events if e["coll_type"] == "DYNAMIC"]
    stat = [e for e in events if e["coll_type"] == "STATIC"]
    from collections import Counter

    print("\n========== COLLISION DIAGNOSTIC ==========")
    print(f"Checkpoint: {checkpoint}")
    print(f"Episodes: {episode_count}")
    print(f"SUCCESS {counts['success']}  DYN {counts['coll_dynamic']}  STATIC {counts['coll_static']}  "
          f"BOUNDARY {counts['coll_boundary']}  TIMEOUT {counts['timeout']}  TUMBLE {counts['tumble']}")
    print(f"CSV: {csv_path}")

    if dyn:
        n = len(dyn)
        zc = Counter(e["cyl_zone"] for e in dyn)
        fast    = sum(1 for e in dyn if e["robot_speed"] > fast_speed)
        closing = sum(1 for e in dyn if e["cyl_closing"] > 0.0)
        pinned  = sum(1 for e in dyn if 0 <= e["nearest_wall"] < 0.5)
        print(f"\n-- DYNAMIC (cylinder) collisions: {n} --")
        for z in ("front", "side", "behind"):
            v = zc.get(z, 0)
            print(f"  cylinder approached from {z:6s}: {v:3d}  ({100.0*v/n:.1f}%)")
        print(f"  robot was CLOSING on the cylinder:  {closing:3d}  ({100.0*closing/n:.1f}%)  — robot drove in (avoidable)")
        print(f"  robot moving FAST (>{fast_speed}) at hit:{fast:3d}  ({100.0*fast/n:.1f}%)")
        print(f"  robot pinned near a wall (<0.5m):   {pinned:3d}  ({100.0*pinned/n:.1f}%)  — trapped geometry")

    if stat:
        n = len(stat)
        zc = Counter(e["wall_zone"] for e in stat)
        fast  = sum(1 for e in stat if e["robot_speed"] > fast_speed)
        dodge = sum(1 for e in stat if 0 <= e["nearest_cyl"] < 0.8)
        print(f"\n-- STATIC (wall) collisions: {n} --")
        for z in ("front", "side", "behind"):
            v = zc.get(z, 0)
            print(f"  wall in {z:6s}: {v:3d}  ({100.0*v/n:.1f}%)")
        print(f"  robot moving FAST (>{fast_speed}) at hit:{fast:3d}  ({100.0*fast/n:.1f}%)")
        print(f"  cylinder nearby (<0.8m):            {dodge:3d}  ({100.0*dodge/n:.1f}%)  — dodged a cylinder INTO the wall")
    print("==========================================\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
