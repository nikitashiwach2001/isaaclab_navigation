# scripts/play_td3.py

import argparse
import json
import math
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
parser.add_argument("--use_gru", action="store_true", default=False, help="Use GRU actor.")
parser.add_argument("--use_conv", action="store_true", default=False, help="Use 1D conv lidar encoder. Must match the checkpoint's architecture.")
parser.add_argument("--record_failures", action="store_true", default=False,
                    help="Record per-failure diagnostic data to JSON for offline analysis.")
parser.add_argument("--failure_log", type=str, default="failures.json",
                    help="Output JSON path used with --record_failures.")
# Isaac Lab launcher args (includes --width / --height for render resolution)
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

    # Deterministic eval: fix RNG so cylinder phase randomization and goal
    # selection give the same sequence across runs. Without this, eval-to-eval
    # SR variance is ±1.5-2% from phase sampling noise alone — masking real
    # checkpoint differences.
    torch.manual_seed(0)
    np.random.seed(0)

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

    # Observation layout (284 dims):
    #   0  – 269 : lidar_stacked (90×3)       (270)
    #   270– 277 : lidar_temporal_sector_diff  (8)
    #   278      : goal distance               (1)
    #   279      : goal angle                  (1)
    #   280– 281 : robot_velocity              (2)
    #   282– 283 : previous actions            (2)
    GOAL_DIST_IDX  = 270 + 8          # 278
    GOAL_ANGLE_IDX = 270 + 8 + 1      # 279
    MAX_GOAL_DIST  = 7.07106781187   # sqrt(5² + 5²)

    # -------------------------
    # TD3 Agent
    # -------------------------
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        hidden_dim=args_cli.hidden_dim,
        use_gru=args_cli.use_gru,
        use_conv=args_cli.use_conv,
    )

    # Actor-only load: eval runs the lidar-only actor. Skipping the critic
    # makes this work for both normal (554-critic) and asymmetric-actor-critic
    # (562-critic privileged) checkpoints — the critic is never used at eval.
    agent.load_actor_only(args_cli.checkpoint)
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

    from collections import defaultdict
    per_goal_counts = defaultdict(lambda: defaultdict(int))

    # Failure diagnostics — populated only when --record_failures is set.
    obs_keys = sorted(k for k in env.unwrapped.scene.keys() if k.startswith("obstacle_"))
    failure_records: list[dict] = []
    last_actions = torch.zeros((num_envs, action_dim), device=device)
    # Pre-step caches — Isaac Lab resets the scene in-place for done envs
    # during env.step(), so collision-time positions must be captured BEFORE
    # the step or we'd record the post-reset positions of the *next* episode.
    cached_robot_pos = None
    cached_robot_vel = None
    cached_robot_quat = None
    cached_obstacles: dict = {}
    cached_lidar_pos = None
    cached_lidar_hits = None

    # -------------------------
    # Play / Evaluate loop
    # -------------------------
    while simulation_app.is_running():
        with torch.inference_mode():
            action = agent.select_action(state)

            # DEBUG: compares lidar 0° direction (body X axis) with the actual
            # motion direction (body XY velocity). If body X is aligned with
            # forward motion, body_vel_X dominates (X/speed close to 1.0).
            # If body frame is 90° rotated, body_vel_Y dominates instead.
            if total_steps % 30 == 0:
                last_lidar = state[0, 0:90]
                closest_ray = int(last_lidar.argmin().item())
                closest_dist_m = float(last_lidar.min().item()) * 3.5
                angle_deg = closest_ray * 4.0
                robot_e = env.unwrapped.scene["robot"]
                bvel = robot_e.data.root_lin_vel_b[0]   # body-frame [vx, vy, vz]
                bvx, bvy = float(bvel[0].item()), float(bvel[1].item())
                speed = (bvx * bvx + bvy * bvy) ** 0.5
                xfrac = bvx / max(speed, 1e-6)
                print(f"[DBG] ray{closest_ray:>3}({angle_deg:>3.0f}°) d={closest_dist_m:.2f}m | "
                      f"body_vel X={bvx:+.3f} Y={bvy:+.3f} speed={speed:.3f}  "
                      f"X/speed={xfrac:+.2f}")

            if args_cli.record_failures:
                robot_pre = env.unwrapped.scene["robot"]
                cached_robot_pos = robot_pre.data.root_pos_w[:, :2].clone()
                cached_robot_vel = robot_pre.data.root_lin_vel_w[:, :2].clone()
                cached_robot_quat = robot_pre.data.root_quat_w.clone()
                cached_obstacles = {
                    okey: env.unwrapped.scene[okey].data.root_pos_w[:, :2].clone()
                    for okey in obs_keys
                }
                lidar_pre = env.unwrapped.scene["lidar"]
                cached_lidar_pos = lidar_pre.data.pos_w.clone()
                cached_lidar_hits = lidar_pre.data.ray_hits_w.clone()

            obs, reward, terminated, truncated, _ = env.step(action)
            last_actions = action.detach().clone()
            state = obs["policy"]

            done = terminated | truncated

            episode_reward_sum += reward
            episode_step_count += 1
            total_steps += num_envs

            if done.any():
                done_env_ids = torch.where(done)[0]
                agent.reset_hidden(done_env_ids)

                goal_reached_buf       = get_env_buffer(env, "goal_reached_buf",       num_envs, device)
                collision_buf          = get_env_buffer(env, "collision_buf",           num_envs, device)
                collision_dynamic_buf  = get_env_buffer(env, "collision_dynamic_buf",   num_envs, device)
                collision_static_buf   = get_env_buffer(env, "collision_static_buf",    num_envs, device)
                collision_boundary_buf = get_env_buffer(env, "collision_boundary_buf",  num_envs, device)
                tumble_buf             = get_env_buffer(env, "tumble_buf",              num_envs, device)

                env_origins = env.unwrapped.scene.env_origins[:, :2]
                goal_pos_w  = env.unwrapped.goal_pos_w
                goal_local  = goal_pos_w - env_origins

                for env_id in done_env_ids.tolist():
                    episode_count += 1

                    epi_reward = episode_reward_sum[env_id].item()
                    epi_steps  = episode_step_count[env_id].item()

                    goal_dist_real  = state[env_id, GOAL_DIST_IDX].item()  * MAX_GOAL_DIST
                    goal_angle_norm = state[env_id, GOAL_ANGLE_IDX].item()

                    gx = round(goal_local[env_id, 0].item(), 1)
                    gy = round(goal_local[env_id, 1].item(), 1)
                    goal_key = (gx, gy)

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

                    per_goal_counts[goal_key]["total"] += 1
                    per_goal_counts[goal_key][outcome] += 1

                    print(
                        f"Epi: {episode_count:<5} "
                        f"env: {env_id:<3} "
                        f"R: {epi_reward:<8.0f} "
                        f"outcome: {outcome:<14} "
                        f"steps: {epi_steps:<6} "
                        f"goal_dist: {goal_dist_real:<6.3f} "
                        f"goal_angle: {goal_angle_norm:<6.3f}"
                    )

                    if args_cli.record_failures and outcome != "SUCCESS" and cached_robot_pos is not None:
                        rpos_w = cached_robot_pos[env_id]
                        rpos_local = (rpos_w - env_origins[env_id]).tolist()
                        rspeed = float(torch.norm(cached_robot_vel[env_id]).item())
                        q = cached_robot_quat[env_id]
                        qw, qx, qy, qz = q[0].item(), q[1].item(), q[2].item(), q[3].item()
                        ryaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

                        obstacles_info = {}
                        closest_obs = None
                        closest_dist = float("inf")
                        for okey in obs_keys:
                            opos_w = cached_obstacles[okey][env_id]
                            opos_local = (opos_w - env_origins[env_id]).tolist()
                            dist = float(torch.norm(opos_w - rpos_w).item())
                            obstacles_info[okey] = {
                                "pos_local": [round(x, 3) for x in opos_local],
                                "dist": round(dist, 3),
                            }
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_obs = okey

                        lr = torch.norm(cached_lidar_hits[env_id] - cached_lidar_pos[env_id].unsqueeze(0), dim=-1)
                        lr = torch.nan_to_num(lr, nan=3.5, posinf=3.5, neginf=3.5).clamp(0.0, 3.5)
                        lidar_min_val = float(lr.min().item())
                        lidar_min_idx = int(lr.argmin().item())

                        last_act = last_actions[env_id].tolist()
                        goal_local_full = (goal_pos_w[env_id] - env_origins[env_id]).tolist()
                        dist_to_goal = float(torch.norm(goal_pos_w[env_id] - rpos_w).item())

                        failure_records.append({
                            "episode": int(episode_count),
                            "env_id": int(env_id),
                            "outcome": outcome,
                            "step_count": int(epi_steps),
                            "robot_pos_local": [round(x, 3) for x in rpos_local],
                            "robot_yaw_rad": round(ryaw, 3),
                            "robot_speed": round(rspeed, 3),
                            "last_action": [round(a, 3) for a in last_act],
                            "goal_local": [round(goal_local_full[0], 3), round(goal_local_full[1], 3)],
                            "dist_to_goal": round(dist_to_goal, 3),
                            "closest_obstacle": closest_obs,
                            "closest_dist": round(closest_dist, 3),
                            "obstacles": obstacles_info,
                            "lidar_min_dist": round(lidar_min_val, 3),
                            "lidar_min_ray_idx": lidar_min_idx,
                        })

                    episode_reward_sum[env_id] = 0.0
                    episode_step_count[env_id] = 0

                    if episode_count >= args_cli.eval_episodes:
                        print_eval_summary(
                            checkpoint=args_cli.checkpoint,
                            episode_count=episode_count,
                            counts=counts,
                        )

                        # print("\n========== PER-GOAL BREAKDOWN ==========")
                        # print(f"{'goal (x,y)':<14} {'N':>4} {'SR':>7} {'STATIC':>7} {'DYN':>5} {'TIMEOUT':>8}")
                        for g, cs in sorted(per_goal_counts.items(), key=lambda kv: kv[1].get("SUCCESS", 0) / max(kv[1]["total"], 1)):
                            n = cs["total"]
                            sr = 100.0 * cs.get("SUCCESS", 0) / max(n, 1)
                            cs_static = 100.0 * cs.get("COLL_STATIC", 0) / max(n, 1)
                            cs_dyn = 100.0 * cs.get("COLL_DYNAMIC", 0) / max(n, 1)
                            cs_to = 100.0 * cs.get("TIMEOUT", 0) / max(n, 1)
                            # print(f"({g[0]:>4.1f},{g[1]:>4.1f})  {n:>4} {sr:>6.1f}% {cs_static:>6.1f}% {cs_dyn:>4.1f}% {cs_to:>7.1f}%")
                        # print("========================================\n")

                        if args_cli.record_failures:
                            with open(args_cli.failure_log, "w") as f:
                                json.dump(failure_records, f, indent=2)
                            print(f"========== FAILURE DIAGNOSTICS ==========")
                            print(f"Wrote {len(failure_records)} failure records to {args_cli.failure_log}")
                            if failure_records:
                                from collections import Counter
                                by_obs = Counter(r["closest_obstacle"] for r in failure_records)
                                by_outcome = Counter(r["outcome"] for r in failure_records)

                                def loc_bucket(r):
                                    x = r["robot_pos_local"][0]
                                    if x < -1.0: return "back_corridor"
                                    if x < 0.5:  return "mid_corridor"
                                    if x < 1.0:  return "near_exit"
                                    return "goal_area"
                                by_loc = Counter(loc_bucket(r) for r in failure_records)

                                # "Frozen near obstacle" heuristic: low robot speed + close obstacle
                                frozen_n = sum(1 for r in failure_records
                                               if r["robot_speed"] < 0.05 and r["closest_dist"] < 0.6)
                                # "Rushing into obstacle": high speed + close obstacle
                                rushing_n = sum(1 for r in failure_records
                                                if r["robot_speed"] > 0.15 and r["closest_dist"] < 0.4)

                                n = len(failure_records)
                                avg_speed = sum(r["robot_speed"] for r in failure_records) / n
                                avg_lin = sum(r["last_action"][0] for r in failure_records) / n
                                avg_abs_ang = sum(abs(r["last_action"][1]) for r in failure_records) / n

                                print(f"By closest obstacle:  {dict(by_obs)}")
                                print(f"By outcome:           {dict(by_outcome)}")
                                print(f"By location:          {dict(by_loc)}")
                                print(f"Frozen-near-obstacle: {frozen_n}/{n}  ({100.0*frozen_n/n:.1f}%)")
                                print(f"Rushing-into-obstacle: {rushing_n}/{n}  ({100.0*rushing_n/n:.1f}%)")
                                print(f"Avg robot speed:      {avg_speed:.3f} m/s")
                                print(f"Avg action linear:    {avg_lin:.3f}")
                                print(f"Avg |action angular|: {avg_abs_ang:.3f}")
                            print(f"==========================================\n")

                        env.close()
                        return

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
