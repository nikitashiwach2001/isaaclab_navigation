# scripts/play_td3_waypoints.py
#
# Step 1 of the planner pipeline: a waypoint manager, no planner yet.
# Each episode every env gets a RANDOM 4-point path (drawn from the known
# wall-clear goal positions). The manager feeds the policy one waypoint at a
# time as its goal; when the robot gets near it, the manager advances to the
# next. Waypoints are drawn as spheres (green = upcoming/done, yellow =
# current target). The env's own random goal marker is hidden.
# Eval-only — training code is untouched.

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Test waypoint-following with random hardcoded paths.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--eval_episodes", type=int, default=100)
parser.add_argument("--use_gru", action="store_true", default=False)
parser.add_argument("--use_conv", action="store_true", default=False)
parser.add_argument("--n_waypoints", type=int, default=4,
                    help="Waypoints per random path (the last one is the goal).")
parser.add_argument("--switch_radius", type=float, default=0.35,
                    help="Advance to the next waypoint when the robot is within this distance (m). "
                         "Must stay > THRESHOLD_GOAL (0.25) so intermediate waypoints don't end the episode.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_scene.tasks  # noqa: F401

from td3.td3_agent import TD3Agent


# Wall-clear positions (local arena coords) — copied from STAGE4_GOAL_POSITIONS
# in mdp/goals.py. Random paths are built by chaining a few of these, so every
# waypoint is guaranteed to sit in free space.
GOAL_POOL = [
    ( 2.0,  1.5), ( 1.8,  2.0), ( 1.5,  0.8),
    ( 2.0, -1.5), ( 1.5, -1.8), ( 0.5, -2.0),
    (-1.0, -2.0), (-1.8, -2.0), (-2.0, -0.8),
    (-2.0,  0.8), (-2.0,  1.8), (-1.5,  2.0),
    ( 0.0,  2.0), ( 0.5,  1.5), ( 1.8, -0.5),
]

MARKER_Z = 0.3   # height to draw the waypoint spheres at


def make_waypoint_markers():
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/waypoints",
        markers={
            "upcoming": sim_utils.SphereCfg(
                radius=0.10,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1)),
            ),
            "current": sim_utils.SphereCfg(
                radius=0.16,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def main():
    device = args_cli.device
    torch.manual_seed(2)
    np.random.seed(0)
    n_wp = args_cli.n_waypoints

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
    print(f"[INFO] Loaded {args_cli.checkpoint}")
    print(f"[INFO] random {n_wp}-waypoint paths, switch_radius={args_cli.switch_radius}")

    pool = torch.tensor(GOAL_POOL, dtype=torch.float32, device=device)        # [P, 2]

    def random_paths(num):
        # num paths, each n_wp distinct points drawn at random from the pool
        perm = torch.rand(num, len(GOAL_POOL), device=device).argsort(dim=-1)[:, :n_wp]
        return pool[perm]                                                     # [num, n_wp, 2]

    paths = random_paths(num_envs)                                            # [E, n_wp, 2]
    wp_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
    last_idx = n_wp - 1

    env_origins = env.unwrapped.scene.env_origins[:, :2]
    arange_e = torch.arange(num_envs, device=device)
    w_range = torch.arange(n_wp, device=device).unsqueeze(0)                  # [1, n_wp]
    markers = make_waypoint_markers()

    has_goal_marker = "goal_marker" in env.unwrapped.scene.keys()

    episode_count = 0
    counts = dict(success=0, timeout=0, other=0)

    while simulation_app.is_running():
        with torch.inference_mode():
            robot_xy = env.unwrapped.scene["robot"].data.root_pos_w[:, :2]

            # advance to the next waypoint once the robot is close enough
            cur_wp = env_origins + paths[arange_e, wp_idx]
            dist = torch.norm(robot_xy - cur_wp, dim=-1)
            advance = (dist < args_cli.switch_radius) & (wp_idx < last_idx)
            wp_idx = wp_idx + advance.long()

            # feed the current waypoint to the policy as its goal
            env.unwrapped.goal_pos_w = env_origins + paths[arange_e, wp_idx]

            # hide the env's own random goal marker so only the waypoints show
            if has_goal_marker:
                gm = env.unwrapped.scene["goal_marker"]
                hide = gm.data.root_pose_w.clone()
                hide[:, 2] = -5.0
                gm.write_root_pose_to_sim(hide)

            # draw the path: yellow = each env's current target, green = the rest
            marker_xyz = torch.zeros(num_envs, n_wp, 3, device=device)
            marker_xyz[:, :, :2] = env_origins.unsqueeze(1) + paths
            marker_xyz[:, :, 2] = MARKER_Z
            marker_ids = (w_range == wp_idx.unsqueeze(1)).long().reshape(-1)
            markers.visualize(translations=marker_xyz.reshape(-1, 3), marker_indices=marker_ids)

            action = agent.select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            state = obs["policy"]
            done = terminated | truncated

            if done.any():
                done_ids = torch.where(done)[0]
                agent.reset_hidden(done_ids)

                goal_reached_buf = getattr(
                    env.unwrapped, "goal_reached_buf",
                    torch.zeros(num_envs, dtype=torch.bool, device=device))

                for env_id in done_ids.tolist():
                    episode_count += 1
                    reached = wp_idx[env_id].item()
                    if goal_reached_buf[env_id]:
                        outcome = "SUCCESS"; counts["success"] += 1
                    elif truncated[env_id]:
                        outcome = "TIMEOUT"; counts["timeout"] += 1
                    else:
                        outcome = "COLLISION/OTHER"; counts["other"] += 1
                    print(f"Epi {episode_count:<4} env {env_id:<3} {outcome:<16} "
                          f"reached waypoint {reached}/{last_idx}")

                    if episode_count >= args_cli.eval_episodes:
                        print(f"\nSUCCESS {counts['success']}  TIMEOUT {counts['timeout']}  "
                              f"COLLISION/OTHER {counts['other']}  of {episode_count}\n")
                        env.close()
                        return

                # fresh random path for each env that just finished
                paths[done_ids] = random_paths(len(done_ids))
                wp_idx[done_ids] = 0

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
