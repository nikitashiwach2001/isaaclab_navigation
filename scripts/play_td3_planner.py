# scripts/play_td3_planner.py
#
# Planner pipeline: an A* planner + the waypoint-following policy.
# Every few steps the planner checks whether its current path is still clear; it
# replans ONLY when a sensed obstacle actually blocks the path, otherwise it
# keeps the path it has (no gratuitous shortest-path swapping). The static grid
# holds the walls only; moving obstacles are sensed from the robot's live lidar
# scan, so the planner reacts to whatever the robot can actually see (no
# privileged obstacle positions). The policy follows that path (drawn as a green
# line) and does the fine reactive dodging. Goals are a fixed set cycled
# deterministically, so the experiment is repeatable. Eval-only.

import argparse
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run the A* planner + waypoint-following policy.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--eval_episodes", type=int, default=100)
parser.add_argument("--use_gru", action="store_true", default=False)
parser.add_argument("--use_conv", action="store_true", default=False)
parser.add_argument("--switch_radius", type=float, default=0.15,
                    help="Advance to the next waypoint within this distance (m).")
parser.add_argument("--inflate", type=float, default=0.30,
                    help="Clearance the planner keeps the path away from walls (m, static grid).")
parser.add_argument("--block_radius", type=float, default=0.20,
                    help="Disk radius around each lidar-sensed obstacle hit (m, dynamic blocks). "
                         "Decoupled from --inflate so the static grid can have generous wall "
                         "clearance while dynamic disks stay tight — prevents over-blocking "
                         "corridors when a transient lidar hit lands near a wall.")
parser.add_argument("--cluster_dist", type=float, default=0.30,
                    help="Lidar hits within this distance are merged to a single block disk (m). "
                         "A single moving cylinder hits ~5 rays — without clustering they create "
                         "5 overlapping disks that block more area than the obstacle's footprint.")
parser.add_argument("--replan_every", type=int, default=3,
                    help="Check the path for blockage every N steps; replan if a sensed obstacle is "
                         "on it AND it's near the robot (see --replan_lookahead_m). 0 = plan once.")
parser.add_argument("--replan_lookahead_m", type=float, default=1.5,
                    help="Only trigger a replan when the blocked waypoint is within this distance "
                         "from the robot. Obstacles on far-future waypoints often move away before "
                         "the robot reaches them, so replanning early picks needlessly long routes.")
parser.add_argument("--proximity_replan_m", type=float, default=0.8,
                    help="Also trigger a replan whenever ANY sensed obstacle is within this radius "
                         "of the robot, even if it's not on the planned path. Stops the policy from "
                         "looping/circling around a nearby obstacle the planner hasn't routed around. "
                         "0 = disabled.")
parser.add_argument("--enable_backward_recovery", action="store_true", default=False,
                    help="When the robot is nose-pinned and pin_steps reaches the trigger, override "
                         "the policy action with a straight backward command for a few steps so the "
                         "robot can back out of the pin and resume normal policy control.")
parser.add_argument("--recovery_trigger_steps", type=int, default=3,
                    help="Consecutive pin steps before backward recovery kicks in (default 3 "
                         "≈ 0.1 s at 30 Hz). Recovery also fires on no_progress for "
                         "recovery_trigger_steps × 4 steps so non-front-blocked stalls also "
                         "trigger reverse. Must be < PIN_STEPS to recover before STUCK fires.")
parser.add_argument("--recovery_duration_steps", type=int, default=30,
                    help="Max steps of backward motion during recovery (default 30 = 1 s). "
                         "Recovery also exits early once the front clears.")
parser.add_argument("--recovery_action_linear", type=float, default=-0.8,
                    help="Normalized linear action during recovery (negative = backward). Default "
                         "-0.8 → ~80%% of MAX_LINEAR_SPEED in reverse.")
parser.add_argument("--waypoint_spacing", type=float, default=0.1,
                    help="Spacing between policy waypoints (m). Smaller = tighter path-following and a closer first waypoint.")
parser.add_argument("--episode_length_s", type=float, default=75.0,
                    help="Per-episode time budget (s). Default 75 — longer than the env's 50 s "
                         "so detour-heavy paths in dense scenes have time to complete. "
                         "Robots that aren't progressing get caught earlier by --no_progress_timeout_s.")
parser.add_argument("--no_progress_timeout_s", type=float, default=15.0,
                    help="If goal-distance doesn't drop by >0.05 m within this many seconds, "
                         "mark the episode as STALLED (separate bucket from TIMEOUT). Catches "
                         "wandering / circling without waiting for the hard time budget.")
parser.add_argument("--show_path", action="store_true", default=False,
                    help="Draw the planned path as a green line. Off by default.")
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
from planner import PathPlanner, build_arena_grid


# Fixed wall-clear goals (local arena coords), cycled deterministically per env.
# Works for Stage 5/6 (open arena with short inner walls).
FIXED_GOALS = [
    ( 2.0,  1.5),
    (-2.0,  0.8),
    ( 1.5, -1.8),
    (-1.8, -2.0),
    ( 0.0,  2.0),
    (-2.0,  1.8),
    ( 1.8, -0.5),
]

MAX_WP = 96           # max policy waypoints stored per env (covers ~9.6 m at 0.1 m spacing)
PATH_Z = 0.10         # m — height the green path line is drawn at
LINE_WIDTH = 0.05     # m — thickness of the green path line
FINAL_REACH = 0.25    # robot within this of the goal counts as success
OBSTACLE_RADIUS = 0.16
BLOCK_SIDE_OFFSET = 1.3   # m — blocker starts this far to the side of its stop point
CONTACT_THRESH = 0.30     # m — lidar closer than this counts as a contact (any direction)
PIN_THRESH = 0.30         # m — something this close dead-ahead = a head-on (nose) pin
STUCK_SPEED = 0.05        # m/s — below this the robot is "not moving"
# Soft-recovery: brief contacts are tolerated. Only persistent "stuck while
# nose-pinned" terminates the episode as a failure. 60 steps = 2 s at 30 Hz.
# Below this, the robot can brush an obstacle, replan, and continue.
PIN_STEPS = 60            # consecutive stuck + front-close steps to flag a nose pin (failure)
LIDAR_SENSE_RANGE = 3.0   # m — the planner only trusts lidar hits within this range


def make_path_markers():
    """One thin green cuboid per path segment — the segments butt end-to-end
    into a single continuous line. Each instance is scaled in X to its segment
    length and yawed to align with it (set per frame in visualize())."""
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/path_line",
        markers={
            "segment": sim_utils.CuboidCfg(
                size=(1.0, LINE_WIDTH, LINE_WIDTH),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1)),
            ),
        },
    )
    return VisualizationMarkers(cfg)


# Marker dimensions
GOAL_MARKER_RADIUS = 0.20    # final-goal sphere (larger, always visible)
WAYPOINT_MARKER_RADIUS = 0.07  # path-waypoint spheres (smaller, intermediate)
GOAL_MARKER_Z = 0.30           # height at which markers are drawn


def make_goal_marker():
    """Large yellow sphere for the final goal — always visible per env, larger
    than the waypoint markers so the eye can lock onto the destination."""
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/final_goal",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=GOAL_MARKER_RADIUS,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def make_waypoint_markers():
    """Small green spheres along the planner's path waypoints — visually
    smaller than the final-goal sphere so the goal stays the obvious target."""
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/waypoints",
        markers={
            "wp": sim_utils.SphereCfg(
                radius=WAYPOINT_MARKER_RADIUS,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.1)),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def _yaw_from_quat(q):
    qw, qx, qy, qz = (float(v) for v in q)
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def main():
    device = args_cli.device
    torch.manual_seed(2)
    np.random.seed(0)

    env_cfg = parse_env_cfg(
        args_cli.task, device=device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # disable the env's auto-success — the demo detects reaching the goal
    # itself via FINAL_REACH so the scoring is uniform across stages. The
    # env's collision termination is also disabled so brief contacts don't
    # end the episode (soft-recovery semantics — a real robot would brush an
    # obstacle and continue). Persistent nose-pin (PIN_STEPS) is then the
    # script's own "unrecoverable stuck" criterion. Tumble + time_out remain
    # active as legitimate hard-failure signals.
    env_cfg.terminations.goal_reached = None
    env_cfg.terminations.collision = None
    env_cfg.episode_length_s = float(args_cli.episode_length_s)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.action_space = spaces.Box(low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32)

    obs, _ = env.reset()
    state = obs["policy"]
    num_envs = env.unwrapped.num_envs
    scene = env.unwrapped.scene
    max_ep = int(env.unwrapped.max_episode_length)

    agent = TD3Agent(
        state_dim=state.shape[1], action_dim=env.action_space.shape[1], device=device,
        hidden_dim=args_cli.hidden_dim, use_gru=args_cli.use_gru, use_conv=args_cli.use_conv,
    )
    agent.load_actor_only(args_cli.checkpoint)
    agent.actor.eval()

    # --- static occupancy grid from the env's inner walls ---
    # Wall dimensions come from each wall's spawn cfg, so all inner-wall
    # geometries are rasterised at their true size.
    env_origins = scene.env_origins[:, :2]
    origin0 = env_origins[0]
    walls = []
    for key in sorted(scene.keys()):
        if key.startswith("inner_wall"):
            w = scene[key]
            pos = w.data.root_pos_w[0, :2] - origin0
            yaw = _yaw_from_quat(w.data.root_quat_w[0])
            size = w.cfg.spawn.size
            walls.append((pos[0].item(), pos[1].item(), float(size[0]) / 2.0, float(size[1]) / 2.0, yaw))
    grid = build_arena_grid(walls, inflate=args_cli.inflate)
    planner = PathPlanner(grid)
    obs_keys = sorted(k for k in scene.keys() if k.startswith("obstacle_"))
    # Dynamic-block disk radius (now separate from --inflate). Smaller default
    # so transient lidar hits don't over-block corridors — a single cylinder
    # hits 4-5 rays, and without clustering each ray makes its own 0.4 m disk
    # creating ~2 m² of false blockage. With block_radius=0.20 and clustering
    # below, one cylinder blocks ~0.13 m² which matches its actual footprint.
    block_radius = args_cli.block_radius
    print(f"[INFO] Loaded {args_cli.checkpoint}")
    print(f"[INFO] grid {grid.nx}x{grid.ny}, {len(walls)} inner walls (static); "
          f"{len(obs_keys)} obstacles sensed via lidar, replan_every={args_cli.replan_every}")

    # per-env state
    paths = torch.zeros(num_envs, MAX_WP, 2, device=device)
    path_len = torch.ones(num_envs, dtype=torch.long, device=device)
    wp_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
    true_goal = torch.zeros(num_envs, 2, device=device)
    goal_cycle = torch.arange(num_envs, device=device) % len(FIXED_GOALS)
    succeeded = torch.zeros(num_envs, dtype=torch.bool, device=device)
    no_path = torch.zeros(num_envs, dtype=torch.bool, device=device)
    contacts = torch.zeros(num_envs, dtype=torch.long, device=device)
    in_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
    pin_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    had_pin = torch.zeros(num_envs, dtype=torch.bool, device=device)
    best_goal_dist = torch.full((num_envs,), float("inf"), device=device)
    no_progress_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    stalled = torch.zeros(num_envs, dtype=torch.bool, device=device)
    prev_wp_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
    prev_path_len = torch.zeros(num_envs, dtype=torch.long, device=device)
    progress_thresh_steps = max(int(args_cli.no_progress_timeout_s / float(env.unwrapped.step_dt)), 1)
    PROGRESS_DELTA = 0.05  # m — improvement under this doesn't reset the no-progress counter
    env.unwrapped.blk_start = torch.zeros(num_envs, 2, device=device)
    env.unwrapped.blk_stop = torch.zeros(num_envs, 2, device=device)

    # ── Backward-recovery state ───────────────────────────────────────────────
    # Two distinct modes:
    #   recovery_active (backward)  — reverse out of a front pin; only entered
    #                                 when the rear sector is clear so we don't
    #                                 back into another obstacle.
    #   waiting_active              — both front AND rear are blocked; stop and
    #                                 wait for any direction to clear (e.g. a
    #                                 moving obstacle passing by).
    recovery_active     = torch.zeros(num_envs, dtype=torch.bool, device=device)
    recovery_steps_left = torch.zeros(num_envs, dtype=torch.long, device=device)
    waiting_active      = torch.zeros(num_envs, dtype=torch.bool, device=device)
    waiting_steps_left  = torch.zeros(num_envs, dtype=torch.long, device=device)
    env.unwrapped.allow_backward = torch.zeros(num_envs, dtype=torch.bool, device=device)
    RECOVERY_CLEAR_THRESH = PIN_THRESH * 1.5  # front >= this = free to drive forward
    REAR_CLEAR_THRESH     = 0.45              # rear must exceed this to reverse safely

    def _sensed_blocks(e):
        """Obstacle disks for env e built from the live lidar scan — NOT from
        privileged obstacle positions. Lidar hits within sensing range that
        land in free space (i.e. not a wall the static grid already knows)
        are clustered to a coarse grid (--cluster_dist) so a single cylinder's
        4-5 adjacent ray hits collapse to one disk instead of creating an
        over-inflated stack."""
        lidar = scene["lidar"]
        hits_w = lidar.data.ray_hits_w[e, :, :2]
        rng = torch.norm(hits_w - lidar.data.pos_w[e, :2], dim=-1)
        valid = torch.isfinite(hits_w).all(dim=-1) & torch.isfinite(rng) & (rng < LIDAR_SENSE_RANGE)
        cluster_cell = args_cli.cluster_dist
        seen = set()
        blocks = []
        for hx, hy in (hits_w[valid] - env_origins[e]).tolist():
            # Spatial dedupe: hits sharing the same coarse cell are one obstacle.
            key = (int(math.floor(hx / cluster_cell)), int(math.floor(hy / cluster_cell)))
            if key in seen:
                continue
            ci, cj = grid.world_to_cell(hx, hy)
            if grid.is_free(ci, cj):
                seen.add(key)
                blocks.append((hx, hy, block_radius))
        return blocks

    def _store(e, wps):
        no_path[e] = wps is None
        if not wps:
            wps = [(true_goal[e, 0].item(), true_goal[e, 1].item())]
        # Preserve the goal as the final stored waypoint even if the dense
        # path exceeds MAX_WP — otherwise truncation silently drops the goal
        # and the robot stops short of it.
        if len(wps) > MAX_WP:
            wps = wps[: MAX_WP - 1] + [wps[-1]]
        n = len(wps)
        for k in range(n):
            paths[e, k, 0], paths[e, k, 1] = wps[k][0], wps[k][1]
        path_len[e] = n
        wp_idx[e] = 0
        return wps

    def plan_path(e, blocks=None):
        """Replan env e from the robot's current pose to its goal, routing
        around the obstacles sensed by its live lidar scan."""
        if blocks is None:
            blocks = _sensed_blocks(e)
        robot_l = scene["robot"].data.root_pos_w[e, :2] - env_origins[e]
        g = (true_goal[e, 0].item(), true_goal[e, 1].item())
        return _store(e, planner.plan((robot_l[0].item(), robot_l[1].item()), g,
                                      spacing=args_cli.waypoint_spacing,
                                      extra_blocks=blocks))

    def _segment_intersects_disk(ax, ay, bx, by, cx, cy, r):
        """True if the line segment from (ax, ay) to (bx, by) passes within
        distance r of the point (cx, cy). Computed by clamping the projection
        of (cx, cy) onto the segment and checking distance squared."""
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-9:
            # Degenerate (zero-length) segment — endpoint check.
            return (cx - ax) ** 2 + (cy - ay) ** 2 < r * r
        t = ((cx - ax) * dx + (cy - ay) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        px, py = ax + t * dx, ay + t * dy
        return (cx - px) ** 2 + (cy - py) ** 2 < r * r

    def _obstacle_near_robot(e, blocks):
        """True if any sensed obstacle disk is within --proximity_replan_m
        of the robot — independent of whether it touches the planned path.
        Forces a replan so the route reroutes around close obstacles before
        the policy starts orbit-dodging them."""
        prox = args_cli.proximity_replan_m
        if prox <= 0.0:
            return False
        robot_l = scene["robot"].data.root_pos_w[e, :2] - env_origins[e]
        rx, ry = robot_l[0].item(), robot_l[1].item()
        for cx, cy, r in blocks:
            # disk surface within prox of the robot
            if (cx - rx) ** 2 + (cy - ry) ** 2 < (prox + r) ** 2:
                return True
        return False

    def _path_blocked(e, blocks):
        """True if a sensed obstacle intersects the path between the robot's
        current position and any near-future waypoint (within
        --replan_lookahead_m). Previously this checked only waypoint endpoints,
        which let obstacles slip between two waypoints (e.g. a stopped cylinder
        sitting halfway along a 0.25-m segment with a 0.18-m disk would clear
        both endpoints by 0.07 m — no replan triggered, robot drove into it).
        Now checks line segments via point-to-segment distance, so any
        obstacle the path physically crosses triggers a replan."""
        lookahead_sq = args_cli.replan_lookahead_m ** 2
        robot_l = scene["robot"].data.root_pos_w[e, :2] - env_origins[e]
        rx, ry = robot_l[0].item(), robot_l[1].item()
        wpi = int(wp_idx[e].item())
        plen = int(path_len[e].item())
        # Walk segments: robot -> wp[wpi] -> wp[wpi+1] -> ... until out of lookahead.
        ax, ay = rx, ry
        for k in range(wpi, plen):
            bx, by = paths[e, k, 0].item(), paths[e, k, 1].item()
            # Once a waypoint is beyond lookahead, stop scanning further segments.
            if (bx - rx) ** 2 + (by - ry) ** 2 > lookahead_sq:
                break
            for cx, cy, r in blocks:
                if _segment_intersects_disk(ax, ay, bx, by, cx, cy, r):
                    return True
            ax, ay = bx, by
        return False

    def compute_blocker(wps):
        """Pick the scripted blocker's stop point P (middle of the path) and a
        start point S off to a wall-clear side. Returns (S, P) in local coords."""
        mid = len(wps) // 2
        P = wps[mid]
        a, b = wps[max(mid - 1, 0)], wps[min(mid + 1, len(wps) - 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = math.hypot(dx, dy) or 1.0
        px, py = -dy / norm, dx / norm
        for sign in (1.0, -1.0):
            S = (P[0] + sign * px * BLOCK_SIDE_OFFSET, P[1] + sign * py * BLOCK_SIDE_OFFSET)
            if abs(S[0]) < 2.2 and abs(S[1]) < 2.2 and grid.is_free(*grid.world_to_cell(*S)):
                return S, P
        return P, P

    def new_episode(e):
        """Set the env's fixed goal, plan, and place the scripted blocker."""
        goal_l = FIXED_GOALS[int(goal_cycle[e].item())]
        true_goal[e, 0], true_goal[e, 1] = goal_l[0], goal_l[1]
        succeeded[e] = False
        contacts[e] = 0
        in_contact[e] = False
        pin_steps[e] = 0
        had_pin[e] = False
        best_goal_dist[e] = float("inf")
        no_progress_steps[e] = 0
        stalled[e] = False
        prev_wp_idx[e] = 0
        prev_path_len[e] = 0
        recovery_active[e] = False
        recovery_steps_left[e] = 0
        waiting_active[e] = False
        waiting_steps_left[e] = 0
        env.unwrapped.allow_backward[e] = False
        wps = plan_path(e)
        S, P = compute_blocker(wps)
        env.unwrapped.blk_start[e] = env_origins[e] + torch.tensor(S, device=device)
        env.unwrapped.blk_stop[e] = env_origins[e] + torch.tensor(P, device=device)

    for e in range(num_envs):
        new_episode(e)

    markers = make_path_markers() if args_cli.show_path else None
    waypoint_markers = make_waypoint_markers() if args_cli.show_path else None
    # Final-goal sphere is ALWAYS created (independent of --show_path).
    # Large yellow sphere makes the destination visually unambiguous.
    goal_markers = make_goal_marker()
    seg_slot = torch.arange(MAX_WP - 1, device=device).unsqueeze(0)
    wp_slot = torch.arange(MAX_WP, device=device).unsqueeze(0)
    arange_e = torch.arange(num_envs, device=device)
    has_goal_marker = "goal_marker" in scene.keys()

    episode_count = 0
    # Outcomes:
    #   success         — reached the final goal (may include episodes with brief
    #                     recoverable contact events; recovered count is a subset)
    #   recovered       — subset of success: had contacts during the episode but
    #                     recovered and still reached the goal
    #   success_clean   — subset of success: zero contact events all episode
    #   stuck           — nose-pinned long enough (PIN_STEPS) to be unrecoverable
    #   timeout         — ran out of time without reaching goal and without stuck
    #   tumble          — robot tipped over (env's tumble termination fired)
    #   no_path         — planner couldn't find a path (tracked separately)
    counts = dict(success=0, success_clean=0, recovered=0,
                  stuck=0, stalled=0, tumble=0, timeout=0, no_path=0)
    total_contacts = 0
    pinned = 0
    step_count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            step_count += 1
            robot_xy = scene["robot"].data.root_pos_w[:, :2]

            # contact + head-on (nose) pin tracking
            lidar = scene["lidar"]
            lr = torch.norm(lidar.data.ray_hits_w - lidar.data.pos_w.unsqueeze(1), dim=-1)
            lr = torch.nan_to_num(lr, nan=3.5, posinf=3.5, neginf=3.5).clamp(0.0, 3.5).reshape(num_envs, -1)
            nr = lr.shape[1]
            kf = max(int(nr * 30 / 360), 1)                      # rays within ±30° of forward
            front_min = torch.cat([lr[:, :kf], lr[:, nr - kf:]], dim=1).min(dim=1).values
            # Rear sector: ±30° around 180° (rays nr/2 - kf .. nr/2 + kf).
            # Used by the backward-recovery logic to decide if reversing is safe.
            rear_min = lr[:, nr // 2 - kf:nr // 2 + kf].min(dim=1).values
            min_range = lr.min(dim=1).values
            speed = torch.norm(scene["robot"].data.root_lin_vel_w[:, :2], dim=-1)
            now_contact = min_range < CONTACT_THRESH
            contacts = contacts + (now_contact & ~in_contact).long()
            in_contact = now_contact
            is_pin = (speed < STUCK_SPEED) & (front_min < PIN_THRESH)
            pin_steps = torch.where(is_pin, pin_steps + 1, torch.zeros_like(pin_steps))
            had_pin = had_pin | (pin_steps >= PIN_STEPS)

            # replan every replan_every steps if a sensed obstacle is on the
            # current path OR the previous plan failed (no_path). Without the
            # no_path retry, a failed A* leaves the env with an empty path that
            # _path_blocked can never flag, so the planner never recovers.
            if args_cli.replan_every > 0 and step_count % args_cli.replan_every == 0:
                for e in range(num_envs):
                    blocks = _sensed_blocks(e)
                    if (
                        no_path[e]
                        or _path_blocked(e, blocks)
                        or _obstacle_near_robot(e, blocks)
                    ):
                        plan_path(e, blocks)

            # advance to the next waypoint once the robot is close enough
            cur_wp = env_origins + paths[arange_e, wp_idx]
            dist = torch.norm(robot_xy - cur_wp, dim=-1)
            advance = (dist < args_cli.switch_radius) & (wp_idx < path_len - 1)
            wp_idx = wp_idx + advance.long()

            # success: robot reached the final goal -> force this env to end
            final_world = env_origins + true_goal
            cur_goal_dist = torch.norm(robot_xy - final_world, dim=-1)
            newly = (cur_goal_dist < FINAL_REACH) & ~succeeded
            succeeded = succeeded | newly
            if newly.any():
                env.unwrapped.episode_length_buf[newly] = max_ep

            # Progress tracking: the robot is making progress if ANY of:
            #   (a) it advanced to a new waypoint,
            #   (b) the planner gave it a new/different path (replan),
            #   (c) goal-distance dropped by more than PROGRESS_DELTA below best.
            # Otherwise increment no_progress_steps; STALL when threshold reached.
            wp_changed = (wp_idx != prev_wp_idx) | (path_len != prev_path_len)
            prev_wp_idx[:] = wp_idx
            prev_path_len[:] = path_len
            goal_improved = cur_goal_dist < (best_goal_dist - PROGRESS_DELTA)
            best_goal_dist = torch.where(goal_improved, cur_goal_dist, best_goal_dist)
            improved = wp_changed | goal_improved
            no_progress_steps = torch.where(
                improved, torch.zeros_like(no_progress_steps), no_progress_steps + 1
            )
            newly_stalled = (no_progress_steps >= progress_thresh_steps) & ~succeeded & ~stalled
            stalled = stalled | newly_stalled
            if newly_stalled.any():
                env.unwrapped.episode_length_buf[newly_stalled] = max_ep

            # Stuck failure (soft-recovery boundary): a nose-pin that's lasted
            # PIN_STEPS consecutive steps is unrecoverable. Force the episode
            # to end so it's counted as STUCK rather than running to timeout.
            newly_stuck = had_pin & ~succeeded
            if newly_stuck.any():
                env.unwrapped.episode_length_buf[newly_stuck] = max_ep

            # feed the current waypoint to the policy as its goal
            env.unwrapped.goal_pos_w = env_origins + paths[arange_e, wp_idx]

            # hide the env's own goal marker — our yellow sphere replaces it
            if has_goal_marker:
                gm = scene["goal_marker"]
                hide = gm.data.root_pose_w.clone()
                hide[:, 2] = -10.0
                gm.write_root_pose_to_sim(hide)

            # FINAL-GOAL marker: large yellow sphere at each env's true goal.
            # Always shown so the destination is visually obvious regardless of
            # --show_path. Drawn at GOAL_MARKER_Z height so it floats above the
            # path-line / waypoint spheres.
            goal_xyz = torch.zeros(num_envs, 3, device=device)
            goal_xyz[:, :2] = env_origins + true_goal
            goal_xyz[:, 2] = GOAL_MARKER_Z
            goal_markers.visualize(translations=goal_xyz)

            # draw the path as one continuous green line — a thin box per
            # segment, scaled to the segment length and yawed to align with it
            if args_cli.show_path:
                p0, p1 = paths[:, :-1, :], paths[:, 1:, :]
                mid = 0.5 * (p0 + p1)
                delta = p1 - p0
                seg_len = torch.norm(delta, dim=-1)
                yaw = torch.atan2(delta[..., 1], delta[..., 0])
                seg_used = seg_slot < (path_len - 1).unsqueeze(1)

                sxyz = torch.zeros(num_envs, MAX_WP - 1, 3, device=device)
                sxyz[:, :, :2] = env_origins.unsqueeze(1) + mid
                sxyz[:, :, 2] = torch.where(seg_used,
                                            torch.full_like(seg_len, float(PATH_Z)),
                                            torch.full_like(seg_len, -10.0))
                squat = torch.zeros(num_envs, MAX_WP - 1, 4, device=device)
                squat[:, :, 0] = torch.cos(yaw * 0.5)
                squat[:, :, 3] = torch.sin(yaw * 0.5)
                sscale = torch.ones(num_envs, MAX_WP - 1, 3, device=device)
                sscale[:, :, 0] = seg_len.clamp(min=1e-3)
                markers.visualize(
                    translations=sxyz.reshape(-1, 3),
                    orientations=squat.reshape(-1, 4),
                    scales=sscale.reshape(-1, 3),
                )

                # Waypoint spheres (smaller than the yellow goal marker) at
                # each path point. The final waypoint coincides with the goal
                # position but is drawn below the goal sphere so the goal
                # remains the larger / more obvious marker.
                wp_xyz = torch.zeros(num_envs, MAX_WP, 3, device=device)
                wp_xyz[:, :, :2] = env_origins.unsqueeze(1) + paths
                wp_used = wp_slot < path_len.unsqueeze(1)
                wp_xyz[:, :, 2] = torch.where(
                    wp_used,
                    torch.full_like(wp_used, float(PATH_Z), dtype=torch.float32),
                    torch.full_like(wp_used, -10.0, dtype=torch.float32),
                )
                waypoint_markers.visualize(translations=wp_xyz.reshape(-1, 3))

            action = agent.select_action(state)

            # ── Backward-recovery / wait override ──────────────────────────────
            # Trigger condition (env is "stuck"):
            #   pin_steps ≥ trigger  OR  no_progress_steps ≥ trigger × 4
            # Decision when trigger fires:
            #   rear clear (> REAR_CLEAR_THRESH)  → BACKWARD recovery (reverse)
            #   rear blocked                       → WAIT (zero action)
            # Exit:
            #   BACKWARD: front clears OR duration runs out OR rear becomes blocked
            #             (in which case we switch to WAIT for the remaining time)
            #   WAIT:     ANY side clears OR duration runs out
            if args_cli.enable_backward_recovery:
                pin_trigger = pin_steps >= args_cli.recovery_trigger_steps
                noprog_trigger = no_progress_steps >= (args_cli.recovery_trigger_steps * 4)
                triggered = (pin_trigger | noprog_trigger) & ~recovery_active & ~waiting_active

                rear_clear = rear_min > REAR_CLEAR_THRESH
                enter_backward = triggered & rear_clear
                enter_wait     = triggered & ~rear_clear

                if enter_backward.any():
                    for e in enter_backward.nonzero(as_tuple=False).flatten().tolist():
                        reason = "pin" if pin_trigger[e] else "no_progress"
                        print(f"[RECOVERY] env {e} BACKWARD (reason={reason}, "
                              f"front_min={float(front_min[e].item()):.2f}m, "
                              f"rear_min={float(rear_min[e].item()):.2f}m, "
                              f"speed={float(speed[e].item()):.3f} m/s)")
                if enter_wait.any():
                    for e in enter_wait.nonzero(as_tuple=False).flatten().tolist():
                        print(f"[RECOVERY] env {e} WAIT (front and rear blocked, "
                              f"front_min={float(front_min[e].item()):.2f}m, "
                              f"rear_min={float(rear_min[e].item()):.2f}m)")

                recovery_active = recovery_active | enter_backward
                waiting_active  = waiting_active  | enter_wait

                # If we are mid-reverse and the rear becomes blocked, switch to wait.
                rear_blocked_now = recovery_active & ~rear_clear
                if rear_blocked_now.any():
                    for e in rear_blocked_now.nonzero(as_tuple=False).flatten().tolist():
                        print(f"[RECOVERY] env {e} BACKWARD → WAIT "
                              f"(rear blocked mid-recovery, rear_min={float(rear_min[e].item()):.2f}m)")
                recovery_active = recovery_active & ~rear_blocked_now
                waiting_active  = waiting_active  | rear_blocked_now

                # Duration timers
                recovery_steps_left = torch.where(
                    enter_backward,
                    torch.full_like(recovery_steps_left, args_cli.recovery_duration_steps),
                    recovery_steps_left,
                )
                waiting_steps_left = torch.where(
                    enter_wait | rear_blocked_now,
                    torch.full_like(waiting_steps_left, args_cli.recovery_duration_steps),
                    waiting_steps_left,
                )
                recovery_steps_left = torch.where(
                    recovery_active, recovery_steps_left - 1, recovery_steps_left
                )
                waiting_steps_left = torch.where(
                    waiting_active, waiting_steps_left - 1, waiting_steps_left
                )

                # Exit conditions
                exit_recovery = recovery_active & (
                    (recovery_steps_left <= 0) | (front_min > RECOVERY_CLEAR_THRESH)
                )
                exit_wait = waiting_active & (
                    (waiting_steps_left <= 0)
                    | (front_min > RECOVERY_CLEAR_THRESH)
                    | (rear_min > REAR_CLEAR_THRESH)
                )
                if exit_recovery.any():
                    for e in exit_recovery.nonzero(as_tuple=False).flatten().tolist():
                        print(f"[RECOVERY] env {e} exiting backward "
                              f"(front_min={float(front_min[e].item()):.2f}m, "
                              f"steps_left={int(recovery_steps_left[e].item())})")
                if exit_wait.any():
                    for e in exit_wait.nonzero(as_tuple=False).flatten().tolist():
                        print(f"[RECOVERY] env {e} exiting wait "
                              f"(front_min={float(front_min[e].item()):.2f}m, "
                              f"rear_min={float(rear_min[e].item()):.2f}m)")
                recovery_active = recovery_active & ~exit_recovery
                waiting_active  = waiting_active  & ~exit_wait

                # Reset no_progress on entry into either mode so the env doesn't
                # get killed as STALLED while reversing or waiting.
                entered_either = enter_backward | enter_wait
                no_progress_steps = torch.where(
                    entered_either, torch.zeros_like(no_progress_steps), no_progress_steps
                )

                env.unwrapped.allow_backward[:] = recovery_active

                # Override action: reverse during BACKWARD, zero during WAIT.
                if recovery_active.any():
                    rec_action = torch.tensor(
                        [args_cli.recovery_action_linear, 0.0],
                        device=device,
                        dtype=action.dtype,
                    )
                    action = torch.where(
                        recovery_active.unsqueeze(-1),
                        rec_action.unsqueeze(0).expand_as(action),
                        action,
                    )
                if waiting_active.any():
                    # During wait we send the forward-only "minimum" — linear_norm=-1
                    # maps to 0 m/s (forward-only clamp). angular = 0 → no spin.
                    wait_action = torch.tensor(
                        [-1.0, 0.0], device=device, dtype=action.dtype
                    )
                    action = torch.where(
                        waiting_active.unsqueeze(-1),
                        wait_action.unsqueeze(0).expand_as(action),
                        action,
                    )

            obs, reward, terminated, truncated, _ = env.step(action)
            state = obs["policy"]
            done = terminated | truncated

            if done.any():
                done_ids = torch.where(done)[0]
                agent.reset_hidden(done_ids)

                for env_id in done_ids.tolist():
                    episode_count += 1
                    epi_contacts = int(contacts[env_id].item())
                    # Priority: stuck > tumble > success > timeout. STUCK takes
                    # priority over SUCCESS because we force-end stuck envs.
                    if had_pin[env_id] and not succeeded[env_id]:
                        outcome = "STUCK"; counts["stuck"] += 1
                    elif stalled[env_id] and not succeeded[env_id]:
                        outcome = "STALLED"; counts["stalled"] += 1
                    elif succeeded[env_id]:
                        outcome = "SUCCESS"; counts["success"] += 1
                        if epi_contacts == 0:
                            counts["success_clean"] += 1
                        else:
                            counts["recovered"] += 1
                    elif terminated[env_id]:
                        # env-side terminations remaining: tumble only (collision off)
                        outcome = "TUMBLE"; counts["tumble"] += 1
                    elif truncated[env_id]:
                        outcome = "TIMEOUT"; counts["timeout"] += 1
                    else:
                        outcome = "OTHER"
                    if no_path[env_id]:
                        counts["no_path"] += 1
                    total_contacts += epi_contacts
                    if had_pin[env_id]:
                        pinned += 1
                    print(f"Epi {episode_count:<4} env {env_id:<3} {outcome:<10} "
                          f"goal {int(goal_cycle[env_id].item())}  "
                          f"contacts={epi_contacts}"
                          f"{'  [NOSE-PIN]' if had_pin[env_id] else ''}")

                    if episode_count >= args_cli.eval_episodes:
                        total = max(episode_count, 1)
                        pct = lambda n: f"{100.0 * n / total:.1f}%"
                        print(f"\n========== PLANNER EVAL SUMMARY ==========")
                        print(f"Total episodes: {episode_count}")
                        print(f"SUCCESS:       {counts['success']:<4} ({pct(counts['success'])})")
                        print(f"  └─ clean:    {counts['success_clean']:<4} ({pct(counts['success_clean'])})")
                        print(f"  └─ recovered:{counts['recovered']:<4} ({pct(counts['recovered'])})")
                        print(f"STUCK:         {counts['stuck']:<4} ({pct(counts['stuck'])})")
                        print(f"STALLED:       {counts['stalled']:<4} ({pct(counts['stalled'])})")
                        print(f"TUMBLE:        {counts['tumble']:<4} ({pct(counts['tumble'])})")
                        print(f"TIMEOUT:       {counts['timeout']:<4} ({pct(counts['timeout'])})")
                        print(f"NO_PATH:       {counts['no_path']:<4} (planner failed to route)")
                        print(f"contacts total {total_contacts}  |  episodes with a nose-pin: "
                              f"{pinned}")
                        print(f"==========================================\n")
                        env.close()
                        return

                # cycle each finished env to the next fixed goal and replan
                for env_id in done_ids.tolist():
                    goal_cycle[env_id] = (goal_cycle[env_id] + 1) % len(FIXED_GOALS)
                    new_episode(env_id)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
