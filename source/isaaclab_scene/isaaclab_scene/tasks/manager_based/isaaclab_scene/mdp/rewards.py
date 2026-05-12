import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass


LIDAR_DISTANCE_CAP = 3.5
MAX_LINEAR_SPEED   = 0.22
MAX_ANGULAR_SPEED  = 2.0
ENABLE_BACKWARD    = False
THRESHOLD_GOAL      = 0.25
THRESHOLD_COLLISION = 0.30
SUCCESS_REWARD    = 500.0
COLLISION_PENALTY = 300.0


def _get_lidar_distances(env: ManagerBasedRLEnv):
    lidar = env.scene["lidar"]
    ranges = torch.norm(lidar.data.ray_hits_w - lidar.data.pos_w.unsqueeze(1), dim=-1)
    ranges = torch.nan_to_num(ranges, nan=LIDAR_DISTANCE_CAP,
                               posinf=LIDAR_DISTANCE_CAP, neginf=LIDAR_DISTANCE_CAP)
    ranges = torch.clamp(ranges, 0.0, LIDAR_DISTANCE_CAP).reshape(env.num_envs, -1)

    min_obstacle_dist = ranges.min(dim=1).values
    num_rays   = ranges.shape[1]
    fw         = max(1, int(num_rays * 24.0 / 360.0))
    front_min  = torch.cat([ranges[:, :fw], ranges[:, -fw:]], dim=1).min(dim=1).values
    sw         = max(1, int(num_rays * 45.0 / 360.0))
    rc         = int(num_rays * 0.25)
    lc         = int(num_rays * 0.75)
    right_min  = ranges[:, max(0, rc - sw // 2):min(num_rays, rc + sw // 2)].min(dim=1).values
    left_min   = ranges[:, max(0, lc - sw // 2):min(num_rays, lc + sw // 2)].min(dim=1).values
    return min_obstacle_dist, front_min, right_min, left_min


def _get_goal_distance_and_angle(env: ManagerBasedRLEnv):
    robot    = env.scene["robot"]
    robot_xy = robot.data.root_pos_w[:, :2]
    goal_xy  = env.goal_pos_w
    diff     = goal_xy - robot_xy
    goal_dist        = torch.norm(diff, dim=-1)
    heading_to_goal  = torch.atan2(diff[:, 1], diff[:, 0])
    quat = robot.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw  = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    goal_angle = torch.atan2(torch.sin(heading_to_goal - yaw), torch.cos(heading_to_goal - yaw))
    return goal_dist, goal_angle


def _get_real_actions(env: ManagerBasedRLEnv):
    action = env.action_manager.action
    if action is None:
        action = torch.zeros((env.num_envs, 2), device=env.device)
    action = torch.clamp(action, -1.0, 1.0)
    if ENABLE_BACKWARD:
        action_linear = action[:, 0] * MAX_LINEAR_SPEED
    else:
        action_linear = (action[:, 0] + 1.0) * 0.5 * MAX_LINEAR_SPEED
    action_angular = action[:, 1] * MAX_ANGULAR_SPEED
    return action_linear, action_angular


def _ensure_reward_buffers(env: ManagerBasedRLEnv, goal_dist: torch.Tensor):
    if not hasattr(env, "goal_dist_initial"):
        env.goal_dist_initial = goal_dist.clone()
    if not hasattr(env, "goal_dist_prev"):
        env.goal_dist_prev = goal_dist.clone()


def _ensure_stage3_buffers(env, goal_dist, min_obstacle_dist):
    _ensure_reward_buffers(env, goal_dist)
    if (not hasattr(env, "prev_min_obstacle_dist")
            or env.prev_min_obstacle_dist.shape != min_obstacle_dist.shape):
        env.prev_min_obstacle_dist = min_obstacle_dist.clone()


def _get_lidar_temporal_sector_diff_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Raw per-sector lidar diff (negative = obstacle closer than last step)."""
    lidar  = env.scene["lidar"]
    ranges = torch.norm(lidar.data.ray_hits_w - lidar.data.pos_w.unsqueeze(1), dim=-1)
    ranges = torch.nan_to_num(ranges, nan=LIDAR_DISTANCE_CAP,
                               posinf=LIDAR_DISTANCE_CAP, neginf=LIDAR_DISTANCE_CAP)
    ranges = torch.clamp(ranges, 0.0, LIDAR_DISTANCE_CAP) / LIDAR_DISTANCE_CAP
    current_lidar = ranges.reshape(env.num_envs, -1)

    if (not hasattr(env, "reward_prev_lidar_scan")
            or env.reward_prev_lidar_scan.shape != current_lidar.shape):
        env.reward_prev_lidar_scan = current_lidar.clone()
        return torch.zeros((env.num_envs, 8), device=env.device)

    diff = torch.clamp(current_lidar - env.reward_prev_lidar_scan, -1.0, 1.0)
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        if reset_mask.any():
            diff[reset_mask] = 0.0

    sectors = torch.tensor_split(diff, 8, dim=1)
    sector_diff = torch.cat([s.min(dim=1, keepdim=True).values for s in sectors], dim=1)
    env.reward_prev_lidar_scan = current_lidar.clone()
    return sector_diff


def navigation_reward_stage4(env: ManagerBasedRLEnv) -> torch.Tensor:
    min_obstacle_dist, front_min, right_min, left_min = _get_lidar_distances(env)
    goal_dist, goal_angle = _get_goal_distance_and_angle(env)
    action_linear, action_angular = _get_real_actions(env)
    _ensure_stage3_buffers(env, goal_dist, min_obstacle_dist)

    temporal_diff  = _get_lidar_temporal_sector_diff_reward(env)
    front_temporal = temporal_diff[:, 0]
    left_temporal  = torch.minimum(temporal_diff[:, 1], temporal_diff[:, 2])
    right_temporal = torch.minimum(temporal_diff[:, 6], temporal_diff[:, 7])

    # Ego-motion compensation: subtract robot's forward contribution from front diff
    # so front_closing detects actual obstacle motion, not robot approaching a wall.
    CLOSING_THRESH    = -0.0001
    robot_fwd         = env.scene["robot"].data.root_lin_vel_b[:, 0]
    robot_front_diff  = robot_fwd * env.step_dt / LIDAR_DISTANCE_CAP
    front_closing = (front_temporal + robot_front_diff) < CLOSING_THRESH
    left_closing  = left_temporal  < CLOSING_THRESH
    right_closing = right_temporal < CLOSING_THRESH

    front_blocked = front_min < 0.40
    side_tight    = (left_min < 0.35) | (right_min < 0.35)

    front_dynamic_risk = (front_min < 0.50) & front_closing
    left_dynamic_risk  = (left_min  < 0.40) & left_closing
    right_dynamic_risk = (right_min < 0.40) & right_closing
    front_early_risk   = (front_min < 0.75) & front_closing

    dynamic_blocked = front_dynamic_risk | left_dynamic_risk | right_dynamic_risk | front_early_risk
    tight_space     = front_blocked | side_tight | dynamic_blocked
    detour_needed   = front_blocked | front_dynamic_risk | left_dynamic_risk | right_dynamic_risk

    # ── 1. Yaw ────────────────────────────────────────────────────────────────
    near_goal_heading  = goal_dist < 0.40
    corner_trapped     = front_blocked & side_tight
    has_escape_room    = (left_min > 0.40) | (right_min > 0.40)
    open_dodge_active  = front_early_risk & has_escape_room & ~near_goal_heading
    # Static wall with escape room: robot needs to detour freely, so yaw pull is near-zero.
    static_wall_detour = front_blocked & ~front_closing & has_escape_room & ~near_goal_heading
    r_yaw = torch.where(
        static_wall_detour, -0.20 * torch.abs(goal_angle),
        torch.where(open_dodge_active,  -0.40 * torch.abs(goal_angle),
        torch.where(corner_trapped, -4.00 * torch.abs(goal_angle),
        torch.where(near_goal_heading & ~detour_needed, -4.00 * torch.abs(goal_angle),
        torch.where(detour_needed, -1.50 * torch.abs(goal_angle),
                    -2.00 * torch.abs(goal_angle))))))

    # ── 2. Angular penalty ────────────────────────────────────────────────────
    misaligned  = torch.abs(goal_angle) > 0.5
    r_vangular  = torch.where(misaligned, -0.10 * (action_angular ** 2),
                  torch.where(detour_needed, -0.50 * (action_angular ** 2),
                              -1.00 * (action_angular ** 2)))

    # ── 3. Goal progress ──────────────────────────────────────────────────────
    progress   = env.goal_dist_prev - goal_dist
    r_distance = progress * 60.0
    env.goal_dist_prev[:] = goal_dist

    # ── 4. Obstacle soft penalty (safe_dist=0.65 — do not lower) ─────────────
    safe_dist       = 0.65
    terminal_dist   = 0.30
    r_obstacle_soft = -6.0 * torch.clamp(
        (safe_dist - min_obstacle_dist) / (safe_dist - terminal_dist), 0.0, 1.0) ** 2
    r_obstacle_hard = torch.where(
        min_obstacle_dist < terminal_dist,
        torch.full_like(min_obstacle_dist, -20.0),
        torch.zeros_like(min_obstacle_dist))
    r_obstacle = r_obstacle_soft + r_obstacle_hard

    # ── 5. Linear speed (halved in tight/dynamic zone) ────────────────────────
    near_goal      = goal_dist < 0.40
    r_vlinear_open = -(((MAX_LINEAR_SPEED - action_linear) * 10.0) ** 2)
    r_vlinear      = torch.where(near_goal, torch.zeros_like(action_linear),
                     torch.where(tight_space, 0.50 * r_vlinear_open, r_vlinear_open))

    # ── 6. Front push penalty (static block only) ─────────────────────────────
    r_front_push = torch.where(
        (front_min < 0.35) & (action_linear > 0.08),
        torch.full_like(goal_dist, -2.0),
        torch.zeros_like(goal_dist))

    # ── 7. Clearance recovery ─────────────────────────────────────────────────
    clearance_delta = min_obstacle_dist - env.prev_min_obstacle_dist
    env.prev_min_obstacle_dist[:] = min_obstacle_dist
    r_clearance_recovery = torch.where(
        front_min < 0.50,
        torch.clamp(clearance_delta, -0.03, 0.03) * 80.0,
        torch.zeros_like(goal_dist))

    # ── 8. Anti-stuck ─────────────────────────────────────────────────────────
    genuinely_trapped = ~((left_min > 0.40) | (right_min > 0.40))
    yielding = (front_dynamic_risk & genuinely_trapped) | left_dynamic_risk | right_dynamic_risk
    no_progress = (torch.abs(progress) < 0.003) & (goal_dist > 0.28) & ~yielding
    # Near obstacles: only penalize when also moving slowly (avoid forcing into walls).
    r_stuck  = torch.where(
        no_progress & (action_linear < 0.06),
        torch.full_like(goal_dist, -4.0),
        torch.zeros_like(goal_dist))
    # Open space: penalize any no-progress regardless of action (spinning after dodge).
    r_open_stuck = torch.where(
        no_progress & (min_obstacle_dist > 0.50),
        torch.full_like(goal_dist, -3.0),
        torch.zeros_like(goal_dist))

    # ── 9. Front-dodge (corridor-aware) ───────────────────────────────────────
    open_side_turn = torch.where(
        left_min > right_min,
        torch.clamp(action_angular,  0.0, 1.0),
        torch.clamp(-action_angular, 0.0, 1.0))

    # Dynamic obstacle approaching from front.
    r_front_dodge = torch.where(
        front_early_risk & has_escape_room & ~near_goal_heading,
        open_side_turn * 0.30,
        torch.zeros_like(goal_dist))

    # Static wall ahead but escape room exists: reward turning toward open side.
    # Ego-motion compensation makes front_closing=False for static walls, so
    # r_front_dodge never fires — this term fills that gap.
    static_wall_blocked = front_blocked & ~front_closing & has_escape_room & ~near_goal_heading
    r_static_dodge = torch.where(
        static_wall_blocked,
        open_side_turn * 0.50,
        torch.zeros_like(goal_dist))

    # ── 10. Goal-approach brake ───────────────────────────────────────────────
    r_goal_brake = torch.where(
        goal_dist < 0.35,
        -action_linear * 0.5,
        torch.zeros_like(goal_dist))

    reward = (
        r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular
        + r_front_push + r_clearance_recovery + r_stuck + r_open_stuck + r_front_dodge + r_static_dodge + r_goal_brake
        - 1.0
    )
    return reward


def navigation_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    return navigation_reward_stage4(env)


def terminal_reward_ppo(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminal bonus/penalty for PPO (TD3 applies these externally)."""
    n, device = env.num_envs, env.device
    goal_buf = getattr(env, "goal_reached_buf", torch.zeros(n, dtype=torch.bool, device=device))
    coll_buf = getattr(env, "collision_buf",    torch.zeros(n, dtype=torch.bool, device=device))
    r = torch.zeros(n, device=device)
    r = torch.where(goal_buf, r + SUCCESS_REWARD, r)
    r = torch.where(coll_buf, r - COLLISION_PENALTY, r)
    return r


@configclass
class RewardsCfg:
    navigation = RewTerm(func=navigation_reward, weight=1.0)


@configclass
class PPORewardsCfg(RewardsCfg):
    terminal = RewTerm(func=terminal_reward_ppo, weight=1.0)
