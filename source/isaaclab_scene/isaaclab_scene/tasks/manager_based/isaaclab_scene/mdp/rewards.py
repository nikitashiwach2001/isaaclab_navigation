import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass


# -------------------------
# Gazebo-style constants
# -------------------------
LIDAR_DISTANCE_CAP = 3.5

MAX_LINEAR_SPEED = 0.22   # must match actions.py
MAX_ANGULAR_SPEED = 2.0   # must match actions.py

ENABLE_BACKWARD = False

THRESHOLD_GOAL = 0.25
THRESHOLD_COLLISION = 0.30

SUCCESS_REWARD = 500.0
COLLISION_PENALTY = 300.0

REWARD_FUNCTION = "stage4"


def _get_lidar_distances(env: ManagerBasedRLEnv):
    """
    Return global min LiDAR distance and sector min distances.

    Assumption:
    - Around 90 rays
    - ray_alignment = "yaw"
    - circular 360-degree lidar

    Sectors are approximate:
    - front: around ray 0 and ray end
    - right: one side sector
    - left: opposite side sector
    """

    lidar = env.scene["lidar"]

    ray_hits_w = lidar.data.ray_hits_w
    sensor_pos_w = lidar.data.pos_w

    ranges = torch.norm(ray_hits_w - sensor_pos_w.unsqueeze(1), dim=-1)

    ranges = torch.nan_to_num(
        ranges,
        nan=LIDAR_DISTANCE_CAP,
        posinf=LIDAR_DISTANCE_CAP,
        neginf=LIDAR_DISTANCE_CAP,
    )

    ranges = torch.clamp(ranges, 0.0, LIDAR_DISTANCE_CAP)
    ranges = ranges.reshape(env.num_envs, -1)

    min_obstacle_dist = ranges.min(dim=1).values

    num_rays = ranges.shape[1]

    # For 90 rays:
    # 1 ray ≈ 4 degrees.
    # Front sector: roughly +/- 24 degrees around forward.
    front_width = max(1, int(num_rays * 24.0 / 360.0))  # around 6 rays for 90

    front_min = torch.cat(
        [ranges[:, :front_width], ranges[:, -front_width:]],
        dim=1,
    ).min(dim=1).values

    # Side sectors:
    # Approx right side around 90 degrees.
    # Approx left side around 270 degrees.
    # These may swap depending on ray ordering, but both sides are used symmetrically.
    side_width = max(1, int(num_rays * 45.0 / 360.0))  # around 11 rays for 90

    right_center = int(num_rays * 0.25)
    left_center = int(num_rays * 0.75)

    right_start = max(0, right_center - side_width // 2)
    right_end = min(num_rays, right_center + side_width // 2)

    left_start = max(0, left_center - side_width // 2)
    left_end = min(num_rays, left_center + side_width // 2)

    right_min = ranges[:, right_start:right_end].min(dim=1).values
    left_min = ranges[:, left_start:left_end].min(dim=1).values

    return min_obstacle_dist, front_min, right_min, left_min


def _get_goal_distance_and_angle(env: ManagerBasedRLEnv):
    """Return goal distance and goal angle per env."""

    robot = env.scene["robot"]

    robot_xy = robot.data.root_pos_w[:, :2]
    goal_xy = env.goal_pos_w

    diff = goal_xy - robot_xy

    goal_dist = torch.norm(diff, dim=-1)
    heading_to_goal = torch.atan2(diff[:, 1], diff[:, 0])

    quat = robot.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

    goal_angle = heading_to_goal - yaw
    goal_angle = torch.atan2(torch.sin(goal_angle), torch.cos(goal_angle))

    return goal_dist, goal_angle


def _get_real_actions(env: ManagerBasedRLEnv):
    """Convert normalized action [-1, 1] to real linear/angular velocity."""

    action = env.action_manager.action

    if action is None:
        action = torch.zeros((env.num_envs, 2), device=env.device)

    action = torch.clamp(action, -1.0, 1.0)

    linear_norm = action[:, 0]
    angular_norm = action[:, 1]

    if ENABLE_BACKWARD:
        action_linear = linear_norm * MAX_LINEAR_SPEED
    else:
        action_linear = (linear_norm + 1.0) * 0.5 * MAX_LINEAR_SPEED

    action_angular = angular_norm * MAX_ANGULAR_SPEED

    return action_linear, action_angular


def _ensure_reward_buffers(env: ManagerBasedRLEnv, goal_dist: torch.Tensor):
    """Create reward distance buffers if missing."""

    if not hasattr(env, "goal_dist_initial"):
        env.goal_dist_initial = goal_dist.clone()

    if not hasattr(env, "goal_dist_prev"):
        env.goal_dist_prev = goal_dist.clone()


def _get_lidar_temporal_sector_diff_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Reward-side temporal LiDAR sector diff.

    Shape:
        [num_envs, 8]

    Sector order:
        0 front
        1 front-left
        2 left
        3 back-left
        4 back
        5 back-right
        6 right
        7 front-right

    Meaning:
        negative = free-space reduced / obstacle closer
        positive = free-space increased / obstacle farther
    """

    lidar = env.scene["lidar"]

    ray_hits_w = lidar.data.ray_hits_w
    sensor_pos_w = lidar.data.pos_w

    ranges = torch.norm(ray_hits_w - sensor_pos_w.unsqueeze(1), dim=-1)

    ranges = torch.nan_to_num(
        ranges,
        nan=LIDAR_DISTANCE_CAP,
        posinf=LIDAR_DISTANCE_CAP,
        neginf=LIDAR_DISTANCE_CAP,
    )

    ranges = torch.clamp(ranges, 0.0, LIDAR_DISTANCE_CAP)
    ranges = ranges / LIDAR_DISTANCE_CAP

    current_lidar = ranges.reshape(env.num_envs, -1)

    if (
        not hasattr(env, "reward_prev_lidar_scan")
        or env.reward_prev_lidar_scan.shape != current_lidar.shape
    ):
        env.reward_prev_lidar_scan = current_lidar.clone()
        return torch.zeros((env.num_envs, 8), device=env.device)

    prev_lidar = env.reward_prev_lidar_scan

    diff = current_lidar - prev_lidar
    diff = torch.clamp(diff, -1.0, 1.0)

    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        if reset_mask.any():
            diff[reset_mask] = 0.0

    sectors = torch.tensor_split(diff, 8, dim=1)

    sector_features = []
    for sector in sectors:
        sector_min = torch.min(sector, dim=1, keepdim=True).values
        sector_features.append(sector_min)

    sector_diff = torch.cat(sector_features, dim=1)

    # No deadband: 0.0003 threshold zeroed slow obstacles (0.075 m/s) when robot
    # was stopped (raw diff 0.000321 < 0.0003). CLOSING_THRESH in the reward
    # function serves as the noise floor instead.

    env.reward_prev_lidar_scan = current_lidar.clone()

    return sector_diff


def _ensure_stage3_buffers(env, goal_dist, min_obstacle_dist):
    _ensure_reward_buffers(env, goal_dist)

    if (
        not hasattr(env, "prev_min_obstacle_dist")
        or env.prev_min_obstacle_dist.shape != min_obstacle_dist.shape
    ):
        env.prev_min_obstacle_dist = min_obstacle_dist.clone()



def navigation_reward_A_stage3_temporal(env: ManagerBasedRLEnv) -> torch.Tensor:
    min_obstacle_dist, front_min, right_min, left_min = _get_lidar_distances(env)
    goal_dist, goal_angle = _get_goal_distance_and_angle(env)
    action_linear, action_angular = _get_real_actions(env)

    near_goal = goal_dist < 0.80


    _ensure_stage3_buffers(env, goal_dist, min_obstacle_dist)

    success = goal_dist < THRESHOLD_GOAL
    collision = min_obstacle_dist < THRESHOLD_COLLISION


    temporal_diff = _get_lidar_temporal_sector_diff_reward(env)

    front_temporal = temporal_diff[:, 0]
    right_temporal = torch.minimum(temporal_diff[:, 1], temporal_diff[:, 2])
    left_temporal = torch.minimum(temporal_diff[:, 6], temporal_diff[:, 7])


    front_blocked = front_min < 0.65
    side_tight = (left_min < 0.45) | (right_min < 0.45)
    tight_space = front_blocked | side_tight

    # stage4: lowered to match 0.001 dead-zone (was -0.007, unreachable from continuous motion)
    # front_closing = front_temporal < -0.007
    # left_closing = left_temporal < -0.007
    # right_closing = right_temporal < -0.007
    front_closing = front_temporal < -0.0003
    left_closing = left_temporal < -0.0003
    right_closing = right_temporal < -0.0003

    front_danger = front_min < 0.50
    left_danger = left_min < 0.40
    right_danger = right_min < 0.40

    fast_forward = action_linear > 0.12

    front_dynamic_risk = front_danger & front_closing
    left_dynamic_risk = left_danger & left_closing
    right_dynamic_risk = right_danger & right_closing

    # fix: fire anywhere
    dynamic_blocked = front_dynamic_risk | left_dynamic_risk | right_dynamic_risk
    detour_needed = front_blocked | dynamic_blocked

    r_yaw = torch.where(
        detour_needed,
        -0.20 * torch.abs(goal_angle),
        -1.00 * torch.abs(goal_angle),
    )

    r_vangular = torch.where(
        detour_needed,
        -0.10 * (action_angular ** 2),
        -1.00 * (action_angular ** 2),
    )

    progress = env.goal_dist_prev - goal_dist
    r_distance = progress * 30.0
    env.goal_dist_prev[:] = goal_dist

    safe_dist = 0.45
    critical_dist = 0.22

    r_obstacle = -3.0 * torch.clamp(
        (safe_dist - min_obstacle_dist) / safe_dist,
        0.0,
        1.0,
    ) ** 2

    r_obstacle = torch.where(
        min_obstacle_dist < critical_dist,
        r_obstacle - 20.0,
        r_obstacle,
    )

    r_vlinear_open = -(((MAX_LINEAR_SPEED - action_linear) * 10.0) ** 2)

    r_vlinear_tight = torch.where(
        action_linear > 0.06,
        action_linear * 0.5,
        torch.full_like(action_linear, -0.4),
    )

    r_vlinear = torch.where(
        tight_space,
        r_vlinear_tight,
        r_vlinear_open,
    )



    closing_danger = (
        (front_danger & front_closing)
        | (left_danger & left_closing)
        | (right_danger & right_closing)
    )

    r_closing_fast = torch.where(
        closing_danger & fast_forward,
        torch.full_like(goal_dist, -0.2),
        torch.zeros_like(goal_dist),
    )

    clearance_delta = min_obstacle_dist - env.prev_min_obstacle_dist
    env.prev_min_obstacle_dist[:] = min_obstacle_dist

    very_close = min_obstacle_dist < 0.40

    r_clearance_recovery = torch.where(
        very_close,
        torch.clamp(clearance_delta, -0.03, 0.03) * 40.0,
        torch.zeros_like(goal_dist),
    )

    # goal_area_blocked = near_goal & (front_dynamic_risk | ...)


    # r_near_goal_rush = torch.where(
    #     goal_area_blocked & (action_linear > 0.08),
    #     torch.full_like(goal_dist, -1.0),
    #     torch.zeros_like(goal_dist),
    # )

    no_progress = torch.abs(progress) < 0.003
    not_near_goal = goal_dist > 0.40
    low_linear = action_linear < 0.06
    low_angular = torch.abs(action_angular) < 0.08

    r_stuck = torch.where(
        no_progress & not_near_goal & low_linear & (min_obstacle_dist > 0.55),
        -1.0, 0.0
    )


    reward = (
        r_yaw
        + r_distance
        + r_obstacle
        + r_vlinear
        + r_vangular
        + r_closing_fast
        + r_clearance_recovery
        # + r_near_goal_rush
        + r_stuck
        - 1.0
    )

    reward = torch.where(success, reward + SUCCESS_REWARD, reward)
    reward = torch.where(collision, reward - COLLISION_PENALTY, reward)

    return reward


def navigation_reward_A(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gazebo get_reward_B converted to Isaac Lab vectorized reward."""

    min_obstacle_dist, front_min, right_min, left_min = _get_lidar_distances(env)
    goal_dist, goal_angle = _get_goal_distance_and_angle(env)
    action_linear, action_angular = _get_real_actions(env)

    _ensure_reward_buffers(env, goal_dist)

    success = goal_dist < THRESHOLD_GOAL
    collision = min_obstacle_dist < THRESHOLD_COLLISION

    near_obstacle = min_obstacle_dist < 0.55

    # [-3.14, 0]
    r_yaw = -torch.abs(goal_angle)
    r_yaw = torch.where(success, torch.zeros_like(r_yaw), r_yaw)

    # Same as Gazebo: [-4, 0] when angular max is 2.0
    # r_vangular = -1.0 * (action_angular**2)
    r_vangular = torch.where(
        near_obstacle,
        -0.10 * (action_angular ** 2),
        -1.00 * (action_angular ** 2),
    )

    # Delta-based distance reward
    progress = env.goal_dist_prev - goal_dist
    r_distance = progress * 30.0
    env.goal_dist_prev[:] = goal_dist

    # Same as Gazebo: obstacle penalty when below 0.22m
    r_obstacle = torch.where(
        min_obstacle_dist < 0.22,
        torch.full_like(min_obstacle_dist, -20.0),
        torch.zeros_like(min_obstacle_dist),
    )

    # Same as Gazebo: prefer max forward speed 0.22
    r_vlinear = -(((MAX_LINEAR_SPEED - action_linear) * 10.0) ** 2)


    reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular  - 1.0

    reward = torch.where(success, reward + SUCCESS_REWARD, reward)
    reward = torch.where(collision, reward - COLLISION_PENALTY, reward)

    return reward


def navigation_reward_B(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 4B reward: obstacle-avoidance-first navigation without temporal diff."""

    min_obstacle_dist, front_min, right_min, left_min = _get_lidar_distances(env)
    goal_dist, goal_angle = _get_goal_distance_and_angle(env)
    action_linear, action_angular = _get_real_actions(env)

    _ensure_reward_buffers(env, goal_dist)

    success = goal_dist < THRESHOLD_GOAL
    collision = min_obstacle_dist < THRESHOLD_COLLISION

    # Main obstacle zones
    near_obstacle = min_obstacle_dist < 0.65
    front_blocked = front_min < 0.55

    # --------------------------------------------------
    # 1. Yaw reward
    # Open space: face goal.
    # Near obstacle: weak goal pressure.
    # Front blocked: no goal-facing pressure, allow detour.
    # --------------------------------------------------
    r_yaw = torch.where(
        front_blocked,
        torch.zeros_like(goal_angle),
        torch.where(
            near_obstacle,
            -0.03 * torch.abs(goal_angle),
            -1.00 * torch.abs(goal_angle),
        ),
    )

    r_yaw = torch.where(
        success,
        torch.zeros_like(r_yaw),
        r_yaw,
    )

    # --------------------------------------------------
    # 2. Angular penalty
    # Near obstacles, allow sharp turns.
    # --------------------------------------------------
    r_vangular = torch.where(
        near_obstacle,
        -0.001 * (action_angular ** 2),
        -0.50 * (action_angular ** 2),
    )

    # --------------------------------------------------
    # 3. Goal progress reward
    # Near obstacles, reduce goal pressure so it does not push into pillars/walls.
    # --------------------------------------------------
    progress = env.goal_dist_prev - goal_dist

    r_distance = torch.where(
        near_obstacle,
        progress * 8.0,
        progress * 20.0,
    )

    env.goal_dist_prev[:] = goal_dist

    # --------------------------------------------------
    # 4. Smooth obstacle penalty
    # Gives warning before actual collision.
    # --------------------------------------------------
    safe_dist = 0.55
    critical_dist = THRESHOLD_COLLISION

    r_obstacle_soft = -6.0 * torch.clamp(
        (safe_dist - min_obstacle_dist) / (safe_dist - critical_dist),
        0.0,
        1.0,
    ) ** 2

    r_obstacle_hard = torch.where(
        min_obstacle_dist < THRESHOLD_COLLISION,
        torch.full_like(min_obstacle_dist, -25.0),
        torch.zeros_like(min_obstacle_dist),
    )

    r_obstacle = r_obstacle_soft + r_obstacle_hard

    # --------------------------------------------------
    # 5. Linear velocity reward
    # Open space: encourage normal forward motion.
    # Near obstacle: remove max-speed pressure.
    # --------------------------------------------------
    r_vlinear_open = -(((MAX_LINEAR_SPEED - action_linear) * 10.0) ** 2)

    r_vlinear = torch.where(
        near_obstacle,
        torch.zeros_like(action_linear),
        r_vlinear_open,
    )

    # --------------------------------------------------
    # 6. Penalize pushing forward into front obstacle
    # This is not forced left/right action.
    # It only says: don't drive straight into a front obstacle.
    # --------------------------------------------------
    r_front_push = torch.where(
        front_blocked & (action_linear > 0.08),
        torch.full_like(goal_dist, -2.0),
        torch.zeros_like(goal_dist),
    )

    # --------------------------------------------------
    # 7. Anti-stuck and anti-freeze
    # --------------------------------------------------
    not_near_goal = goal_dist > 0.40
    no_progress = torch.abs(progress) < 0.005

    low_linear = action_linear < 0.08
    low_angular = torch.abs(action_angular) < 0.08
    truly_frozen = low_linear & low_angular

    # Only penalize stuck in open space.
    # Near obstacles, no progress can happen during valid turning/maneuvering.
    r_stuck = torch.where(
        no_progress & not_near_goal & (~near_obstacle),
        torch.full_like(goal_dist, -0.5),
        torch.zeros_like(goal_dist),
    )

    # Only penalize true freezing near obstacles.
    # Do not punish turning/maneuvering near obstacles.
    r_freeze = torch.where(
        near_obstacle & truly_frozen & no_progress & not_near_goal,
        torch.full_like(goal_dist, -1.5),
        torch.zeros_like(goal_dist),
    )
    # --------------------------------------------------
    # Final reward
    # --------------------------------------------------
    reward = (
        r_yaw
        + r_distance
        + r_obstacle
        + r_vlinear
        + r_vangular
        + r_stuck
        + r_freeze
        + r_front_push
        - 1.0
    )

    reward = torch.where(success, reward + SUCCESS_REWARD, reward)
    reward = torch.where(collision, reward - COLLISION_PENALTY, reward)

    return reward

def _debug_lidar_vs_obstacle(env, min_obstacle_dist, front_min):
    """
    Verification helper: compare LiDAR-reported distances to true obstacle
    distances computed from physics positions.  Run for ~200 steps then remove.

    Expected output when LiDAR is BROKEN (merge_prim_meshes=True):
        actual_obs1=0.42  lidar_min=2.31  front=2.60   << obstacle invisible
    Expected output when LiDAR is FIXED  (merge_prim_meshes=False):
        actual_obs1=0.42  lidar_min=0.43  front=0.44   << obstacle tracked
    """
    if not hasattr(env, "_dbg_step"):
        env._dbg_step = 0
    env._dbg_step += 1
    if env._dbg_step > 200 or env._dbg_step % 10 != 0:
        return

    robot  = env.scene["robot"]
    obs1   = env.scene["obstacle_1"]
    obs2   = env.scene["obstacle_2"]

    rp  = robot.data.root_pos_w[0, :2]
    o1p = obs1.data.root_pos_w[0, :2]
    o2p = obs2.data.root_pos_w[0, :2]

    d1 = torch.norm(o1p - rp).item()
    d2 = torch.norm(o2p - rp).item()

    print(
        f"[LIDAR_DBG step={env._dbg_step:3d}] "
        f"actual_obs1={d1:.2f}  actual_obs2={d2:.2f} | "
        f"lidar_global_min={min_obstacle_dist[0].item():.2f}  "
        f"lidar_front={front_min[0].item():.2f}"
    )


def navigation_reward_stage4(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 4 reward: static maze + two moving obstacles, temporal-diff aware."""

    min_obstacle_dist, front_min, right_min, left_min = _get_lidar_distances(env)
    goal_dist, goal_angle = _get_goal_distance_and_angle(env)
    action_linear, action_angular = _get_real_actions(env)

    _ensure_stage3_buffers(env, goal_dist, min_obstacle_dist)

    # _debug_lidar_vs_obstacle(env, min_obstacle_dist, front_min)

    # ── Temporal diff ────────────────────────────────────────────────────────
    temporal_diff  = _get_lidar_temporal_sector_diff_reward(env)
    # sector order: 0=front, 1=front-left, 2=left, 3=back-left,
    #               4=back,  5=back-right, 6=right, 7=front-right
    front_temporal = temporal_diff[:, 0]
    left_temporal  = torch.minimum(temporal_diff[:, 1], temporal_diff[:, 2])
    right_temporal = torch.minimum(temporal_diff[:, 6], temporal_diff[:, 7])

    # ── Closing flags (obstacle moving toward robot in that sector) ───────────
    CLOSING_THRESH = -0.0003
    front_closing = front_temporal < CLOSING_THRESH
    left_closing  = left_temporal  < CLOSING_THRESH
    right_closing = right_temporal < CLOSING_THRESH

    # ── Distance zones ────────────────────────────────────────────────────────
    front_blocked = front_min < 0.50
    side_tight    = (left_min < 0.35) | (right_min < 0.35)

    # dynamic risk: obstacle close AND closing.
    front_dynamic_risk = (front_min < 0.50) & front_closing
    left_dynamic_risk  = (left_min  < 0.40) & left_closing
    right_dynamic_risk = (right_min < 0.40) & right_closing

    # early warning: moderately close AND closing — only used to slow down,
    # NOT to relax heading (prevents constant detour mode in the small arena)
    front_early_risk = (front_min < 0.75) & front_closing

    dynamic_blocked = (
        front_dynamic_risk | left_dynamic_risk | right_dynamic_risk | front_early_risk
    )

    # tight_space: slow down for static proximity OR any dynamic approach
    tight_space = front_blocked | side_tight | dynamic_blocked

    # detour_needed: only relax heading when obstacle is genuinely close/dangerous.
    # Excludes front_early_risk — early warning slows speed but keeps goal heading.
    detour_needed = front_blocked | front_dynamic_risk | left_dynamic_risk | right_dynamic_risk

    # Both sides > 0.60m → no corridor walls nearby → closing front object must
    # be a dynamic obstacle. Defined here so both r_yaw and r_front_dodge use it.
    in_open_space = (left_min > 0.60) & (right_min > 0.60)

    # ── 1. Yaw reward ─────────────────────────────────────────────────────────
    # Near goal: stronger heading so robot doesn't orbit the goal.
    near_goal_heading = goal_dist < 0.40
    corner_trapped    = front_blocked & side_tight   # both front and side walls close
    # During an open-space dynamic dodge, reduce yaw penalty so r_front_dodge
    # (magnitude 1.0) can override heading and commit to a decisive evasive turn.
    open_dodge_active = front_early_risk & in_open_space & ~near_goal_heading
    r_yaw = torch.where(
        open_dodge_active,
        -0.40 * torch.abs(goal_angle),   # allow decisive evasive turn
        torch.where(
            corner_trapped,
            -4.00 * torch.abs(goal_angle),   # urgent: face goal to escape corner
            torch.where(
                near_goal_heading & ~detour_needed,
                -4.00 * torch.abs(goal_angle),
                torch.where(
                    detour_needed,
                    -1.50 * torch.abs(goal_angle),
                    -2.00 * torch.abs(goal_angle),  # open space: 2x stronger to compete with r_vlinear
                ),
            ),
        ),
    )

    # ── 2. Angular velocity penalty ───────────────────────────────────────────
    # When robot is misaligned with goal (needs to turn), reduce angular penalty
    # so it can make sharp corrections. When well-aligned, penalise spinning.
    misaligned = torch.abs(goal_angle) > 0.5   # ~30 degrees off
    r_vangular = torch.where(
        misaligned,
        -0.10 * (action_angular ** 2),   # allow sharp turns to re-acquire goal
        torch.where(
            detour_needed,
            -0.50 * (action_angular ** 2),
            -1.00 * (action_angular ** 2),
        ),
    )

    # ── 3. Goal progress ──────────────────────────────────────────────────────
    progress   = env.goal_dist_prev - goal_dist
    r_distance = progress * 60.0
    env.goal_dist_prev[:] = goal_dist

    # ── 4. Obstacle penalty (soft gradient + hard step at termination boundary)
    # Termination fires at 0.30 m (terminations.py THRESHOLD_COLLISION).
    # Uses global min_obstacle_dist — required so the robot avoids side walls in
    # maze corridors. safe_dist=0.65 was established in v6 and dropped COLL_STATIC
    # from 14.6% → 2.7%; do not lower it.
    safe_dist        = 0.65
    terminal_dist    = 0.30
    r_obstacle_soft  = -6.0 * torch.clamp(
        (safe_dist - min_obstacle_dist) / (safe_dist - terminal_dist),
        0.0, 1.0,
    ) ** 2
    r_obstacle_hard = torch.where(
        min_obstacle_dist < terminal_dist,
        torch.full_like(min_obstacle_dist, -20.0),
        torch.zeros_like(min_obstacle_dist),
    )
    r_obstacle = r_obstacle_soft + r_obstacle_hard

    # ── 5. Linear speed ───────────────────────────────────────────────────────
    # Open space: full pressure toward max speed.
    # Tight / dynamic zone: 50% pressure — prevents speed collapse and timeout
    # while still allowing the robot to slow when needed.
    # Near goal: no pressure so robot doesn't overshoot.
    near_goal = goal_dist < 0.40
    r_vlinear_open = -(((MAX_LINEAR_SPEED - action_linear) * 10.0) ** 2)
    r_vlinear = torch.where(
        near_goal,
        torch.zeros_like(action_linear),
        torch.where(
            tight_space,
            0.50 * r_vlinear_open,
            r_vlinear_open,
        ),
    )

    # ── 6. Penalise rushing into a closing obstacle ───────────────────────────
    closing_danger = front_dynamic_risk | left_dynamic_risk | right_dynamic_risk
    r_closing_fast = torch.where(
        closing_danger & (action_linear > 0.12),
        torch.full_like(goal_dist, -1.5),
        torch.zeros_like(goal_dist),
    )

    # ── 7. Penalise pushing straight into a static front block ────────────────
    # Only fires at 0.35m (not 0.50m) to avoid constant stop-penalty that
    # causes speed collapse in the confined 5×5m arena.
    front_blocked_push = front_min < 0.35
    r_front_push = torch.where(
        front_blocked_push & (action_linear > 0.08),
        torch.full_like(goal_dist, -2.0),
        torch.zeros_like(goal_dist),
    )

    # ── 8. Clearance recovery (front only) ───────────────────────────────────
    # Only fires when front is tight (<0.50m). Side walls in maze corridors are
    # constantly at 0.40-0.45m — triggering on side_close fires every step and
    # creates contradictory signals (global min is the wall, reward is noisy).
    clearance_delta = min_obstacle_dist - env.prev_min_obstacle_dist
    env.prev_min_obstacle_dist[:] = min_obstacle_dist
    r_clearance_recovery = torch.where(
        front_min < 0.50,
        torch.clamp(clearance_delta, -0.03, 0.03) * 80.0,
        torch.zeros_like(goal_dist),
    )

    # ── 9. Anti-stuck ────────────────────────────────────────────────────────────
    # min_obstacle_dist > 0.40 gate is kept: removing it caused r_stuck to fire
    # in normal corridor navigation (walls at 0.38m) → COLL_STATIC 30%+ (v27).
    # yielding: front obstacle excused only when genuinely trapped (both sides
    # tight); side risks always excuse stopping.
    no_progress       = torch.abs(progress) < 0.003
    not_near_goal     = goal_dist > 0.40
    has_dodge_room    = (left_min > 0.40) | (right_min > 0.40)
    genuinely_trapped = ~has_dodge_room
    yielding          = (front_dynamic_risk & genuinely_trapped) | left_dynamic_risk | right_dynamic_risk
    r_stuck = torch.where(
        no_progress & not_near_goal & (action_linear < 0.06) & (min_obstacle_dist > 0.40) & ~yielding,
        torch.full_like(goal_dist, -1.0),
        torch.zeros_like(goal_dist),
    )

    # ── 10. Open-space front-dodge ────────────────────────────────────────────
    # Triggers at front_early_risk (0.75m) not front_dynamic_risk (0.50m):
    #   at 0.50 m/s combined closing speed, 0.75m trigger → 73 reaction steps
    #   vs only 40 steps from 0.50m — an extra 0.5 seconds to commit to the turn.
    # Magnitude 1.0 (was 0.20): overrides the r_yaw penalty (reduced to -0.20
    #   via open_dodge_active) so the robot commits to a decisive evasive turn.
    # ~near_goal_heading: suppress during intentional goal approach near walls.
    open_side_turn = torch.where(
        left_min > right_min,
        torch.clamp(action_angular,  0.0, 1.0),   # left more open → CCW
        torch.clamp(-action_angular, 0.0, 1.0),   # right more open → CW
    )
    r_front_dodge = torch.where(
        front_early_risk & in_open_space & ~near_goal_heading,
        open_side_turn * 0.30,
        torch.zeros_like(goal_dist),
    )

    # ── 11. Goal-approach brake ───────────────────────────────────────────────
    # When very close to goal (< 0.35m), penalise high linear speed to prevent
    # the robot from overshooting the goal and hitting the boundary wall behind.
    r_goal_brake = torch.where(
        goal_dist < 0.35,
        -action_linear * 0.5,
        torch.zeros_like(goal_dist),
    )

    # ── Final ─────────────────────────────────────────────────────────────────
    # Terminal rewards (SUCCESS_REWARD / COLLISION_PENALTY) are applied once
    # by the training script on done steps — do NOT add them here to avoid
    # double-counting in the replay buffer.
    reward = (
        r_yaw
        + r_distance
        + r_obstacle
        + r_vlinear
        + r_vangular
        + r_closing_fast
        + r_front_push
        + r_clearance_recovery
        + r_stuck
        + r_front_dodge
        + r_goal_brake
        - 1.0
    )

    return reward


def navigation_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Select reward function."""

    if REWARD_FUNCTION == "stage4":
        return navigation_reward_stage4(env)
    elif REWARD_FUNCTION == "B":
        return navigation_reward_A_stage3_temporal(env)
    elif REWARD_FUNCTION == "A":
        return navigation_reward_B(env)

    raise ValueError(f"Unknown REWARD_FUNCTION: {REWARD_FUNCTION}")


def terminal_reward_ppo(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminal bonus/penalty for on-policy (PPO) training.

    train_td3.py applies SUCCESS_REWARD / COLLISION_PENALTY externally after env.step().
    RSL-RL's runner.learn() never does — so this RewTerm must be added to the env
    rewards config for any PPO task to receive the terminal signal.
    """
    n, device = env.num_envs, env.device
    goal_buf = getattr(env, "goal_reached_buf", torch.zeros(n, dtype=torch.bool, device=device))
    coll_buf = getattr(env, "collision_buf",    torch.zeros(n, dtype=torch.bool, device=device))
    r = torch.zeros(n, device=device)
    r = torch.where(goal_buf, r + SUCCESS_REWARD,    r)
    r = torch.where(coll_buf, r - COLLISION_PENALTY, r)
    return r


@configclass
class RewardsCfg:
    navigation = RewTerm(
        func=navigation_reward,
        weight=1.0,
    )


@configclass
class PPORewardsCfg(RewardsCfg):
    """RewardsCfg with terminal signals for on-policy (PPO) training."""
    terminal = RewTerm(
        func=terminal_reward_ppo,
        weight=1.0,
    )