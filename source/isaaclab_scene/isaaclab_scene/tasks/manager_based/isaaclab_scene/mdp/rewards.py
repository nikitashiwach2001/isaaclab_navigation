import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass


# -------------------------
# Gazebo-style constants
# -------------------------
LIDAR_DISTANCE_CAP = 3.5

MAX_LINEAR_SPEED = 0.35
MAX_ANGULAR_SPEED = 1.5

ENABLE_BACKWARD = False

THRESHOLD_GOAL = 0.25
THRESHOLD_COLLISION = 0.22

SUCCESS_REWARD = 2500.0
COLLISION_PENALTY = 2000.0

REWARD_FUNCTION = "B"


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

    # stage4: matches observation threshold
    # sector_diff = torch.where(
    #     torch.abs(sector_diff) < 0.005,
    #     torch.zeros_like(sector_diff),
    #     sector_diff,
    # )
    sector_diff = torch.where(
        torch.abs(sector_diff) < 0.001,
        torch.zeros_like(sector_diff),
        sector_diff,
    )

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
    front_closing = front_temporal < -0.0015
    left_closing = left_temporal < -0.0015
    right_closing = right_temporal < -0.0015

    front_danger = front_min < 0.50
    left_danger = left_min < 0.40
    right_danger = right_min < 0.40

    fast_forward = action_linear > 0.12

    front_dynamic_risk = front_danger & front_closing
    left_dynamic_risk = left_danger & left_closing
    right_dynamic_risk = right_danger & right_closing

    goal_area_blocked = near_goal & (
        front_dynamic_risk | left_dynamic_risk | right_dynamic_risk
    )

    detour_needed = front_blocked | goal_area_blocked

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

    r_near_goal_rush = torch.where(
        goal_area_blocked & (action_linear > 0.08),
        torch.full_like(goal_dist, -1.0),
        torch.zeros_like(goal_dist),
    )

    no_progress = torch.abs(progress) < 0.003
    not_near_goal = goal_dist > 0.40
    low_linear = action_linear < 0.06
    low_angular = torch.abs(action_angular) < 0.08

    r_stuck = torch.where(
        no_progress & not_near_goal & low_linear,
        torch.full_like(goal_dist, -1.0),
        torch.zeros_like(goal_dist),
    )

    reward = (
        r_yaw
        + r_distance
        + r_obstacle
        + r_vlinear
        + r_vangular
        # + r_closing_fast
        + r_clearance_recovery
        + r_near_goal_rush
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

    # not_near_goal = goal_dist > 0.40
    # near_obstacle = min_obstacle_dist < 0.65

    # # no or negative progress
    # no_progress = progress < 0.0002

    # low_linear = action_linear < 0.06
    # low_angular = torch.abs(action_angular) < 0.08
    # true_freeze = low_linear & low_angular

    # r_stuck_near = torch.where(
    #     near_obstacle & not_near_goal & no_progress & true_freeze,
    #     torch.full_like(goal_dist, -2.0),
    #     torch.zeros_like(goal_dist),
    # )


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

def navigation_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Select reward function."""

    if REWARD_FUNCTION == "A":
        return navigation_reward_A_stage3_temporal(env)
    elif REWARD_FUNCTION == "B":
        return navigation_reward_B(env)
    # elif REWARD_FUNCTION == "A1":
    #     return local_avoidance_reward_A1(env)


    raise ValueError(f"Unknown REWARD_FUNCTION: {REWARD_FUNCTION}")


@configclass
class RewardsCfg:
    navigation = RewTerm(
        func=navigation_reward,
        weight=1.0,
    )