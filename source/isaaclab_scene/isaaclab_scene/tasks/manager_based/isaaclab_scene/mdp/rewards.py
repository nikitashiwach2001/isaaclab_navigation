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
    - 40 rays
    - ray_alignment = "yaw"
    - ray 0 is robot forward
    - rays are ordered circularly around the robot

    Sectors:
    - front: rays 0-4 and 36-39
    - right: rays 5-14
    - left: rays 26-35
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

    min_obstacle_dist = ranges.min(dim=1).values

    front_min = torch.cat(
        [ranges[:, :5], ranges[:, 36:]],
        dim=1,
    ).min(dim=1).values

    right_min = ranges[:, 5:15].min(dim=1).values
    left_min = ranges[:, 26:36].min(dim=1).values

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


def navigation_reward_B(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gazebo-style Reward B with front/side LiDAR proximity shaping."""

    min_obstacle_dist, front_min, right_min, left_min = _get_lidar_distances(env)
    goal_dist, goal_angle = _get_goal_distance_and_angle(env)
    action_linear, action_angular = _get_real_actions(env)

    _ensure_reward_buffers(env, goal_dist)

    success = goal_dist < THRESHOLD_GOAL
    collision = min_obstacle_dist < THRESHOLD_COLLISION

    near_obstacle = min_obstacle_dist < 0.65

    # In open space, face the goal.
    # Near obstacles, allow free detour direction.
    r_yaw = torch.where(
        near_obstacle,
        torch.zeros_like(goal_angle),
        -1.0 * torch.abs(goal_angle),
    )
    r_yaw = torch.where(success, torch.zeros_like(r_yaw), r_yaw)

    # Allow sharper turns near obstacles, discourage spinning in open space.
    r_vangular = torch.where(
        near_obstacle,
        -0.1 * (action_angular ** 2),
        -0.5 * (action_angular ** 2),
    )

    # Gazebo Reward B progress term.
    progress = env.goal_dist_prev - goal_dist
    r_distance = progress * 30.0
    env.goal_dist_prev[:] = goal_dist

    # Emergency near-collision penalty.
    r_obstacle = torch.where(
        min_obstacle_dist < THRESHOLD_COLLISION,
        torch.full_like(min_obstacle_dist, -20.0),
        torch.zeros_like(min_obstacle_dist),
    )

    # Keep forward movement, but reduce forward incentive near obstacles.
    r_forward = torch.where(
        near_obstacle,
        0.10 * action_linear,
        0.30 * action_linear,
    )

    # Directional proximity shaping.
    # Front danger: obstacle in path.
    r_prox_front = torch.where(
        front_min < 0.50,
        -8.0 * (0.50 - front_min),
        torch.zeros_like(front_min),
    )

    # Side danger: obstacle crossing from left/right.
    side_min = torch.minimum(right_min, left_min)

    r_prox_sides = torch.where(
        side_min < 0.45,
        -6.0 * (0.45 - side_min),
        torch.zeros_like(side_min),
    )

    r_proximity = r_prox_front + r_prox_sides

    r_cp = torch.zeros_like(goal_dist)

    # Stage 3: single moving obstacle_3
    # Stage 4: two moving obstacles (obstacle_1, obstacle_2)
    _cp_obstacle_names = []
    if "obstacle_3" in env.scene.keys():
        _cp_obstacle_names = ["obstacle_3"]
    elif "obstacle_1" in env.scene.keys() and "obstacle_2" in env.scene.keys():
        _cp_obstacle_names = ["obstacle_1", "obstacle_2"]

    if _cp_obstacle_names:
        robot = env.scene["robot"]
        robot_pos_xy = robot.data.root_pos_w[:, :2]
        robot_vel_xy = robot.data.root_lin_vel_w[:, :2]

        for obs_name in _cp_obstacle_names:
            obstacle = env.scene[obs_name]

            obstacle_pos_xy = obstacle.data.root_pos_w[:, :2]
            obstacle_vel_xy = obstacle.data.root_lin_vel_w[:, :2]

            rel_pos = obstacle_pos_xy - robot_pos_xy
            dist = torch.norm(rel_pos, dim=-1)

            rel_vel = obstacle_vel_xy - robot_vel_xy
            obstacle_speed = torch.norm(obstacle_vel_xy, dim=-1)

            closing_speed = -torch.sum(rel_pos * rel_vel, dim=-1) / torch.clamp(dist, min=1e-6)

            approaching = closing_speed > 0.0
            moving_obstacle = obstacle_speed > 0.02

            ttc = dist / torch.clamp(closing_speed, min=1e-6)

            pc_ttc = torch.where(
                approaching,
                torch.clamp(0.15 / torch.clamp(ttc, min=1e-6), 0.0, 1.0),
                torch.zeros_like(dist),
            )

            pc_dist = torch.clamp(
                (0.80 - dist) / (0.80 - THRESHOLD_COLLISION),
                0.0,
                1.0,
            )

            cp = 0.5 * pc_ttc + 0.5 * pc_dist

            r_cp += torch.where(
                moving_obstacle,
                -1.5 * cp,
                torch.zeros_like(cp),
            )

    reward = (
        r_yaw
        + r_distance
        + r_obstacle
        + r_forward
        + r_proximity
        + r_vangular
        + r_cp
        - 1.0
    )

    reward = torch.where(success, reward + SUCCESS_REWARD, reward)
    reward = torch.where(collision, reward - COLLISION_PENALTY, reward)

    return reward


def navigation_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Select reward function."""

    if REWARD_FUNCTION == "B":
        return navigation_reward_B(env)

    raise ValueError(f"Unknown REWARD_FUNCTION: {REWARD_FUNCTION}")


@configclass
class RewardsCfg:
    navigation = RewTerm(
        func=navigation_reward,
        weight=1.0,
    )