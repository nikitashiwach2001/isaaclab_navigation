import math
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass


# Match Gazebo constants
LIDAR_DISTANCE_CAP = 3.5

MAX_LINEAR_SPEED = 0.35
MAX_ANGULAR_SPEED = 1.5

ENABLE_BACKWARD = False

THRESHOLD_GOAL = 0.25
THRESHOLD_COLLISION = 0.22

SUCCESS_REWARD = 2500.0
COLLISION_PENALTY = 2000.0

# Choose reward function here: "A" or "B"
REWARD_FUNCTION = "B"


def _get_lidar_min_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return min lidar distance per env. Shape: [num_envs]."""

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

    return torch.min(ranges, dim=1).values

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
    """Convert normalized action to real linear/angular velocities."""

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


def navigation_reward_A(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gazebo get_reward_A converted to Isaac Lab vectorized reward."""

    goal_dist, goal_angle = _get_goal_distance_and_angle(env)
    min_obstacle_dist = _get_lidar_min_distance(env)
    action_linear, action_angular = _get_real_actions(env)

    _ensure_reward_buffers(env, goal_dist)

    success = goal_dist < THRESHOLD_GOAL
    collision = min_obstacle_dist < THRESHOLD_COLLISION

    # [-3.14, 0] - Don't penalize yaw if goal is reached
    r_yaw = -torch.abs(goal_angle)
    r_yaw = torch.where(success, torch.zeros_like(r_yaw), r_yaw)

    # [-4, 0]
    r_vangular = -1.5 * (action_angular**2)

    # [-1, 1]
    denom = env.goal_dist_initial + goal_dist
    denom = torch.clamp(denom, min=1e-6)
    r_distance = (2.0 * env.goal_dist_initial) / denom - 1.0

    # [-20, 0]
    r_obstacle = torch.where(
        min_obstacle_dist < 0.22,
        torch.full_like(min_obstacle_dist, -20.0),
        torch.zeros_like(min_obstacle_dist),
    )
    desired_linear = torch.where(
    goal_dist < 0.50,
    torch.full_like(goal_dist, 0.10),
    torch.full_like(goal_dist, MAX_LINEAR_SPEED),)

    # Penalize low forward speed, same as Gazebo
    # r_vlinear = -(((0.22 - action_linear) * 10.0) ** 2)
    r_vlinear = -(((desired_linear - action_linear) * 10.0) ** 2)

    reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular - 1.0

    reward = torch.where(success, reward + SUCCESS_REWARD, reward)
    reward = torch.where(collision, reward - COLLISION_PENALTY, reward)

    return reward


def navigation_reward_B(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gazebo get_reward_B converted to Isaac Lab vectorized reward."""

    goal_dist, goal_angle = _get_goal_distance_and_angle(env)
    min_obstacle_dist = _get_lidar_min_distance(env)
    action_linear, action_angular = _get_real_actions(env)

    _ensure_reward_buffers(env, goal_dist)

    success = goal_dist < THRESHOLD_GOAL
    collision = min_obstacle_dist < THRESHOLD_COLLISION

    near_obstacle = min_obstacle_dist < 0.65

    # [-3.14, 0]
    r_yaw = -torch.abs(goal_angle)
    r_yaw = torch.where(success, torch.zeros_like(r_yaw), r_yaw)

    # Same as Gazebo: [-4, 0] when angular max is 2.0
    r_vangular = -1.0 * (action_angular**2)
    # near_obstacle = min_obstacle_dist < 0.55

    # r_vangular = torch.where(
    #     near_obstacle,
    #     -0.2 * (action_angular**2),   # allow sharp turns near obstacle
    #     -1.0 * (action_angular**2),   # still discourage spinning in open space
    # )

    # Delta-based distance reward
    r_distance = (env.goal_dist_prev - goal_dist) * 30.0
    env.goal_dist_prev[:] = goal_dist

    # no progress penealty
    no_progress = r_distance < 0.005
    not_near_goal = goal_dist > 0.40

    r_stuck = torch.where(
        no_progress & not_near_goal,
        torch.full_like(goal_dist, -0.25),
        torch.zeros_like(goal_dist),
    )

    
    low_linear = action_linear < 0.08
    not_near_goal = goal_dist > 0.40
    no_progress = r_distance < 0.005

    r_freeze = torch.where(
        near_obstacle & low_linear & not_near_goal & no_progress,
        torch.full_like(goal_dist, -1.0),
        torch.zeros_like(goal_dist),
    )

    # Same as Gazebo: obstacle penalty when below 0.22m
    r_obstacle = torch.where(
        min_obstacle_dist < 0.22,
        torch.full_like(min_obstacle_dist, -20.0),
        torch.zeros_like(min_obstacle_dist),
    )

    # gradual penelty
    # safe_dist = 0.50
    # danger_dist = 0.25

    # r_obstacle = -4.0 * torch.clamp(
    #     (safe_dist - min_obstacle_dist) / safe_dist,
    #     0.0,
    #     1.0,
    # ) ** 2

    # r_obstacle = torch.where(
    #     min_obstacle_dist < danger_dist,
    #     r_obstacle - 15.0,
    #     r_obstacle,
    # )

    # Same as Gazebo: prefer max forward speed 0.22
    r_vlinear = -(((MAX_LINEAR_SPEED - action_linear) * 10.0) ** 2)

    reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular + r_stuck + r_freeze - 1.0

    reward = torch.where(success, reward + SUCCESS_REWARD, reward)
    reward = torch.where(collision, reward - COLLISION_PENALTY, reward)

    return reward

def navigation_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Select reward function A or B."""

    if REWARD_FUNCTION == "A":
        return navigation_reward_A(env)

    elif REWARD_FUNCTION == "B":
        return navigation_reward_B(env)

    else:
        raise ValueError(f"Unknown REWARD_FUNCTION: {REWARD_FUNCTION}")


@configclass
class RewardsCfg:
    navigation = RewTerm(
        func=navigation_reward,
        weight=1.0,
    )


##################################################
# with this reward function the agent learns the partial behviour to navigate around the movinf obstacle but stuck near the static obstacle
# after this I added a cp reward 
# def navigation_reward_B(env: ManagerBasedRLEnv) -> torch.Tensor:
    # """Gazebo get_reward_B converted to Isaac Lab vectorized reward."""

    # goal_dist, goal_angle = _get_goal_distance_and_angle(env)
    # min_obstacle_dist = _get_lidar_min_distance(env)
    # action_linear, action_angular = _get_real_actions(env)

    # _ensure_reward_buffers(env, goal_dist)

    # success = goal_dist < THRESHOLD_GOAL
    # collision = min_obstacle_dist < THRESHOLD_COLLISION

    # near_obstacle = min_obstacle_dist < 0.65

    # # [-3.14, 0]
    # r_yaw = -torch.abs(goal_angle)
    # r_yaw = torch.where(success, torch.zeros_like(r_yaw), r_yaw)

    # # Same as Gazebo: [-4, 0] when angular max is 2.0

    # r_vangular = -0.5 * (action_angular**2) # was -1.0
    # # near_obstacle = min_obstacle_dist < 0.55

    # # r_vangular = torch.where(
    # #     near_obstacle,
    # #     -0.2 * (action_angular**2),   # allow sharp turns near obstacle
    # #     -1.0 * (action_angular**2),   # still discourage spinning in open space
    # # )

    # progress = env.goal_dist_prev - goal_dist

    # # 2. Add spin_penalty after r_freeze block
    # spin_penalty = torch.where(
    #     (torch.abs(action_angular) > 0.7) & (action_linear < 0.08) & (goal_dist > 0.40),
    #     torch.full_like(goal_dist, -1.0),
    #     torch.zeros_like(goal_dist),
    # )

    # deadlock_penalty = torch.where(
    #     (torch.abs(progress) < 0.0003) & (goal_dist > 0.40), # was 0.0005
    #     torch.full_like(goal_dist, -0.35),
    #     torch.zeros_like(goal_dist),
    # )


    # # Delta-based distance reward
    # r_distance = progress * 30.0
    # env.goal_dist_prev[:] = goal_dist

    # # no progress penealty
    # no_progress = r_distance < 0.005
    # not_near_goal = goal_dist > 0.40

    # r_stuck = torch.where(
    #     no_progress & not_near_goal,
    #     torch.full_like(goal_dist, -0.5), # was -0.25
    #     torch.zeros_like(goal_dist),
    # )

    
    # low_linear = action_linear < 0.08
    # not_near_goal = goal_dist > 0.40
    # no_progress = r_distance < 0.005

    # r_freeze = torch.where(
    #     near_obstacle & low_linear & not_near_goal & no_progress,
    #     torch.full_like(goal_dist, -1.0),
    #     torch.zeros_like(goal_dist),
    # )

    # # Same as Gazebo: obstacle penalty when below 0.22m
    # r_obstacle = torch.where(
    #     min_obstacle_dist < 0.22,
    #     torch.full_like(min_obstacle_dist, -20.0),
    #     torch.zeros_like(min_obstacle_dist),
    # )

    # # gradual penelty
    # # safe_dist = 0.50
    # # danger_dist = 0.25

    # # r_obstacle = -4.0 * torch.clamp(
    # #     (safe_dist - min_obstacle_dist) / safe_dist,
    # #     0.0,
    # #     1.0,
    # # ) ** 2

    # # r_obstacle = torch.where(
    # #     min_obstacle_dist < danger_dist,
    # #     r_obstacle - 15.0,
    # #     r_obstacle,
    # # )

    # # Same as Gazebo: prefer max forward speed 0.22
    # r_vlinear = -(((MAX_LINEAR_SPEED - action_linear) * 10.0) ** 2)

    # reward = deadlock_penalty + r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular + r_stuck + r_freeze + spin_penalty - 1.0

    # reward = torch.where(success, reward + SUCCESS_REWARD, reward)
    # reward = torch.where(collision, reward - COLLISION_PENALTY, reward)

    # return reward



    ## original reward_B without spin penalty and deadlock penalty, which still learns to navigate around the moving obstacle but gets stuck near the static one

    # def navigation_reward_B(env: ManagerBasedRLEnv) -> torch.Tensor:
    # """Gazebo get_reward_B converted to Isaac Lab vectorized reward."""

    # goal_dist, goal_angle = _get_goal_distance_and_angle(env)
    # min_obstacle_dist = _get_lidar_min_distance(env)
    # action_linear, action_angular = _get_real_actions(env)

    # _ensure_reward_buffers(env, goal_dist)

    # success = goal_dist < THRESHOLD_GOAL
    # collision = min_obstacle_dist < THRESHOLD_COLLISION

    # # [-3.14, 0]
    # r_yaw = -torch.abs(goal_angle)
    # r_yaw = torch.where(success, torch.zeros_like(r_yaw), r_yaw)

    # # [-4, 0]
    # r_vangular = -1.0 * (action_angular ** 2)

    # # Delta-based progress reward
    # r_distance = (env.goal_dist_prev - goal_dist) * 30.0
    # env.goal_dist_prev[:] = goal_dist

    # # [-20, 0]
    # r_obstacle = torch.where(
    #     min_obstacle_dist < 0.22,
    #     torch.full_like(min_obstacle_dist, -20.0),
    #     torch.zeros_like(min_obstacle_dist),
    # )

    # # Gazebo preferred forward velocity = 0.22
    # r_vlinear = -1.0 * (((0.22 - action_linear) * 10.0) ** 2)

    # reward = (
    #     r_yaw
    #     + r_distance
    #     + r_obstacle
    #     + r_vlinear
    #     + r_vangular
    #     - 1.0
    # )

    # reward = torch.where(success, reward + SUCCESS_REWARD, reward)
    # reward = torch.where(collision, reward - COLLISION_PENALTY, reward)

    # return reward



def navigation_reward_C(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gazebo get_reward_B converted to Isaac Lab vectorized reward."""

    goal_dist, goal_angle = _get_goal_distance_and_angle(env)
    min_obstacle_dist = _get_lidar_min_distance(env)
    action_linear, action_angular = _get_real_actions(env)

    _ensure_reward_buffers(env, goal_dist)

    success = goal_dist < THRESHOLD_GOAL
    collision = min_obstacle_dist < THRESHOLD_COLLISION

    near_obstacle = min_obstacle_dist < 0.65

    # [-3.14, 0]
    r_yaw = -torch.abs(goal_angle)
    r_yaw = torch.where(success, torch.zeros_like(r_yaw), r_yaw)

    r_vangular = -1.0 * (action_angular**2)

    # Delta-based distance reward
    r_distance = (env.goal_dist_prev - goal_dist) * 30.0
    env.goal_dist_prev[:] = goal_dist

    # no progress penealty
    no_progress = r_distance < 0.005
    not_near_goal = goal_dist > 0.40

    r_stuck = torch.where(
        no_progress & not_near_goal,
        torch.full_like(goal_dist, -0.25),
        torch.zeros_like(goal_dist),
    )

    
    low_linear = action_linear < 0.08
    not_near_goal = goal_dist > 0.40
    no_progress = r_distance < 0.005

    r_freeze = torch.where(
        near_obstacle & low_linear & not_near_goal & no_progress,
        torch.full_like(goal_dist, -1.0),
        torch.zeros_like(goal_dist),
    )

    # Same as Gazebo: obstacle penalty when below 0.22m
    r_obstacle = torch.where(
        min_obstacle_dist < 0.22,
        torch.full_like(min_obstacle_dist, -20.0),
        torch.zeros_like(min_obstacle_dist),
    )

    r_vlinear = -(((MAX_LINEAR_SPEED - action_linear) * 10.0) ** 2)

    reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular + r_stuck + r_freeze - 1.0

    reward = torch.where(success, reward + SUCCESS_REWARD, reward)
    reward = torch.where(collision, reward - COLLISION_PENALTY, reward)

    return reward