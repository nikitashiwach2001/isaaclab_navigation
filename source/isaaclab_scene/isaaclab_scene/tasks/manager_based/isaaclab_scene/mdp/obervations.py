import math
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass


LIDAR_DISTANCE_CAP = 3.5

# Fixed goal for now.
# Later this will be replaced by random goal spawning per environment.
# GOAL_POS_W = torch.tensor([0.0, 0.0])
def get_goal_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return per-env goal positions. If missing, create zero goals."""

    if not hasattr(env, "goal_pos_w"):
        env.goal_pos_w = torch.zeros((env.num_envs, 2), device=env.device)

    return env.goal_pos_w

def lidar_scan(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Normalized lidar ranges: shape [num_envs, 40]."""

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

    return ranges


def goal_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Normalized distance to goal: shape [num_envs, 1]."""

    robot = env.scene["robot"]

    robot_xy = robot.data.root_pos_w[:, :2]
    goal_xy = get_goal_pos_w(env)

    diff = goal_xy - robot_xy
    distance = torch.norm(diff, dim=-1, keepdim=True)

    max_goal_distance = math.sqrt(5.0**2 + 5.0**2)

    distance = torch.clamp(distance / max_goal_distance, 0.0, 1.0)

    return distance


def goal_angle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Normalized goal angle: shape [num_envs, 1], range [-1, 1]."""

    robot = env.scene["robot"]

    robot_xy = robot.data.root_pos_w[:, :2]
    goal_xy = get_goal_pos_w(env)

    diff = goal_xy - robot_xy
    heading_to_goal = torch.atan2(diff[:, 1], diff[:, 0])

    quat = robot.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

    angle = heading_to_goal - yaw
    angle = torch.atan2(torch.sin(angle), torch.cos(angle))
    angle = angle / math.pi

    return angle.unsqueeze(-1)


def previous_actions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous action: shape [num_envs, 2]."""

    if env.action_manager.action is None:
        return torch.zeros((env.num_envs, 2), device=env.device)

    return env.action_manager.action


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        lidar_scan = ObsTerm(func=lidar_scan)
        goal_distance = ObsTerm(func=goal_distance)
        goal_angle = ObsTerm(func=goal_angle)
        previous_actions = ObsTerm(func=previous_actions)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()