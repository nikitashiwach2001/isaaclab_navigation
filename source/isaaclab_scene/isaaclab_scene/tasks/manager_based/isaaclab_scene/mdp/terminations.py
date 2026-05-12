import math
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import time_out
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass


LIDAR_DISTANCE_CAP = 3.5

# Tune these later after testing
THRESHOLD_GOAL = 0.25
THRESHOLD_COLLISION = 0.30
TUMBLE_THRESHOLD = 0.20
RESET_GRACE_STEPS = 30


def _grace_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return True for envs still inside reset grace period."""
    return env.episode_length_buf <= RESET_GRACE_STEPS


def reached_goal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when robot reaches the goal."""

    if not hasattr(env, "goal_pos_w"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    robot = env.scene["robot"]

    robot_xy = robot.data.root_pos_w[:, :2]
    goal_xy = env.goal_pos_w

    goal_dist = torch.norm(goal_xy - robot_xy, dim=-1)

    done = goal_dist < THRESHOLD_GOAL

    # Same as Gazebo: no termination during first 30 steps
    done = torch.where(_grace_mask(env), torch.zeros_like(done), done)

    env.goal_reached_buf = done

    return done


def collision_from_lidar(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when lidar detects obstacle/wall too close."""

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
    min_range = torch.min(ranges, dim=1).values

    done = min_range < THRESHOLD_COLLISION

    # Same as Gazebo: no termination during first 30 steps
    done = torch.where(_grace_mask(env), torch.zeros_like(done), done)

    env.collision_buf = done

    return done

def collision_from_lidar_stage4(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate on collision and classify source."""

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

    min_range = torch.min(ranges, dim=1).values

    done = min_range < THRESHOLD_COLLISION
    done = torch.where(_grace_mask(env), torch.zeros_like(done), done)

    env.collision_buf = done

    env.collision_dynamic_buf = torch.zeros_like(done)
    env.collision_static_buf = torch.zeros_like(done)
    env.collision_boundary_buf = torch.zeros_like(done)

    if not done.any():
        return done

    robot_xy = env.scene["robot"].data.root_pos_w[:, :2]

    dynamic_dist = torch.full((env.num_envs,), 999.0, device=env.device)
    static_dist = torch.full((env.num_envs,), 999.0, device=env.device)
    boundary_dist = torch.full((env.num_envs,), 999.0, device=env.device)

    for name in ["obstacle_1", "obstacle_2", "obstacle_3"]:
        if name in env.scene.keys():
            obs_xy = env.scene[name].data.root_pos_w[:, :2]
            dist = torch.norm(robot_xy - obs_xy, dim=-1)
            dynamic_dist = torch.minimum(dynamic_dist, dist)

    for name in [
        "inner_wall_1",
        "inner_wall_2",
        "inner_wall_3",
        "inner_wall_4",
        "inner_wall_5",
        "inner_wall_6",
        "inner_wall_7",
    ]:
        if name in env.scene.keys():
            wall_xy = env.scene[name].data.root_pos_w[:, :2]
            dist = torch.norm(robot_xy - wall_xy, dim=-1)
            static_dist = torch.minimum(static_dist, dist)

    for name in ["wall_1", "wall_2", "wall_3", "wall_4"]:
        if name in env.scene.keys():
            wall_xy = env.scene[name].data.root_pos_w[:, :2]
            dist = torch.norm(robot_xy - wall_xy, dim=-1)
            boundary_dist = torch.minimum(boundary_dist, dist)

    nearest_dist = torch.minimum(dynamic_dist, torch.minimum(static_dist, boundary_dist))

    env.collision_dynamic_buf = done & (dynamic_dist == nearest_dist)
    env.collision_static_buf = done & (static_dist == nearest_dist)
    env.collision_boundary_buf = done & (boundary_dist == nearest_dist)

    return done


def tumble(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when robot tilts too much."""

    robot = env.scene["robot"]

    quat = robot.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Roll
    roll = torch.atan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )

    # Pitch
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    sin_pitch = torch.clamp(sin_pitch, -1.0, 1.0)
    pitch = torch.asin(sin_pitch)

    done = (torch.abs(roll) > TUMBLE_THRESHOLD) | (torch.abs(pitch) > TUMBLE_THRESHOLD)

    # Same as Gazebo: no termination during first 30 steps
    done = torch.where(_grace_mask(env), torch.zeros_like(done), done)
    env.tumble_buf = done

    return done


@configclass
class TerminationsCfg:
    goal_reached = DoneTerm(func=reached_goal)
    collision = DoneTerm(func=collision_from_lidar_stage4)
    tumble = DoneTerm(func=tumble)
    time_out = DoneTerm(func=time_out, time_out=True)