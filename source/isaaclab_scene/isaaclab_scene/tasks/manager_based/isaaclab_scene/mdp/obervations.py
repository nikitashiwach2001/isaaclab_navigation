import math
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass


LIDAR_DISTANCE_CAP = 3.5


def get_goal_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return per-env goal positions. If missing, create zero goals."""

    if not hasattr(env, "goal_pos_w"):
        env.goal_pos_w = torch.zeros((env.num_envs, 2), device=env.device)

    return env.goal_pos_w


def _compute_lidar_scan(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Compute normalized lidar ranges.

    Expected shape:
        [num_envs, 90]

    Values:
        0.0 = very close obstacle
        1.0 = no hit / max distance
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

    # Ensure shape is [num_envs, num_rays]
    ranges = ranges.reshape(env.num_envs, -1)

    # if not hasattr(env, "debug_lidar_range_printed"):
    # #     print("[DEBUG] lidar min:", torch.min(ranges).item())
    # #     print("[DEBUG] lidar max:", torch.max(ranges).item())
    # #     print("[DEBUG] lidar env0 first 10:", ranges[0, :10].detach().cpu().numpy())
    #     print("[DEBUG] lidar_scan shape:", ranges.shape)
    #     env.debug_lidar_shape_printed = True

    return ranges


def lidar_scan(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Current normalized lidar ranges.

    Shape:
        [num_envs, 90]
    """

    return _compute_lidar_scan(env)


def lidar_temporal_sector_diff(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Sector-based temporal difference of lidar.

    Uses 8 sectors:
        front, front-left, left, back-left,
        back, back-right, right, front-right

    Shape:
        [num_envs, 8]

    Meaning:
        Negative value = obstacle/free-space distance reduced in that sector.
        Positive value = obstacle moved away / free space increased.
        Near zero = no major change.
    """

    
    enable_temporal = getattr(env.cfg, "enable_lidar_temporal_diff", False)

    if not enable_temporal:
        if not hasattr(env, "debug_temporal_disabled_printed"):
            print("[DEBUG] temporal lidar disabled: returning zeros")
            env.debug_temporal_disabled_printed = True
        return torch.zeros((env.num_envs, 8), device=env.device)
    
    current_lidar = _compute_lidar_scan(env)

    if (
        not hasattr(env, "prev_lidar_scan")
        or env.prev_lidar_scan.shape != current_lidar.shape
    ):
        env.prev_lidar_scan = current_lidar.clone()
        return torch.zeros((env.num_envs, 8), device=env.device)

    prev_lidar = env.prev_lidar_scan

    diff = current_lidar - prev_lidar
    diff = torch.clamp(diff, -1.0, 1.0)

    # Reset temporal diff for newly reset environments
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        if reset_mask.any():
            diff[reset_mask] = 0.0


    # For 90 rays and 360 degrees:
    # 90 rays / 8 sectors = 11.25 rays per sector.
    # torch.tensor_split handles uneven split safely.
    sectors = torch.tensor_split(diff, 8, dim=1)

    sector_features = []

    for sector in sectors:
        # Use min change per sector
        sector_min = torch.min(sector, dim=1, keepdim=True).values
        sector_features.append(sector_min)

    sector_diff = torch.cat(sector_features, dim=1)

    # Deadband: 0.001 passes obstacle motion (~0.0015/step); clamp large spikes from robot rotation
    sector_diff = torch.where(
        torch.abs(sector_diff) < 0.001,
        torch.zeros_like(sector_diff),
        sector_diff,
    )
    sector_diff = torch.clamp(sector_diff, -0.05, 0.05)

    env.prev_lidar_scan = current_lidar.clone()

    if not hasattr(env, "_temporal_debug_step"):
        env._temporal_debug_step = 0
    env._temporal_debug_step += 1
    if env._temporal_debug_step % 500 == 0:
        nonzero_frac = (sector_diff.abs() > 0).float().mean().item()
        print(f"[temporal] non-zero fraction: {nonzero_frac:.3f}, max: {sector_diff.abs().max().item():.5f}")

    return sector_diff


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

    angle_out = angle.unsqueeze(-1)

    # if not hasattr(env, "debug_goal_angle_printed"):
    #     print("[DEBUG] goal_angle shape:", angle_out.shape)
    #     print("[DEBUG] goal_angle min:", torch.min(angle_out).item())
    #     print("[DEBUG] goal_angle max:", torch.max(angle_out).item())
    #     print("[DEBUG] goal_angle has_nan:", torch.isnan(angle_out).any().item())
    #     env.debug_goal_angle_printed = True

    return angle_out

def previous_actions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous action: shape [num_envs, 2]."""

    if env.action_manager.action is None:
        action = torch.zeros((env.num_envs, 2), device=env.device)
    else:
        action = env.action_manager.action

    return action


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        lidar_scan = ObsTerm(func=lidar_scan)
        lidar_temporal_sector_diff = ObsTerm(func=lidar_temporal_sector_diff)
        goal_distance = ObsTerm(func=goal_distance)
        goal_angle = ObsTerm(func=goal_angle)
        previous_actions = ObsTerm(func=previous_actions)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()