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


N_LIDAR_HISTORY = 6  # frames stacked: t, t-1, t-2, t-3, t-4, t-5
# Extended from 3 → 6 to give the policy enough history to estimate obstacle
# velocity AND acceleration for trajectory prediction (5 velocity samples + 4 accel).


def lidar_stacked(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Stacked LiDAR frames [t, t-1, ..., t-(N_LIDAR_HISTORY-1)] concatenated along the ray axis.

    Shape: [num_envs, 90 * N_LIDAR_HISTORY]  →  [num_envs, 540] for N_LIDAR_HISTORY=6

    Why this beats temporal_sector_diff alone:
      - Cross-path obstacles (moving left-to-right) keep near-constant range →
        temporal diff ≈ 0 even though collision is imminent.
      - With two past frames the network sees the obstacle shift across rays
        and can infer its trajectory and speed directly.
      - Stacking also gives obstacle acceleration (d²range/dt²) free of charge.

    Stage 1 (no moving obstacles): all three frames are identical (walls static).
    The network learns "identical frames → static env". When frames differ in
    Stage 4, it knows a dynamic obstacle is present.
    """
    current = _compute_lidar_scan(env)  # [num_envs, 90]

    if (
        not hasattr(env, "lidar_stack")
        or env.lidar_stack.shape[0] != env.num_envs
    ):
        env.lidar_stack = current.unsqueeze(1).repeat(1, N_LIDAR_HISTORY, 1).clone()
        return env.lidar_stack.reshape(env.num_envs, -1)

    # Reset history for newly-reset envs so past-episode frames don't leak in
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        if reset_mask.any():
            env.lidar_stack = env.lidar_stack.clone()
            env.lidar_stack[reset_mask] = (
                current[reset_mask].unsqueeze(1).repeat(1, N_LIDAR_HISTORY, 1)
            )

    # Shift: drop oldest frame, prepend current  →  [t, t-1, t-2]
    env.lidar_stack = torch.cat(
        [current.unsqueeze(1), env.lidar_stack[:, :-1, :]], dim=1
    ).clone()

    return env.lidar_stack.reshape(env.num_envs, -1)


def lidar_temporal_sector_diff(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Sector-based obstacle approach velocity from LiDAR temporal difference.

    Uses 8 sectors (45° each): front, front-left, left, back-left,
        back, back-right, right, front-right

    Shape: [num_envs, 8], range [-1, 1]

    Interpretation:
        -1.0 = obstacle approaching at >= 0.15 m/s in that sector
        +1.0 = obstacle receding at >= 0.15 m/s
         0.0 = no motion

    Scales the raw normalized diff to velocity units so the actor gets a
    signal proportional to actual obstacle speed rather than near-zero raw diffs
    (~0.00064/step at max obstacle speed vs ±0.05 clamp = 1.3% of range).
    EMA (alpha=0.8) smooths rotation noise without losing the motion signal.
    """
    enable_temporal = getattr(env.cfg, "enable_lidar_temporal_diff", True)

    if not enable_temporal:
        if not hasattr(env, "debug_temporal_disabled_printed"):
            print("[DEBUG] temporal lidar disabled: returning zeros")
            env.debug_temporal_disabled_printed = True
        return torch.zeros((env.num_envs, 8), device=env.device)

    current_lidar = _compute_lidar_scan(env)  # [num_envs, 90], normalized by 3.5m

    if (
        not hasattr(env, "prev_lidar_scan")
        or env.prev_lidar_scan.shape != current_lidar.shape
    ):
        env.prev_lidar_scan = current_lidar.clone()
        env.temporal_diff_ema = torch.zeros((env.num_envs, 8), device=env.device)
        return torch.zeros((env.num_envs, 8), device=env.device)

    # Raw diff in normalized units (negative = obstacle getting closer)
    diff = current_lidar - env.prev_lidar_scan

    # Reset diff for newly reset environments
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        if reset_mask.any():
            diff = diff.clone()
            diff[reset_mask] = 0.0

    # Min per 45° sector (most-approaching ray in each sector)
    sectors = torch.tensor_split(diff, 8, dim=1)
    raw_sector_diff = torch.cat(
        [s.min(dim=1, keepdim=True).values for s in sectors], dim=1
    )

    # Scale so the full ±0.05 output range is utilized.
    # Old raw diff at max obstacle speed (0.15 m/s, dt=0.015s): 0.00064 = 1.3% of ±0.05.
    # Slow obstacles (0.075 m/s) produced 0.00032 — below the old deadband, zeroed entirely.
    # New scale: multiply by (0.05 / max_expected_diff) so max obstacle speed → ±0.05.
    # max_expected_diff = 0.15 * 0.015 / 3.5 = 0.000643 → scale = 0.05 / 0.000643 ≈ 77.8
    _SCALE = 77.8
    scaled = raw_sector_diff * _SCALE

    # EMA (alpha=0.8) to suppress robot-rotation noise while preserving motion signal
    if not hasattr(env, "temporal_diff_ema"):
        env.temporal_diff_ema = torch.zeros((env.num_envs, 8), device=env.device)

    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        if reset_mask.any():
            env.temporal_diff_ema = env.temporal_diff_ema.clone()
            env.temporal_diff_ema[reset_mask] = 0.0

    env.temporal_diff_ema = 0.8 * env.temporal_diff_ema + 0.2 * scaled

    env.prev_lidar_scan = current_lidar.clone()

    return torch.clamp(env.temporal_diff_ema / 0.05, -1.0, 1.0)


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


_N_PRIV_OBSTACLES = 3   # Stage 5 has three cylinders; Stage 4 (two) zero-pads the extra slot
_PRIV_POS_NORM = 5.0    # keeps relative position roughly in [-1, 1]
_PRIV_VEL_NORM = 1.0    # cylinder speeds are well under 1 m/s


def privileged_obstacle_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Critic-only privileged observation: per-cylinder relative position and
    relative velocity.

    Shape: [num_envs, _N_PRIV_OBSTACLES * 4], zero-padded if fewer cylinders.

    For asymmetric actor-critic — given to the CRITIC only. The deployed actor
    never sees this; it stays lidar-only. Lets the critic compute exact
    collision risk so it gives the lidar actor a clean learning signal.
    """
    robot = env.scene["robot"]
    robot_xy  = robot.data.root_pos_w[:, :2]
    robot_vel = robot.data.root_lin_vel_w[:, :2]

    out = torch.zeros((env.num_envs, _N_PRIV_OBSTACLES * 4), device=env.device)
    obs_keys = sorted(k for k in env.scene.keys() if k.startswith("obstacle_"))
    for i, okey in enumerate(obs_keys[:_N_PRIV_OBSTACLES]):
        cyl = env.scene[okey]
        rel_pos = (cyl.data.root_pos_w[:, :2] - robot_xy) / _PRIV_POS_NORM
        rel_vel = (cyl.data.root_lin_vel_w[:, :2] - robot_vel) / _PRIV_VEL_NORM
        out[:, i * 4 + 0:i * 4 + 2] = torch.clamp(rel_pos, -1.0, 1.0)
        out[:, i * 4 + 2:i * 4 + 4] = torch.clamp(rel_vel, -1.0, 1.0)
    return out


_MAX_LIN_VEL = 0.22   # matches actions.py
_MAX_ANG_VEL = 2.0    # matches actions.py

def robot_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Actual body-frame forward speed and yaw rate, normalised to [-1, 1].

    Shape: [num_envs, 2]
    Dim 0: forward velocity  / MAX_LINEAR_SPEED
    Dim 1: yaw rate          / MAX_ANGULAR_SPEED

    Lets the LSTM correlate commanded actions with true motion — useful when
    wheels slip or the actuator doesn't respond instantly.
    """
    robot = env.scene["robot"]
    lin = robot.data.root_lin_vel_b[:, 0:1]   # forward (body x)
    ang = robot.data.root_ang_vel_b[:, 2:3]   # yaw rate (body z)
    lin_norm = torch.clamp(lin / _MAX_LIN_VEL, -1.0, 1.0)
    ang_norm = torch.clamp(ang / _MAX_ANG_VEL, -1.0, 1.0)
    return torch.cat([lin_norm, ang_norm], dim=-1)


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        lidar_stacked = ObsTerm(func=lidar_stacked)                          # [540] 6 frames × 90 rays
        lidar_temporal_sector_diff = ObsTerm(func=lidar_temporal_sector_diff)  # [8]   motion hint
        goal_distance = ObsTerm(func=goal_distance)                          # [1]
        goal_angle = ObsTerm(func=goal_angle)                                # [1]
        robot_velocity = ObsTerm(func=robot_velocity)                        # [2]
        previous_actions = ObsTerm(func=previous_actions)                    # [2]
        # total: 554 dims

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Critic-only privileged observation (asymmetric actor-critic).
        Consumed only by the critic during training — the deployed actor
        stays lidar-only and never reads this group."""
        obstacle_state = ObsTerm(func=privileged_obstacle_state)   # [12] up to 3 cylinders × (rel_pos, rel_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()