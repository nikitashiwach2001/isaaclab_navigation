import os
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


# ── Obstacle spawn configs ────────────────────────────────────────────────────

STAGE4_OBSTACLE_1_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Obstacle_1",
    spawn=sim_utils.CylinderCfg(
        radius=0.16, height=0.50, axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, 
                                                     disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 2.0, 0.25), rot=(1.0, 0.0, 0.0, 0.0)),
)

STAGE4_OBSTACLE_2_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Obstacle_2",
    spawn=sim_utils.CylinderCfg(
        radius=0.16, height=0.50, axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, 
                                                     disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.0, -2.0, 0.25), rot=(1.0, 0.0, 0.0, 0.0)),
)


# ── Keyframe tables (local arena coordinates) ─────────────────────────────────
#
# Arena inner bounds: ±2.35 m  (walls at ±2.425, wall half-thickness 0.075)
# All XY values are offsets from each env's origin (local coords).
#
# Speed curriculum: scale all keyframe times by this factor.
# 1.0 = full speed, 3.0 = 3× slower (good starting point), ramp down toward 1.0.
OBSTACLE_SPEED_SCALE = float(os.environ.get("OBSTACLE_SPEED_SCALE", "2.0"))

# Obstacle 1 — pendulum, 140 s loop, stays mostly in upper half (y ≈ 1.0)
_OBS1_TIMES_BASE = [0.0, 10.0, 50.0, 70.0, 90.0, 130.0, 140.0]
_OBS1_TIMES = [t * OBSTACLE_SPEED_SCALE for t in _OBS1_TIMES_BASE]
_OBS1_XY = [
    [ 2.0,  2.0],   # 0 s   — loop start
    [ 1.5,  1.0],   # 10 s  — diagonal toward center
    [-1.5,  1.0],   # 50 s  — end of left sweep
    [-1.7, -1.0],   # 70 s  — dip down-left
    [-1.5,  1.0],   # 90 s  — bounce back up
    [ 1.5,  1.0],   # 130 s — end of right sweep
    [ 2.0,  2.0],   # 140 s — back to start, loop immediately
]
_OBS1_PERIOD = 140.0 * OBSTACLE_SPEED_SCALE

# Obstacle 2 — erratic, 130 s loop, crosses full arena twice per loop
_OBS2_TIMES_BASE = [0.0, 10.0, 40.0, 55.0, 85.0, 100.0, 110.0, 115.0, 120.0, 125.0, 130.0]
_OBS2_TIMES = [t * OBSTACLE_SPEED_SCALE for t in _OBS2_TIMES_BASE]
_OBS2_XY = [
    [-2.0, -2.0],   # 0 s   — loop start
    [-1.3, -1.8],   # 10 s  — nudges out of corner
    [ 0.5,  1.5],   # 40 s  — long diagonal across arena
    [-1.7,  1.5],   # 55 s  — horizontal sweep left at top
    [ 1.5, -0.2],   # 85 s  — long diagonal down-right through center
    [ 1.5, -2.0],   # 100 s — vertical drop to bottom-right
    [ 0.0, -1.5],   # 110 s — moves left along bottom
    [-0.5, -1.0],   # 115 s — zigzag 1
    [-1.0, -1.5],   # 120 s — zigzag 2
    [-1.5, -1.9],   # 125 s — zigzag 3
    [-2.0, -2.0],   # 130 s — back to start, loop immediately
]
_OBS2_PERIOD = 130.0 * OBSTACLE_SPEED_SCALE


# ── Interpolation helper ──────────────────────────────────────────────────────

def _interp_keyframes(
    t: torch.Tensor,
    times: list,
    xy: list,
    device: str,
) -> torch.Tensor:
    """
    Vectorised linear interpolation over a keyframe table.

    Args:
        t:      (N,) current time within the period, in [0, period).
        times:  list of K monotonically increasing floats.
        xy:     list of K [x, y] pairs (local arena coords).
        device: torch device string.

    Returns:
        (N, 2) interpolated XY positions (local coords).
    """
    times_t = torch.tensor(times, device=device, dtype=torch.float32)   # (K,)
    pos_t   = torch.tensor(xy,    device=device, dtype=torch.float32)   # (K, 2)

    # idx: index of the segment-end keyframe (1 … K-1)
    idx = torch.searchsorted(times_t, t, right=True).clamp(1, len(times) - 1)  # (N,)

    t0  = times_t[idx - 1]     # (N,)
    t1  = times_t[idx]          # (N,)
    p0  = pos_t[idx - 1]        # (N, 2)
    p1  = pos_t[idx]            # (N, 2)

    alpha = ((t - t0) / (t1 - t0).clamp(min=1e-6)).clamp(0.0, 1.0).unsqueeze(-1)  # (N, 1)
    return p0 + alpha * (p1 - p0)   # (N, 2)


# ── Per-step update ───────────────────────────────────────────────────────────

def update_moving_obstacles_stage4(env, _env_ids=None):
    """Advance both keyframe obstacles. Phase is re-randomized per-episode by
    randomize_obstacle_phases_stage4; this function just ticks the clocks
    forward and interpolates positions."""

    dt = env.step_dt

    if not hasattr(env, "s4_obs1_time"):
        env.s4_obs1_time = torch.rand(env.num_envs, device=env.device) * _OBS1_PERIOD
        env.s4_obs2_time = torch.rand(env.num_envs, device=env.device) * _OBS2_PERIOD

    env.s4_obs1_time = (env.s4_obs1_time + dt) % _OBS1_PERIOD
    env.s4_obs2_time = (env.s4_obs2_time + dt) % _OBS2_PERIOD

    _move_obstacle(env, "obstacle_1", env.s4_obs1_time, _OBS1_TIMES, _OBS1_XY)
    _move_obstacle(env, "obstacle_2", env.s4_obs2_time, _OBS2_TIMES, _OBS2_XY)


def randomize_obstacle_phases_stage4(env, env_ids=None):
    """Re-randomize cylinder phases at episode reset.

    Without this, each cylinder's clock just drifts deterministically — the
    replay buffer over-samples some (goal × phase) pairs and under-samples
    others, capping policy performance on the rare ones. Calling this at every
    reset gives uniform phase coverage across episodes.

    Run order matters: this must run BEFORE reset_goal_position so the goal
    selector sees the new obstacle positions and can reject goals that landed
    under a cylinder."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(dtype=torch.long, device=env.device)
    n = len(env_ids)

    if not hasattr(env, "s4_obs1_time"):
        env.s4_obs1_time = torch.zeros(env.num_envs, device=env.device)
        env.s4_obs2_time = torch.zeros(env.num_envs, device=env.device)

    env.s4_obs1_time[env_ids] = torch.rand(n, device=env.device) * _OBS1_PERIOD
    env.s4_obs2_time[env_ids] = torch.rand(n, device=env.device) * _OBS2_PERIOD

    for name, t, times, xy in (
        ("obstacle_1", env.s4_obs1_time, _OBS1_TIMES, _OBS1_XY),
        ("obstacle_2", env.s4_obs2_time, _OBS2_TIMES, _OBS2_XY),
    ):
        obs = env.scene[name]
        xy_local = _interp_keyframes(t[env_ids], times, xy, env.device)
        pos = torch.zeros((n, 3), device=env.device)
        pos[:, :2] = env.scene.env_origins[env_ids, :2] + xy_local
        pos[:, 2] = 0.25
        quat = obs.data.root_quat_w[env_ids].clone()
        pose = torch.cat([pos, quat], dim=-1)
        obs.write_root_pose_to_sim(pose, env_ids=env_ids)
        obs.write_root_velocity_to_sim(torch.zeros((n, 6), device=env.device), env_ids=env_ids)


def _move_obstacle(env, name: str, time: torch.Tensor, times: list, xy: list):
    obs      = env.scene[name]
    xy_local = _interp_keyframes(time, times, xy, env.device)   # (N, 2) local coords

    # Convert local → world by adding each env's origin
    pos_w = env.scene.env_origins.clone()   # (N, 3)
    pos_w[:, :2] += xy_local
    pos_w[:, 2]   = 0.25                    # fixed height

    # Compute real velocity from position delta so root_lin_vel_w is correct.
    prev_pos_w = obs.data.root_pos_w.clone()
    vel_xyz = (pos_w - prev_pos_w) / max(env.step_dt, 1e-6)

    vel_w = torch.zeros((env.num_envs, 6), device=env.device)
    vel_w[:, :3] = vel_xyz

    pose = torch.cat([pos_w, obs.data.root_quat_w.clone()], dim=-1)
    obs.write_root_pose_to_sim(pose)
    obs.write_root_velocity_to_sim(vel_w)
