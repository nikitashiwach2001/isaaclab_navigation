import os
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


# ── Obstacle spawn configs ────────────────────────────────────────────────────

STAGE5_OBSTACLE_1_CFG = RigidObjectCfg(
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

STAGE5_OBSTACLE_2_CFG = RigidObjectCfg(
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

STAGE5_OBSTACLE_3_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Obstacle_3",
    spawn=sim_utils.CylinderCfg(
        radius=0.16, height=0.50, axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,
                                                     disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.9)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 1.5, 0.25), rot=(1.0, 0.0, 0.0, 0.0)),
)


# ── Keyframe tables (local arena coordinates) ─────────────────────────────────
#
# Arena inner bounds: ±2.35 m  (walls at ±2.425, wall half-thickness 0.075)
# All XY values are offsets from each env's origin (local coords).

OBSTACLE_SPEED_SCALE = float(os.environ.get("OBSTACLE_SPEED_SCALE", "2.0"))

# Obstacle 1 — pendulum, 140 s loop, stays mostly in upper half (y ≈ 1.0)
_OBS1_TIMES_BASE = [0.0, 10.0, 50.0, 70.0, 90.0, 130.0, 140.0]
_OBS1_TIMES = [t * OBSTACLE_SPEED_SCALE for t in _OBS1_TIMES_BASE]
_OBS1_XY = [
    [ 2.0,  2.0],
    [ 1.5,  1.0],
    [-1.5,  1.0],
    [-1.7, -1.0],
    [-1.5,  1.0],
    [ 1.5,  1.0],
    [ 2.0,  2.0],
]
_OBS1_PERIOD = 140.0 * OBSTACLE_SPEED_SCALE

# Obstacle 2 — erratic, 130 s loop, crosses full arena twice per loop
_OBS2_TIMES_BASE = [0.0, 10.0, 40.0, 55.0, 85.0, 100.0, 110.0, 115.0, 120.0, 125.0, 130.0]
_OBS2_TIMES = [t * OBSTACLE_SPEED_SCALE for t in _OBS2_TIMES_BASE]
_OBS2_XY = [
    [-2.0, -2.0],
    [-1.3, -1.8],
    [ 0.5,  1.5],
    [-1.7,  1.5],
    [ 1.5, -0.2],
    [ 1.5, -2.0],
    [ 0.0, -1.5],
    [-0.5, -1.0],
    [-1.0, -1.5],
    [-1.5, -1.9],
    [-2.0, -2.0],
]
_OBS2_PERIOD = 130.0 * OBSTACLE_SPEED_SCALE

# Obstacle 3 — right-side sweeper, 110 s loop (desynchronized from obs1/obs2).
# Covers right half and center, creating head-on scenarios from a third angle.
_OBS3_TIMES_BASE = [0.0, 20.0, 40.0, 55.0, 75.0, 90.0, 110.0]
_OBS3_TIMES = [t * OBSTACLE_SPEED_SCALE for t in _OBS3_TIMES_BASE]
_OBS3_XY = [
    [ 2.0,  1.5],   # 0 s   — start top-right
    [-0.5,  1.8],   # 20 s  — sweep left across the top
    [ 0.5,  0.0],   # 40 s  — diagonal down to center
    [ 1.8, -1.5],   # 55 s  — right side lower
    [-1.5, -1.8],   # 75 s  — sweep across bottom
    [ 0.0,  0.5],   # 90 s  — back up through center
    [ 2.0,  1.5],   # 110 s — back to start, loop immediately
]
_OBS3_PERIOD = 110.0 * OBSTACLE_SPEED_SCALE


# ── Interpolation helper ──────────────────────────────────────────────────────

def _interp_keyframes(
    t: torch.Tensor,
    times: list,
    xy: list,
    device: str,
) -> torch.Tensor:
    times_t = torch.tensor(times, device=device, dtype=torch.float32)
    pos_t   = torch.tensor(xy,    device=device, dtype=torch.float32)

    idx = torch.searchsorted(times_t, t, right=True).clamp(1, len(times) - 1)

    t0  = times_t[idx - 1]
    t1  = times_t[idx]
    p0  = pos_t[idx - 1]
    p1  = pos_t[idx]

    alpha = ((t - t0) / (t1 - t0).clamp(min=1e-6)).clamp(0.0, 1.0).unsqueeze(-1)
    return p0 + alpha * (p1 - p0)


# ── Per-step update ───────────────────────────────────────────────────────────

def _move_obstacle(env, name: str, time: torch.Tensor, times: list, xy: list):
    obs      = env.scene[name]
    xy_local = _interp_keyframes(time, times, xy, env.device)

    pos_w = env.scene.env_origins.clone()
    pos_w[:, :2] += xy_local
    pos_w[:, 2]   = 0.25

    prev_pos_w = obs.data.root_pos_w.clone()
    vel_xyz = (pos_w - prev_pos_w) / max(env.step_dt, 1e-6)

    vel_w = torch.zeros((env.num_envs, 6), device=env.device)
    vel_w[:, :3] = vel_xyz

    pose = torch.cat([pos_w, obs.data.root_quat_w.clone()], dim=-1)
    obs.write_root_pose_to_sim(pose)
    obs.write_root_velocity_to_sim(vel_w)


def update_moving_obstacles_stage5(env, _env_ids=None):
    """Advance all three keyframe obstacles. Each env starts at a random phase."""

    dt = env.step_dt

    if not hasattr(env, "s5_obs1_time"):
        env.s5_obs1_time = torch.rand(env.num_envs, device=env.device) * _OBS1_PERIOD
        env.s5_obs2_time = torch.rand(env.num_envs, device=env.device) * _OBS2_PERIOD
        env.s5_obs3_time = torch.rand(env.num_envs, device=env.device) * _OBS3_PERIOD

    env.s5_obs1_time = (env.s5_obs1_time + dt) % _OBS1_PERIOD
    env.s5_obs2_time = (env.s5_obs2_time + dt) % _OBS2_PERIOD
    env.s5_obs3_time = (env.s5_obs3_time + dt) % _OBS3_PERIOD

    _move_obstacle(env, "obstacle_1", env.s5_obs1_time, _OBS1_TIMES, _OBS1_XY)
    _move_obstacle(env, "obstacle_2", env.s5_obs2_time, _OBS2_TIMES, _OBS2_XY)
    _move_obstacle(env, "obstacle_3", env.s5_obs3_time, _OBS3_TIMES, _OBS3_XY)
