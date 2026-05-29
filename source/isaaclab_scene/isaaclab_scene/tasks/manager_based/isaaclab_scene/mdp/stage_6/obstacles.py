"""Stage 6 obstacles: six cylinders spread across the maze —
  obstacle_1, obstacle_2, obstacle_3 : continuous movers (never stop),
  obstacle_4, obstacle_5             : stop-and-move (pause on dwell keyframes),
  obstacle_6                          : slow race-track loop around the upper arena.
Each starts at a random phase, so they desynchronise and cover the whole arena.
Self-contained; Stage 5 code is untouched.
"""

import os
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


def _cylinder(color):
    return sim_utils.CylinderCfg(
        radius=0.16, height=0.50, axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,
                                                     disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


STAGE6_OBSTACLE_1_CFG = RigidObjectCfg(   # continuous
    prim_path="{ENV_REGEX_NS}/Obstacle_1", spawn=_cylinder((0.8, 0.1, 0.1)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 2.0, 0.25), rot=(1.0, 0.0, 0.0, 0.0)),
)
STAGE6_OBSTACLE_2_CFG = RigidObjectCfg(   # continuous
    prim_path="{ENV_REGEX_NS}/Obstacle_2", spawn=_cylinder((0.8, 0.1, 0.1)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.0, -2.0, 0.25), rot=(1.0, 0.0, 0.0, 0.0)),
)
STAGE6_OBSTACLE_3_CFG = RigidObjectCfg(   # continuous
    prim_path="{ENV_REGEX_NS}/Obstacle_3", spawn=_cylinder((0.8, 0.1, 0.1)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 1.5, 0.25), rot=(1.0, 0.0, 0.0, 0.0)),
)
STAGE6_OBSTACLE_4_CFG = RigidObjectCfg(   # stop-and-move (left-interior)
    prim_path="{ENV_REGEX_NS}/Obstacle_4", spawn=_cylinder((0.9, 0.5, 0.1)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.8, -1.0, 0.25), rot=(1.0, 0.0, 0.0, 0.0)),
)
STAGE6_OBSTACLE_5_CFG = RigidObjectCfg(   # stop-and-move (right-interior)
    prim_path="{ENV_REGEX_NS}/Obstacle_5", spawn=_cylinder((0.9, 0.5, 0.1)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, -0.2, 0.25), rot=(1.0, 0.0, 0.0, 0.0)),
)
STAGE6_OBSTACLE_6_CFG = RigidObjectCfg(   # slow race-track (upper-arena loop)
    prim_path="{ENV_REGEX_NS}/Obstacle_6", spawn=_cylinder((0.1, 0.3, 0.9)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.8, 1.0, 0.25), rot=(1.0, 0.0, 0.0, 0.0)),
)


# ── Keyframe tables (local arena coords) ──────────────────────────────────────
OBSTACLE_SPEED_SCALE = float(os.environ.get("OBSTACLE_SPEED_SCALE", "2.0"))

# Continuous movers (no dwells) — upper, whole-arena, and right/center/bottom.
_OBS1_TIMES_BASE = [0.0, 10.0, 50.0, 70.0, 90.0, 130.0, 140.0]
_OBS1_XY = [[2.0, 2.0], [1.5, 1.0], [-1.5, 1.0], [-1.7, -1.0], [-1.5, 1.0], [1.5, 1.0], [2.0, 2.0]]

_OBS2_TIMES_BASE = [0.0, 10.0, 40.0, 55.0, 85.0, 100.0, 110.0, 115.0, 120.0, 125.0, 130.0]
_OBS2_XY = [[-2.0, -2.0], [-1.3, -1.8], [0.5, 1.5], [-1.7, 1.5], [1.5, -0.2], [1.5, -2.0],
            [0.0, -1.5], [-0.5, -1.0], [-1.0, -1.5], [-1.5, -1.9], [-2.0, -2.0]]

_OBS3_TIMES_BASE = [0.0, 20.0, 40.0, 55.0, 75.0, 90.0, 110.0]
_OBS3_XY = [[2.0, 1.5], [-0.5, 1.8], [0.5, 0.0], [1.8, -1.5], [-1.5, -1.8], [0.0, 0.5], [2.0, 1.5]]

# Stop-and-move — vertical patrols through the interior, where they cross the
# robot's routes and block them repeatedly. Repeated keyframes are dwells (the
# obstacle holds still there, then moves on).
_OBS4_TIMES_BASE = [0.0, 20.0, 35.0, 55.0, 70.0, 90.0]
_OBS4_XY = [[-0.8, -1.0], [-0.8, 0.8], [-0.8, 0.8], [-0.8, -1.0], [-0.8, -1.0], [-0.8, -1.0]]

_OBS5_TIMES_BASE = [0.0, 20.0, 35.0, 55.0, 70.0, 90.0]
_OBS5_XY = [[0.7, -0.2], [0.7, 1.3], [0.7, 1.3], [0.7, -0.2], [0.7, -0.2], [0.7, -0.2]]

# Obstacle 6 — slow race-track loop around the upper-arena perimeter. Path clear
# of all four inner walls (inner_wall_6 at (-0.5, 1.5) sits below y=2.0; the loop
# travels at y in {1.0, 2.0}). Base period 200 s — ~45% slower than obstacle_1
# at 140 s, the slowest existing continuous mover.
_OBS6_TIMES_BASE = [0.0, 50.0, 100.0, 150.0, 200.0]
_OBS6_XY = [
    [ 1.8,  1.0],   # bottom-right of loop
    [ 1.8,  2.0],   # top-right
    [-2.0,  2.0],   # top-left
    [-2.0,  1.0],   # bottom-left
    [ 1.8,  1.0],   # close the loop
]


def _scaled(base):
    return [t * OBSTACLE_SPEED_SCALE for t in base]


# (name, env-attr clock, scaled times, xy table, period)
_MOVERS = [
    ("obstacle_1", "s6_obs1_time", _scaled(_OBS1_TIMES_BASE), _OBS1_XY,
     _OBS1_TIMES_BASE[-1] * OBSTACLE_SPEED_SCALE),
    ("obstacle_2", "s6_obs2_time", _scaled(_OBS2_TIMES_BASE), _OBS2_XY,
     _OBS2_TIMES_BASE[-1] * OBSTACLE_SPEED_SCALE),
    ("obstacle_3", "s6_obs3_time", _scaled(_OBS3_TIMES_BASE), _OBS3_XY,
     _OBS3_TIMES_BASE[-1] * OBSTACLE_SPEED_SCALE),
    ("obstacle_4", "s6_obs4_time", _scaled(_OBS4_TIMES_BASE), _OBS4_XY,
     _OBS4_TIMES_BASE[-1] * OBSTACLE_SPEED_SCALE),
    ("obstacle_5", "s6_obs5_time", _scaled(_OBS5_TIMES_BASE), _OBS5_XY,
     _OBS5_TIMES_BASE[-1] * OBSTACLE_SPEED_SCALE),
    ("obstacle_6", "s6_obs6_time", _scaled(_OBS6_TIMES_BASE), _OBS6_XY,
     _OBS6_TIMES_BASE[-1] * OBSTACLE_SPEED_SCALE),
]


def _interp_keyframes(t, times, xy, device):
    times_t = torch.tensor(times, device=device, dtype=torch.float32)
    pos_t = torch.tensor(xy, device=device, dtype=torch.float32)
    idx = torch.searchsorted(times_t, t, right=True).clamp(1, len(times) - 1)
    t0, t1 = times_t[idx - 1], times_t[idx]
    p0, p1 = pos_t[idx - 1], pos_t[idx]
    alpha = ((t - t0) / (t1 - t0).clamp(min=1e-6)).clamp(0.0, 1.0).unsqueeze(-1)
    return p0 + alpha * (p1 - p0)


def _move_obstacle(env, name, time, times, xy):
    obs = env.scene[name]
    xy_local = _interp_keyframes(time, times, xy, env.device)
    pos_w = env.scene.env_origins.clone()
    pos_w[:, :2] += xy_local
    pos_w[:, 2] = 0.25
    vel_xyz = (pos_w - obs.data.root_pos_w.clone()) / max(env.step_dt, 1e-6)
    vel_w = torch.zeros((env.num_envs, 6), device=env.device)
    vel_w[:, :3] = vel_xyz
    pose = torch.cat([pos_w, obs.data.root_quat_w.clone()], dim=-1)
    obs.write_root_pose_to_sim(pose)
    obs.write_root_velocity_to_sim(vel_w)


def update_moving_obstacles_stage6(env, _env_ids=None):
    """Advance all five obstacles. Dwell keyframes make obstacle_4/5 pause in
    place (velocity ~0) before moving on — a stop-and-move pattern."""
    dt = env.step_dt
    for _name, attr, _times, _xy, period in _MOVERS:
        if not hasattr(env, attr):
            setattr(env, attr, torch.rand(env.num_envs, device=env.device) * period)
    for name, attr, times, xy, period in _MOVERS:
        setattr(env, attr, (getattr(env, attr) + dt) % period)
        _move_obstacle(env, name, getattr(env, attr), times, xy)


def randomize_obstacle_phases_stage6(env, env_ids=None):
    """Re-randomize obstacle phases at episode reset (training).
    Disable with STAGE6_OBSTACLE_PHASE_RESET=0 (obstacles drift, for eval)."""
    if os.environ.get("STAGE6_OBSTACLE_PHASE_RESET", "1") != "1":
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(dtype=torch.long, device=env.device)
    n = len(env_ids)
    for name, attr, times, xy, period in _MOVERS:
        if not hasattr(env, attr):
            setattr(env, attr, torch.zeros(env.num_envs, device=env.device))
        clk = getattr(env, attr)
        clk[env_ids] = torch.rand(n, device=env.device) * period
        obs = env.scene[name]
        xy_local = _interp_keyframes(clk[env_ids], times, xy, env.device)
        pos = torch.zeros((n, 3), device=env.device)
        pos[:, :2] = env.scene.env_origins[env_ids, :2] + xy_local
        pos[:, 2] = 0.25
        quat = obs.data.root_quat_w[env_ids].clone()
        obs.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        obs.write_root_velocity_to_sim(torch.zeros((n, 6), device=env.device), env_ids=env_ids)
