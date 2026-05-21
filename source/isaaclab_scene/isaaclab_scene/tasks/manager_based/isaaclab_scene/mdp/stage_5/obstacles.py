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
_OBS3_XY = [
    [ 2.0,  1.5],   # 0 s   — start top-right
    [-0.5,  1.8],   # 20 s  — sweep left across the top
    [ 0.5,  0.0],   # 40 s  — diagonal down to center
    [ 1.8, -1.5],   # 55 s  — right side lower
    [-1.5, -1.8],   # 75 s  — sweep across bottom
    [ 0.0,  0.5],   # 90 s  — back up through center
    [ 2.0,  1.5],   # 110 s — back to start, loop immediately
]

# Obstacle 3 — eval-only alternate path (STAGE5_OBS3_ALT_TRAJ=1): a right-edge
# vertical patrol. Speed matches trained obstacle_3.
_OBS3_TIMES_BASE_ALT = [0.0, 31.0, 62.0]
_OBS3_XY_ALT = [
    [ 2.0, -2.1],   # 0 s   — bottom-right
    [ 2.0,  2.1],   # 31 s  — straight up the right edge
    [ 2.0, -2.1],   # 62 s  — back down, loop
]

if os.environ.get("STAGE5_OBS3_ALT_TRAJ", "0") == "1":
    _OBS3_TIMES_BASE, _OBS3_XY = _OBS3_TIMES_BASE_ALT, _OBS3_XY_ALT

_OBS3_TIMES = [t * OBSTACLE_SPEED_SCALE for t in _OBS3_TIMES_BASE]
_OBS3_PERIOD = _OBS3_TIMES_BASE[-1] * OBSTACLE_SPEED_SCALE

# Obstacle 3 — eval-only blocking diagnostic (STAGE5_OBS3_BLOCK=1): obstacle_3
# holds a point ahead of the robot ON the robot->goal line, then freezes once the
# robot is partway there — a dead blocker squarely on the path, every episode.
_OBS3_BLOCK = os.environ.get("STAGE5_OBS3_BLOCK", "0") == "1"
_OBS3_BLOCK_SPEED = 0.35             # m/s — fast enough to hold position ahead of the robot
_OBS3_BLOCK_LEAD = 1.2               # m — parks this far ahead of the robot on the goal line
_OBS3_BLOCK_FREEZE_PROGRESS = 0.35   # freezes once the robot is this fraction of the way to the goal


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


def _block_update_obstacle_3(env):
    """Eval-only (STAGE5_OBS3_BLOCK=1): obstacle_3 holds a point ahead of the
    robot on the robot->goal line, then freezes once the robot is partway there —
    a dead blocker squarely on the path, every episode."""
    obs   = env.scene["obstacle_3"]
    robot = env.scene["robot"]
    n, dev = env.num_envs, env.device

    robot_xy = robot.data.root_pos_w[:, :2]
    goal_xy  = env.goal_pos_w

    if not hasattr(env, "s5_obs3_frozen"):
        env.s5_obs3_frozen = torch.zeros(n, dtype=torch.bool, device=dev)
        env.s5_obs3_d0     = torch.ones(n, device=dev)

    to_goal   = goal_xy - robot_xy
    dist_goal = torch.norm(to_goal, dim=-1, keepdim=True).clamp(min=1e-6)
    dir_goal  = to_goal / dist_goal

    # block point: on the robot->goal line, _OBS3_BLOCK_LEAD ahead of the robot
    lead     = torch.clamp(dist_goal - 0.3, min=0.0, max=_OBS3_BLOCK_LEAD)
    block_pt = robot_xy + dir_goal * lead

    # episode reset: unfreeze and record the initial robot->goal distance
    if hasattr(env, "episode_length_buf"):
        reset = env.episode_length_buf <= 1
    else:
        reset = torch.zeros(n, dtype=torch.bool, device=dev)
    if reset.any():
        env.s5_obs3_frozen = env.s5_obs3_frozen.clone()
        env.s5_obs3_d0     = env.s5_obs3_d0.clone()
        env.s5_obs3_frozen[reset] = False
        env.s5_obs3_d0[reset]     = dist_goal.squeeze(-1)[reset].clamp(min=0.5)

    # freeze once the robot is far enough along to its goal
    progress = 1.0 - dist_goal.squeeze(-1) / env.s5_obs3_d0
    env.s5_obs3_frozen = env.s5_obs3_frozen | (progress >= _OBS3_BLOCK_FREEZE_PROGRESS)

    # reset envs snap straight onto the block point; others home in on it
    cur   = torch.where(reset.unsqueeze(-1), block_pt, obs.data.root_pos_w[:, :2])
    to_bp = block_pt - cur
    d     = torch.norm(to_bp, dim=-1, keepdim=True).clamp(min=1e-6)
    move  = to_bp / d * torch.clamp(d, max=_OBS3_BLOCK_SPEED * env.step_dt)
    new_xy = torch.where(env.s5_obs3_frozen.unsqueeze(-1), cur, cur + move)

    # keep inside the arena (local ±2.3 around each env origin)
    origin = env.scene.env_origins[:, :2]
    new_xy = (new_xy - origin).clamp(-2.3, 2.3) + origin

    pos_w = torch.cat([new_xy, torch.full((n, 1), 0.25, device=dev)], dim=-1)
    vel_xyz = (pos_w - obs.data.root_pos_w) / max(env.step_dt, 1e-6)
    vel_xyz = torch.where(reset.unsqueeze(-1), torch.zeros_like(vel_xyz), vel_xyz)
    vel_w = torch.zeros((n, 6), device=dev)
    vel_w[:, :3] = vel_xyz
    pose = torch.cat([pos_w, obs.data.root_quat_w.clone()], dim=-1)
    obs.write_root_pose_to_sim(pose)
    obs.write_root_velocity_to_sim(vel_w)


def update_moving_obstacles_stage5(env, _env_ids=None):
    """Advance the keyframe obstacles (each env at a random phase). With
    STAGE5_OBS3_BLOCK=1, obstacle_3 instead seeks-and-freezes in front of the
    robot (eval-only); obstacle_1/2 stay on their keyframe paths as the control."""

    dt = env.step_dt

    if not hasattr(env, "s5_obs1_time"):
        env.s5_obs1_time = torch.rand(env.num_envs, device=env.device) * _OBS1_PERIOD
        env.s5_obs2_time = torch.rand(env.num_envs, device=env.device) * _OBS2_PERIOD
        env.s5_obs3_time = torch.rand(env.num_envs, device=env.device) * _OBS3_PERIOD

    env.s5_obs1_time = (env.s5_obs1_time + dt) % _OBS1_PERIOD
    env.s5_obs2_time = (env.s5_obs2_time + dt) % _OBS2_PERIOD
    _move_obstacle(env, "obstacle_1", env.s5_obs1_time, _OBS1_TIMES, _OBS1_XY)
    _move_obstacle(env, "obstacle_2", env.s5_obs2_time, _OBS2_TIMES, _OBS2_XY)

    if _OBS3_BLOCK:
        _block_update_obstacle_3(env)
    else:
        env.s5_obs3_time = (env.s5_obs3_time + dt) % _OBS3_PERIOD
        _move_obstacle(env, "obstacle_3", env.s5_obs3_time, _OBS3_TIMES, _OBS3_XY)


def randomize_obstacle_phases_stage5(env, env_ids=None):
    """Re-randomize the three cylinder phases at episode reset for uniform phase
    coverage. Must run before reset_goal_position. Disable with env var
    STAGE5_OBSTACLE_PHASE_RESET=0 (obstacles drift across episodes, for eval)."""
    if os.environ.get("STAGE5_OBSTACLE_PHASE_RESET", "1") != "1":
        return

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(dtype=torch.long, device=env.device)
    n = len(env_ids)

    # Lazy-init clocks if the interval update has not run yet this session
    if not hasattr(env, "s5_obs1_time"):
        env.s5_obs1_time = torch.zeros(env.num_envs, device=env.device)
        env.s5_obs2_time = torch.zeros(env.num_envs, device=env.device)
        env.s5_obs3_time = torch.zeros(env.num_envs, device=env.device)

    env.s5_obs1_time[env_ids] = torch.rand(n, device=env.device) * _OBS1_PERIOD
    env.s5_obs2_time[env_ids] = torch.rand(n, device=env.device) * _OBS2_PERIOD
    env.s5_obs3_time[env_ids] = torch.rand(n, device=env.device) * _OBS3_PERIOD

    triples = [
        ("obstacle_1", env.s5_obs1_time, _OBS1_TIMES, _OBS1_XY),
        ("obstacle_2", env.s5_obs2_time, _OBS2_TIMES, _OBS2_XY),
        ("obstacle_3", env.s5_obs3_time, _OBS3_TIMES, _OBS3_XY),
    ]
    for name, t, times, xy in triples:
        obs = env.scene[name]
        xy_local = _interp_keyframes(t[env_ids], times, xy, env.device)
        pos = torch.zeros((n, 3), device=env.device)
        pos[:, :2] = env.scene.env_origins[env_ids, :2] + xy_local
        pos[:, 2] = 0.25
        quat = obs.data.root_quat_w[env_ids].clone()
        pose = torch.cat([pos, quat], dim=-1)
        obs.write_root_pose_to_sim(pose, env_ids=env_ids)
        obs.write_root_velocity_to_sim(torch.zeros((n, 6), device=env.device), env_ids=env_ids)
