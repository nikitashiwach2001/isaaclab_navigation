import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from . import mdp

from .isaaclab_scene_env_cfg import BaseSceneCfg, BaseEventsCfg, BaseEnvCfg
from .mdp.rewards import Stage4RewardsCfgV2
from .mdp.stage_5.robot import LIDAR_CFG_5
from .mdp.goals import ensure_goal_pos_w, MIN_START_GOAL_DIST, OBSTACLE_CLEARANCE
from .mdp.robot import TURTLEBOT3_BURGER_CFG, CONTACT_SENSOR_CFG
from .mdp.robot_registry import CFG as _ROBOT_CFG
from .mdp.goal_marker import GOAL_MARKER_CFG
from .mdp.stage_6.obstacles import (
    STAGE6_OBSTACLE_1_CFG,
    STAGE6_OBSTACLE_2_CFG,
    STAGE6_OBSTACLE_3_CFG,
    STAGE6_OBSTACLE_4_CFG,
    STAGE6_OBSTACLE_5_CFG,
    STAGE6_OBSTACLE_6_CFG,
    update_moving_obstacles_stage6,
    randomize_obstacle_phases_stage6,
)


def _inner_wall(prim_path, pos, rot):
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.15, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )



@configclass
class Stage6SceneCfg(BaseSceneCfg):
    """Stage 6: Stage 5 arena (four inner walls) + five obstacles spread across
    the maze — obstacle_1/2/3 continuous movers, obstacle_4/5 stop-and-move."""

    lidar: RayCasterCfg = LIDAR_CFG_5.replace(offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)))

    inner_wall_2: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_2", (-0.5, -2.0, 0.25), (0.707, 0.0, 0.0, -0.707))
    inner_wall_3: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_3", ( 1.0, -1.0, 0.25), (0.707, 0.0, 0.0,  0.707))
    inner_wall_6: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_6", (-0.5,  1.5, 0.25), (1.0,   0.0, 0.0,  0.0))
    inner_wall_7: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_7", (-1.5,  0.0, 0.25), (0.707, 0.0, 0.0,  0.707))
    # inner_wall_8: vertical wall in the right-center, forms a routing pinch point
    # between obstacle_4 (x=-0.8) and obstacle_5 (x=0.7). Covers y∈[0.0, 1.0] at x≈0.5.
    inner_wall_8: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_8", ( 0.5,  0.5, 0.25), (0.707, 0.0, 0.0,  0.707))

    obstacle_1: RigidObjectCfg = STAGE6_OBSTACLE_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_1")
    obstacle_2: RigidObjectCfg = STAGE6_OBSTACLE_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_2")
    obstacle_3: RigidObjectCfg = STAGE6_OBSTACLE_3_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_3")
    obstacle_4: RigidObjectCfg = STAGE6_OBSTACLE_4_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_4")
    obstacle_5: RigidObjectCfg = STAGE6_OBSTACLE_5_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_5")
    obstacle_6: RigidObjectCfg = STAGE6_OBSTACLE_6_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_6")


@configclass
class Stage6EventsCfg(BaseEventsCfg):

    randomize_obstacle_phases = EventTerm(
        func=randomize_obstacle_phases_stage6,
        mode="reset",
    )

    reset_goal_position = EventTerm(
        func=mdp.randomize_goal_positions_stage4,
        mode="reset",
    )

    move_obstacles = EventTerm(
        func=update_moving_obstacles_stage6,
        mode="interval",
        interval_range_s=(0.01, 0.01),
    )


@configclass
class Stage6EnvCfg(BaseEnvCfg):
    """Stage 6: outer walls + four inner walls + five obstacles (four moving,
    one static). Same observation/reward as Stage 5, so a Stage 5 policy runs
    on it unchanged — used to stress-test the planner with a denser field."""

    scene: Stage6SceneCfg = Stage6SceneCfg()
    events: Stage6EventsCfg = Stage6EventsCfg()
    rewards: Stage4RewardsCfgV2 = Stage4RewardsCfgV2()
    enable_lidar_temporal_diff: bool = True


# ───────────────────────────── Stage 6a ─────────────────────────────
# Open-world evaluation scene: no outer boundary walls. Robot spawns
# inside a corridor on the left side of the arena and must exit east to
# reach a goal in the obstacle field. Same six obstacles + motion
# patterns as Stage 6. Designed for planner+policy evaluation only.

# Corridor geometry (local arena coords). Kept inside arena_half=2.35 so
# the planner's default static-grid extent still covers it.
# Long, wide corridor running west→east; robot enters from -X side and
# must exit east into the open obstacle field.
S6A_CORRIDOR_CENTER_X = -0.7
S6A_CORRIDOR_LEN_X    = 3.0    # side-wall length along X (longer than v1's 1.8 m)
S6A_CORRIDOR_WIDTH_Y  = 2.2    # outer Y-span (~2.05 m inside)
S6A_CORRIDOR_BACK_X   = S6A_CORRIDOR_CENTER_X - S6A_CORRIDOR_LEN_X / 2.0   # -2.2
S6A_CORRIDOR_EXIT_X   = S6A_CORRIDOR_CENTER_X + S6A_CORRIDOR_LEN_X / 2.0   # +0.8
S6A_ROBOT_SPAWN_X     = -1.7
S6A_ROBOT_SPAWN_Y     = 0.0

# Spawn keep-out radius — obstacles never enter this disk around the
# robot's spawn position. Stage 6 motion is unchanged; we just push any
# obstacle that interpolates inside the disk back out to its boundary.
S6A_SPAWN_KEEPOUT     = 1.0

# Goal keep-out: obstacles may *pass through* the goal disk, but if one
# loiters (or stops dead) inside the radius for more than the dwell
# limit the goal becomes unreachable. We push it out radially in that
# case so the robot always has at least one window to score.
S6A_GOAL_KEEPOUT      = 0.6
S6A_GOAL_DWELL_LIMIT  = 2.0    # seconds an obstacle may sit on/near goal

# Goal positions east of the corridor exit (x > +0.8). All are
# >= MIN_START_GOAL_DIST from the spawn at (-1.7, 0).
S6A_GOAL_POSITIONS = [
    ( 1.2,  0.0),
    ( 1.5,  0.5), ( 1.5, -0.5),
    ( 2.0,  1.0), ( 2.0, -1.0),
    ( 2.0,  1.8), ( 2.0, -1.8),
    ( 1.0,  2.0), ( 1.0, -2.0),
    ( 1.5,  1.8), ( 1.5, -1.8),
]


# Lidar config for Stage 6a: no outer walls, so the wall_* target from
# LIDAR_CFG_5 would fail to resolve. Keep only InnerWall_* and Obstacle_*.
LIDAR_CFG_6A = MultiMeshRayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/" + _ROBOT_CFG["base_link"],
    offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)),
    ray_alignment="yaw",
    pattern_cfg=patterns.LidarPatternCfg(
        channels=1,
        vertical_fov_range=(0.0, 0.0),
        horizontal_fov_range=(0.0, 360.0),
        horizontal_res=3.96,
    ),
    max_distance=3.5,
    debug_vis=False,
    mesh_prim_paths=[
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/InnerWall_.*",
            is_shared=True,
            merge_prim_meshes=True,
            track_mesh_transforms=False,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/Obstacle_*",
            track_mesh_transforms=True,
            merge_prim_meshes=False,
        ),
    ],
)


def _stage6a_corridor_wall(prim_path, pos, size, rot=(1.0, 0.0, 0.0, 0.0)):
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def reset_robot_stage6a(env, env_ids=None):
    """Place the robot at the back of the Stage 6a corridor, facing the exit (+X)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(dtype=torch.long, device=env.device)
    n = len(env_ids)
    robot = env.scene["robot"]
    env_origins_xy = env.scene.env_origins[env_ids, :2]

    pos_w = robot.data.root_pos_w.clone()
    quat_w = robot.data.root_quat_w.clone()
    pos_w[env_ids, 0] = env_origins_xy[:, 0] + S6A_ROBOT_SPAWN_X
    pos_w[env_ids, 1] = env_origins_xy[:, 1] + S6A_ROBOT_SPAWN_Y
    # face +X: yaw = 0 → quat (w=1, x=0, y=0, z=0)
    quat_w[env_ids, 0] = 1.0
    quat_w[env_ids, 1] = 0.0
    quat_w[env_ids, 2] = 0.0
    quat_w[env_ids, 3] = 0.0
    pose = torch.cat([pos_w[env_ids], quat_w[env_ids]], dim=-1)
    robot.write_root_pose_to_sim(pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros((n, 6), device=env.device), env_ids=env_ids)


def randomize_goal_positions_stage6a(env, env_ids=None):
    """Pick a Stage 6a goal from the predefined list outside the corridor."""
    goal_pos_w = ensure_goal_pos_w(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(dtype=torch.long, device=env.device)
    n = len(env_ids)

    positions = torch.tensor(S6A_GOAL_POSITIONS, device=env.device, dtype=torch.float32)
    env_origins_xy = env.scene.env_origins[env_ids, :2]
    robot = env.scene["robot"]
    robot_xy_local = robot.data.root_pos_w[env_ids, :2] - env_origins_xy

    obstacle_keys = [k for k in env.scene.keys() if k.startswith("obstacle_")]
    random_xy_local = positions[torch.randint(len(positions), (n,), device=env.device)]
    for _ in range(20):
        too_close = torch.norm(random_xy_local - robot_xy_local, dim=-1) < MIN_START_GOAL_DIST
        for obs_name in obstacle_keys:
            obs_xy_w = env.scene[obs_name].data.root_pos_w[env_ids, :2]
            obs_xy_local = obs_xy_w - env_origins_xy
            too_close = too_close | (
                torch.norm(random_xy_local - obs_xy_local, dim=-1) < OBSTACLE_CLEARANCE
            )
        if not too_close.any():
            break
        idx = torch.randint(len(positions), (int(too_close.sum()),), device=env.device)
        random_xy_local[too_close] = positions[idx]

    goal_pos_w[env_ids] = env_origins_xy + random_xy_local

    init_dist = torch.norm(goal_pos_w[env_ids] - robot.data.root_pos_w[env_ids, :2], dim=-1)
    if not hasattr(env, "goal_dist_initial"):
        env.goal_dist_initial = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "goal_dist_prev"):
        env.goal_dist_prev = torch.zeros(env.num_envs, device=env.device)
    env.goal_dist_initial[env_ids] = init_dist
    env.goal_dist_prev[env_ids] = init_dist

    if "goal_marker" in env.scene.keys():
        goal_marker = env.scene["goal_marker"]
        goal_pose_w = goal_marker.data.root_pose_w.clone()
        goal_pose_w[env_ids, 0] = goal_pos_w[env_ids, 0]
        goal_pose_w[env_ids, 1] = goal_pos_w[env_ids, 1]
        goal_pose_w[env_ids, 2] = 0.08
        goal_marker.write_root_pose_to_sim(goal_pose_w[env_ids], env_ids=env_ids)
    return goal_pos_w


def _push_obstacles_out_of_spawn(env, env_ids=None):
    """Push every obstacle radially out of the S6A spawn keep-out disk so
    obstacles never overlap the robot at spawn. Called after the Stage 6
    motion/reset updates have placed the obstacles."""
    spawn_local = torch.tensor(
        [S6A_ROBOT_SPAWN_X, S6A_ROBOT_SPAWN_Y], device=env.device, dtype=torch.float32
    )
    keepout = S6A_SPAWN_KEEPOUT

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(dtype=torch.long, device=env.device)

    env_origins_xy = env.scene.env_origins[env_ids, :2]
    for okey in [k for k in env.scene.keys() if k.startswith("obstacle_")]:
        obs = env.scene[okey]
        pos_w = obs.data.root_pos_w[env_ids].clone()
        local_xy = pos_w[:, :2] - env_origins_xy
        rel = local_xy - spawn_local
        dist = torch.norm(rel, dim=-1, keepdim=True).clamp(min=1e-6)
        too_close = (dist.squeeze(-1) < keepout)
        if not too_close.any():
            continue
        safe_local = spawn_local + (rel / dist) * keepout
        new_local = torch.where(too_close.unsqueeze(-1), safe_local, local_xy)
        pos_w[:, 0] = env_origins_xy[:, 0] + new_local[:, 0]
        pos_w[:, 1] = env_origins_xy[:, 1] + new_local[:, 1]
        quat_w = obs.data.root_quat_w[env_ids].clone()
        obs.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1), env_ids=env_ids)


def _push_obstacles_off_goal(env, env_ids=None):
    """Per-obstacle dwell timer: any obstacle that spends > dwell_limit
    seconds within S6A_GOAL_KEEPOUT of the goal gets pushed radially out
    to the keep-out boundary, so the robot always has a chance to score."""
    if not hasattr(env, "goal_pos_w"):
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(dtype=torch.long, device=env.device)

    dt = float(env.step_dt)
    keepout = S6A_GOAL_KEEPOUT
    limit = S6A_GOAL_DWELL_LIMIT
    goal_w_xy = env.goal_pos_w[env_ids]

    for okey in [k for k in env.scene.keys() if k.startswith("obstacle_")]:
        attr = f"s6a_goal_dwell_{okey}"
        if not hasattr(env, attr):
            setattr(env, attr, torch.zeros(env.num_envs, device=env.device))
        dwell_all = getattr(env, attr)

        obs = env.scene[okey]
        pos_w = obs.data.root_pos_w[env_ids].clone()
        diff = pos_w[:, :2] - goal_w_xy
        dist = torch.norm(diff, dim=-1)
        near = dist < keepout

        dwell = dwell_all[env_ids]
        dwell = torch.where(near, dwell + dt, torch.zeros_like(dwell))

        over_limit = dwell > limit
        if over_limit.any():
            # Push direction: from goal outwards along current offset.
            # If obstacle sits exactly on goal (diff≈0), default to +X.
            safe_dir = diff / dist.unsqueeze(-1).clamp(min=1e-6)
            fallback = torch.zeros_like(safe_dir)
            fallback[:, 0] = 1.0
            safe_dir = torch.where(
                dist.unsqueeze(-1) > 1e-6, safe_dir, fallback
            )
            new_xy = goal_w_xy + safe_dir * keepout
            pos_w[over_limit, 0] = new_xy[over_limit, 0]
            pos_w[over_limit, 1] = new_xy[over_limit, 1]
            quat_w = obs.data.root_quat_w[env_ids].clone()
            obs.write_root_pose_to_sim(
                torch.cat([pos_w, quat_w], dim=-1), env_ids=env_ids
            )
            dwell = torch.where(over_limit, torch.zeros_like(dwell), dwell)

        dwell_all[env_ids] = dwell


def update_moving_obstacles_stage6a(env, _env_ids=None):
    """Stage 6a obstacle motion: run Stage 6 interpolation, then enforce
    the spawn keep-out disk and the goal-dwell limit. Obstacle keyframes
    are unchanged."""
    update_moving_obstacles_stage6(env, _env_ids)
    _push_obstacles_out_of_spawn(env)
    _push_obstacles_off_goal(env)


def randomize_obstacle_phases_stage6a(env, env_ids=None):
    """Stage 6a obstacle reset: run Stage 6 randomization, then enforce
    the spawn keep-out disk and reset goal-dwell timers."""
    randomize_obstacle_phases_stage6(env, env_ids)
    _push_obstacles_out_of_spawn(env, env_ids)
    # Reset goal-dwell timers for newly-reset envs so an obstacle that
    # was previously camping doesn't carry its old timer into a new goal.
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(dtype=torch.long, device=env.device)
    for okey in [k for k in env.scene.keys() if k.startswith("obstacle_")]:
        attr = f"s6a_goal_dwell_{okey}"
        if hasattr(env, attr):
            getattr(env, attr)[env_ids] = 0.0


@configclass
class Stage6aSceneCfg(InteractiveSceneCfg):
    """Stage 6a: open arena with no outer walls. A three-walled corridor on
    the left holds the spawn position; the +X side is open and full of the
    same six Stage 6 obstacles. Used for planner evaluation in an
    open-world setting (no boundary walls to funnel the robot)."""

    num_envs: int = 4
    env_spacing: float = 10.0

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=2000.0),
    )

    # Corridor walls — named "inner_wall_*" so the planner picks them up
    # for the static occupancy grid (same convention as Stage 5/6).
    inner_wall_corridor_north: RigidObjectCfg = _stage6a_corridor_wall(
        "{ENV_REGEX_NS}/InnerWall_corridor_north",
        pos=(S6A_CORRIDOR_CENTER_X, S6A_CORRIDOR_WIDTH_Y / 2.0, 0.25),
        size=(S6A_CORRIDOR_LEN_X, 0.15, 0.5),
    )
    inner_wall_corridor_south: RigidObjectCfg = _stage6a_corridor_wall(
        "{ENV_REGEX_NS}/InnerWall_corridor_south",
        pos=(S6A_CORRIDOR_CENTER_X, -S6A_CORRIDOR_WIDTH_Y / 2.0, 0.25),
        size=(S6A_CORRIDOR_LEN_X, 0.15, 0.5),
    )
    inner_wall_corridor_back: RigidObjectCfg = _stage6a_corridor_wall(
        "{ENV_REGEX_NS}/InnerWall_corridor_back",
        pos=(S6A_CORRIDOR_BACK_X, 0.0, 0.25),
        size=(S6A_CORRIDOR_WIDTH_Y, 0.15, 0.5),
        rot=(0.707, 0.0, 0.0, 0.707),
    )

    obstacle_1: RigidObjectCfg = STAGE6_OBSTACLE_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_1")
    obstacle_2: RigidObjectCfg = STAGE6_OBSTACLE_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_2")
    obstacle_3: RigidObjectCfg = STAGE6_OBSTACLE_3_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_3")
    obstacle_4: RigidObjectCfg = STAGE6_OBSTACLE_4_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_4")
    obstacle_5: RigidObjectCfg = STAGE6_OBSTACLE_5_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_5")
    obstacle_6: RigidObjectCfg = STAGE6_OBSTACLE_6_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_6")

    robot: ArticulationCfg = TURTLEBOT3_BURGER_CFG
    lidar: RayCasterCfg = LIDAR_CFG_6A
    contact_forces: ContactSensorCfg = CONTACT_SENSOR_CFG
    goal_marker: RigidObjectCfg = GOAL_MARKER_CFG


@configclass
class Stage6aEventsCfg:
    """Reset and interval events for Stage 6a. No inherited
    reset_scene_to_default — robot spawn is fully controlled by
    reset_robot_stage6a so the corridor pose is deterministic."""

    reset_robot_position = EventTerm(
        func=reset_robot_stage6a,
        mode="reset",
    )

    randomize_obstacle_phases = EventTerm(
        func=randomize_obstacle_phases_stage6a,
        mode="reset",
    )

    reset_goal_position = EventTerm(
        func=randomize_goal_positions_stage6a,
        mode="reset",
    )

    move_obstacles = EventTerm(
        func=update_moving_obstacles_stage6a,
        mode="interval",
        interval_range_s=(0.01, 0.01),
    )


@configclass
class Stage6aEnvCfg(BaseEnvCfg):
    """Stage 6a: corridor spawn + open arena (no outer walls) + Stage 6
    obstacles. Drop-in replacement for Stage6EnvCfg in planner eval to
    stress-test behaviour when there is no outer-wall safety net."""

    scene: Stage6aSceneCfg = Stage6aSceneCfg()
    events: Stage6aEventsCfg = Stage6aEventsCfg()
    rewards: Stage4RewardsCfgV2 = Stage4RewardsCfgV2()
    enable_lidar_temporal_diff: bool = True
