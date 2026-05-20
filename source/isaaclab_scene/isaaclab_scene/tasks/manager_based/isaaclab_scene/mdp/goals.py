import torch

from isaaclab.envs import ManagerBasedRLEnv

# Walls sit at ±2.425 m; keep 0.5 m clearance so the goal is never inside a wall
ARENA_LIMIT = 1.9
MIN_START_GOAL_DIST = 1.0

# local XY positions of static obstacles inside each arena
OBSTACLE_POSITIONS_LOCAL = torch.zeros((0, 2))

# goal must be at least this far from any obstacle center
# obstacle radius 0.16 + goal threshold 0.25 + buffer = 0.5
OBSTACLE_CLEARANCE = 0.5

# obstacle names to avoid when placing goals
_STAGE_OBSTACLE_NAMES = ["obstacle_1", "obstacle_2", "obstacle_3", "obstacle_4"]

STAGE4_GOAL_POSITIONS = [
    ( 2.0,  1.5), ( 1.8,  2.0), ( 1.5,  0.8),
    ( 2.0, -1.5), ( 1.5, -1.8), ( 0.5, -2.0),
    (-1.0, -2.0), (-1.8, -2.0), (-2.0, -0.8),
    (-2.0,  0.8), (-2.0,  1.8), (-1.5,  2.0),
    ( 0.0,  2.0), ( 0.5,  1.5), ( 1.8, -0.5),
]


def ensure_goal_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "goal_pos_w"):
        env.goal_pos_w = torch.zeros((env.num_envs, 2), device=env.device)
    return env.goal_pos_w


INNER_WALL_KEEPOUT = 0.75   # min distance from any inner wall center to a safe spawn
OBSTACLE_KEEPOUT   = 0.6    # min distance from any obstacle initial position


def randomize_robot_positions(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """Place the robot at a random position + orientation in the arena.

    Avoids:
        - Inner walls (any scene entry whose name starts with 'inner_wall_')
        - Moving obstacles (entries starting with 'obstacle_') at their current poses

    Random yaw in [-π, π] so the robot does not always start facing the same direction.

    Stage 1 (open arena) automatically has empty wall list — no special handling needed.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(dtype=torch.long, device=env.device)
    num_reset_envs = len(env_ids)

    robot = env.scene["robot"]
    env_origins_xy = env.scene.env_origins[env_ids, :2]

    # Collect keep-out points from current scene state
    wall_keys     = [k for k in env.scene.keys() if k.startswith("inner_wall_")]
    obstacle_keys = [k for k in env.scene.keys() if k.startswith("obstacle_")]

    # Rejection sampling: random XY in arena, resample if too close to walls/obstacles
    random_xy_local = (torch.rand((num_reset_envs, 2), device=env.device) * 2.0 - 1.0) * ARENA_LIMIT
    for _ in range(20):
        bad = torch.zeros(num_reset_envs, dtype=torch.bool, device=env.device)
        for wkey in wall_keys:
            wall_xy_w     = env.scene[wkey].data.root_pos_w[env_ids, :2]
            wall_xy_local = wall_xy_w - env_origins_xy
            bad = bad | (torch.norm(random_xy_local - wall_xy_local, dim=-1) < INNER_WALL_KEEPOUT)
        for okey in obstacle_keys:
            obs_xy_w     = env.scene[okey].data.root_pos_w[env_ids, :2]
            obs_xy_local = obs_xy_w - env_origins_xy
            bad = bad | (torch.norm(random_xy_local - obs_xy_local, dim=-1) < OBSTACLE_KEEPOUT)
        if not bad.any():
            break
        resampled = (torch.rand((int(bad.sum()), 2), device=env.device) * 2.0 - 1.0) * ARENA_LIMIT
        random_xy_local[bad] = resampled

    # Random yaw in [-π, π], encoded as quaternion (w, x, y, z) for rotation about Z
    yaw = (torch.rand(num_reset_envs, device=env.device) * 2.0 - 1.0) * 3.14159265
    qw = torch.cos(yaw * 0.5)
    qz = torch.sin(yaw * 0.5)
    qx = torch.zeros_like(qw)
    qy = torch.zeros_like(qw)

    # Build 7-D pose [x, y, z, qw, qx, qy, qz]
    pos_w  = robot.data.root_pos_w.clone()
    quat_w = robot.data.root_quat_w.clone()
    pos_w[env_ids, 0] = env_origins_xy[:, 0] + random_xy_local[:, 0]
    pos_w[env_ids, 1] = env_origins_xy[:, 1] + random_xy_local[:, 1]
    # leave z (height) as default
    quat_w[env_ids, 0] = qw
    quat_w[env_ids, 1] = qx
    quat_w[env_ids, 2] = qy
    quat_w[env_ids, 3] = qz
    pose = torch.cat([pos_w[env_ids], quat_w[env_ids]], dim=-1)
    robot.write_root_pose_to_sim(pose, env_ids=env_ids)

    # Zero out velocity so robot starts from rest
    vel = torch.zeros((num_reset_envs, 6), device=env.device)
    robot.write_root_velocity_to_sim(vel, env_ids=env_ids)


def randomize_goal_positions(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """Place the goal at a random position at least MIN_START_GOAL_DIST away from the robot."""

    goal_pos_w = ensure_goal_pos_w(env)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    env_ids = env_ids.to(dtype=torch.long, device=env.device)
    num_reset_envs = len(env_ids)

    robot = env.scene["robot"]
    env_origins_xy = env.scene.env_origins[env_ids, :2]
    robot_xy_local = robot.data.root_pos_w[env_ids, :2] - env_origins_xy

    random_xy_local = (torch.rand((num_reset_envs, 2), device=env.device) * 2.0 - 1.0) * ARENA_LIMIT

    for _ in range(20):
        too_close = torch.norm(random_xy_local - robot_xy_local, dim=-1) < MIN_START_GOAL_DIST

        # check against each obstacle's current world position
        for obs_name in _STAGE_OBSTACLE_NAMES:
            if obs_name not in env.scene.keys():
                continue
            obs_xy_w = env.scene[obs_name].data.root_pos_w[env_ids, :2]
            obs_xy_local = obs_xy_w - env_origins_xy
            dist = torch.norm(random_xy_local - obs_xy_local, dim=-1)
            too_close = too_close | (dist < OBSTACLE_CLEARANCE)

        if not too_close.any():
            break
        resampled = (torch.rand((int(too_close.sum()), 2), device=env.device) * 2.0 - 1.0) * ARENA_LIMIT
        random_xy_local[too_close] = resampled

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




def randomize_goal_positions_stage4(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """Pick a random goal from the predefined safe list for stage 4."""

    goal_pos_w = ensure_goal_pos_w(env)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    env_ids = env_ids.to(dtype=torch.long, device=env.device)
    num_reset_envs = len(env_ids)

    positions = torch.tensor(STAGE4_GOAL_POSITIONS, device=env.device, dtype=torch.float32)
    env_origins_xy = env.scene.env_origins[env_ids, :2]
    robot = env.scene["robot"]
    robot_xy_local = robot.data.root_pos_w[env_ids, :2] - env_origins_xy

    for _ in range(20):
        idx = torch.randint(len(positions), (num_reset_envs,), device=env.device)
        random_xy_local = positions[idx]
        too_close = torch.norm(random_xy_local - robot_xy_local, dim=-1) < MIN_START_GOAL_DIST
        for obs_name in _STAGE_OBSTACLE_NAMES:
            if obs_name not in env.scene.keys():
                continue
            obs_xy_w = env.scene[obs_name].data.root_pos_w[env_ids, :2]
            obs_xy_local = obs_xy_w - env_origins_xy
            too_close = too_close | (torch.norm(random_xy_local - obs_xy_local, dim=-1) < OBSTACLE_CLEARANCE)
        if not too_close.any():
            break

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
