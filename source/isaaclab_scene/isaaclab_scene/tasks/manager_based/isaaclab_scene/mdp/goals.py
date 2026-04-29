import torch

from isaaclab.envs import ManagerBasedRLEnv


GOAL_X_LIMIT = 2.0
GOAL_Y_LIMIT = 2.0

SAFE_GOAL_POINTS_LOCAL = torch.tensor([

    # [ 1.45, 0.90], [ 0.90, 1.45], [-1.45, 0.65], [-0.90, 1.15], [ 1.45, -0.90], [ 0.90, -1.45], [-1.45, -0.90], [-0.90, -1.45],
    # Original trained goals
    [ 1.80,  0.90],
    [ 0.90,  1.80],

    [-1.80,  0.70],
    [-0.90,  1.80],

    [ 1.80, -0.90],
    [ 0.90, -1.80],

    [-1.80, -0.90],
    [-0.90, -1.80],

    # Medium side goals
    [ 1.60,  0.40],
    [ 1.60, -0.40],
    [-1.60,  0.40],
    [-1.60, -0.40],

    [ 0.40,  1.60],
    [-0.40,  1.60],
    [ 0.40, -1.60],
    [-0.40, -1.60],

    # Crossing-path goals for moving-obstacle learning
    [ 1.80,  0.00],
    [-1.80,  0.00],
    [ 0.00,  1.80],
    [ 0.00, -1.80],

    [ 1.60,  1.60],
    [-1.60,  1.60],
    [ 1.60, -1.60],
    [-1.60, -1.60],

    # # Obstacle-passing goals - near path but not too close behind obstacles
    # [ 1.80,  0.90],
    # [ 0.90,  1.80],
    # [-1.80,  0.70],
    # [-0.90,  1.80],
    # [ 1.80, -0.90],
    # [ 0.90, -1.80],
    # [-1.80, -0.90],
    # [-0.90, -1.80],

    # # Easier corner goals for stability
    # [ 1.80,  1.80],
    # [ 1.80, -1.80],
    # [-1.80,  1.80],
    # [-1.80, -1.80],
])

# SAFE_GOAL_POINTS_LOCAL = torch.tensor([
#     # Hard goals near / behind obstacles
#     [ 1.25,  0.80],
#     [ 0.80,  1.25],

#     [-1.25,  0.50],
#     [-0.80,  0.95],

#     [ 1.25, -0.80],
#     [ 0.80, -1.25],

#     [-1.25, -0.80],
#     [-0.80, -1.25],

    # Medium difficulty side goals
    # [ 1.60,  0.40],
    # [ 1.60, -0.40],
    # [-1.60,  0.40],
    # [-1.60, -0.40],
    # [ 0.40,  1.60],
    # [-0.40,  1.60],
    # [ 0.40, -1.60],
    # [-0.40, -1.60],

    # Easier corner goals for stability
    # [ 1.80,  1.80],
    # [ 1.80, -1.80],
    # [-1.80,  1.80],
    # [-1.80, -1.80],
# ])

def ensure_goal_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Create goal_pos_w if it does not exist yet."""

    if not hasattr(env, "goal_pos_w"):
        env.goal_pos_w = torch.zeros((env.num_envs, 2), device=env.device)

    return env.goal_pos_w


def randomize_goal_positions(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """Randomize goal position for selected envs and move goal marker."""

    goal_pos_w = ensure_goal_pos_w(env)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    env_ids = env_ids.to(dtype=torch.long, device=env.device)
    num_reset_envs = len(env_ids)

    # Random goal inside each local arena
    safe_points = SAFE_GOAL_POINTS_LOCAL.to(env.device)
    random_ids = torch.randint(
        low=0,
        high=safe_points.shape[0],
        size=(num_reset_envs,),
        device=env.device,
    )
    random_xy_local = safe_points[random_ids]

    # Convert local env goal to world position
    env_origins_xy = env.scene.env_origins[env_ids, :2]
    goal_pos_w[env_ids] = env_origins_xy + random_xy_local

    # Initialize reward distance buffers for selected envs
    robot = env.scene["robot"]
    robot_xy = robot.data.root_pos_w[env_ids, :2]
    goal_xy = goal_pos_w[env_ids]

    init_dist = torch.norm(goal_xy - robot_xy, dim=-1)

    if not hasattr(env, "goal_dist_initial"):
        env.goal_dist_initial = torch.zeros(env.num_envs, device=env.device)

    if not hasattr(env, "goal_dist_prev"):
        env.goal_dist_prev = torch.zeros(env.num_envs, device=env.device)

    env.goal_dist_initial[env_ids] = init_dist
    env.goal_dist_prev[env_ids] = init_dist

    # Move visual goal marker to goal world position
    if "goal_marker" in env.scene.keys():
        goal_marker = env.scene["goal_marker"]

        goal_pose_w = goal_marker.data.root_pose_w.clone()

        goal_pose_w[env_ids, 0] = goal_pos_w[env_ids, 0]
        goal_pose_w[env_ids, 1] = goal_pos_w[env_ids, 1]
        goal_pose_w[env_ids, 2] = 0.08

        goal_marker.write_root_pose_to_sim(goal_pose_w[env_ids], env_ids=env_ids)

    return goal_pos_w