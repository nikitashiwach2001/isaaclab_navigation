from cmath import phase

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


OBSTACLE_1_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Obstacle_1",
    spawn=sim_utils.CylinderCfg(
        radius=0.16,
        height=0.50,
        axis="Z",                          # upright cylinder
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,       # physics engine controls this body
            disable_gravity=True,          # floats at spawn height, won't fall
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),  # red
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.8, 0.8, 0.25),
        #  pos=(1.5, 1.5, 0.25)
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

OBSTACLE_2_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Obstacle_2",
    spawn=sim_utils.CylinderCfg(
        radius=0.16,
        height=0.50,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-0.8, 0.5, 0.25),
        # pos=(-1.5, -1.5, 0.25),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)


OBSTACLE_3_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Obstacle_3",
    spawn=sim_utils.CylinderCfg(
        radius=0.16,
        height=0.50,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.8, -0.8, 0.25),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)


OBSTACLE_4_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Obstacle_4",
    spawn=sim_utils.CylinderCfg(
        radius=0.16,
        height=0.50,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-0.8, -0.8, 0.25),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

# Note: Obstacle_3 is kinematic and will be moved in the update loop to create a dynamic obstacle. The others are static.

def update_moving_obstacle_3(env, env_ids=None):
    """Move Obstacle_3 slowly left-right along local X axis inside each env."""

    obstacle = env.scene["obstacle_3"]

    pos = obstacle.data.root_pos_w.clone()
    quat = obstacle.data.root_quat_w.clone()

    # Store each env's original obstacle position once.
    # This avoids using global/world x_min/x_max.
    if not hasattr(env, "moving_obstacle_3_base_pos"):
        env.moving_obstacle_3_base_pos = obstacle.data.root_pos_w.clone()

    if not hasattr(env, "moving_obstacle_3_dir"):
        env.moving_obstacle_3_dir = torch.ones(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

    dt = env.step_dt

    # Movement settings
    # phase 1 - slow speed, small range
    # speed = 0.05          # m/s
    # travel_range = 0.30  # moves +/- 0.45m from original position

    # phase 2 - faster speed, larger range
    speed = 0.12          # m/s
    travel_range = 0.50  # move

    base_pos = env.moving_obstacle_3_base_pos

    x_min = base_pos[:, 0] - travel_range
    x_max = base_pos[:, 0] + travel_range

    pos[:, 0] += env.moving_obstacle_3_dir * speed * dt

    hit_right = pos[:, 0] >= x_max
    hit_left = pos[:, 0] <= x_min

    env.moving_obstacle_3_dir = torch.where(
        hit_right,
        torch.full_like(env.moving_obstacle_3_dir, -1.0),
        env.moving_obstacle_3_dir,
    )

    env.moving_obstacle_3_dir = torch.where(
        hit_left,
        torch.full_like(env.moving_obstacle_3_dir, 1.0),
        env.moving_obstacle_3_dir,
    )

    pos[:, 0] = torch.clamp(pos[:, 0], x_min, x_max)

    root_pose = torch.cat([pos, quat], dim=-1)
    obstacle.write_root_pose_to_sim(root_pose)

    # Keep velocity clean for kinematic obstacle
    root_vel = torch.zeros((env.num_envs, 6), device=env.device)
    obstacle.write_root_velocity_to_sim(root_vel)