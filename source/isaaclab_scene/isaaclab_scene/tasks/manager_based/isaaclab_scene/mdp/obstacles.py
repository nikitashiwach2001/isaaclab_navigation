from cmath import phase
import math

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
            kinematic_enabled=True,       # physics engine controls this body
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
            kinematic_enabled=True,
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
            kinematic_enabled=True,
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



ROTATION_PERIOD = 40.0                          # seconds per full rotation
ORBIT_RADIUS    = math.sqrt(2.0)                # distance from center to each cylinder
ANGULAR_VEL     = 2.0 * math.pi / ROTATION_PERIOD

# angle of each cylinder at t=0 (matching +x+y, -x+y, -x-y, +x-y)
_BASE_ANGLES = [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]
_OBSTACLE_NAMES = ["obstacle_1", "obstacle_2", "obstacle_3", "obstacle_4"]


def update_rotating_obstacles(env, env_ids=None):
    """Rotate all 4 obstacles together as a pinwheel around each env's center."""

    if not hasattr(env, "pinwheel_angle"):
        env.pinwheel_angle = torch.zeros(env.num_envs, device=env.device)

    env.pinwheel_angle += ANGULAR_VEL * env.step_dt

    env_origins = env.scene.env_origins  # (num_envs, 3)

    for i, name in enumerate(_OBSTACLE_NAMES):
        obstacle = env.scene[name]
        angle = env.pinwheel_angle + _BASE_ANGLES[i]  # (num_envs,)

        pos  = obstacle.data.root_pos_w.clone()
        quat = obstacle.data.root_quat_w.clone()

        pos[:, 0] = env_origins[:, 0] + ORBIT_RADIUS * torch.cos(angle)
        pos[:, 1] = env_origins[:, 1] + ORBIT_RADIUS * torch.sin(angle)
        pos[:, 2] = 0.25

        obstacle.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        obstacle.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))




# Note: Obstacle_3 is kinematic and will be moved in the update loop to create a dynamic obstacle. The others are static.
# single moving obstacle
# def update_moving_obstacle_3(env, env_ids=None):
#     """Move Obstacle_3 slowly left-right along local X axis inside each env."""

#     obstacle = env.scene["obstacle_3"]

#     pos = obstacle.data.root_pos_w.clone()
#     quat = obstacle.data.root_quat_w.clone()

#     # Store each env's original obstacle position once.
#     # This avoids using global/world x_min/x_max.
#     if not hasattr(env, "moving_obstacle_3_base_pos"):
#         env.moving_obstacle_3_base_pos = obstacle.data.root_pos_w.clone()

#     if not hasattr(env, "moving_obstacle_3_dir"):
#         env.moving_obstacle_3_dir = torch.ones(
#             env.num_envs,
#             device=env.device,
#             dtype=torch.float32,
#         )

#     dt = env.step_dt

#     # Movement settings
#     # phase 1 - slow speed, small range
#     # speed = 0.05          # m/s
#     # travel_range = 0.30  # moves +/- 0.45m from original position

#     # phase 2 - faster speed, larger range
#     speed = 0.12          # m/s
#     travel_range = 0.50  # move

#     base_pos = env.moving_obstacle_3_base_pos

#     x_min = base_pos[:, 0] - travel_range
#     x_max = base_pos[:, 0] + travel_range

#     pos[:, 0] += env.moving_obstacle_3_dir * speed * dt

#     hit_right = pos[:, 0] >= x_max
#     hit_left = pos[:, 0] <= x_min

#     env.moving_obstacle_3_dir = torch.where(
#         hit_right,
#         torch.full_like(env.moving_obstacle_3_dir, -1.0),
#         env.moving_obstacle_3_dir,
#     )

#     env.moving_obstacle_3_dir = torch.where(
#         hit_left,
#         torch.full_like(env.moving_obstacle_3_dir, 1.0),
#         env.moving_obstacle_3_dir,
#     )

#     pos[:, 0] = torch.clamp(pos[:, 0], x_min, x_max)

#     root_pose = torch.cat([pos, quat], dim=-1)
#     obstacle.write_root_pose_to_sim(root_pose)

#     # Keep velocity clean for kinematic obstacle
#     root_vel = torch.zeros((env.num_envs, 6), device=env.device)
#     obstacle.write_root_velocity_to_sim(root_vel)