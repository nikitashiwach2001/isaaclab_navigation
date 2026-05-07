# mdp/robot.py

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import MultiMeshRayCasterCfg, ContactSensorCfg, patterns

TURTLEBOT3_BURGER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/user/Documents/isaaclab_scene/assets/robots/turtlebot3_burger.usd",
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # arena origin
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
            stiffness=0.0,
            damping=50.0,
            velocity_limit=10.0,
        ),
    },
)

LIDAR_CFG = MultiMeshRayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",
    offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
    # attach_yaw_only=True,
    ray_alignment="yaw",
    pattern_cfg=patterns.LidarPatternCfg(
        channels=1,
        vertical_fov_range=(0.0, 0.0),
        horizontal_fov_range=(0.0, 360.0),
        horizontal_res=3.96,  # 360 / 90 rays = 4.0 deg resolution
        # horizontal_res = 8.9,  # 360 / 40 rays = 8.9 deg resolution

    ),
    max_distance=3.5,
    debug_vis=False,
    mesh_prim_paths=[
        # MultiMeshRayCasterCfg.RaycastTargetCfg(
        #     prim_expr="{ENV_REGEX_NS}/(?!Robot$).*",
        #     is_shared=False,
        #     merge_prim_meshes=True,
        #     track_mesh_transforms=False,
        # ),

        # static walls
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/wall.*",
            is_shared=True,
            merge_prim_meshes=True,
            track_mesh_transforms=False
        ),

        # MultiMeshRayCasterCfg.RaycastTargetCfg(
        #     prim_expr="{ENV_REGEX_NS}/Obstacle_.*",
        #     track_mesh_transforms=False,
        #     merge_prim_meshes=True,
        # ),

        # Static obstacles: Obstacle_1, Obstacle_2, Obstacle_4...
        # Excludes Obstacle_3
        # MultiMeshRayCasterCfg.RaycastTargetCfg(
        #     prim_expr="{ENV_REGEX_NS}/Obstacle_(?!3$).*",
        #     track_mesh_transforms=False,
        #     merge_prim_meshes=True,
        # ),

        # Moving obstacle only
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/Obstacle_*",
            track_mesh_transforms=True,
            merge_prim_meshes=True,
        ),
    ],
)

CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",
    update_period=0.0,
    history_length=2,
    track_air_time=False,
)



