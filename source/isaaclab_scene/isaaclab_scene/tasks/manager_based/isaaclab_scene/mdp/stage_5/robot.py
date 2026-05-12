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
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
            stiffness=0.0,
            damping=50.0,
            velocity_limit=10.0,
        ),
    },
)

LIDAR_CFG_5 = MultiMeshRayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",
    offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
    ray_alignment="yaw",
    pattern_cfg=patterns.LidarPatternCfg(
        channels=1,
        vertical_fov_range=(0.0, 0.0),
        horizontal_fov_range=(0.0, 360.0),
        horizontal_res=3.96,  # 90 rays
    ),
    max_distance=3.5,
    debug_vis=False,
    mesh_prim_paths=[
        # static outer walls
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/wall.*",
            is_shared=True,
            merge_prim_meshes=True,
            track_mesh_transforms=False,
        ),
        # inner maze walls
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/InnerWall_.*",
            is_shared=True,
            merge_prim_meshes=True,
            track_mesh_transforms=False,
        ),
        # three moving obstacles — each tracked independently (merge=False required)
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/Obstacle_*",
            track_mesh_transforms=True,
            merge_prim_meshes=False,
        ),
    ],
)

CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",
    update_period=0.0,
    history_length=2,
    track_air_time=False,
)
