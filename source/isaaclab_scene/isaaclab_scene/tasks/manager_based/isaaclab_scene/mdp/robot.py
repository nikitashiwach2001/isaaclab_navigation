# mdp/robot.py

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import MultiMeshRayCasterCfg, ContactSensorCfg, patterns

from .robot_registry import CFG, FORWARD_AXIS

# Lidar yaw rotation derived from the robot's forward axis. Identity for
# TurtleBot (forward = body +X); -π/2 about Z for ict_bot (forward = body -Y).
_AXIS_YAW = {
    "X":   0.0,
    "Y":   math.pi / 2.0,
    "-X":  math.pi,
    "-Y": -math.pi / 2.0,
}.get(FORWARD_AXIS, 0.0)
_LIDAR_ROT = (math.cos(_AXIS_YAW / 2.0), 0.0, 0.0, math.sin(_AXIS_YAW / 2.0))

_BASE_LINK = CFG["base_link"]

TURTLEBOT3_BURGER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=CFG["usd_path"],
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
            joint_names_expr=[CFG["left_wheel_joint"], CFG["right_wheel_joint"]],
            stiffness=0.0,
            damping=50.0,
            velocity_limit=10.0,
        ),
    },
)

LIDAR_CFG = MultiMeshRayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/" + _BASE_LINK,
    offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, CFG["lidar_offset_z"]), rot=_LIDAR_ROT),
    ray_alignment="yaw",
    pattern_cfg=patterns.LidarPatternCfg(
        channels=1,
        vertical_fov_range=(0.0, 0.0),
        horizontal_fov_range=(0.0, 360.0),
        horizontal_res=3.96,  # 360 / 90 rays = 4.0 deg resolution
    ),
    max_distance=3.5,
    debug_vis=False,
    mesh_prim_paths=[
        # static walls
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/wall_*",
            is_shared=True,
            merge_prim_meshes=True,
            track_mesh_transforms=False,
        ),
        # Moving obstacles
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/Obstacle_*",
            track_mesh_transforms=True,
            merge_prim_meshes=True,
        ),
    ],
)

CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/" + _BASE_LINK,
    update_period=0.0,
    history_length=2,
    track_air_time=False,
)
