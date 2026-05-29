"""Stage 5 robot/lidar config.

Robot articulation and contact-sensor configs come from the shared
registry-driven ``mdp/robot.py`` (TurtleBot ↔ ict_bot via ``ROBOT`` env var).
This module only overrides the lidar's mesh-target list to also pick up
``InnerWall_*`` prims that Stage 5 introduces.
"""

from isaaclab.sensors import MultiMeshRayCasterCfg, patterns

from ..robot_registry import CFG as _ROBOT_CFG
from ..robot import TURTLEBOT3_BURGER_CFG, CONTACT_SENSOR_CFG  # re-export


_BASE_LINK = _ROBOT_CFG["base_link"]


LIDAR_CFG_5 = MultiMeshRayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/" + _BASE_LINK,
    offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, _ROBOT_CFG["lidar_offset_z"])),
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
        # moving obstacles — each tracked independently (merge=False required)
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/Obstacle_*",
            track_mesh_transforms=True,
            merge_prim_meshes=False,
        ),
    ],
)
