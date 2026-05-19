from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import RayCasterCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.envs.mdp import reset_scene_to_default

from .isaaclab_scene_env_cfg import BaseSceneCfg, BaseEnvCfg
from .mdp.rewards import StaticRewardsCfgV4
from . import mdp


# Walls-only LiDAR — no obstacle or inner-wall tracking needed for Stage 1
_LIDAR_STAGE1 =  MultiMeshRayCasterCfg(
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
        # static walls
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/wall.*",
            is_shared=True,
            merge_prim_meshes=True,
            track_mesh_transforms=False
        ),

    ],
)

@configclass
class Stage1SceneCfg(BaseSceneCfg):
    """Plain arena: four outer walls only. No inner walls, no obstacles."""
    lidar: RayCasterCfg = _LIDAR_STAGE1.replace(offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)))


@configclass
class Stage1EventsCfg:
    reset_robot_position = EventTerm(
        func=reset_scene_to_default,
        mode="reset",
    )
    reset_goal_position = EventTerm(
        func=mdp.randomize_goal_positions,
        mode="reset",
    )


@configclass
class Stage1EnvCfg(BaseEnvCfg):
    """Stage 1: open arena, goal-seeking only. Used as warm-up before Stage 3.1 / Stage 4."""
    scene: Stage1SceneCfg = Stage1SceneCfg()
    events: Stage1EventsCfg = Stage1EventsCfg()
    rewards: StaticRewardsCfgV4 = StaticRewardsCfgV4()
    enable_lidar_temporal_diff: bool = False


