from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .isaaclab_scene_env_cfg import BaseSceneCfg, BaseEventsCfg, BaseEnvCfg
from .mdp import (
    OBSTACLE_1_CFG,
    OBSTACLE_2_CFG,
    OBSTACLE_3_CFG,
    OBSTACLE_4_CFG,
    LIDAR_CFG,
    update_rotating_obstacles,
)


@configclass
class Stage3SceneCfg(BaseSceneCfg):
    """Stage 3: four walls + pinwheel (4 kinematic cylinders rotating together)."""

    lidar = LIDAR_CFG.replace(offset=LIDAR_CFG.OffsetCfg(pos=(0.0, 0.0, 0.25)))

    obstacle_1: RigidObjectCfg = OBSTACLE_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_1")
    obstacle_2: RigidObjectCfg = OBSTACLE_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_2")
    obstacle_3: RigidObjectCfg = OBSTACLE_3_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_3")
    obstacle_4: RigidObjectCfg = OBSTACLE_4_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_4")


@configclass
class Stage3EventsCfg(BaseEventsCfg):
    """Stage 3 events: base resets + pinwheel rotation."""

    rotate_obstacles = EventTerm(
        func=update_rotating_obstacles,
        mode="interval",
        interval_range_s=(0.01, 0.01),
    )


@configclass
class Stage3EnvCfg(BaseEnvCfg):
    """Stage 3 environment: navigate to goal while avoiding the rotating pinwheel."""

    scene: Stage3SceneCfg = Stage3SceneCfg()
    events: Stage3EventsCfg = Stage3EventsCfg()
    enable_lidar_temporal_diff: bool = True
