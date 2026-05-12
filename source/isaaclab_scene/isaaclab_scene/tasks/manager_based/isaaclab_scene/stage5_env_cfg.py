import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils import configclass
from . import mdp

from .isaaclab_scene_env_cfg import BaseSceneCfg, BaseEventsCfg, BaseEnvCfg
from .mdp.rewards import PPORewardsCfg
from .mdp.stage_5.obstacles import (
    STAGE5_OBSTACLE_1_CFG,
    STAGE5_OBSTACLE_2_CFG,
    STAGE5_OBSTACLE_3_CFG,
    update_moving_obstacles_stage5,
)
from .mdp.stage_5.robot import LIDAR_CFG_5


def _inner_wall(prim_path, pos, rot):
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.15, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


@configclass
class Stage5SceneCfg(BaseSceneCfg):
    """Stage 5: four outer walls + four inner walls + three moving obstacles."""

    # Override lidar to use stage_5 config (same rays, explicitly named for stage 5)
    lidar: RayCasterCfg = LIDAR_CFG_5.replace(offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)))

    # Inherited walls from stage 4
    inner_wall_2: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_2", (-0.5, -2.0, 0.25), (0.707, 0.0, 0.0, -0.707))
    inner_wall_3: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_3", ( 1.0, -1.0, 0.25), (0.707, 0.0, 0.0,  0.707))
    inner_wall_6: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_6", (-0.5,  1.5, 0.25), (1.0,   0.0, 0.0,  0.0))

    # New wall for stage 5 — left-center vertical, creates tighter corridor near obstacle_2 start
    inner_wall_7: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_7", (-1.5,  0.0, 0.25), (0.707, 0.0, 0.0,  0.707))

    obstacle_1: RigidObjectCfg = STAGE5_OBSTACLE_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_1")
    obstacle_2: RigidObjectCfg = STAGE5_OBSTACLE_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_2")
    obstacle_3: RigidObjectCfg = STAGE5_OBSTACLE_3_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_3")


@configclass
class Stage5EventsCfg(BaseEventsCfg):

    reset_goal_position = EventTerm(
        func=mdp.randomize_goal_positions_stage4,
        mode="reset",
    )

    move_obstacles = EventTerm(
        func=update_moving_obstacles_stage5,
        mode="interval",
        interval_range_s=(0.01, 0.01),
    )


@configclass
class Stage5EnvCfg(BaseEnvCfg):
    """Stage 5: full arena with outer walls, four inner walls, and three moving obstacles."""

    scene: Stage5SceneCfg = Stage5SceneCfg()
    events: Stage5EventsCfg = Stage5EventsCfg()
    enable_lidar_temporal_diff: bool = True


@configclass
class Stage5PPOEnvCfg(Stage5EnvCfg):
    """Stage5EnvCfg with terminal rewards added for PPO (RSL-RL does not apply them externally)."""
    rewards: PPORewardsCfg = PPORewardsCfg()
