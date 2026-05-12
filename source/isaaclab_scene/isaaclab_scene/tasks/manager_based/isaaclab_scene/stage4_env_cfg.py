import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from . import mdp

from .isaaclab_scene_env_cfg import BaseSceneCfg, BaseEventsCfg, BaseEnvCfg
from .mdp.rewards import PPORewardsCfg
from .mdp.stage_4.obstacles import (
    STAGE4_OBSTACLE_1_CFG,
    STAGE4_OBSTACLE_2_CFG,
    update_moving_obstacles_stage4,
)


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
class Stage4SceneCfg(BaseSceneCfg):
    """Stage 4: four outer walls + three inner walls + two moving obstacles."""

    # # inner_wall_1: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_1", (-2.0, -1.5, 0.25), (1.0, 0.0, 0.0, 0.0))
    inner_wall_2: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_2", (-0.5, -2.0, 0.25), (0.707, 0.0, 0.0, -0.707))
    inner_wall_3: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_3", ( 1.0, -1.0, 0.25), (0.707, 0.0, 0.0,  0.707))
    # # inner_wall_4: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_4", ( 1.2,  1.9, 0.25), (0.707, 0.0, 0.0, -0.707))
    # # inner_wall_5: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_5", ( 1.9,  0.4, 0.25), (1.0, 0.0, 0.0,  0.0))
    inner_wall_6: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_6", (-0.5,  1.5, 0.25), (1.0, 0.0, 0.0,  0.0))
    # inner_wall_7: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_7", (-1.2,  0.092, 0.25), (0.707, 0.0, 0.0, -0.707))

    # obstacle_1: RigidObjectCfg = STAGE4_OBSTACLE_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_1")
    # obstacle_2: RigidObjectCfg = STAGE4_OBSTACLE_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_2")

    # obstacle_3: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Obstacle_3",
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.09,
    #         height=1.5,
    #         axis="Z",
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.9)),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.5, 0.6, 0.75),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #     ),
    # )

@configclass
class Stage31EventsCfg(BaseEventsCfg):
    """Stage 3.1 events: reset robot + randomize goal, no obstacle movement."""

    reset_goal_position = EventTerm(
        func=mdp.randomize_goal_positions_stage4,
        mode="reset",
    )


@configclass
class Stage4EventsCfg(BaseEventsCfg):

    reset_goal_position = EventTerm(
        func=mdp.randomize_goal_positions_stage4,
        mode="reset",
    )

    move_obstacles = EventTerm(
        func=update_moving_obstacles_stage4,
        mode="interval",
        interval_range_s=(0.01, 0.01),
    )


@configclass
class Stage31EnvCfg(BaseEnvCfg):
    """Stage 3.1: Stage4 maze (inner walls + static obstacle cylinders), no movement.

    Curriculum step between Stage1 (open arena) and Stage4 (moving obstacles).
    Robot masters static maze navigation before learning dynamic avoidance.
    """

    scene: Stage4SceneCfg = Stage4SceneCfg()
    events: Stage31EventsCfg = Stage31EventsCfg()
    enable_lidar_temporal_diff: bool = True


@configclass
class Stage4EnvCfg(BaseEnvCfg):
    """Stage 4: full arena with outer walls, inner walls, and two moving obstacles."""

    scene: Stage4SceneCfg = Stage4SceneCfg()
    events: Stage4EventsCfg = Stage4EventsCfg()
    enable_lidar_temporal_diff: bool = True


@configclass
class Stage4PPOEnvCfg(Stage4EnvCfg):
    """Stage4EnvCfg with terminal rewards added for PPO (RSL-RL does not apply them externally)."""
    rewards: PPORewardsCfg = PPORewardsCfg()
