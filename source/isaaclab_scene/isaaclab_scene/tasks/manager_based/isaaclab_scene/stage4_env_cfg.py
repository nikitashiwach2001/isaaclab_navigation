import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from . import mdp

from .isaaclab_scene_env_cfg import BaseSceneCfg, BaseEventsCfg, BaseEnvCfg
from .mdp.rewards import StaticRewardsCfg, Stage4CleanRewardsCfg, RewardsCfg
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

    obstacle_1: RigidObjectCfg = STAGE4_OBSTACLE_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_1")
    obstacle_2: RigidObjectCfg = STAGE4_OBSTACLE_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_2")

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
class Stage31SceneCfg(BaseSceneCfg):
    """Stage 3.1 scene: walls only (3 inner walls), no cylinder obstacles.

    Removing cylinders is important here because Stage 3.1 doesn't move them —
    if they stayed at (2.0, 2.0) and (-2.0, -2.0) they'd permanently block goal
    positions (1.8, 2.0) and (-1.8, -2.0), wasting ~13% of episodes.
    """

    inner_wall_2: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_2", (-0.5, -2.0, 0.25), (0.707, 0.0, 0.0, -0.707))
    inner_wall_3: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_3", ( 1.0, -1.0, 0.25), (0.707, 0.0, 0.0,  0.707))
    inner_wall_6: RigidObjectCfg = _inner_wall("{ENV_REGEX_NS}/InnerWall_6", (-0.5,  1.5, 0.25), (1.0,   0.0, 0.0,  0.0))


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
    """Stage 3.1: walls-only maze (no cylinder obstacles).

    Curriculum step between Stage1 (open arena) and Stage4 (moving obstacles).
    Robot masters static maze navigation before learning dynamic avoidance.
    Cylinders excluded because they'd permanently block ~13% of goal positions.

    Uses StaticRewardsCfg (navigation_reward_static) which excludes Stage 5-specific
    dynamic-obstacle reward terms that fire wrongly for static walls during turns.
    """

    scene: Stage31SceneCfg = Stage31SceneCfg()
    events: Stage31EventsCfg = Stage31EventsCfg()
    rewards: StaticRewardsCfg = Stage4CleanRewardsCfg()
    enable_lidar_temporal_diff: bool = True


@configclass
class Stage4EnvCfg(BaseEnvCfg):
    """Stage 4: full arena with outer walls, inner walls, and two moving obstacles.

    Uses RewardsCfg (navigation_reward → navigation_reward_stage4) — the full reward
    with r_closing_proximity and r_left_dodge / r_right_dodge.
    """

    scene: Stage4SceneCfg = Stage4SceneCfg()
    events: Stage4EventsCfg = Stage4EventsCfg()
    rewards: RewardsCfg = Stage4CleanRewardsCfg()
    enable_lidar_temporal_diff: bool = True


# @configclass
# class Stage4PPOEnvCfg(Stage4EnvCfg):
#     """Stage4EnvCfg with terminal rewards added for PPO (RSL-RL does not apply them externally)."""
#     rewards: PPORewardsCfg = PPORewardsCfg()
