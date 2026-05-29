import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils import configclass
from . import mdp

from .isaaclab_scene_env_cfg import BaseSceneCfg, BaseEventsCfg, BaseEnvCfg
from .mdp.rewards import Stage4RewardsCfgV2, Stage4RewardsCfgV2Orbit, Stage4RewardsCfgV2Smooth
from .mdp.stage_5.obstacles import (
    STAGE5_OBSTACLE_1_CFG,
    STAGE5_OBSTACLE_2_CFG,
    STAGE5_OBSTACLE_3_CFG,
    update_moving_obstacles_stage5,
    randomize_obstacle_phases_stage5,
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

    # Runs before reset_goal_position so the goal selector sees new obstacle positions
    randomize_obstacle_phases = EventTerm(
        func=randomize_obstacle_phases_stage5,
        mode="reset",
    )

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
    rewards: Stage4RewardsCfgV2 = Stage4RewardsCfgV2()
    enable_lidar_temporal_diff: bool = True


_MIXED_SPAWN_RANDOM_FRAC = 0.5   # fraction of episodes that get a random spawn


def reset_stage5_mixedspawn(env, env_ids=None):
    """Combined Stage 5 reset, mixed spawn. Per episode, per env: ~50% start at
    a random arena position, ~50% stay at the fixed origin. Order: obstacle
    phases -> robot spawn -> goal (goal selector needs the new obstacle positions)."""
    randomize_obstacle_phases_stage5(env, env_ids)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    rand_mask = torch.rand(len(env_ids), device=env.device) < _MIXED_SPAWN_RANDOM_FRAC
    rand_ids = env_ids[rand_mask]
    if len(rand_ids) > 0:
        mdp.randomize_robot_positions(env, rand_ids)
    mdp.randomize_goal_positions_stage4(env, env_ids)


@configclass
class Stage5MixedSpawnEventsCfg(BaseEventsCfg):
    """Stage 5 events with mixed (50% fixed / 50% random) robot spawn.
    Base reset_robot_position runs first, then reset_goal_position runs the
    combined obstacle + mixed-robot + goal reset."""

    reset_goal_position = EventTerm(
        func=reset_stage5_mixedspawn,
        mode="reset",
    )

    move_obstacles = EventTerm(
        func=update_moving_obstacles_stage5,
        mode="interval",
        interval_range_s=(0.01, 0.01),
    )


@configclass
class Stage5MixedSpawnEnvCfg(Stage5EnvCfg):
    """Stage 5, mixed robot spawn — ~50% fixed-origin, ~50% random per episode.
    Inherits Stage5EnvCfg's scene and Stage4RewardsCfgV2 reward; only the spawn
    differs. The finetune target for the random-spawn Stage 4 checkpoint."""

    events: Stage5MixedSpawnEventsCfg = Stage5MixedSpawnEventsCfg()


@configclass
class Stage5MixedSpawnOrbitEnvCfg(Stage5MixedSpawnEnvCfg):
    """Stage 5 mixed-spawn + orbit-penalty reward variant.
    Same scene/events as Stage5MixedSpawnEnvCfg; only the reward differs.
    Used to finetune a policy whose dodge collapses into spinning around
    obstacles instead of side-weaving past them."""

    rewards: Stage4RewardsCfgV2Orbit = Stage4RewardsCfgV2Orbit()


@configclass
class Stage5MixedSpawnSmoothEnvCfg(Stage5MixedSpawnEnvCfg):
    """Stage 5 mixed-spawn + orbit penalty + gap-through bonus.
    Same scene/events as Stage5MixedSpawnEnvCfg; only the reward differs.
    The "best behavior" finetune target — stops orbiting AND pays for
    side-weave, so the policy translates past obstacles smoothly."""

    rewards: Stage4RewardsCfgV2Smooth = Stage4RewardsCfgV2Smooth()


