import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from . import mdp

from .isaaclab_scene_env_cfg import BaseSceneCfg, BaseEventsCfg, BaseEnvCfg
from .mdp.rewards import RewardsCfg, Stage4RewardsCfgV2, Stage4RewardsCfgV3
from .mdp.stage_4.obstacles import (
    STAGE4_OBSTACLE_1_CFG,
    STAGE4_OBSTACLE_2_CFG,
    STAGE4_OBSTACLE_3_CFG,
    update_moving_obstacles_stage4,
    randomize_obstacle_phases_stage4,
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
    reset_goal_position = EventTerm(
        func=mdp.randomize_goal_positions_stage4,
        mode="reset",
    )


@configclass
class Stage4EventsCfg(BaseEventsCfg):
    randomize_obstacle_phases = EventTerm(
        func=randomize_obstacle_phases_stage4,
        mode="reset",
    )

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
    """Stage 3.1: walls-only maze (no cylinder obstacles)."""

    scene: Stage31SceneCfg = Stage31SceneCfg()
    events: Stage31EventsCfg = Stage31EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    enable_lidar_temporal_diff: bool = True


@configclass
class Stage4EnvCfg(BaseEnvCfg):
    """Stage 4: full arena with outer walls, inner walls, and two moving obstacles."""

    scene: Stage4SceneCfg = Stage4SceneCfg()
    events: Stage4EventsCfg = Stage4EventsCfg()
    rewards: Stage4RewardsCfgV2 = Stage4RewardsCfgV2()
    enable_lidar_temporal_diff: bool = True


def reset_stage4_randspawn(env, env_ids=None):
    """Combined Stage 4 reset with random robot spawn, in a guaranteed order:
    obstacle phases -> random robot position -> goal placement. Using one reset
    event keeps the order deterministic regardless of event-manager field
    iteration. Robot placement avoids walls/obstacles; goal avoids robot/obstacles."""
    randomize_obstacle_phases_stage4(env, env_ids)
    mdp.randomize_robot_positions(env, env_ids)
    mdp.randomize_goal_positions_stage4(env, env_ids)


@configclass
class Stage4RandSpawnEventsCfg(BaseEventsCfg):
    """Stage 4 events with random robot spawn (freeze-fix retrain variant).
    Base reset_robot_position (reset_scene_to_default) runs first; then
    reset_goal_position runs the combined obstacle + random-robot + goal reset."""

    reset_goal_position = EventTerm(
        func=reset_stage4_randspawn,
        mode="reset",
    )

    move_obstacles = EventTerm(
        func=update_moving_obstacles_stage4,
        mode="interval",
        interval_range_s=(0.01, 0.01),
    )


@configclass
class Stage4RandSpawnEnvCfg(Stage4EnvCfg):
    """Stage 4, random robot spawn — freeze-fix variant. Identical to
    Stage4EnvCfg except the robot starts at a random arena position each
    episode, flooding training with diverse open-space states to train out
    the open-space freeze. Fixed-spawn Stage4EnvCfg is left untouched."""

    events: Stage4RandSpawnEventsCfg = Stage4RandSpawnEventsCfg()


_MIXED_SPAWN_RANDOM_FRAC = 0.5   # fraction of episodes that get a random spawn


def reset_stage4_mixedspawn(env, env_ids=None):
    """Combined Stage 4 reset, mixed spawn. Per episode, per env: ~50% start at
    a random arena position, ~50% stay at the fixed origin (already placed
    there by reset_scene_to_default). Fixed-spawn episodes keep cylinder-timing
    sharp (low DYN); random-spawn episodes keep position robustness (low
    TIMEOUT/STATIC). Order: obstacle phases -> robot spawn -> goal."""
    randomize_obstacle_phases_stage4(env, env_ids)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    rand_mask = torch.rand(len(env_ids), device=env.device) < _MIXED_SPAWN_RANDOM_FRAC
    rand_ids = env_ids[rand_mask]
    if len(rand_ids) > 0:
        mdp.randomize_robot_positions(env, rand_ids)
    mdp.randomize_goal_positions_stage4(env, env_ids)


@configclass
class Stage4MixedSpawnEventsCfg(BaseEventsCfg):
    """Stage 4 events with mixed (50% fixed / 50% random) robot spawn.
    Base reset_robot_position (reset_scene_to_default) runs first; then
    reset_goal_position runs the combined obstacle + mixed-robot + goal reset."""

    reset_goal_position = EventTerm(
        func=reset_stage4_mixedspawn,
        mode="reset",
    )

    move_obstacles = EventTerm(
        func=update_moving_obstacles_stage4,
        mode="interval",
        interval_range_s=(0.01, 0.01),
    )


@configclass
class Stage4ThreePillarSceneCfg(Stage4SceneCfg):
    """Stage 4 scene + a third moving pillar (obstacle_3, blue, simple slow
    motion). Generalization test — the policy never trained on a 3rd pillar."""

    obstacle_3: RigidObjectCfg = STAGE4_OBSTACLE_3_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_3")


@configclass
class Stage4ThreePillarEnvCfg(Stage4EnvCfg):
    """Stage 4 with a third moving pillar — for evaluating whether the current
    policy avoids a never-trained-on obstacle (genuine perception-based
    avoidance vs memorization). Eval-only; same events/reward as Stage4EnvCfg.
    update_moving_obstacles_stage4 moves obstacle_3 automatically when present."""

    scene: Stage4ThreePillarSceneCfg = Stage4ThreePillarSceneCfg()


@configclass
class Stage4MixedSpawnEnvCfg(Stage4EnvCfg):
    """Stage 4, mixed robot spawn. Each episode is ~50% fixed-origin spawn,
    ~50% random spawn — combining the cylinder-timing sharpness of fixed spawn
    with the position robustness of random spawn. Uses Stage4RewardsCfgV3
    (adds the side-push + cylinder-proximity collision fixes). Fixed-spawn
    Stage4EnvCfg and Stage4RandSpawnEnvCfg are left untouched."""

    events: Stage4MixedSpawnEventsCfg = Stage4MixedSpawnEventsCfg()
    rewards: Stage4RewardsCfgV3 = Stage4RewardsCfgV3()
