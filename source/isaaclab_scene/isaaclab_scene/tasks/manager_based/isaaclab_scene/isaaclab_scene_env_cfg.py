import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from .mdp import (
    OBSTACLE_1_CFG,
    OBSTACLE_2_CFG,
    OBSTACLE_3_CFG,
    OBSTACLE_4_CFG,
    TerminationsCfg,
    ObservationsCfg,        
    RewardsCfg,
    ActionsCfg,
    update_moving_obstacle_3,
)

from .mdp import TURTLEBOT3_BURGER_CFG, LIDAR_CFG, CONTACT_SENSOR_CFG
from isaaclab.sensors import RayCasterCfg, ContactSensorCfg
from .mdp import GOAL_MARKER_CFG
from isaaclab.envs.mdp import reset_scene_to_default


##
# Scene definition
##


@configclass
class IsaaclabSceneSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # num of envs and spacing
    num_envs: int = 4
    env_spacing: float = 4.0

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    # robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=2000.0),
    )

    # outer walls
    wall_1 = RigidObjectCfg(
        prim_path = "{ENV_REGEX_NS}/wall_1",
        spawn=sim_utils.CuboidCfg(size=(5.0, 0.15, 0.5),
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                  mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
                                  collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.25, 0.1)),
                                  ),
    init_state = RigidObjectCfg.InitialStateCfg(pos=(-2.425, 0.0, 0.25),
                                             rot=(0.707, 0.0, 0.0, 0.707), # 90 degrees around x-axis
                                             ),
    )

    wall_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall_2",
        spawn=sim_utils.CuboidCfg(
            size=(5.0, 0.15, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.25, 0.1)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 2.425, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),       # 0°
        ),
    )

    wall_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall_3",
        spawn=sim_utils.CuboidCfg(
            size=(5.0, 0.15, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.25, 0.1)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.425, 0.0, 0.25),
            rot=(0.707, 0.0, 0.0, -0.707),  # -90°
        ),
    )


    wall_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall_4",
        spawn=sim_utils.CuboidCfg(
            size=(5.0, 0.15, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.25, 0.1)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -2.425, 0.25),
            rot=(0.0, 0.0, 0.0, 1.0),  # 180°
        ),
    )


    # inner walls
    # inner_wall_1 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/InnerWall_1",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.0, 0.15, 0.5),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.5, 0.3, 0.1)
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(-2.0, -1.5, 0.25),
    #         rot=(1.0, 0.0, 0.0, 0.0),       # 0°
    #     ),
    # )

    # inner_wall_2 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/InnerWall_2",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.0, 0.15, 0.5),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.5, 0.3, 0.1)
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(-0.5, -2.0, 0.25),
    #         rot=(0.707, 0.0, 0.0, -0.707),  # -90°
    #     ),
    # )

    # inner_wall_3 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/InnerWall_3",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.0, 0.15, 0.5),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.5, 0.3, 0.1)
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(1.0, -1.0, 0.25),
    #         rot=(0.707, 0.0, 0.0, 0.707),   # 90°
    #     ),
    # )

    # inner_wall_4 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/InnerWall_4",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.0, 0.15, 0.5),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.5, 0.3, 0.1)
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(1.2, 1.9, 0.25),
    #         rot=(0.707, 0.0, 0.0, -0.707),  # -90°
    #     ),
    # )

    # inner_wall_5 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/InnerWall_5",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.0, 0.15, 0.5),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.5, 0.3, 0.1)
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(1.9, 0.4, 0.25),
    #         rot=(1.0, 0.0, 0.0, 0.0),       # 0°
    #     ),
    # )

    # inner_wall_6 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/InnerWall_6",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.0, 0.15, 0.5),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.5, 0.3, 0.1)
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(-0.5, 1.5, 0.25),
    #         rot=(1.0, 0.0, 0.0, 0.0),       # 0°
    #     ),
    # )

    # inner_wall_7 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/InnerWall_7",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.0, 0.15, 0.5),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.5, 0.3, 0.1)
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(-1.2, 0.092, 0.25),
    #         rot=(0.707, 0.0, 0.0, -0.707),  # -90°
    #     ),
    # )

    # # dynamic obstacles 
    obstacle_1: RigidObjectCfg = OBSTACLE_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_1")
    obstacle_2: RigidObjectCfg = OBSTACLE_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_2") 
    obstacle_3: RigidObjectCfg = OBSTACLE_3_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_3")
    obstacle_4: RigidObjectCfg = OBSTACLE_4_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_4")

    # robot and sensors
    robot: ArticulationCfg = TURTLEBOT3_BURGER_CFG
    lidar: RayCasterCfg =  LIDAR_CFG.replace(offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)))
    contact_forces: ContactSensorCfg = CONTACT_SENSOR_CFG

    # goal marker
    goal_marker: RigidObjectCfg = GOAL_MARKER_CFG

##
# Environment configuration
##

# reset event
@configclass
class EventsCfg:

    reset_robot_position = EventTerm(
        func=reset_scene_to_default,
        mode="reset",
    )

    reset_goal_position = EventTerm(
        func=mdp.randomize_goal_positions,
        mode="reset",
    )

    move_obstacle_3 = EventTerm(
        func=update_moving_obstacle_3,
        mode="interval",
        interval_range_s=(0.01, 0.01),
    )

@configclass
class IsaaclabSceneEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: IsaaclabSceneSceneCfg = IsaaclabSceneSceneCfg(num_envs=4, env_spacing=8.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 50.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 15.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation