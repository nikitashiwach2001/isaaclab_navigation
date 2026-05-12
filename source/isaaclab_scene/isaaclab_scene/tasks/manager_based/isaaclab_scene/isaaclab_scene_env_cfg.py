import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import reset_scene_to_default

from . import mdp
from .mdp import (
    TerminationsCfg,
    ObservationsCfg,
    RewardsCfg,
    ActionsCfg,
    TURTLEBOT3_BURGER_CFG,
    LIDAR_CFG,
    CONTACT_SENSOR_CFG,
    GOAL_MARKER_CFG,
    OBSTACLE_1_CFG,
    OBSTACLE_2_CFG,
    OBSTACLE_3_CFG,
    OBSTACLE_4_CFG,
)

from .mdp.stage_4.robot import LIDAR_CFG_4


@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    """Shared arena: ground, lights, four outer walls, robot, sensors, goal marker.
    Stage-specific configs inherit this and add their own obstacles."""

    num_envs: int = 4
    env_spacing: float = 8.0

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=2000.0),
    )

    wall_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_1",
        spawn=sim_utils.CuboidCfg(
            size=(5.0, 0.15, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.25, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-2.425, 0.0, 0.25),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )

    wall_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_2",
        spawn=sim_utils.CuboidCfg(
            size=(5.0, 0.15, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.25, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 2.425, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    wall_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_3",
        spawn=sim_utils.CuboidCfg(
            size=(5.0, 0.15, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.25, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.425, 0.0, 0.25),
            rot=(0.707, 0.0, 0.0, -0.707),
        ),
    )

    wall_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_4",
        spawn=sim_utils.CuboidCfg(
            size=(5.0, 0.15, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.25, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -2.425, 0.25),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # obstacle_1: RigidObjectCfg = OBSTACLE_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_1")
    # obstacle_2: RigidObjectCfg = OBSTACLE_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_2")
    # obstacle_3: RigidObjectCfg = OBSTACLE_3_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_3")
    # obstacle_4: RigidObjectCfg = OBSTACLE_4_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_4")
    
    robot: ArticulationCfg = TURTLEBOT3_BURGER_CFG
    # lidar: RayCasterCfg = LIDAR_CFG.replace(offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)))
    lidar: RayCasterCfg = LIDAR_CFG_4.replace(offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)))
    contact_forces: ContactSensorCfg = CONTACT_SENSOR_CFG
    goal_marker: RigidObjectCfg = GOAL_MARKER_CFG


@configclass
class BaseEventsCfg:
    """Reset events shared by all stages."""

    reset_robot_position = EventTerm(
        func=reset_scene_to_default,
        mode="reset",
    )

    reset_goal_position = EventTerm(
        func=mdp.randomize_goal_positions,
        mode="reset",
    )


@configclass
class BaseEnvCfg(ManagerBasedRLEnvCfg):
    """Shared RL settings: observations, actions, rewards, terminations.
    Stage configs inherit this and plug in their own scene and events."""

    # # for stage1 and stage 2
    # scene: BaseSceneCfg = BaseSceneCfg()
    # events: BaseEventsCfg = BaseEventsCfg()

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    enable_lidar_temporal_diff: bool = False

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 50.0
        self.viewer.eye = (8.0, 0.0, 15.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation
