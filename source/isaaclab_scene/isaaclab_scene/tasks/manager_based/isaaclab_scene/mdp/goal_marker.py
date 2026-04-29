import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


GOAL_MARKER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Goal",
    spawn=sim_utils.SphereCfg(
        radius=0.08,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0)
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.08),
    ),
)