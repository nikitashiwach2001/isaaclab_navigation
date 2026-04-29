import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.16

MAX_LINEAR_SPEED = 0.22
MAX_ANGULAR_SPEED = 2.0

ENABLE_BACKWARD = False


class DifferentialDriveAction(ActionTerm):
    """Gazebo-style normalized linear/angular action for TurtleBot3.

    Input action:
        action[:, 0] = linear normalized  [-1, 1]
        action[:, 1] = angular normalized [-1, 1]

    Output:
        left/right wheel velocity target in rad/s.
    """

    cfg: "DifferentialDriveActionCfg"

    def __init__(self, cfg: "DifferentialDriveActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._asset: Articulation = env.scene[cfg.asset_name]

        self._joint_ids, self._joint_names = self._asset.find_joints(
            [cfg.left_wheel_joint_name, cfg.right_wheel_joint_name]
        )

        if len(self._joint_ids) != 2:
            raise RuntimeError(
                f"Expected 2 wheel joints but found {len(self._joint_ids)}: {self._joint_names}"
            )

        self._raw_actions = torch.zeros((env.num_envs, 2), device=env.device)
        self._processed_actions = torch.zeros((env.num_envs, 2), device=env.device)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)

        linear_norm = self._raw_actions[:, 0]
        angular_norm = self._raw_actions[:, 1]

        if ENABLE_BACKWARD:
            linear = linear_norm * MAX_LINEAR_SPEED
        else:
            linear = (linear_norm + 1.0) * 0.5 * MAX_LINEAR_SPEED

        angular = angular_norm * MAX_ANGULAR_SPEED

        left_wheel_vel = (
            linear - angular * WHEEL_SEPARATION * 0.5
        ) / WHEEL_RADIUS

        right_wheel_vel = (
            linear + angular * WHEEL_SEPARATION * 0.5
        ) / WHEEL_RADIUS

        self._processed_actions[:, 0] = left_wheel_vel
        self._processed_actions[:, 1] = right_wheel_vel

    def apply_actions(self):
        self._asset.set_joint_velocity_target(
            self._processed_actions,
            joint_ids=self._joint_ids,
        )


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    class_type: type = DifferentialDriveAction

    asset_name: str = "robot"

    left_wheel_joint_name: str = "wheel_left_joint"
    right_wheel_joint_name: str = "wheel_right_joint"


@configclass
class ActionsCfg:
    base_velocity: DifferentialDriveActionCfg = DifferentialDriveActionCfg(
        asset_name="robot",
        left_wheel_joint_name="wheel_left_joint",
        right_wheel_joint_name="wheel_right_joint",
    )