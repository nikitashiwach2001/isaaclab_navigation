import os
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .robot_registry import CFG as _ROBOT_CFG

# Wheel geometry comes from the robot registry — set ROBOT=turtlebot or
# ROBOT=ictbot to switch. TurtleBot: 0.033 m; ict_bot: 0.05 m.
WHEEL_RADIUS = _ROBOT_CFG["wheel_radius"]
WHEEL_SEPARATION = _ROBOT_CFG["wheel_separation"]

# Deployment-time tuning gains. Default 1.0 matches training. Set these via
# env vars to make the robot react faster or take sharper turns without
# retraining the policy:
#   LINEAR_GAIN < 1   → slower forward → smaller turn radius for same angular
#   ANGULAR_GAIN > 1  → faster turning → sharper response to dodge commands
# Combined example: LINEAR_GAIN=0.8 ANGULAR_GAIN=1.4
MAX_LINEAR_SPEED = 0.22 * float(os.environ.get("LINEAR_GAIN", "1.0"))
MAX_ANGULAR_SPEED = 2.0 * float(os.environ.get("ANGULAR_GAIN", "1.0"))

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

        # Forward-only mapping (default): linear_norm ∈ [-1, 1] is rescaled to [0, MAX].
        # Per-env backward override: if env.allow_backward[e] is True, env e is
        # allowed to issue genuine backward motion (linear_norm < 0 → negative speed).
        # Used by play_td3_planner.py's nose-pin recovery, where the script writes
        # a backward action AND sets the flag for the same env. Other envs stay
        # forward-only so the policy's training-time action distribution is
        # preserved during evaluation.
        forward_linear = (linear_norm + 1.0) * 0.5 * MAX_LINEAR_SPEED
        if ENABLE_BACKWARD:
            linear = linear_norm * MAX_LINEAR_SPEED
        elif hasattr(self._env, "allow_backward"):
            full_linear = linear_norm * MAX_LINEAR_SPEED
            linear = torch.where(self._env.allow_backward, full_linear, forward_linear)
        else:
            linear = forward_linear

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

    left_wheel_joint_name: str = _ROBOT_CFG["left_wheel_joint"]
    right_wheel_joint_name: str = _ROBOT_CFG["right_wheel_joint"]


@configclass
class ActionsCfg:
    base_velocity: DifferentialDriveActionCfg = DifferentialDriveActionCfg(
        asset_name="robot",
        left_wheel_joint_name=_ROBOT_CFG["left_wheel_joint"],
        right_wheel_joint_name=_ROBOT_CFG["right_wheel_joint"],
    )