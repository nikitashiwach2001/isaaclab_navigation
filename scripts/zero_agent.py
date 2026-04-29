# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_scene.tasks  # noqa: F401


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    print("CREATING ENVIRONMENT")
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("AFTER ENV CREATE")

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    print("BEFORE RESET")   
    env.reset()
    print("AFTER RESET")   
    # FIX WHEEL AXIS
    from pxr import UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "wheel_left_joint" in path or "wheel_right_joint" in path:
            joint = UsdPhysics.RevoluteJoint(prim)
            joint.GetAxisAttr().Set("Y")
            print("[FIXED AXIS]:", path)

    step_count = 0

    for _ in range(1000):
        with torch.inference_mode():
            print("LOOP RUNNING", step_count)

            actions = torch.tensor([[5.0, 5.0]], device=env.unwrapped.device).repeat(env.num_envs, 1)
            env.step(actions)

            step_count += 1
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
