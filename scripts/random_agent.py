# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
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
    """Goal-following sanity test agent."""

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)

    import numpy as np
    from gymnasium import spaces

    env.action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=env.action_space.shape,
        dtype=np.float32,
    )

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    obs, _ = env.reset()

    print("POLICY OBS SHAPE:", obs["policy"].shape)
    print("GOAL POS W:", env.unwrapped.goal_pos_w)

    step_count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            goal_dist = obs["policy"][:, 40]
            goal_angle = obs["policy"][:, 41]

            # Simple manual controller:
            # if goal is mostly ahead, move forward
            # otherwise rotate toward the goal
            linear = torch.where(
                torch.abs(goal_angle) < 0.25,
                torch.ones_like(goal_angle),
                torch.zeros_like(goal_angle),
            )

            angular = torch.clamp(goal_angle * 2.0, -1.0, 1.0)

            actions = torch.stack([linear, angular], dim=-1)

            obs, reward, terminated, truncated, info = env.step(actions)

            step_count += 1

            if step_count % 20 == 0:
                goal_dist_obs = obs["policy"][:, 40]
                goal_angle_obs = obs["policy"][:, 41]
                goal_dist_real = goal_dist_obs * 7.07106781187

                print("\n--- STEP", step_count, "---")
                print("GOAL DIST REAL:", goal_dist_real)
                print("GOAL ANGLE OBS:", goal_angle_obs)
                print("ACTION:", actions)
                print("REWARD:", reward)
                print("TERMINATED:", terminated)
                print("TRUNCATED:", truncated)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
