"""Robot registry — switch between TurtleBot3 Burger and ict_bot via ROBOT env var.

Usage (training or eval):
    ROBOT=turtlebot python scripts/...   # default
    ROBOT=ictbot    python scripts/...

Every robot-specific value (USD path, base link, joint names, wheel geometry,
forward-axis convention) is read from a single dict so swapping robots is a
one-flag change for both training and evaluation scripts.

FORWARD_AXIS env var still wins if set explicitly — useful when temporarily
testing one robot's policy on the other's URDF frame.
"""
import os

ROBOT_CFGS = {
    "turtlebot": {
        "usd_path":          "/home/user/Documents/isaaclab_scene/assets/robots/turtlebot3_burger.usd",
        "base_link":         "base_link",
        "left_wheel_joint":  "wheel_left_joint",
        "right_wheel_joint": "wheel_right_joint",
        "wheel_radius":      0.033,
        "wheel_separation":  0.16,
        "forward_axis":      "X",
        "lidar_offset_z":    0.20,
    },
    "ictbot": {
        "usd_path":          "/home/user/Documents/isaaclab_scene/assets/ict_bot_robot/ict_bot.usd",
        "base_link":         "link_base",
        "left_wheel_joint":  "left_wheel_joint",
        "right_wheel_joint": "right_wheel_joint",
        "wheel_radius":      0.05,
        "wheel_separation":  0.16,
        "forward_axis":      "-Y",
        "lidar_offset_z":    0.20,
    },
}

ROBOT = os.environ.get("ROBOT", "turtlebot").lower()
if ROBOT not in ROBOT_CFGS:
    raise ValueError(
        f"Unknown ROBOT='{ROBOT}'. Set ROBOT to one of: {list(ROBOT_CFGS)}"
    )

CFG = ROBOT_CFGS[ROBOT]

# Forward axis: explicit FORWARD_AXIS env var wins, otherwise use the robot's
# natural axis from the registry.
FORWARD_AXIS = os.environ.get("FORWARD_AXIS", CFG["forward_axis"]).upper()
