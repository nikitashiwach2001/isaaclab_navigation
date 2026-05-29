"""Turns the arena walls + a (start, goal) pair into a dense waypoint path."""

import math

from .occupancy_grid import OccupancyGrid
from .astar import astar


def build_arena_grid(walls, cell_size=0.1, inflate=0.30, arena_half=2.35):
    """Build a static occupancy grid for an arena bounded by outer walls.

    Args:
        walls:      list of (cx, cy, half_x, half_y, yaw) in local arena coords.
                    half_x/half_y are the wall's half-extents BEFORE inflation,
                    so this handles walls of any size — Stage 5's short 1.0 m
                    inner walls, Stage 7's long corridor walls, etc.
        cell_size:  grid resolution in metres.
        inflate:    clearance grown around every wall (>= robot clearance).
        arena_half: half-width of the arena (outer walls).
    """
    grid = OccupancyGrid(-arena_half, arena_half, -arena_half, arena_half, cell_size)
    grid.add_border(inflate)
    for cx, cy, half_x, half_y, yaw in walls:
        grid.add_rect(cx, cy, half_x, half_y, yaw, inflate)
    return grid


def build_stage5_grid(inner_walls, cell_size=0.1, inflate=0.30, arena_half=2.35):
    """Legacy back-compat builder — assumes every wall is the Stage 5 inner-wall
    size (1.0 x 0.15). Prefer build_arena_grid() which takes per-wall half-extents."""
    walls = [(cx, cy, 0.5, 0.075, yaw) for cx, cy, yaw in inner_walls]
    return build_arena_grid(walls, cell_size=cell_size, inflate=inflate, arena_half=arena_half)


class PathPlanner:
    """Plans a dense waypoint path on a fixed occupancy grid."""

    def __init__(self, grid):
        self.grid = grid

    def plan(self, start_xy, goal_xy, spacing=0.4, extra_blocks=None):
        """Return a list of (x, y) waypoints from start to goal — dense
        (~spacing apart), last point = the exact goal — or None if no
        path exists.

        extra_blocks: optional list of (cx, cy, radius) circles blocked only
        for this plan (e.g. a stopped obstacle). The static grid is untouched.
        """
        grid = self.grid
        if extra_blocks:
            grid = self.grid.copy()
            for cx, cy, r in extra_blocks:
                grid.add_disk(cx, cy, r)
        start = grid.nearest_free(*grid.world_to_cell(*start_xy))
        goal = grid.nearest_free(*grid.world_to_cell(*goal_xy))
        if start is None or goal is None:
            return None
        cells = astar(grid, start, goal)
        if cells is None:
            return None
        pts = [grid.cell_to_world(i, j) for i, j in cells]
        return _thin(pts, spacing, goal_xy)


def _thin(pts, spacing, goal_xy):
    """Keep grid points ~spacing apart; drop the start cell; end on the exact
    goal. Always returns at least one waypoint (the goal)."""
    kept = []
    last = pts[0]
    for p in pts[1:-1]:
        if math.hypot(p[0] - last[0], p[1] - last[1]) >= spacing:
            kept.append(p)
            last = p
    kept.append(tuple(goal_xy))
    return kept
