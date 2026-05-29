"""Static occupancy grid of the arena.

Pure Python / numpy — no Isaac Lab or training dependency. The caller passes
wall rectangles (centre, half-extents, yaw); the grid rasterises them, grown
by an inflation margin so a planned point-path keeps real robot clearance.
"""

import numpy as np


class OccupancyGrid:
    def __init__(self, x_min, x_max, y_min, y_max, cell_size):
        self.x_min = x_min
        self.y_min = y_min
        self.cell = cell_size
        self.nx = int(round((x_max - x_min) / cell_size))
        self.ny = int(round((y_max - y_min) / cell_size))
        self.blocked = np.zeros((self.nx, self.ny), dtype=bool)
        # cached cell-centre coordinates, for vectorised rasterisation
        self._cx = x_min + (np.arange(self.nx) + 0.5) * cell_size
        self._cy = y_min + (np.arange(self.ny) + 0.5) * cell_size

    def world_to_cell(self, x, y):
        return (int((x - self.x_min) / self.cell),
                int((y - self.y_min) / self.cell))

    def cell_to_world(self, i, j):
        return (self.x_min + (i + 0.5) * self.cell,
                self.y_min + (j + 0.5) * self.cell)

    def in_bounds(self, i, j):
        return 0 <= i < self.nx and 0 <= j < self.ny

    def is_free(self, i, j):
        return self.in_bounds(i, j) and not self.blocked[i, j]

    def add_rect(self, cx, cy, half_x, half_y, yaw, inflate=0.0):
        """Block every cell whose centre falls inside a rotated rectangle,
        grown by `inflate` on every side."""
        gx, gy = np.meshgrid(self._cx, self._cy, indexing="ij")
        dx, dy = gx - cx, gy - cy
        c, s = np.cos(-yaw), np.sin(-yaw)
        lx = dx * c - dy * s          # cell centre in the rectangle's frame
        ly = dx * s + dy * c
        self.blocked |= (np.abs(lx) <= half_x + inflate) & (np.abs(ly) <= half_y + inflate)

    def add_border(self, inflate=0.0):
        """Block a ring of cells around the arena edge (the outer walls)."""
        x_max = self.x_min + self.nx * self.cell
        y_max = self.y_min + self.ny * self.cell
        gx, gy = np.meshgrid(self._cx, self._cy, indexing="ij")
        self.blocked |= (
            (gx <= self.x_min + inflate) | (gx >= x_max - inflate) |
            (gy <= self.y_min + inflate) | (gy >= y_max - inflate)
        )

    def add_disk(self, cx, cy, radius):
        """Block every cell whose centre falls inside a circle (a round
        obstacle, e.g. a stopped blocker when replanning)."""
        gx, gy = np.meshgrid(self._cx, self._cy, indexing="ij")
        self.blocked |= ((gx - cx) ** 2 + (gy - cy) ** 2) <= radius ** 2

    def copy(self):
        """Return a copy with an independent blocked array — used to replan
        around a temporary obstacle without mutating the static grid."""
        g = OccupancyGrid(self.x_min, self.x_min + self.nx * self.cell,
                          self.y_min, self.y_min + self.ny * self.cell, self.cell)
        g.blocked = self.blocked.copy()
        return g

    def nearest_free(self, i, j, max_radius=12):
        """Nearest free cell to (i, j), searched in expanding rings.
        Returns (i, j) or None. Handles a start/goal that fell inside
        an inflated wall."""
        if self.is_free(i, j):
            return (i, j)
        for r in range(1, max_radius + 1):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    if self.is_free(i + di, j + dj):
                        return (i + di, j + dj)
        return None
