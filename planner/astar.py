"""A* path search on an OccupancyGrid (8-connected, no diagonal corner-cutting)."""

import heapq
import math


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1)]


def astar(grid, start, goal):
    """Find a path of grid cells from start to goal.

    Args:
        grid:  an OccupancyGrid.
        start: (i, j) start cell — must be free.
        goal:  (i, j) goal cell — must be free.

    Returns:
        list of (i, j) cells including both ends, or None if no path exists.
    """
    if not (grid.is_free(*start) and grid.is_free(*goal)):
        return None

    def h(c):
        return math.hypot(c[0] - goal[0], c[1] - goal[1])

    open_heap = [(h(start), 0.0, start)]
    came_from = {start: None}
    cost = {start: 0.0}

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            return path[::-1]
        if g > cost[cur]:
            continue
        ci, cj = cur
        for di, dj in _DIRS:
            ni, nj = ci + di, cj + dj
            if not grid.is_free(ni, nj):
                continue
            # a diagonal step must not clip the corner of a blocked cell
            if di != 0 and dj != 0:
                if not (grid.is_free(ci + di, cj) and grid.is_free(ci, cj + dj)):
                    continue
            ng = g + math.hypot(di, dj)
            nxt = (ni, nj)
            if ng < cost.get(nxt, math.inf):
                cost[nxt] = ng
                came_from[nxt] = cur
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
    return None
