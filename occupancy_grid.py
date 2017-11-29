"""
Implements occupancy grid mapping.
"""

import numpy as np
import rangefinder


def cells_on_line(p0, d, g0=None, g1=None):
    """
    Get all cells covered by a line segment stretching from `p0` to
    `p0+d`.

    The returned line can optionally be constrained to the rectangle defined
    by `g0` (as the lower-left corner) and `g1` (as the upper-right corner).
    Both corners are included in this rectangle.
    """

    n = np.abs(d)
    s = np.sign(d)

    i = [0, 0]
    cur = np.array([int(p0[0]), int(p0[1])], dtype=np.int32)
    while i[0] <= n[0] and i[1] <= n[1]:
        o = [(0.5 + i[0]) / n[0], (0.5 + i[1]) / n[1]]
        if abs(o[0] - o[1]) < 0.0000001:
            cur += s
            i[0] += 1
            i[1] += 1
        elif o[0] < o[1]:
            cur[0] += s[0]
            i[0] += 1
        else:
            cur[1] += s[1]
            i[1] += 1

        if g0 is not None and g1 is not None:
            if (
                cur[0] < g0[0]
                or cur[0] > g1[0]
                or cur[1] < g0[0]
                or cur[1] > g1[1]
            ):
                break

        yield np.copy(cur)


def draw_binary_line(p0, p1, grid, value):
    for cell in cells_on_line(p0, p1-p0, np.zeros(2), grid.shape):
        grid[cell[0]][cell[1]] = value


def get_raycast_distance(p0, theta, max_dist, grid):
    d = np.array([np.cos(theta), np.sin(theta)]) * max_dist
    for cell in cells_on_line(p0, d, np.zeros(2), grid.shape):
        if grid[cell[0]][cell[1]]:
            return np.sqrt(np.sum((p0 - cell)**2))

    return max_dist


class OccupancyGrid(object):
    """docstring for OccupancyGrid."""
    def __init__(self, grid_scale, grid_size):
        """
        grid_scale = scale factor (actual units per grid cell length)
        """
        self.grid = np.zeros(grid_size, dtype=np.float32)
        self.scale = grid_scale

    def sensor_update(self, p0, theta, dist, model, mix):
        """
        Update the occupancy grid with a beam-based model.
        """
        d = np.array([np.cos(theta), np.sin(theta)]) * dist / self.scale

        for cell in cells_on_line(p0, d, np.zeros(2), self.grid.shape):
            grid_dist = np.sqrt(np.sum((p0 - cell)**2))
            z_exp = grid_dist * self.scale

            likelihood = rangefinder.rangefinder_model(
                dist,
                z_exp,
                model,
                mix
            )

            modifier = np.log(likelihood / (1-likelihood))

            self.grid[cell[0]][cell[1]] += modifier
