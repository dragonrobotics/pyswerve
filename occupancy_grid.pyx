# cython: profile=True
"""
Implements occupancy grid mapping.
"""
import math
cimport cython
import numpy as np
cimport numpy as np
import rangefinder

np_dtype = np.float32
ctypedef np.float32_t np_dtype_t
ctypedef float num_dtype_t

def cells_on_line(p0, d, g0=None, g1=None):
    """
    Get all cells covered by a line segment stretching from `p0` to
    `p0+d`.

    The returned line can optionally be constrained to the rectangle defined
    by `g0` (as the lower-left corner) and `g1` (as the upper-right corner).
    Both corners are included in this rectangle.
    """

    n = np.abs(d)
    s = np.sign(d, dtype=np.int32, casting='unsafe')

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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray list_cells_on_line(
    num_dtype_t[:] p0,
    num_dtype_t[:] delta,
    int[:] boundary
):
    #n = np.abs(delta)
    #s = np.sign(delta, dtype=np.int32, casting='unsafe')

    #cdef num_dtype_t[:] nv = n
    cdef num_dtype_t nv[2]
    nv = [abs(delta[0]), abs(delta[1])]

    #cdef int[:] sv = s
    cdef int sv[2]
    sv = [np.sign(<int>delta[0]), np.sign(<int>delta[1])]

    cdef int i[2]
    cdef int cur[2]
    cdef num_dtype_t o[2]

    i = [0, 0]
    cur = [int(p0[0]), int(p0[1])]

    # max_n is upper bound on cell list length
    cdef int max_n = math.ceil(nv[0]) + math.ceil(nv[1]) + 1
    cdef np.ndarray out_array = np.zeros([max_n, 2], dtype=np.int32)
    cdef int[:, :] cell_list = out_array
    #cell_list = [(cur[0], cur[1])]

    cdef int t = 0
    while i[0] <= nv[0] and i[1] <= nv[1]:
        if (
            cur[0] <= 0 or cur[0] >= boundary[0]
            or cur[1] <= 0 or cur[1] >= boundary[1]
        ):
            break

        cell_list[t][0] = cur[0]
        cell_list[t][1] = cur[1]
        t += 1

        if nv[0] == 0:
            cur[1] += sv[1]
            i[1] += 1
            continue

        if nv[1] == 0:
            cur[0] += sv[0]
            i[0] += 1
            continue

        o = [(0.5 + i[0]) / nv[0], (0.5 + i[1]) / nv[1]]

        #if abs(o[0] - o[1]) < 0.0000001:
        #    cur[0] += s[0]
        #    cur[1] += s[1]
        #    i[0] += 1
        #    i[1] += 1
        #elif o[0] < o[1]:
        if o[0] < o[1]:
            cur[0] += sv[0]
            i[0] += 1
        else:
            cur[1] += sv[1]
            i[1] += 1

        #cell_list.append((cur[0], cur[1]))

    #out_array = np.asarray(cell_list, dtype=np.int32)
    out_array.resize((t, 2), refcheck=False)

    return out_array


def cells_in_ray(p0, theta, max_dist, grid):
    d = np.array([np.cos(theta), np.sin(theta)]) * max_dist
    return cells_on_line(p0, d, np.zeros(2), grid.shape)


def draw_binary_line(p0, p1, grid, value):
    for cell in cells_on_line(p0, p1-p0, np.zeros(2), grid.shape):
        grid[cell[0]][cell[1]] = value


cpdef num_dtype_t get_raycast_distance(
    num_dtype_t[:] p0,
    num_dtype_t theta,
    num_dtype_t max_dist,
    np.uint8_t[:, :] grid
) except? 0 :
    d = np.array(
        [np.cos(theta), np.sin(theta)], dtype=np_dtype
    ) * max_dist

    cdef int grid_size[2]
    grid_size = [grid.shape[0], grid.shape[1]]

    cdef np.ndarray cell_matx = list_cells_on_line(
        p0,
        d,
        grid_size
    )

    cdef int[:, :] cell_view = cell_matx
    cdef int i
    cdef int n_cells = cell_view.shape[0]

    for i in range(n_cells):
        if grid[cell_view[i][0]][cell_view[i][1]] > 0:
            return np.sqrt(np.sum((p0 - cell_matx[i])**2))

    return max_dist


cdef class OccupancyGrid(object):
    """docstring for OccupancyGrid."""

    cdef num_dtype_t[:, :] grid
    cdef num_dtype_t scale
    cdef int grid_size[2]

    def __init__(self, num_dtype_t grid_scale, grid_size):
        """
        grid_scale = scale factor (actual units per grid cell length)
        """
        self.grid = np.zeros(grid_size, dtype=np_dtype)
        self.scale = grid_scale
        self.grid_size = grid_size  # [grid_size[0], grid_size[1]]

    cpdef np.ndarray get_occupied_cells(self):
        return np.greater(self.grid, 0)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void sensor_update(
        self,
        num_dtype_t[:] p0,
        num_dtype_t theta,
        num_dtype_t dist,
        num_dtype_t[:] model
    ) except *:
        """
        Update the occupancy grid with a beam-based model.
        """

        d = np.array(
            [np.cos(theta), np.sin(theta)], dtype=np_dtype
        ) * dist / self.scale

        cell_matx = list_cells_on_line(
            p0,
            d,
            self.grid_size
        )

        z_exps = np.sqrt(np.sum((p0 - cell_matx)**2, axis=1)) * self.scale

        modifiers = rangefinder.log_lambda(
            dist,
            z_exps,
            model
        ).astype(np_dtype)

        cdef num_dtype_t[:] mod_view = modifiers
        cdef int[:, :] cell_view = cell_matx
        cdef int i
        cdef int n_cells = cell_view.shape[0]

        for i in range(n_cells):
            self.grid[cell_view[i][0]][cell_view[i][1]] += mod_view[i]
