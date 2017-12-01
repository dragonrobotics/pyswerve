#!/usr/bin/env python3
import math
import pstats
import cProfile
import numpy as np
import pyximport

pyximport.install(
    setup_args={
        "include_dirs": np.get_include(),
        "extra_compile_args": ['/openmp']
    },
    reload_support=True
)

import occupancy_grid  # noqa: E402

grid_size = (600, 600)
map_scale = 10

n_measurements = 1000
rad_per_measurement = (2 * math.pi) / n_measurements

actual_grid = np.zeros(grid_size, dtype=np.uint8)
current_pos = np.array([200, 200], dtype=np.float32)

rf_mix = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
rf_model = np.array([10/2, 0, 10, 4000], dtype=np.float32)

c0 = np.array([50, 50], dtype=np.int32)
c1 = np.array([550, 50], dtype=np.int32)
c2 = np.array([50, 550], dtype=np.int32)
c3 = np.array([550, 550], dtype=np.int32)

occupancy_grid.draw_binary_line(c0, c1, actual_grid, 255)
occupancy_grid.draw_binary_line(c0, c2, actual_grid, 255)
occupancy_grid.draw_binary_line(c3, c1, actual_grid, 255)
occupancy_grid.draw_binary_line(c3, c2, actual_grid, 255)

pr = cProfile.Profile()
inferred_map = occupancy_grid.OccupancyGrid(map_scale, actual_grid.shape)
sensor_angle = 0
for i in range(n_measurements):
    raycast_dist = occupancy_grid.get_raycast_distance(
        current_pos,
        sensor_angle,
        rf_model[3] / map_scale,
        actual_grid
    )

    pr.enable()
    inferred_map.sensor_update(
        current_pos,
        sensor_angle,
        np.random.normal(raycast_dist * map_scale, rf_model[0]),
        rf_model
    )
    pr.disable()

    sensor_angle += rad_per_measurement

pr.disable()
ps = pstats.Stats(pr)
ps.strip_dirs().sort_stats("time").print_stats()
