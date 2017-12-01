"""
Simulates and displays an occupancy grid.
"""

import math
import sys
import numpy as np
import pygame
import pyximport

pyximport.install(
    setup_args={
        "include_dirs": np.get_include()
    },
    reload_support=True
)

import occupancy_grid  # noqa: E402

grid_size = (600, 600)
map_scale = 10
draw_radius = 2
erase_radius = 3

actual_grid = np.zeros(grid_size, dtype=np.uint8)
current_pos = np.array([200, 200], dtype=np.float32)
current_vel = np.zeros(2, dtype=np.float32)

move_speed = 100

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

pygame.init()
screen = pygame.display.set_mode(((grid_size[0]*2)+5, grid_size[1]))

grid_surf = pygame.Surface(grid_size)
grid_surf = grid_surf.convert()
grid_surf.fill((250,)*3)

for x in range(int(grid_size[0] / 10)):
    pygame.draw.line(grid_surf, (223,)*3, (x*10, 0), (x*10, grid_size[1]), 1)

for y in range(int(grid_size[1] / 10)):
    pygame.draw.line(grid_surf, (223,)*3, (0, y*10), (grid_size[0], y*10), 1)

clock = pygame.time.Clock()

actual_surf = pygame.Surface(grid_size)
actual_surf = actual_surf.convert()
actual_surf.set_colorkey(0xFFFFFF)

map_surf = pygame.Surface(grid_size)
map_surf = map_surf.convert()
map_surf.set_colorkey(0xFFFFFF)


sensor_sweeping = False
sensor_angle = 0

sensor_speed = math.pi / 6
measurements_per_frame = 5

inferred_map = occupancy_grid.OccupancyGrid(map_scale, actual_grid.shape)

frame_no = 0
mouse_down = False
draw_state = 0
while True:
    dt = clock.tick(60)  # ms

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit(0)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                sys.exit(0)
            elif event.key == pygame.K_SPACE:
                sensor_sweeping = not sensor_sweeping
            elif event.key == pygame.K_LEFT:
                current_vel[0] = -move_speed
            elif event.key == pygame.K_RIGHT:
                current_vel[0] = move_speed
            elif event.key == pygame.K_UP:
                current_vel[1] = move_speed
            elif event.key == pygame.K_DOWN:
                current_vel[1] = -move_speed
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                current_vel[0] = 0
            elif event.key == pygame.K_RIGHT:
                current_vel[0] = 0
            elif event.key == pygame.K_UP:
                current_vel[1] = 0
            elif event.key == pygame.K_DOWN:
                current_vel[1] = 0
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_down = True
                mx, my = pygame.mouse.get_pos()
                draw_state = 255 - actual_grid[mx][grid_size[1] - my]
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_down = False
        elif event.type == pygame.MOUSEMOTION:
            if mouse_down:
                mx, my = pygame.mouse.get_pos()
                my = grid_size[1] - my  # flip coordinates vertically
                r = draw_radius if draw_state == 255 else erase_radius
                actual_grid[mx-r:mx+r, my-r:my+r] = draw_state

    current_pos += current_vel * (dt / 1000)

    if sensor_sweeping:
        old_angle = sensor_angle
        sensor_angle += sensor_speed * dt / 1000

    # get sensor ray endpoint (not taking obstacles into account)
    max_endpt = (
        np.array([np.cos(sensor_angle), np.sin(sensor_angle)])
        * rf_model[3] / map_scale
    ) + current_pos

    if sensor_sweeping:
        # simulate sensor update
        n_measurements = min(dt, 25)
        for i in range(n_measurements):
            # interpolate between the two angles
            t = i / (n_measurements-1)
            intermed_angle = (t * sensor_angle) + ((1-t) * old_angle)

            # raycast and update
            intermed_rc = occupancy_grid.get_raycast_distance(
                current_pos,
                intermed_angle,
                rf_model[3] / map_scale,
                actual_grid
            )

            inferred_map.sensor_update(
                current_pos,
                intermed_angle,
                np.random.normal(intermed_rc * map_scale, rf_model[0]),
                rf_model
            )

    # blit background to screen first
    screen.blit(grid_surf, (0, 0))
    screen.blit(grid_surf, (grid_size[0]+5, 0))

    # directly blit grid to actual_surf (occupied = black, free = transparent)
    pygame.surfarray.blit_array(
        actual_surf,
        np.where(actual_grid, 0, 0xFFFFFF)
    )

    # get occupied cells in inferred map and blit to map_surf
    pygame.surfarray.blit_array(
        map_surf,
        np.where(inferred_map.get_occupied_cells(), 0, 0xFFFFFF)
    )

    # draw rays
    pygame.draw.line(actual_surf, (0, 0, 255), current_pos, max_endpt)
    pygame.draw.line(map_surf, (0, 0, 255), current_pos, max_endpt)

    actual_surf = pygame.transform.flip(actual_surf, False, True)
    map_surf = pygame.transform.flip(map_surf, False, True)

    screen.blit(actual_surf, (0, 0))
    screen.blit(map_surf, (grid_size[0]+5, 0))

    pygame.display.flip()

    if frame_no % 15 == 0:
        print("fps: "+str(clock.get_fps()))

    frame_no += 1
