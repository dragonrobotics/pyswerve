"""Simulates and displays an ideal robot's path under pure pursuit control.

This is mainly meant as a debugging and testing aid.
This script takes no arguments.
"""

import traceback
import warnings
import sys
import numpy as np
import matplotlib.pyplot as plt
import pure_pursuit


# Print stack trace on warning
def warn_with_traceback(
        message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))  # noqa: E501
    log.write('\n')


warnings.showwarning = warn_with_traceback

robot_pose = np.array([[0], [0]], dtype=np.float64)
lookahead_dist = 0.25
waypoints = [
    np.array([1, 1], dtype=np.float32),
    np.array([2, 0], dtype=np.float32),
    np.array([4, 2], dtype=np.float32),
    np.array([5, 1.75], dtype=np.float32),
    np.array([4, 1.5], dtype=np.float32),
    np.array([2, 2], dtype=np.float32),
    np.array([0.5, 1], dtype=np.float32)
]

for i, pt in enumerate(waypoints):
    if i < len(waypoints)-1:
        plt.plot([pt[0], waypoints[i+1][0]], [pt[1], waypoints[i+1][1]], 'k-')

t = 0        # s
dt = 0.1     # s
speed = .5   # m/s

controller = pure_pursuit.PurePursuitController(lookahead_dist)
controller.set_path(waypoints, robot_pose)

robot_positions = []
goal_pts_by_case = [[], [], [], [], []]


def plot_points(pts, color='k', linestyle='', marker=None):
    xs = []
    ys = []
    for p in pts:
        xs.append(p[0])
        ys.append(p[1])

    xs = np.array(xs)
    ys = np.array(ys)

    plt.plot(xs, ys, color=color, linestyle=linestyle, marker=marker)


ts = 0
while t < 90:
    robot_loc = np.array([robot_pose[0][0], robot_pose[1][0]])
    print("t={:.2f} loc={}".format(t, str(robot_loc)))

    goal_pt, goal_tuple = controller.get_goal_point(
        robot_pose,
        debug_info=True)
    vel_dir = (goal_pt - robot_loc) / np.sqrt(np.sum((goal_pt - robot_loc)**2))

    robot_positions.append(robot_loc)
    goal_pts_by_case[goal_tuple[2]].append(goal_pt)

    robot_pose += np.expand_dims(vel_dir, 1) * speed * dt

    if ts % 10 == 0:
        plt.annotate('t={:.2f}'.format(t), robot_loc)

    t += dt
    ts += 1

    if (
        controller.end_of_path
        and np.sqrt(np.sum((waypoints[-1] - robot_loc) ** 2)) < 0.05
    ):
        break

plot_points(robot_positions, 'r', '--', 'o')

#  0 = 1st Node outside Virtual Circle
#  1 = 2 Intersection Points
#  2 = No Intersection Points or Goal Point Dead Zone
#  3 = Common Case (1 Intersection Point)
#  4 = Segment Extended to Infinity
plot_points(goal_pts_by_case[0], 'c', marker='.')
plot_points(goal_pts_by_case[1], '#EE82EE', marker='.')
plot_points(goal_pts_by_case[2], 'y', marker='.')
plot_points(goal_pts_by_case[3], 'b', marker='.')
plot_points(goal_pts_by_case[4], '#9400D3', marker='.')

plt.gca().set_aspect('equal', adjustable='box')

plt.show()
