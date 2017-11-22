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
    np.array([1, 1], dtype=np.float64),
    np.array([2, 0], dtype=np.float64),
    np.array([4, 2], dtype=np.float64),
    np.array([5, 1], dtype=np.float64),
    np.array([4, 0], dtype=np.float64),
    np.array([2, 2], dtype=np.float64),
    np.array([0.5, 1], dtype=np.float64)
]

for i, pt in enumerate(waypoints):
    if i < len(waypoints)-1:
        plt.plot([pt[0], waypoints[i+1][0]], [pt[1], waypoints[i+1][1]], 'k-')

xs = []      # m
ys = []      # m
t = 0        # s
dt = 0.1     # s
speed = .5   # m/s

goal_xs = []
goal_ys = []

controller = pure_pursuit.PurePursuitController(lookahead_dist)
controller.set_path(waypoints, robot_pose)

while t < 90:
    robot_loc = pure_pursuit.extract_location(robot_pose)
    goal_pt = controller.get_goal_point(robot_pose)
    vel_dir = (goal_pt - robot_loc) / np.sqrt(np.sum((goal_pt - robot_loc)**2))

    goal_xs.append(goal_pt[0])
    goal_ys.append(goal_pt[1])

    xs.append(robot_loc[0])
    ys.append(robot_loc[1])

    robot_pose += np.expand_dims(vel_dir, 1) * speed * dt
    t += dt

    if controller.end_of_path:
        break

xs = np.array(xs)
ys = np.array(ys)

goal_xs = np.array(goal_xs)
goal_ys = np.array(goal_ys)

plt.plot(xs, ys, 'r--')
plt.plot(goal_xs, goal_ys, 'b.')
plt.show()
