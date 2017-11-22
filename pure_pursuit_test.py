import numpy as np
import matplotlib.pyplot as plt
import pure_pursuit

robot_pose = np.array([[0], [0]], dtype=np.float64)
lookahead_dist = 0.25
waypoints = [
    np.array([1, 0], dtype=np.float64),
    np.array([1.5, 1], dtype=np.float64),
    np.array([2, 1], dtype=np.float64),
    np.array([2.5, 0], dtype=np.float64),
    np.array([3, 0], dtype=np.float64),
    np.array([4, .5], dtype=np.float64)
]

for i, pt in enumerate(waypoints):
    if i < len(waypoints)-1:
        plt.plot([pt[0], waypoints[i+1][0]], [pt[1], waypoints[i+1][1]], 'k-')

xs = []      # m
ys = []      # m
t = 0        # s
dt = 0.1     # s
speed = 0.5  # m/s

goal_xs = []
goal_ys = []

controller = pure_pursuit.PurePursuitController(lookahead_dist)
controller.set_path(waypoints, robot_pose)

while t < 15:
    robot_loc = pure_pursuit.extract_location(robot_pose)
    goal_pt = controller.get_goal_point(robot_pose)
    vel_dir = (goal_pt - robot_loc) / np.sqrt(np.sum((goal_pt - robot_loc)**2))

    goal_xs.append(goal_pt[0])
    goal_ys.append(goal_pt[1])

    xs.append(robot_loc[0])
    ys.append(robot_loc[1])

    robot_pose += np.expand_dims(vel_dir, 1) * speed * dt
    t += dt

    if controller.end_of_path and robot_loc[0] > 4.5:
        break

xs = np.array(xs)
ys = np.array(ys)

goal_xs = np.array(goal_xs)
goal_ys = np.array(goal_ys)

plt.plot(xs, ys, 'r--')
plt.plot(goal_xs, goal_ys, 'b.')
plt.show()
