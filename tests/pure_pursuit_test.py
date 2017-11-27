import traceback
import warnings
import sys
import numpy as np
import pure_pursuit

# Print stack trace on warning
def warn_with_traceback(
        message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
    log.write('\n')

warnings.showwarning = warn_with_traceback

# Testing constants.
sim_robot_velocity = .05
lookahead_dist = 0.25
max_test_timesteps = 1200
initial_robot_pose = np.array([[0], [0]], dtype=np.float64)
max_distance_at_end_of_path = 0.25

def simulate_path(waypoints):
    robot_pose = np.copy(initial_robot_pose)
    controller = pure_pursuit.PurePursuitController(lookahead_dist)
    controller.set_path(waypoints, robot_pose)

    robot_loc = None
    for t in range(max_test_timesteps):
        robot_loc = np.array([robot_pose[0][0], robot_pose[1][0]])
        goal_pt = controller.get_goal_point(robot_pose)
        vel_dir = (
            (goal_pt - robot_loc)
            / np.sqrt(np.sum((goal_pt - robot_loc)**2)))

        robot_pose += np.expand_dims(vel_dir, 1) * sim_robot_velocity

        if controller.end_of_path:
            break

    assert controller.end_of_path, "Robot not at end of path after simulation!"

    # Check distance from robot at end of sim to last waypoint
    d = np.sqrt(np.sum((waypoints[-1] - robot_loc) ** 2))
    assert d <= max_distance_at_end_of_path, "Robot too far from last waypoint!"  # noqa: E501


def test_sharp_angles():
    simulate_path([
        np.array([1, 0], dtype=np.float32),
        np.array([1, 1], dtype=np.float32),
        np.array([2, 1], dtype=np.float32),
        np.array([2, 0], dtype=np.float32),
        np.array([3, 0], dtype=np.float32),
        np.array([3, 1], dtype=np.float32)
    ])


def test_coincident_robot_start():
    simulate_path([
        np.array([0, 0], dtype=np.float32),
        np.array([1, 1], dtype=np.float32),
        np.array([2, 1], dtype=np.float32),
        np.array([2, 0], dtype=np.float32),
        np.array([3, 0], dtype=np.float32),
        np.array([3, 1], dtype=np.float32)
    ])


def test_regular_path():
    simulate_path([
        np.array([1, 1], dtype=np.float32),
        np.array([2, 0], dtype=np.float32),
        np.array([4, 2], dtype=np.float32),
        np.array([5, 1], dtype=np.float32),
        np.array([4, 0], dtype=np.float32),
        np.array([2, 2], dtype=np.float32),
        np.array([0.5, 1], dtype=np.float32)
    ])

def test_tight_turn():
    simulate_path([
        np.array([1, 1], dtype=np.float32),
        np.array([2, 0], dtype=np.float32),
        np.array([4, 2], dtype=np.float32),
        np.array([5, 1.75], dtype=np.float32),
        np.array([4, 1.5], dtype=np.float32),
        np.array([2, 2], dtype=np.float32),
        np.array([0.5, 1], dtype=np.float32)
    ])
