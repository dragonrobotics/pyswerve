"""A Python implementation of the Pure Pursuit controller.

This module implements a basic pure pursuit controller, allowing for
smooth, stable navigation of the robot using waypoints, as opposed to,
say, specific timings.

A reliable source of robot pose / position information is required to
effectively use pure pursuit control. In addition, logic for converting
computed goal points to steering / motor commands is beyond the scope
of this module.
"""

import math
import warnings
import numpy as np

_machine_eps = np.finfo(np.float32).eps
_angle_cmp_tol = math.radians(5)  # angle comparison tolerance


def _extract_location(robot_pose):
    """Convert a robot-pose 9-vector to a robot location 2-vector."""
    if robot_pose.ndim == 2:
        return np.array([robot_pose[0][0], robot_pose[1][0]])
    elif robot_pose.ndim == 1:
        return np.array([robot_pose[0], robot_pose[1]])


# NOTE: using the dot product does _not_ work here.
# using the dot product here leads to nothing but sadness and pain.
def _angle_between(v1, v2):
    """Take the signed angle between two vectors.

    This acts like a heading comparison; for example,
    `_angle_between([1, 0], [1, -1]) = -45 degrees`.
    """
    a1 = np.arctan2(v1[1], v1[0])
    a2 = np.arctan2(v2[1], v2[0])

    return a2 - a1


def _segment_is_relevant(robot_loc, node_list, i):
    """
    Test whether the segment starting at node i in node_list is
    relevant, given the current robot position.
    """
    segment = node_list[i+1] - node_list[i]

    q1 = robot_loc - node_list[i]
    q2 = node_list[i+1] - robot_loc

    a1 = _angle_between(segment, q1)
    a2 = _angle_between(segment, q2)

    theta = _angle_between(
        node_list[i+1] - node_list[i+2],
        node_list[i+1] - node_list[i]
    )

    beta = theta / 2

    if np.abs(a1) - (math.pi / 2) <= _angle_cmp_tol:
        if (
            (a2 <= _angle_cmp_tol and beta > -_angle_cmp_tol)
            or (a2 >= -_angle_cmp_tol and beta < _angle_cmp_tol)
        ):
            return np.abs(a2) <= np.abs(beta)  # Case I
        elif np.abs(a2) <= (math.pi / 2):
            return True  # Case II
        elif np.abs(a2) - np.abs(theta) - (math.pi / 2) < _angle_cmp_tol:
            return True  # Case III
        else:
            return False  # Case II
    else:
        return i == 0  # Case IV


# i == currently relevant segment / node index
def _projected_location(robot_loc, node_list, i):
    """
    Find the closest point from the robot on the path, given the
    current relevant segment.
    """
    segment = node_list[i+1] - node_list[i]

    q1 = robot_loc - node_list[i]
    q2 = node_list[i+1] - robot_loc

    if i < len(node_list) - 2:
        a2 = _angle_between(segment, q2)
        next_segment = node_list[i+2] - node_list[i+1]
        theta = _angle_between(segment, next_segment)
        beta = theta / 2

        if (
            not (
                (a2 <= _angle_cmp_tol and beta > -_angle_cmp_tol)
                or (a2 >= -_angle_cmp_tol and beta < _angle_cmp_tol)
            )
            and np.abs(a2) > (math.pi / 2)
            and np.abs(a2) - np.abs(theta) - (math.pi / 2) < _angle_cmp_tol
        ):
            return node_list[i+1]  # dead zone; select corner point as position

    t = np.sum(q1 * segment) / np.sum(segment * segment)
    return node_list[i] + (t * segment)


def _cross_track_error(robot_loc, node_list, i):
    """
    Find the closest distance between the robot and the path, given
    the current relevant segment.
    """
    segment = node_list[i+1] - node_list[i]

    if i < len(node_list) - 2:
        q2 = node_list[i+1] - robot_loc
        a2 = _angle_between(segment, q2)
        next_segment = node_list[i+2] - node_list[i+1]
        theta = _angle_between(segment, next_segment)
        beta = theta / 2

        if (
            not (
                (a2 <= _angle_cmp_tol and beta > -_angle_cmp_tol)
                or (a2 >= -_angle_cmp_tol and beta < _angle_cmp_tol)
            )
            and np.abs(a2) > (math.pi / 2)
            and np.abs(a2) - np.abs(theta) - (math.pi / 2) < _angle_cmp_tol
        ):
            # dead zone case
            return np.sqrt(np.sum((robot_loc - node_list[i+1])**2))

    q1 = robot_loc - node_list[i]
    a1 = _angle_between(segment, q1)

    return np.abs(-np.sqrt(np.sum(q1 ** 2)) * np.sin(a1))


# Runs whenever a new path is sent to the controller.
# Returns an index to initialize all other searches from.
def _presearch(robot_loc, node_list):
    """
    Search for the relevant segment with least cross-track error; this
    is used to initialize the controller on receipt of a new path.

    This returns an index into node_list; subsequent calls to
    `_find_goal_point` should use this the intial `search_start_idx`.
    """
    # find relevant segment with least cross-track error
    relevant_segments = [
        (i, _cross_track_error(robot_loc, node_list, i))
        for i, node in enumerate(node_list)
        if (
            i < len(node_list) - 2
            and _segment_is_relevant(robot_loc, node_list, i)
        )]

    if len(relevant_segments) == 0:
        # this happens when the beginning and end of a path are the same point
        # or when you have 90 degree turns along paths
        # play it safe, just return node 0 for the index
        warnings.warn(
            "No relevant path segments found in presearch",
            RuntimeWarning)
        return 0

    smallest_err = relevant_segments[0]
    for segment in relevant_segments:
        if segment[1] < smallest_err[1]:
            smallest_err = segment

    return smallest_err[0]


def _find_goal_point(robot_loc, lookahead_dist, node_list, search_start_idx):
    """
    Find a goal point for pure pursuit control.

    Returns a tuple; the first element will be the goal point
    (as a 2-vector), and the second element will be the latest relevant
    segment, which should be passed as search_start_idx in subsequent
    calls. The remaining elements, if present, are debug data and have
    no defined semantics.
    """

    # goal_tuple[2]:
    #  0 = 1st Node outside Virtual Circle
    #  1 = 2 Intersection Points
    #  2 = No Intersection Points or Goal Point Dead Zone
    #  3 = Common Case (1 Intersection Point)
    #  4 = Segment Extended to Infinity
    # Find first relevant segment:
    relevant_segment = None
    for i in range(search_start_idx, len(node_list)-2):
        if _segment_is_relevant(robot_loc, node_list, i):
            relevant_segment = i
            break

    if relevant_segment is None:
        relevant_segment = len(node_list) - 2

    projected_pos = _projected_location(robot_loc, node_list, relevant_segment)
    x_track_err = _cross_track_error(robot_loc, node_list, relevant_segment)

    if relevant_segment == 0:
        segment = node_list[1] - node_list[0]
        q1 = robot_loc - node_list[0]
        q2 = node_list[1] - robot_loc
        a1 = _angle_between(segment, q1)

        if (
            np.sqrt(np.sum(q1 ** 2)) >= lookahead_dist
            and np.sqrt(np.sum(q2 ** 2)) >= lookahead_dist
            and np.abs(a1) > (math.pi / 2)
        ):
            return (node_list[0], 0, 0)  # 1st node outside virtual circle

    for i in range(relevant_segment, len(node_list)-1):
        segment = node_list[i+1] - node_list[i]

        q1 = robot_loc - node_list[i]
        q2 = node_list[i+1] - robot_loc

        magn_q1 = np.sqrt(np.sum(q1 ** 2))
        magn_q2 = np.sqrt(np.sum(q2 ** 2))
        segment_len = np.sqrt(np.sum(segment ** 2))

        if magn_q1 <= lookahead_dist and magn_q2 >= lookahead_dist:
            # common case
            if magn_q1 <= _machine_eps:
                magn_q1 = 0.00001  # very hack-ish but keeps things from breaking  # noqa: E501

            cos_y = (
                (np.sum(q2 ** 2) - np.sum(q1 ** 2) - np.sum(segment ** 2))
                / (-2 * segment_len * magn_q1)
            )

            p = (magn_q1 * cos_y) + np.sqrt(
                (np.sum(q1 ** 2) * ((cos_y ** 2) - 1)) + (lookahead_dist ** 2))

            goal_tuple = (
                node_list[i] + (p * segment / segment_len),
                relevant_segment,
                3
            )

            return goal_tuple
        elif (
            i == relevant_segment
            and magn_q1 >= lookahead_dist
            and magn_q2 >= lookahead_dist
        ):
            if x_track_err < lookahead_dist:
                # 2 intersection points
                p = np.sqrt((lookahead_dist ** 2) - (x_track_err ** 2))
                goal_tuple = (
                    projected_pos + segment*(p / segment_len),
                    relevant_segment,
                    1
                )

                return goal_tuple
            else:
                # no intersection points case and/or dead zone case
                # find closest point on path
                current_xerr = x_track_err
                selected_node = i
                for i2 in range(search_start_idx, len(node_list)-2):
                    if _segment_is_relevant(robot_loc, node_list, i2):
                        xerr = _cross_track_error(
                            robot_loc, node_list, i2)
                        if xerr < current_xerr:
                            selected_node = i2
                            current_xerr = xerr

                p2 = projected_pos = _projected_location(
                    robot_loc, node_list, selected_node)

                goal_tuple = (
                    p2,
                    selected_node,
                    2
                )

                return goal_tuple
        # else go to next segment

    if relevant_segment == len(node_list) - 2:
        # Extend last segment to infinity if necessary
        segment = node_list[-1] - node_list[-2]
        segment_len = np.sqrt(np.sum(segment ** 2))

        q2 = node_list[-1] - robot_loc
        magn_q2 = np.sqrt(np.sum(q2 ** 2))
        eps = _cross_track_error(robot_loc, node_list, len(node_list) - 2)

        if magn_q2 < lookahead_dist:
            p = np.sqrt((lookahead_dist ** 2) - (eps ** 2))
            goal_tuple = (
                projected_pos + (p * segment / segment_len),
                len(node_list)-2,
                4
            )
            return goal_tuple

    raise RuntimeError("Could not find goal point?")


class PurePursuitController(object):
    """Keeps state for a Pure Pursuit Controller."""
    node_list = []
    search_start_index = 0
    end_of_path = False

    def __init__(self, lookahead_dist):
        self.lookahead_dist = lookahead_dist

    def reached_end_of_path(self):
        """Test whether the robot is approaching the end of the path,
        based on the last call to `get_goal_point`.

        Note that this does not neccesarily mean the robot actually
        _has_ reached the end of the path.
        """
        return self.end_of_path

    def get_goal_point(self, robot_pose, debug_info=False):
        """
        Get the next goal point, given a previously set path and the
        current robot position.

        This returns a point (2-vector) by default; however, if
        debug_info is True, this will also return the full
        tuple of values from `_find_goal_point`.
        """
        robot_loc = _extract_location(robot_pose)
        goal_tuple = _find_goal_point(
            robot_loc,
            self.lookahead_dist,
            self.node_list,
            self.search_start_index
        )

        goal_point = goal_tuple[0]
        self.search_start_index = goal_tuple[1]

        if self.search_start_index == len(self.node_list) - 2:
            q2 = self.node_list[-1] - robot_loc
            magn_q2 = np.sqrt(np.sum(q2 ** 2))

            if magn_q2 < self.lookahead_dist:
                self.end_of_path = True

        if debug_info:
            return goal_point, goal_tuple
        else:
            return goal_point

    def set_path(self, new_path, robot_pose):
        """Sets a new path for the controller.

        new_path should be a list of 2-vectors (shape (2,)).
        robot_pose is the current robot pose as a 9-element vector.
        """
        robot_loc = _extract_location(robot_pose)
        self.node_list = new_path
        self.end_of_path = False
        self.search_start_index = _presearch(robot_loc, new_path)

        if self.search_start_index is None:
            # Shouldn't happen... but just in case.
            # Extend the last path segment to infinity.
            self.search_start_index = len(new_path)-2