"""Python implementation of the Pure Pursuit controller.
"""

import numpy as np
import math

machine_eps = np.finfo(np.float64).eps

def extract_location(robot_pose):
    return np.array([robot_pose[0][0], robot_pose[1][0]])


def angle_between(v1, v2):
    magn_prod = np.sum(v1**2) * np.sum(v2**2)  #  |v1|**2 * |v2|**2
    if magn_prod <= machine_eps:
        # somewhat arbitrary and hackish, but to keep things rom breaking
        # just return zero when taking angles involving zero vectors.
        # Could also raise a RuntimeError, but robots don't quit.
        return 0

    cos_theta = np.sum(v1*v2) / np.sqrt(magn_prod)

    if np.abs(cos_theta) > 1:
        # shouldn't be mathematically possible, but happens sometimes
        # (due to rounding / floating point shenanigans?)
        if cos_theta > 0:
            return 0
        else:
            return math.pi

    return np.arccos(cos_theta)


def segment_is_relevant(robot_loc, node_list, i):
    segment = node_list[i+1] - node_list[i]
    next_segment = node_list[i+2] - node_list[i+1]

    q1 = robot_loc - node_list[i]
    q2 = node_list[i+1] - robot_loc

    a1 = angle_between(segment, q1)
    a2 = angle_between(segment, q2)
    theta = angle_between(segment, next_segment)
    beta = theta / 2

    if np.abs(a1) <= (math.pi / 2):
        # bit of a hack here with case I
        # sometimes the angle gets NEAR zero but never actually goes negative
        # so we add some tolerance here to allow for actual inside-turn nav.
        # (hence the <= 1 and >= -1 checks instead of <= 0 and >= 0)
        if (a2 <= 1 and beta > -1) or (a2 >= -1 and beta < 1):
            return np.abs(a2) <= np.abs(beta)  # Case I
        elif np.abs(a2) <= (math.pi / 2):
            return True  # Case II
        elif np.abs(a2) - np.abs(theta) - (math.pi / 2) < 2:
            return True  # Case III
        else:
            return False  # Case II
    else:
        return i == 0  # Case IV


# i == currently relevant segment / node index
def projected_location(robot_loc, node_list, i):
    segment = node_list[i+1] - node_list[i]

    q1 = robot_loc - node_list[i]
    q2 = node_list[i+1] - robot_loc

    if i < len(node_list) - 2:
        a2 = angle_between(segment, q2)
        next_segment = node_list[i+2] - node_list[i+1]
        theta = angle_between(segment, next_segment)
        beta = theta / 2

        if (
            not ((a2 <= 1 and beta > -1) or (a2 >= -1 and beta < 1))
            and np.abs(a2) > (math.pi / 2)
            and np.abs(a2) - np.abs(theta) - (math.pi / 2) < 2
        ):
            return node_list[i+1]  # dead zone; select corner point as position

    t = np.sum(q1 * segment) / np.sum(segment * segment)
    return node_list[i] + (t * segment)


def cross_track_error(robot_loc, node_list, i):
    segment = node_list[i+1] - node_list[i]

    if i < len(node_list) - 2:
        q2 = node_list[i+1] - robot_loc
        a2 = angle_between(segment, q2)
        next_segment = node_list[i+2] - node_list[i+1]
        theta = angle_between(segment, next_segment)
        beta = theta / 2

        if (
            not ((a2 <= 1 and beta > -1) or (a2 >= -1 and beta < 1))
            and np.abs(a2) > (math.pi / 2)
            and np.abs(a2) - np.abs(theta) - (math.pi / 2) < 0
        ):
            # dead zone case
            return np.sqrt(np.sum((robot_loc - node_list[i+1])**2))

    q1 = robot_loc - node_list[i]
    a1 = angle_between(segment, q1)

    return np.abs(-np.sqrt(np.sum(q1 ** 2)) * np.sin(a1))


# Runs whenever a new path is sent to the controller.
# Returns an index to initialize all other searches from.
def presearch(robot_loc, node_list):
    # find relevant segment with least cross-track error
    relevant_segments = [
        (i, cross_track_error(robot_loc, node_list, i))
        for i, node in enumerate(node_list)
        if (
            i < len(node_list) - 2
            and segment_is_relevant(robot_loc, node_list, i)
        )]

    smallest_err = relevant_segments[0]
    for segment in relevant_segments:
        if segment[1] < smallest_err[1]:
            smallest_err = segment

    return smallest_err[0]


def find_goal_point(robot_loc, lookahead_dist, node_list, search_start_idx):
    # Find first relevant segment:
    relevant_segment = None
    for i in range(search_start_idx, len(node_list)-2):
        if segment_is_relevant(robot_loc, node_list, i):
            relevant_segment = i
            break

    if relevant_segment is None:
        relevant_segment = len(node_list) - 2

    projected_pos = projected_location(robot_loc, node_list, relevant_segment)
    x_track_err = cross_track_error(robot_loc, node_list, relevant_segment)

    if relevant_segment == 0:
        segment = node_list[1] - node_list[0]
        q1 = robot_loc - node_list[0]
        q2 = node_list[1] - robot_loc
        a1 = angle_between(segment, q1)

        if (
            np.sqrt(np.sum(q1 ** 2)) >= lookahead_dist
            and np.sqrt(np.sum(q2 ** 2)) >= lookahead_dist
            and np.abs(a1) > (math.pi / 2)
        ):
            return (node_list[0], 0)  # 1st node outside virtual circle

    for i in range(relevant_segment, len(node_list)-1):
        segment = node_list[i+1] - node_list[i]

        q1 = robot_loc - node_list[i]
        q2 = node_list[i+1] - robot_loc

        magn_q1 = np.sqrt(np.sum(q1 ** 2))
        magn_q2 = np.sqrt(np.sum(q2 ** 2))
        segment_len = np.sqrt(np.sum(segment ** 2))

        if magn_q1 <= lookahead_dist and magn_q2 >= lookahead_dist:
            # common case
            cos_y = (
                (np.sum(q2 ** 2) - np.sum(q1 ** 2) - np.sum(segment ** 2))
                / (-2 * segment_len * magn_q1)
            )

            p = (magn_q1 * cos_y) + np.sqrt(
                (np.sum(q1 ** 2) * ((cos_y ** 2) - 1)) + (lookahead_dist ** 2))

            goal_tuple = (
                node_list[i] + (p * segment / segment_len),
                relevant_segment
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
                    relevant_segment
                )

                return goal_tuple
            else:
                # no intersection points case and/or dead zone case
                # find closest point on path
                current_xerr = x_track_err
                selected_node = i
                for i2 in range(search_start_idx, len(node_list)-2):
                    if segment_is_relevant(robot_loc, node_list, i2):
                        xerr = cross_track_error(
                            robot_loc, node_list, i2)
                        if xerr < current_xerr:
                            selected_node = i2
                            current_xerr = xerr

                p2 = projected_pos = projected_location(
                    robot_loc, node_list, selected_node)

                goal_tuple = (
                    p2,
                    selected_node
                )
                return goal_tuple
        # else go to next segment

    if relevant_segment == len(node_list) - 2:
        # Extend last segment to infinity if necessary
        segment = node_list[len(node_list) - 1] - node_list[len(node_list) - 2]
        segment_len = np.sqrt(np.sum(segment ** 2))

        q2 = node_list[-1] - robot_loc
        magn_q2 = np.sqrt(np.sum(q2 ** 2))
        eps = cross_track_error(robot_loc, node_list, len(node_list) - 2)

        if magn_q2 < lookahead_dist:
            p = np.sqrt((lookahead_dist ** 2) - (eps ** 2))
            goal_tuple = (
                projected_pos + (p * segment / segment_len),
                len(node_list)-2
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
        return self.end_of_path

    def get_goal_point(self, robot_pose):
        robot_loc = extract_location(robot_pose)
        goal_point, self.search_start_index = find_goal_point(
            robot_loc,
            self.lookahead_dist,
            self.node_list,
            self.search_start_index
        )

        if self.search_start_index == len(self.node_list) - 2:
            segment = self.node_list[-1] - self.node_list[-2]
            q2 = self.node_list[-1] - robot_loc

            magn_q2 = np.sqrt(np.sum(q2 ** 2))
            a2 = angle_between(segment, q2)

            if a2 >= (math.pi / 2):
                self.end_of_path = True

        return goal_point

    def set_path(self, new_path, robot_pose):
        """Sets a new path for the controller.

        new_path should be a list of rank-1 ndarrays (shape (2,)).
        robot_pose is the current robot pose as a 9-element vector.
        """
        robot_loc = extract_location(robot_pose)
        self.node_list = new_path
        self.end_of_path = False
        self.search_start_index = presearch(robot_loc, new_path)

        if self.search_start_index is None:
            # Shouldn't happen... but just in case.
            # Extend the last path segment to infinity.
            self.search_start_index = len(new_path)-2
