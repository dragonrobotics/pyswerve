"""
Implements estimation of robot pose via a Kalman Filter.

Currently supports the use of information from the NavX and from
onboard drive & steer encoders.
"""

import math
import numpy as np

# X/Y position + theta heading, as well as first / second derivatives for each
# robot_state = np.zeros([9,1])


def linearized_encoder_model(width, length, vx, vy, rcw):
    """Calculates an observation matrix for encoder measurements.

    The observation function for encoder measurements maps from
    R^9 -> R^8 (robot pose vectors to speed / angle vectors), so
    resulting matrices should have shape [8,9].
    Of course, many entries in said matrix will be zero.

    As with SwerveDrive.drive, the returned matrices assume an observation
    vector that puts information in the following order:
        [back-right angle,
        back-left angle,
        front-right angle,
        front-left angle,
        back-right speed,
        back-left speed,
        front-right speed,
        front-left speed]
    """
    radius = math.sqrt((length ** 2) + (width ** 2))

    rel_len = length / radius
    rel_wid = width / radius

    a = (vy - rcw) * rel_len
    b = (vy + rcw) * rel_len
    c = (vx - rcw) * rel_wid
    d = (vx + rcw) * rel_wid

    t1 = np.array([a, a, b, b])  # vy-dependent
    t2 = np.array([d, c, d, c])  # vx-dependent

    # The multipliers / sign-flips here correspond to whether rcw was added or
    # subtracted to the values in t1 and t2
    t1_rcw_sign = np.array([-1, -1, 1, 1])
    t2_rcw_sign = np.array([1, -1, 1, -1])

    # Derivatives for the angles
    d_angle_dvx = -(t1 / (t1**2 + t2**2)) * rel_wid  # t2 path
    d_angle_dvy = (t2 / (t1**2 + t2**2)) * rel_len   # t1 path
    d_angle_d_rcw = (d_angle_dvy * t1_rcw_sign) + (d_angle_dvx * t2_rcw_sign)

    # Derivatives for the speeds
    speeds = np.sqrt((t1 ** 2) + (t2 ** 2))
    d_speed_dvx = t2 * rel_wid / speeds
    d_speed_dvy = t1 * rel_len / speeds
    d_speed_d_rcw = (d_speed_dvy * t1_rcw_sign) + (d_speed_dvx * t2_rcw_sign)

    # Join Vx / Vy / RCW derivatives into column vectors
    dvx = np.concatenate((d_angle_dvx, d_speed_dvx))
    dvy = np.concatenate((d_angle_dvy, d_speed_dvy))
    d_rcw = np.concatenate((d_angle_d_rcw, d_speed_d_rcw))

    # Stack columns into a 2D matrix
    dot_derivs = np.stack((dvx, dvy, d_rcw), axis=-1)

    # Fill in other entries with zero
    return np.concatenate(
        (np.zeros([8, 3]), dot_derivs, np.zeros([8, 3])),
        axis=1)
