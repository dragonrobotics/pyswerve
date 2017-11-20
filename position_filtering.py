"""
Implements estimation of robot pose via a Kalman Filter.

It is planned to support the use of information from the NavX and from
onboard drive & steer encoders.

Robot poses are stored as column vectors (shape [9,1]), containing
robot XY-positions and headings, as well as velocities / accelerations
for each:
pose = [
    x_pos,
    y_pos,
    hdg,
    x_vel,
    y_vel,
    rot_vel,
    x_acc,
    y_acc,
    rot_acc
]

"""

import math
import numpy as np
import wpilib
import kalman_filter


def state_transition(dt):
    """Calculates a state transition matrix given a time delta.
    """
    matx = np.identity(9)

    # velocity terms for new position
    matx[0][3] = dt
    matx[1][4] = dt
    matx[2][5] = dt

    # acceleration terms for new position
    matx[0][6] = (dt**2) / 2
    matx[1][7] = (dt**2) / 2
    matx[2][8] = (dt**2) / 2

    # acceleration terms for new velocity
    matx[3][6] = dt
    matx[4][7] = dt
    matx[5][8] = dt

    return matx


def linearized_swerve_encoder_model(
        chassis_width, chassis_length, robot_pose):
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

    # Get velocity components from robot pose
    vx = np.squeeze(robot_pose[3])
    vy = np.squeeze(robot_pose[4])
    rcw = np.squeeze(robot_pose[5])

    radius = math.sqrt((chassis_length ** 2) + (chassis_width ** 2))

    rel_len = chassis_length / radius
    rel_wid = chassis_width / radius

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
    retn = np.concatenate(
        (np.zeros([8, 3]), dot_derivs, np.zeros([8, 3])),
        axis=1)

    if np.isnan(np.sum(retn)):
        # Don't break the filter
        return np.zeros([8,9])

    return retn


def swerve_encoder_model(chassis_width, chassis_length, robot_pose):
    radius = math.sqrt((chassis_length ** 2) + (chassis_width ** 2))

    rel_len = chassis_length / radius
    rel_wid = chassis_width / radius

    vx = np.squeeze(robot_pose[3])
    vy = np.squeeze(robot_pose[4])
    rcw = np.squeeze(robot_pose[5])

    a = (vy - rcw) * rel_len
    b = (vy + rcw) * rel_len
    c = (vx - rcw) * rel_wid
    d = (vx + rcw) * rel_wid

    t1 = np.array([a, a, b, b])  # vy-dependent
    t2 = np.array([d, c, d, c])  # vx-dependent

    speeds = np.sqrt((t1 ** 2) + (t2 ** 2))
    angles = np.arctan2(t1, t2)

    return np.expand_dims(np.concatenate((angles, speeds)), axis=1)


class PositionFilter(object):
    """Stores state for position filtering.

    All quantities use units of meters and radians.
    Note that robot poses are relative to the world reference frame!
    """
    pose = np.zeros([9, 1])
    covar = np.zeros([9, 9])
    last_predict_time = 0

    def __init__(self, chassis_width, chassis_length):
        self.width = chassis_width
        self.length = chassis_length
        self.last_predict_time = wpilib.Timer.getFPGATimestamp()

    def get_position(self):
        return self.pose[0], self.pose[1]

    def get_heading(self):
        return self.pose[2]

    def predict(self, movement_covariance, dt):
        if dt is None:
            dt = wpilib.Timer.getFPGATimestamp() - self.last_predict_time
            self.last_predict_time = wpilib.Timer.getFPGATimestamp()

        transition_matrix = state_transition(dt)

        self.pose, self.covar = kalman_filter.predict(
            self.pose, self.covar,
            transition_matrix, movement_covariance)

    def swerve_encoder_update(
            self,
            angle_variance, speed_variance,
            module_angles, module_speeds, measurement=None):
        measurement_covar = np.zeros([8, 8])
        for i in range(0, 4):
            measurement_covar[i][i] = angle_variance
        for i in range(4, 8):
            measurement_covar[i][i] = speed_variance

        if measurement is None:
            measurement_vector = np.concatenate((module_angles, module_speeds))
        else:
            measurement_vector = measurement


        assert measurement_vector.shape == (8,1)

        linear_model = linearized_swerve_encoder_model(
            self.width, self.length, self.pose)

        self.pose, self.covar = kalman_filter.ekf_update(
            self.pose, self.covar,
            measurement_vector,
            lambda p: swerve_encoder_model(self.width, self.length, p),
            linear_model, measurement_covar)

    def ahrs_gyro_update(self, ahrs, measurement=None):
        # Model picks out gyro angle and rate (shape = [2, 9])
        model = np.array([
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0]
        ])

        if measurement is None:
            measurement = np.array([
                [ahrs.getYaw()],
                [ahrs.getRate()]
            ])


        assert measurement.shape == (2,1)

        # According to KauaiLabs, the NavX exhibits yaw drift of about
        # 1 degree per minute.
        # thus
        # yaw standard dev = (PI/180) radians * (seconds running / 60) / 2
        # As a side note, the actual ICs used onboard the NavX-MXP probably
        # use an EKF as well.
        #
        # The MPU-9250's gyroscope has a specified total RMS noise
        # of 0.1 degree/second-RMS.
        # It also has a rate noise spectral density
        # of 0.01 degree per second per sqrt(Hz).
        running_time = wpilib.Timer.getFPGATimestamp()  # in seconds
        yaw_sigma = running_time * (math.pi / 180) / 120
        yaw_rate_sigma = 0.1 * (math.pi / 180)
        covariance = np.array([
            [yaw_sigma ** 2, 0],
            [0, yaw_rate_sigma ** 2]
        ])

        self.pose, self.covar = kalman_filter.update(
            self.pose, self.covar,
            measurement, model, covariance)

    def ahrs_accelerometer_update(self, ahrs, measurement=None):
        # Model simply picks the two linear acceleration components in order
        # shape = [2, 9]
        model = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0]
        ])

        # NOTE: double check to see if these are actually field-centric
        if measurement is None:
            measurement = np.array([
                [ahrs.getWorldLinearAccelX()],
                [ahrs.getWorldLinearAccelY()],
            ])

        assert measurement.shape == (2,1)

        # The AHRS accelerometers have a range of +/- 2g with a resolution
        # of 16 bits.
        # The chip in use is the InvenSense MPU-9250; specifications at
        # https://www.invensense.com/wp-content/uploads/2015/02/PS-MPU-9250A-01-v1.1.pdf  # noqa: E501
        # Total RMS noise = 8 milligravities-RMS (0.008g-rms)

        accel_variance = ((0.008 * 9.81) ** 2)

        measurement *= 9.81  # AHRS returns accelerations in units of g

        self.pose, self.covar = kalman_filter.update(
            self.pose, self.covar,
            measurement, model, np.identity(2) * accel_variance)
