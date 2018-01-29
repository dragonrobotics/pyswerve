"""
Implements a full swerve drive.
"""
import wpilib
import math
import numpy as np
from swerve_module import SwerveModule


class SwerveDrive(object):
    def __init__(self, length, width, config_tuples):
        """
        Controls a set of SwerveModules as a coherent drivetrain.

        Args:
            length (number): The length of the chassis.
            width (number): The width of the chassis.
            config_tuples: a list of 3-element tuples of the form
                ``(name, steer_id, drive_id)`` where:

                *   `name` is a human-friendly module name (used for loading
                    and saving config values)
                *   `steer_id` and `drive_id` are the CAN IDs for each
                    module's steer and drive motor controllers (Talons).

                See also :class:`swerve_module.SwerveModule`.

        Note:
            The order of the tuples within ``config_tuples`` *does* matter.
            To be specific, the configurations are assumed to be within the
            following order:

            1. back-right swerve module
            2. back-left swerve module
            3. front-right swerve module
            4. front-left swerve module

            The choice of units for the dimensions of the chassis does not
            matter, as long as they are the *same* units.

        Attributes:
            modules: A list containing each :class:`swerve_module.SwerveModule`
                in this drive.
            radius (number): The length of the chassis diagonal.
            sd_update_timer (:class:`wpilib.timer.Timer`): A timer, used to
                limit the rate at which SmartDashboard is updated.
        """

        self.modules = []
        for config in config_tuples:
            self.modules.append(SwerveModule(*config))

        self.length = length
        self.width = width
        self.radius = math.sqrt((length ** 2) + (width ** 2))

        self.sd_update_timer = wpilib.Timer()
        self.sd_update_timer.start()

    def drive(self, forward, strafe, rotate_cw, max_wheel_speed=370):
        """
        Compute and apply module angles and speeds to achieve a given
        linear / angular velocity.

        All control inputs (arguments) are assumed to be in a robot
        oriented reference frame. In addition, all values are
        (for now) assumed to fall within the range [0, 1].

        Args:
            forward (number): The desired, relative forward motion of the
                robot.
            strafe (number): The desired, relative sideways motion of the
                robot.
            rotate_cw (number): The desired rotational speed of the robot.
        """
        a = (strafe - rotate_cw) * (self.length / self.radius)
        b = (strafe + rotate_cw) * (self.length / self.radius)
        c = (forward - rotate_cw) * (self.width / self.radius)
        d = (forward + rotate_cw) * (self.width / self.radius)

        t1 = np.array([a, a, b, b])
        t2 = np.array([d, c, d, c])

        speeds = np.sqrt((t1 ** 2) + (t2 ** 2))
        angles = np.arctan2(t1, t2)

        if np.amax(speeds) > 1:
            speeds /= np.amax(speeds)

        # back-right, back-left, front-right, front-left?
        for module, angle, speed in zip(self.modules, angles, speeds):
            module.apply_control_values(angle, speed)

    def turn_to_angle(self, navx, target_angle):
        hdg = navx.getFusedHeading() * (math.pi / 180)

        prefs = wpilib.Preferences.getInstance()
        if prefs.getBoolean('Reverse Heading Direction', False):
            hdg *= -1

        if math.degrees(abs(hdg - target_angle)) < 5:
            for module in self.modules:
                module.set_drive_speed(0, True)
            return

        a = -1 * (self.length / self.radius)
        b = (self.length / self.radius)
        c = -1 * (self.width / self.radius)
        d = (self.width / self.radius)

        t1 = np.array([a, a, b, b])
        t2 = np.array([d, c, d, c])

        angles = np.arctan2(t1, t2)

        drv_spd = self.max_speed / 4

        if hdg < target_angle:
            drv_spd *= -1

        for module, angle in zip(self.modules, angles):
            module.set_steer_angle(angle)
            module.set_drive_speed(drv_spd, True)

    def set_all_module_angles(self, angle_rad):
        for module in self.modules:
            module.set_steer_angle(angle_rad)

    def set_all_module_speeds(self, speed, direct=False):
        for module in self.modules:
            module.set_drive_speed(speed, direct)

    def set_all_module_dist_ticks(self, dist_ticks):
        for module in self.modules:
            module.set_drive_distance(dist_ticks)

    def drive_angle_distance(self, angle_rad, dist_ticks):
        for module in self.modules:
            module.set_steer_angle(angle_rad)
            module.set_drive_distance(dist_ticks)

    def get_module_distances(self):
        return [
            abs(module.drive_talon.getQuadraturePosition())
            for module in self.modules
        ]

    def reset_drive_position(self):
        for module in self.modules:
            module.reset_drive_position()

    def save_config_values(self):
        """
        Save configuration values for all modules within this swerve drive.
        """
        for module in self.modules:
            module.save_config_values()

    def load_config_values(self):
        """
        Load configuration values for all modules within this swerve drive.
        """
        for module in self.modules:
            module.load_config_values()

    def update_smart_dashboard(self):
        """
        Update Smart Dashboard for all modules within this swerve drive.
        """
        for module in self.modules:
            module.update_smart_dashboard()

        overall_max_speed = np.amin(np.abs(
            [module.max_observed_speed for module in self.modules]
        ))

        wpilib.SmartDashboard.putNumber(
            'Overall Max Observed Speed',
            overall_max_speed
        )
