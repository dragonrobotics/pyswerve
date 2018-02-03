"""Implements common logic for swerve modules.
"""
from ctre.talonsrx import TalonSRX
import numpy as np
import wpilib
import math

ControlMode = TalonSRX.ControlMode
FeedbackDevice = TalonSRX.FeedbackDevice


# Set to true to add safety margin to steer ranges
_apply_range_hack = False
_acceptable_steer_err_degrees = 1  # degrees
_acceptable_steer_err = _acceptable_steer_err_degrees * (512 / 180)


class SwerveModule(object):
    def __init__(self, name, steer_id, drive_id):
        """
        Performs calculations and bookkeeping for a single swerve module.

        Args:
            name (str): A NetworkTables-friendly name for this swerve
                module. Used for saving and loading configuration data.
            steer_id (number): The CAN ID for the Talon SRX controlling this
                module's steering.
            drive_id (number): The CAN ID for the Talon SRX controlling this
                module's driving.

        Attributes:
            steer_talon (:class:`ctre.cantalon.CANTalon`): The Talon SRX used
                to actuate this module's steering.
            drive_talon (:class:`ctre.cantalon.CANTalon`): The Talon SRX used
                to actuate this module's drive.
            steer_target (number): The current target steering position for
                this module, in radians.
            steer_offset (number): The swerve module's steering zero position.
                This value can be determined by manually steering a swerve
                module so that it faces forwards relative to the chassis, and
                by taking the raw encoder position value (ADC reading); this
                value is the steer offset.
            drive_reversed (boolean): Whether or not the drive motor's output
                is currently reversed.
        """
        self.steer_talon = TalonSRX(steer_id)
        self.drive_talon = TalonSRX(drive_id)

        # Configure steering motors to use abs. encoders
        # and closed-loop control
        self.steer_talon.configSelectedFeedbackSensor(FeedbackDevice.Analog, 0, 0)  # noqa: E501
        self.steer_talon.selectProfileSlot(0, 0)
        self.steer_talon.configAllowableClosedloopError(
            0, math.ceil(_acceptable_steer_err), 0
        )

        self.drive_talon.configSelectedFeedbackSensor(FeedbackDevice.QuadEncoder, 0, 0)  # noqa: E501
        self.drive_talon.setQuadraturePosition(0, 0)

        self.name = name
        self.steer_target = 0
        self.steer_target_native = 0
        self.drive_temp_flipped = False
        self.max_speed = 470  # ticks / 100ms
        self.max_observed_speed = 0
        self.raw_drive_speeds = []

        self.load_config_values()

    def load_config_values(self):
        """
        Load saved configuration values for this module via WPILib's
        Preferences interface.

        The key names are derived from the name passed to the
        constructor.
        """
        self.steer_talon.selectProfileSlot(0, 0)

        preferences = wpilib.Preferences.getInstance()

        self.max_speed = preferences.getFloat(
            self.name+'-Max Wheel Speed',
            370
        )

        sensor_phase = preferences.getBoolean(
            self.name+'-Sensor Reverse',
            False
        )

        self.drive_talon.setSensorPhase(sensor_phase)

        self.steer_offset = preferences.getFloat(self.name+'-offset', 0)
        if _apply_range_hack:
            self.steer_min = preferences.getFloat(self.name+'-min', 0)
            self.steer_max = preferences.getFloat(self.name+'-max', 1024)
            actual_steer_range = int(self.steer_max - self.steer_min)

            self.steer_min += int(actual_steer_range * 0.01)
            self.steer_max -= int(actual_steer_range * 0.01)

            self.steer_offset -= self.steer_min
            self.steer_range = int(self.steer_max - self.steer_min)
        else:
            self.steer_min = 0
            self.steer_max = 1024
            self.steer_range = 1024

        self.drive_reversed = preferences.getBoolean(
            self.name+'-reversed', False
        )

        self.steer_reversed = preferences.getBoolean(
            self.name+'-steer-reversed', False
        )

    def save_config_values(self):
        """
        Save configuration values for this module via WPILib's
        Preferences interface.
        """
        preferences = wpilib.Preferences.getInstance()

        preferences.putFloat(self.name+'-offset', self.steer_offset)
        preferences.putBoolean(self.name+'-reversed', self.drive_reversed)

        if _apply_range_hack:
            preferences.putFloat(self.name+'-min', self.steer_min)
            preferences.putFloat(self.name+'-max', self.steer_max)

    def get_steer_angle(self):
        """
        Get the current angular position of the swerve module in
        radians.
        """
        native_units = self.steer_talon.getSelectedSensorPosition(0)
        native_units -= self.steer_offset

        # Position in rotations
        rotation_pos = native_units / self.steer_range

        return rotation_pos * 2 * math.pi

    def set_steer_angle(self, angle_radians):
        """
        Steer the swerve module to the given angle in radians.
        `angle_radians` should be within :math:`[-2\\pi, 2\\pi]`.
        This method attempts to find the shortest path to the given
        steering angle; thus, it may in actuality servo to the
        position opposite the passed angle and reverse the drive
        direction.
        Args:
            angle_radians (number): The angle to steer towards in radians,
                where 0 points in the chassis forward direction.
        """
        n_rotations = math.trunc(
            (self.steer_talon.getSelectedSensorPosition(0) - self.steer_offset)
            / self.steer_range
        )
        current_angle = self.get_steer_angle()
        adjusted_target = angle_radians + (n_rotations * 2 * math.pi)

        # Perform angle unwrapping for target angles.
        # This prevents issues resulting from discontinuities in input angle
        # ranges.
        if abs(adjusted_target - current_angle) - math.pi > 0.0005:
            if adjusted_target > current_angle:
                adjusted_target -= 2 * math.pi
            else:
                adjusted_target += 2 * math.pi

        # Shortest-path servoing
        should_reverse_drive = False
        if abs(adjusted_target - current_angle) - (math.pi / 2) > 0.0005:
            # shortest path is to move to opposite angle and reverse drive dir
            if adjusted_target > current_angle:
                adjusted_target -= math.pi
            else:  # angle_radians < local_angle
                adjusted_target += math.pi
            should_reverse_drive = True

        self.steer_target = adjusted_target

        # Compute and send actual target to motor controller
        native_units = (self.steer_target * 512 / math.pi) + self.steer_offset
        self.steer_talon.set(ControlMode.Position, native_units)

        self.drive_temp_flipped = should_reverse_drive

    def set_drive_speed(self, speed, direct=False):
        """
        Drive the swerve module wheels at a given percentage of
        maximum power or speed.

        Args:
            percent_speed (number): The speed to drive the module at, expressed
                as a percentage of maximum speed. Negative values drive in
                reverse.
        """
        if self.drive_reversed:
            speed *= -1

        if self.drive_temp_flipped:
            speed *= -1

        self.drive_talon.selectProfileSlot(1, 0)
        self.drive_talon.config_kF(0, 1023 / self.max_speed, 0)

        if direct:
            self.drive_talon.set(ControlMode.Velocity, speed)
        else:
            self.drive_talon.set(ControlMode.Velocity, speed * self.max_speed)

    def set_drive_distance(self, ticks):
        if self.drive_reversed:
            ticks *= -1

        if self.drive_temp_flipped:
            ticks *= -1

        self.drive_talon.selectProfileSlot(0, 0)
        self.drive_talon.set(ControlMode.Position, ticks)

    def reset_drive_position(self):
        self.drive_talon.setQuadraturePosition(0, 0)

    def apply_control_values(self, angle_radians, speed, direct=False):
        """
        Set a steering angle and a drive speed simultaneously.

        Args:
            angle_radians (number): The desired angle to steer towards.
            percent_speed (number): The desired percentage speed to drive at.

        See Also:
            :func:`~set_drive_speed` and :func:`~set_steer_angle`
        """
        self.set_steer_angle(angle_radians)
        self.set_drive_speed(speed, direct)

    def update_smart_dashboard(self):
        """
        Push various pieces of info to the Smart Dashboard.

        This method calls to NetworkTables (eventually), thus it may
        be _slow_.

        As of right now, this displays the current raw absolute encoder reading
        from the steer Talon, and the current target steer position.
        """
        wpilib.SmartDashboard.putNumber(
            self.name+' Position',
            (self.steer_talon.getSelectedSensorPosition(0) - self.steer_offset)
            * 180 / 512
        )

        wpilib.SmartDashboard.putNumber(
            self.name+' ADC', self.steer_talon.getAnalogInRaw())

        wpilib.SmartDashboard.putNumber(
            self.name+' Target',
            self.steer_target * 180 / math.pi)

        wpilib.SmartDashboard.putNumber(
            self.name+' Steer Error',
            self.steer_talon.getClosedLoopError(0))

        wpilib.SmartDashboard.putNumber(
            self.name+' Drive Error',
            self.drive_talon.getClosedLoopError(0))

        wpilib.SmartDashboard.putNumber(
            self.name+' Drive Ticks',
            self.drive_talon.getQuadraturePosition())

        self.raw_drive_speeds.append(self.drive_talon.getQuadratureVelocity())
        if len(self.raw_drive_speeds) > 50:
            self.raw_drive_speeds = self.raw_drive_speeds[-50:]

        self.cur_drive_spd = np.mean(self.raw_drive_speeds)

        if abs(self.cur_drive_spd) > abs(self.max_observed_speed):
            self.max_observed_speed = self.cur_drive_spd

        wpilib.SmartDashboard.putNumber(
            self.name+' Drive Velocity',
            self.cur_drive_spd
        )

        wpilib.SmartDashboard.putNumber(
            self.name+' Drive Velocity (Max)',
            self.max_observed_speed
        )

        if wpilib.RobotBase.isReal():
            wpilib.SmartDashboard.putNumber(
                self.name+' Drive Percent Output',
                self.drive_talon.getMotorOutputPercent()
            )
