"""Implements common logic for swerve modules.
"""
from ctre.cantalon import CANTalon
import wpilib
import math


class SwerveModule(object):
    """
    Command interface for a swerve module.

    Attributes:
        steer_talon (:class:`ctre.cantalon.CANTalon`): The Talon SRX used to
            actuate this module's steering.
        drive_talon (:class:`ctre.cantalon.CANTalon`): The Talon SRX used to
            actuate this module's drive.
        name (string): A NetworkTables-friendly name for this swerve module.
            Used for saving and loading configuration data.
        steer_target (number): The current target steering position for this
            module, in radians.
        steer_offset (number): The swerve module's steering zero position.
            This value can be determined by manually steering a swerve module
            so that it faces forwards relative to the chassis, and by taking
            the raw encoder position value (ADC reading); this value is the
            steer offset.
        drive_reversed (boolean): Whether or not the drive motor's output is
            currently reversed.
    """
    def __init__(self, name, steer_id, drive_id):
        """
        Create a new swerve module object.

        Args:
            name (string): A NetworkTables-friendly name to assign to this
                swerve module; it is used when saving and loading
                configuration data.
            steer_id (number): The CAN ID for the Talon SRX controlling this
                module's steering.
            drive_id (number): The CAN ID for the Talon SRX controlling this
                module's driving.
        """
        self.steer_talon = CANTalon(steer_id)
        self.drive_talon = CANTalon(drive_id)

        # Configure steering motors to use abs. encoders
        # and closed-loop control
        self.steer_talon.changeControlMode(CANTalon.ControlMode.Position)
        self.steer_talon.setFeedbackDevice(CANTalon.FeedbackDevice.AnalogEncoder)  # noqa: E501
        self.steer_talon.setProfile(0)

        self.name = name
        self.steer_target = 0

        self.load_config_values()

    def load_config_values(self):
        """
        Load saved configuration values for this module via WPILib's
        Preferences interface.

        The key names are derived from the name passed to the
        constructor.
        """
        preferences = wpilib.Preferences.getInstance()

        self.steer_offset = preferences.getFloat(self.name+'-offset', 0)
        self.drive_reversed = preferences.getBoolean(
            self.name+'-reversed', False)

    def save_config_values(self):
        """
        Save configuration values for this module via WPILib's
        Preferences interface.
        """
        preferences = wpilib.Preferences.getInstance()

        preferences.putFloat(self.name+'-offset', self.steer_offset)
        preferences.putBoolean(self.name+'-reversed', self.drive_reversed)

    def get_steer_angle(self):
        """
        Get the current angular position of the swerve module in
        radians.
        """
        native_units = self.steer_talon.get()
        return (native_units - self.steer_offset) * math.pi / 512

    def set_steer_angle(self, angle_radians):
        """
        Steer the swerve module to the given angle in radians.
        `angle_radians` should be within [-2pi, 2pi].

        This method attempts to find the shortest path to the given
        steering angle; thus, it may in actuality servo to the
        position opposite the passed angle and reverse the drive
        direction.

        Args:
            angle_radians (number): The angle to steer towards in radians,
                where 0 points in the chassis forward direction.
        """
        # normalize negative angles
        if angle_radians < 0:
            angle_radians += 2 * math.pi

        # get current steering angle, normalized to [0, 2pi)
        local_angle = (self.steer_talon.get() - self.steer_offset) % 1024
        local_angle *= math.pi / 512

        # Shortest-path servoing
        should_reverse_drive = False
        if local_angle < (math.pi / 2) and angle_radians > (3*math.pi / 2):
            # Q1 -> Q4 transition: subtract 1 full rotation from target angle
            angle_radians -= 2*math.pi
        elif local_angle > (3*math.pi / 2) and angle_radians < (math.pi / 2):
            # Q4 -> Q1 transition: add 1 full rotation to target angle
            angle_radians += 2*math.pi
        elif angle_radians - local_angle >= math.pi / 2:
            # shortest path is to move to opposite angle and reverse drive dir
            angle_radians -= math.pi
            should_reverse_drive = True
        elif angle_radians - local_angle <= (-math.pi / 2):
            # same as above
            angle_radians += math.pi
            should_reverse_drive = True

        # Adjust steer target to add to number of rotations of module thus far
        n_rotations = math.trunc(self.steer_talon.get() / 1024)
        self.steer_target = angle_radians + (n_rotations * 2 * math.pi)

        # Compute and send actual target to motor controller
        native_units = (self.steer_target * 512 / math.pi) + self.steer_offset
        self.steer_talon.set(native_units)

        if should_reverse_drive:
            self.drive_reversed = not self.drive_reversed

    def set_drive_speed(self, percent_speed):
        """
        Drive the swerve module wheels at a given percentage of
        maximum power or speed.

        Args:
            percent_speed (number): The speed to drive the module at, expressed
                as a percentage of maximum speed. Negative values drive in
                reverse.
        """
        if self.drive_reversed:
            percent_speed *= -1

        self.drive_talon.set(percent_speed)

    def apply_control_values(self, angle_radians, percent_speed):
        """
        Set a steering angle and a drive speed simultaneously.

        Args:
            angle_radians (number): The desired angle to steer towards.
            percent_speed (number): The desired percentage speed to drive at.

        See Also:
            :func:`~set_drive_speed` and :func:`~set_steer_angle`
        """
        self.set_steer_angle(angle_radians)
        self.set_drive_speed(percent_speed)

    def update_smart_dashboard(self):
        """
        Push various pieces of info to the Smart Dashboard.

        This method calls to NetworkTables (eventually), thus it may
        be _slow_.

        As of right now, this displays the current raw absolute encoder reading
        from the steer Talon, and the current target steer position.
        """
        wpilib.SmartDashboard.putNumber(
            self.name+' Position', self.steer_talon.get())
        wpilib.SmartDashboard.putNumber(self.name+' Target', self.steer_target)
