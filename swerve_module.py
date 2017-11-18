from ctre.cantalon import CANTalon
import wpilib
import math


class SwerveModule(object):
    """docstring for SwerveModule."""
    def __init__(self, name, steer_id, drive_id):
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
        preferences = wpilib.Preferences.getInstance()

        self.steer_offset = preferences.getFloat(self.name+'-offset', 0)
        self.drive_reversed = preferences.getBoolean(
            self.name+'-reversed', False)

    def save_config_values(self):
        preferences = wpilib.Preferences.getInstance()

        preferences.putFloat(self.name+'-offset', self.steer_offset)
        preferences.putBoolean(self.name+'-reversed', self.drive_reversed)

    def get_steer_angle(self):
        native_units = self.steer_talon.get()
        return (native_units - self.steer_offset) * math.pi / 512

    def set_steer_angle(self, angle_radians):
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
        if self.drive_reversed:
            percent_speed *= -1

        self.drive_talon.set(percent_speed)

    def apply_control_values(self, angle_radians, percent_speed):
        self.set_steer_angle(angle_radians)
        self.set_drive_speed(percent_speed)

    def update_smart_dashboard(self):
        wpilib.SmartDashboard.putNumber(
            self.name+' Position', self.steer_talon.get())
        wpilib.SmartDashboard.putNumber(self.name+' Target', self.steer_target)
