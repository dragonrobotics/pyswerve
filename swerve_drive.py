import wpilib
import math
import numpy as np

from swerve_module import SwerveModule

class SwerveDrive(object):
    """docstring for SwerveDrive."""

    # config_tuples is a list of tuples of the form (name, steer_id, drive_id)
    # where each value is passed to the SwerveModule constructor in that order
    def __init__(self, length, width, config_tuples):
        self.modules = []
        for config in config_tuples:
            self.modules.append(SwerveModule(*config))

        self.length = length
        self.width = width
        self.radius = math.sqrt((length ** 2) + (width ** 2))

        self.sd_update_timer = wpilib.Timer()
        self.sd_update_timer.start()

    def drive(self, forward, strafe, rotate_cw):
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

    def save_config_values(self):
        for module in self.modules:
            module.save_config_values()

    def update_smart_dashboard(self):
        if self.sd_update_timer.hasPeriodPassed(0.25):
            for module in self.modules:
                module.update_smart_dashboard()
