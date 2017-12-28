from swerve_drive import SwerveDrive
import wpilib
from robotpy_ext.common_drivers.navx.ahrs import AHRS
from robotpy_ext.control.button_debouncer import ButtonDebouncer
import math
import numpy as np


class Robot(wpilib.IterativeRobot):
    swerve_config = [
        ('Back Right', 15, 12),
        ('Back Left', 4, 10),
        ('Front Right', 1, 2),
        ('Front Left', 5, 13),
    ]

    chassis_length = 24.69
    chassis_width = 22.61

    foc_enabled = False

    def robotInit(self):
        self.control_stick = wpilib.Joystick(0)
        self.save_config_button = ButtonDebouncer(self.control_stick, 0)
        self.toggle_foc_button = ButtonDebouncer(self.control_stick, 1)

        self.drivetrain = SwerveDrive(
            self.chassis_length,
            self.chassis_width,
            self.swerve_config
        )
        try:
            self.navx = AHRS.create_spi()
            self.navx.reset()
        except Exception as e:
            print("Caught exception while trying to initialize AHRS: "+e.args)
            self.navx = None

    def teleopPeriodic(self):
        ctrl = np.array([
            self.control_stick.getAxis(0),
            self.control_stick.getAxis(1)
        ])

        if (self.navx is not None and
                self.navx.isConnected() and self.foc_enabled):
            # perform FOC coordinate transform
            hdg = self.navx.getFusedHeading() * (math.pi / 180)

            # Right-handed passive (alias) transform matrix
            foc_transform = np.array([
                [np.cos(hdg), np.sin(hdg)],
                [-np.sin(hdg), np.cos(hdg)]
            ])

            ctrl = np.squeeze(np.matmul(foc_transform, ctrl))

        self.drivetrain.drive(
            ctrl[0],
            ctrl[1],
            self.control_stick.getAxis(3)
        )

        if self.save_config_button.get():
            self.drivetrain.save_config_values()

        if self.toggle_foc_button.get():
            self.foc_enabled = not self.foc_enabled

        self.drivetrain.update_smart_dashboard()


if __name__ == "__main__":
    wpilib.run(Robot)
