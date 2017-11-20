from swerve_drive import SwerveDrive
from position_filtering import PositionFilter
import wpilib
from robotpy_ext.common_drivers.navx.ahrs import AHRS
from robotpy_ext.control.button_debouncer import ButtonDebouncer
import math
import numpy as np


class Robot(wpilib.IterativeRobot):
    swerve_config = [
        ('swerve-backright', 4, 2),
        ('swerve-backleft', 1, 19),
        ('swerve-frontright', 13, 12),
        ('swerve-frontleft', 10, 15),
    ]

    # NOTE: the actual units for chassis length and width don't matter for
    # the purposes of the swerve drive calculations, as long as they're the
    # _same_ units
    chassis_length = 24.69
    chassis_width = 22.61

    foc_enabled = False

    def robotInit(self):
        self.control_stick = wpilib.Joystick(0)
        self.save_config_button = ButtonDebouncer(self.control_stick, 0)
        self.toggle_foc_button = ButtonDebouncer(self.control_stick, 1)

        self.sd_update_timer = wpilib.Timer()
        self.sd_update_timer.start()

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

        self.position_filter = PositionFilter(
            self.chassis_width,
            self.chassis_length
        )

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

        # I don't really know how to predict / determine robot movement
        # covariances. So I'm just going to assume independent variances
        # of 0.1 (so sigma = sqrt(0.1) ~= 0.316)
        movement_covar = np.identity(9) * 0.1
        self.position_filter.predict(movement_covar)

        if self.navx is not None and self.navx.isConnected():
            self.position_filter.ahrs_gyro_update(self.navx)
            self.position_filter.ahrs_accelerometer_update(self.navx)

        drive_angles = self.drivetrain.get_module_angles()
        drive_speeds = self.drivetrain.get_module_speeds()

        # For now, I'll assume that we're using 4" wheels with CIMcoders.
        # Thus, our drive speed conversion factor =
        # 40 encoder edges per revolution / (2*2*PI) in. per revolution
        # / 0.0254 meters per inch
        drive_speeds *= 40 / (4 * math.pi * 0.0254)

        # The USDigital MA3 Miniature Absolute Magnetic Shaft Encoder
        # has a specified output noise of 220 uV-RMS or:
        # 0.000220V RMS * (1024 units / 5V) = 0.045056 units RMS
        # (max noise = 0.000490V RMS * (1024 units / 5V)) = 0.100352 units RMS

        # Meanwhile, I can't find any information on noise for the CIMCoders.
        # I'll just assume that stddev for speeds is around 10cm/s
        self.position_filter.swerve_encoder_update(
            (0.045056 ** 2), (0.1 ** 2),
            drive_angles, drive_speeds
        )

        # update smart dashboard at 4Hz
        if self.sd_update_timer.hasPeriodPassed(0.25):
            self.drivetrain.update_smart_dashboard()

            wpilib.SmartDashboard.putNumber(
                'Predicted Position X',
                self.position_filter.get_position()[0]
            )

            wpilib.SmartDashboard.putNumber(
                'Predicted Position Y',
                self.position_filter.get_position()[1]
            )

            wpilib.SmartDashboard.putNumber(
                'Predicted Heading',
                self.position_filter.get_heading() * (180 / math.pi)
            )

            if self.navx is not None and self.navx.isConnected():
                wpilib.SmartDashboard.putNumber(
                    'AHRS Fused Heading',
                    self.navx.getFusedHeading()
                )


if __name__ == "__main__":
    wpilib.run(Robot)
