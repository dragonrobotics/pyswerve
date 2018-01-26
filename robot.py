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

    chassis_length = 32
    chassis_width = 28

    foc_enabled = False

    def robotInit(self):
        self.control_stick = wpilib.Joystick(0)
        self.save_config_button = ButtonDebouncer(self.control_stick, 1)
        self.toggle_foc_button = ButtonDebouncer(self.control_stick, 2)
        self.zero_yaw_button = ButtonDebouncer(self.control_stick, 3)

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

    def disabledInit(self):
        self.drivetrain.load_config_values()
        self.drivetrain.reset_drive_position()

    def disabledPeriodic(self):
        self.drivetrain.update_smart_dashboard()
        wpilib.SmartDashboard.putBoolean('FOC Enabled', self.foc_enabled)
        wpilib.SmartDashboard.putNumber('Heading', self.navx.getFusedHeading())
        wpilib.SmartDashboard.putNumber(
            'Accumulated Yaw',
            self.navx.getAngle())

    def autonomousInit(self):
        self.drivetrain.load_config_values()

        self.auto_timer = wpilib.Timer()
        self.auto_timer.reset()
        self.auto_timer.start()
        self.auto_start_time = None

        self.max_left_speed = 0
        self.max_right_speed = 0
        self.max_side_speed_diff = 0
        self.last_countdown_time = 999

        self.drivetrain.reset_drive_position()
        self.drivetrain.set_all_module_angles(0)

        for module in self.drivetrain.modules:
            module.max_observed_speed = 0

        self.drivetrain.update_smart_dashboard()
        self.turn_complete = False

        #self.navx.reset()

    def autonomousSpeedTesting(self):
        countdown_time = math.ceil(wait_time - self.auto_timer.get())
        if self.last_countdown_time > countdown_time and countdown_time >= 0:
            if countdown_time > 0:
                print("{}...".format(countdown_time))
            else:
                print("Go!")

        self.last_countdown_time = countdown_time

        self.drivetrain.set_all_module_angles(0)
        if self.auto_timer.get() > wait_time:
            driving_time = self.auto_timer.get() - wait_time
            if avg_dist < target:
                cur_speed = drive_speed
                if driving_time <= acc_time:
                    cur_speed *= (driving_time / acc_time)

                self.drivetrain.set_all_module_speeds(cur_speed, direct=True)
            else:
                self.drivetrain.set_all_module_speeds(0, direct=True)

    def autonomousTurnTesting(self):
        if not self.turn_complete:
            self.turn_complete = self.drivetrain.turn_to_angle(
                self.navx,
                math.radians(90)
            )

    def autonomousPeriodic(self):
        prefs = wpilib.Preferences.getInstance()
        drive_speed = prefs.getInt('Auto Drive Speed', 100)
        wait_time = prefs.getFloat('Auto Wait Time', 1.0)
        acc_time = prefs.getFloat('Auto Acceleration Time', 2.0)
        target_in = prefs.getFloat('Auto Distance', 120.0)

        target = target_in * ((80 * 6.67) / (4*math.pi))
        avg_dist = np.mean(self.drivetrain.get_module_distances())

        right_speed = np.mean(np.abs([
            self.drivetrain.modules[0].cur_drive_spd,
            self.drivetrain.modules[2].cur_drive_spd
        ]))

        left_speed = np.mean(np.abs([
            self.drivetrain.modules[1].cur_drive_spd,
            self.drivetrain.modules[3].cur_drive_spd
        ]))

        side_speed_diff = abs(left_speed) - abs(right_speed)

        if abs(self.max_left_speed) < abs(left_speed):
            self.max_left_speed = left_speed

        if abs(self.max_right_speed) < abs(right_speed):
            self.max_right_speed = right_speed

        if abs(self.max_side_speed_diff) < abs(side_speed_diff):
            self.max_side_speed_diff = side_speed_diff

        self.autonomousTurnTesting()

        wpilib.SmartDashboard.putNumber('Left Side Speed', left_speed)
        wpilib.SmartDashboard.putNumber('Right Side Speed', right_speed)
        wpilib.SmartDashboard.putNumber(
            'Side Speed Diff',
            side_speed_diff
        )

        wpilib.SmartDashboard.putNumber(
            'Max Left Side Speed',
            self.max_left_speed
        )

        wpilib.SmartDashboard.putNumber(
            'Max Right Side Speed',
            self.max_right_speed
        )

        wpilib.SmartDashboard.putNumber(
            'Max Side Speed Diff',
            self.max_side_speed_diff
        )

        wpilib.SmartDashboard.putNumber('Avg Dist', avg_dist)
        wpilib.SmartDashboard.putBoolean('FOC Enabled', self.foc_enabled)
        wpilib.SmartDashboard.putNumber('Heading', self.navx.getFusedHeading())
        wpilib.SmartDashboard.putNumber(
            'Accumulated Yaw',
            self.navx.getAngle())

        self.drivetrain.update_smart_dashboard()

    def teleopInit(self):
        self.drivetrain.load_config_values()

    def teleopPeriodic(self):
        wpilib.SmartDashboard.putBoolean('FOC Enabled', self.foc_enabled)
        wpilib.SmartDashboard.putNumber('Heading', self.navx.getFusedHeading())
        wpilib.SmartDashboard.putNumber(
            'Accumulated Yaw',
            self.navx.getAngle())

        ctrl = np.array([
            self.control_stick.getRawAxis(1) * -1,
            self.control_stick.getRawAxis(0) * -1
        ])

        pov = self.control_stick.getPOV()
        if pov != -1:
            pov = math.radians(pov)
            ctrl[0] = math.cos(pov)
            ctrl[1] = math.sin(pov)

        if abs(np.sqrt(np.sum(ctrl**2))) < 0.15:
            ctrl[0] = 0
            ctrl[1] = 0

        prefs = wpilib.Preferences.getInstance()
        teleop_max_speed = prefs.getFloat('Teleop Max Speed', 370)

        if (self.navx is not None and
                self.navx.isConnected() and self.foc_enabled):
            # perform FOC coordinate transform
            hdg = self.navx.getFusedHeading() * (math.pi / 180)

            if prefs.getBoolean('Reverse Heading Direction', False):
                hdg *= -1

            # Right-handed passive (alias) transform matrix
            foc_transform = np.array([
                [np.cos(hdg), np.sin(hdg)],
                [-np.sin(hdg), np.cos(hdg)]
            ])

            ctrl = np.squeeze(np.matmul(foc_transform, ctrl))

        tw = self.control_stick.getRawAxis(4) * -1
        if abs(tw) < 0.05:
            tw = 0

        self.drivetrain.drive(
            ctrl[0],
            ctrl[1],
            tw / 2,
            teleop_max_speed
        )

        if self.save_config_button.get():
            self.drivetrain.save_config_values()

        if self.zero_yaw_button.get():
            self.navx.zeroYaw()

        if self.toggle_foc_button.get():
            self.foc_enabled = not self.foc_enabled

        self.drivetrain.update_smart_dashboard()


if __name__ == "__main__":
    wpilib.run(Robot)
