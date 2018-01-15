import wpilib
from ctre.cantalon import CANTalon
import numpy as np


class Robot(wpilib.IterativeRobot):
    active_id = 2
    target = 500

    def robotInit(self):
        self.drive_talon = CANTalon(self.active_id)
        self.drive_talon.setFeedbackDevice(CANTalon.FeedbackDevice.QuadEncoder)
        self.control_stick = wpilib.Joystick(0)
        self.drive_speeds = []
        self.max_vel = 0

    def logSDData(self):
        wpilib.SmartDashboard.putNumber(
            'Drive Ticks',
            self.drive_talon.getEncPosition()
        )

        wpilib.SmartDashboard.putNumber(
            'Drive Rotations',
            self.drive_talon.getEncPosition() / 80
        )

        wpilib.SmartDashboard.putNumber(
            'Drive Velocity',
            self.drive_talon.getEncVelocity()
        )

        wpilib.SmartDashboard.putNumber(
            'Max Drive Velocity',
            self.max_vel
        )

        wpilib.SmartDashboard.putNumber(
            'Avg Drive Velocity',
            np.mean(self.drive_speeds)
        )

        wpilib.SmartDashboard.putNumber(
            'Drive Error',
            self.drive_talon.getClosedLoopError()
        )

    def disabledPeriodic(self):
        self.logSDData()

    def autonomousInit(self):
        self.drive_talon.changeControlMode(CANTalon.ControlMode.Speed)
        self.drive_talon.reverseSensor(False)
        self.drive_talon.setProfile(1)
        self.drive_talon.setEncPosition(0)
        self.drive_talon.set(self.target)
        self.max_vel = 0
        self.drive_speeds = []
        self.logSDData()

    def autonomousPeriodic(self):
        cur_vel = self.drive_talon.getEncVelocity()

        if abs(cur_vel) > abs(self.max_vel):
            self.max_vel = cur_vel

        self.drive_speeds.append(cur_vel)
        if len(self.drive_speeds) > 50:
            self.drive_speeds = self.drive_speeds[-50:]

        self.drive_talon.set(self.target)
        self.logSDData()


if __name__ == "__main__":
    wpilib.run(Robot)
