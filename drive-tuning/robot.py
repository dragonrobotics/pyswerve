import wpilib
from ctre.talonsrx import TalonSRX
import numpy as np

ControlMode = TalonSRX.ControlMode
FeedbackDevice = TalonSRX.FeedbackDevice

class Robot(wpilib.IterativeRobot):
    active_id = 10
    target = 200

    def robotInit(self):
        self.drive_talon = TalonSRX(self.active_id)
        self.drive_talon.configSelectedFeedbackSensor(
            FeedbackDevice.QuadEncoder,
            0, 0
        )
        self.control_stick = wpilib.Joystick(0)
        self.drive_speeds = []
        self.max_vel = 0

    def logSDData(self):
        wpilib.SmartDashboard.putNumber(
            'Drive Ticks',
            self.drive_talon.getQuadraturePosition()
        )

        wpilib.SmartDashboard.putNumber(
            'Drive Rotations',
            self.drive_talon.getQuadraturePosition() / (80 * 6.67)
        )

        wpilib.SmartDashboard.putNumber(
            'Drive Velocity',
            self.drive_talon.getQuadratureVelocity()
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
            self.drive_talon.getClosedLoopError(0)
        )

    def disabledPeriodic(self):
        self.logSDData()

    def autonomousInit(self):
        self.drive_talon.setSensorPhase(True)
        self.drive_talon.selectProfileSlot(1, 0)
        self.drive_talon.setQuadraturePosition(0, 0)
        self.drive_talon.set(ControlMode.Velocity, self.target)
        self.max_vel = 0
        self.drive_speeds = []
        self.logSDData()

    def autonomousPeriodic(self):
        cur_vel = self.drive_talon.getQuadratureVelocity()

        if abs(cur_vel) > abs(self.max_vel):
            self.max_vel = cur_vel

        self.drive_speeds.append(cur_vel)
        if len(self.drive_speeds) > 50:
            self.drive_speeds = self.drive_speeds[-50:]

        self.drive_talon.set(ControlMode.Velocity, self.target)
        self.logSDData()



if __name__ == "__main__":
    wpilib.run(Robot)
