from ctre.cantalon import CANTalon
import wpilib

class Robot(wpilib.IterativeRobot):
    def robotInit(self):
        self.control_stick = wpilib.Joystick(0)
        self.active_talon = CANTalon(0)

        self.active_talon.changeControlMode(CANTalon.ControlMode.Position)
        self.active_talon.setFeedbackDevice(CANTalon.FeedbackDevice.AnalogEncoder)  # noqa: E501
        self.active_talon.setProfile(0)

    def teleopPeriodic(self):
        raw = self.control_stick.getAxis(0)
        tgt = int(raw*16) * 64  # Ranges from 0-1024 in discrete steps
        self.active_talon.set(tgt)

        wpilib.SmartDashboard.putNumber('Ctrl Tgt', tgt)
        wpilib.SmartDashboard.putNumber('Position', self.active_talon.get())

    def autonomousPeriodic(self):
        self.active_talon.set(0)
        wpilib.SmartDashboard.putNumber('Ctrl Tgt', 0)
        wpilib.SmartDashboard.putNumber('Position', self.active_talon.get())

if __name__ == "__main__":
    wpilib.run(Robot)
