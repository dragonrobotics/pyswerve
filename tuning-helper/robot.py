from ctre.cantalon import CANTalon
import wpilib

class Robot(wpilib.IterativeRobot):
    talon_id_set = [
        ('Front Left', 0),
        ('Front Right', 1),
        ('Back Left', 2),
        ('Back Right', 3)
    ]

    def robotInit(self):
        self.control_stick = wpilib.Joystick(0)
        self.talons = [CANTalon(tid) for _, tid in self.talon_id_set]

        for talon in self.talons:
            talon.changeControlMode(CANTalon.ControlMode.Position)
            talon.setFeedbackDevice(CANTalon.FeedbackDevice.AnalogEncoder)
            talon.setProfile(0)

        self.talon_selector = wpilib.SendableChooser()
        for i, talon_cfg in enumerate(self.talon_id_set):
            self.talon_selector.addDefault(talon_cfg[0], str(i))

        wpilib.SmartDashboard.putData('Active Talon', self.talon_selector)

    def teleopPeriodic(self):
        active_talon_idx = int(self.talon_selector.getSelected())

        for idx, talon in enumerate(self.talons):
            if idx == active_talon_idx:
                talon.enable()
            else:
                talon.disable()

        active_talon = self.talons[active_talon_idx]

        raw = self.control_stick.getAxis(0)
        tgt = int(raw*16) * 64  # Ranges from 0-1024 in discrete steps
        active_talon.set(tgt)

        wpilib.SmartDashboard.putNumber('Ctrl Tgt', tgt)
        wpilib.SmartDashboard.putNumber('Position', active_talon.get())

    def autonomousPeriodic(self):
        active_talon_idx = int(self.talon_selector.getSelected())

        for idx, talon in enumerate(self.talons):
            if idx == active_talon_idx:
                talon.enable()
            else:
                talon.disable()

        active_talon = self.talons[active_talon_idx]

        active_talon.set(0)
        wpilib.SmartDashboard.putNumber('Ctrl Tgt', 0)
        wpilib.SmartDashboard.putNumber('Position', active_talon.get())

if __name__ == "__main__":
    wpilib.run(Robot)
