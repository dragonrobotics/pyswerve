from ctre.cantalon import CANTalon
import wpilib

class Robot(wpilib.IterativeRobot):
    talon_id_set = [
        ('Front Left', 5, False),
        ('Front Right', 1, False),
        ('Back Left', 4, False),
        ('Back Right', 15, False)
    ]

    encoder_value_sets = []  # tuples of (home_pos, min_pos, max_pos)
    last_encoder_values = []
    talon_sweeping = []

    def robotInit(self):
        self.control_stick = wpilib.Joystick(0)
        self.talons = [CANTalon(tid) for _, tid, _ in self.talon_id_set]
        self.encoder_value_sets = [[None, None, None] for _ in self.talon_id_set]
        self.last_encoder_values = [None for _ in self.talon_id_set]
        self.talon_sweeping = [True for _ in self.talon_id_set]

        self.talon_selector = wpilib.SendableChooser()
        for i, talon_cfg in enumerate(self.talon_id_set):
            self.talon_selector.addDefault(talon_cfg[0], str(i))

        wpilib.SmartDashboard.putData('Active Talon', self.talon_selector)

    def disabledInit(self):
        for talon in self.talons:
            talon.setFeedbackDevice(CANTalon.FeedbackDevice.AnalogEncoder)

    def disabledPeriodic(self):
        for idx, talon in enumerate(self.talons):
            wpilib.SmartDashboard.putNumber(
                self.talon_id_set[idx][0]+' Encoder',
                talon.getAnalogInRaw()
            )

    def teleopInit(self):
        for talon in self.talons:
            talon.changeControlMode(CANTalon.ControlMode.Position)
            talon.setFeedbackDevice(CANTalon.FeedbackDevice.AnalogEncoder)
            talon.setProfile(0)

    def teleopPeriodic(self):
        for idx, talon in enumerate(self.talons):
            wpilib.SmartDashboard.putNumber(
                self.talon_id_set[idx][0]+' Encoder',
                talon.getAnalogInRaw()
            )

        active_talon_idx = int(self.talon_selector.getSelected())

        for idx, talon in enumerate(self.talons):
            if idx == active_talon_idx:
                talon.enable()
            else:
                talon.disable()

        preferences = wpilib.Preferences.getInstance()
        ctrl_min = preferences.getFloat(
            self.talon_id_set[active_talon_idx][0]+'-min',
            0.0
        )

        ctrl_max = preferences.getFloat(
            self.talon_id_set[active_talon_idx][0]+'-max',
            1024.0
        )

        active_talon = self.talons[active_talon_idx]

        raw = self.control_stick.getAxis(0)
        # Ranges from ctrl_min to ctrl_max in discrete steps
        tgt = (int(raw*20) / 20) * (ctrl_max - ctrl_min)
        tgt += ctrl_min
        active_talon.set(tgt)

        wpilib.SmartDashboard.putNumber('Ctrl Target', tgt)
        wpilib.SmartDashboard.putNumber('Position', active_talon.getAnalogInRaw())

    def autonomousInit(self):
        for idx, talon in enumerate(self.talons):
            talon.changeControlMode(CANTalon.ControlMode.PercentVbus)
            talon.setFeedbackDevice(CANTalon.FeedbackDevice.AnalogEncoder)

            self.encoder_value_sets[idx] = [talon.getAnalogInRaw(), None, None]
            self.last_encoder_values[idx] = talon.getAnalogInRaw()
            self.talon_sweeping[idx] = True
            talon.set(1.0 if not self.talon_id_set[idx][2] else -1.0)

    def autonomousPeriodic(self):
        # Sweep through and get min/max encoder values
        for idx, talon in enumerate(self.talons):
            wpilib.SmartDashboard.putNumber(
                self.talon_id_set[idx][0]+' Encoder',
                talon.getAnalogInRaw()
            )

            if self.talon_sweeping[idx]:
                current_value = talon.getAnalogInRaw()

                if (
                    self.encoder_value_sets[idx][1] is None
                    or current_value < self.encoder_value_sets[idx][1]
                ):
                    self.encoder_value_sets[idx][1] = current_value

                if (
                    self.encoder_value_sets[idx][2] is None
                    or current_value > self.encoder_value_sets[idx][2]
                ):
                    self.encoder_value_sets[idx][2] = current_value

                if (
                    (
                        not self.talon_id_set[idx][2]
                        and current_value < self.last_encoder_values[idx]
                    ) or (
                        self.talon_id_set[idx][2]
                        and current_value > self.last_encoder_values[idx]
                    )
                ):
                    # encoder values wrapped around, stop this talon
                    talon.set(0)
                    self.talon_sweeping[idx] = False

                    preferences = wpilib.Preferences.getInstance()

                    preferences.putFloat(
                        self.talon_id_set[idx][0]+'-offset',
                        self.encoder_value_sets[idx][0]
                    )

                    preferences.putFloat(
                        self.talon_id_set[idx][0]+'-min',
                        self.encoder_value_sets[idx][1]
                    )

                    preferences.putFloat(
                        self.talon_id_set[idx][0]+'-max',
                        self.encoder_value_sets[idx][2]
                    )

                self.last_encoder_values[idx] = current_value

if __name__ == "__main__":
    wpilib.run(Robot)
