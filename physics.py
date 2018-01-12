import math


def get_talon_target(hal_data, talon_id):
    hal_target = hal_data['CAN'][talon_id]['value']
    return hal_target


def copy_talon_analog_pos(hal_data, talon_id):
    hal_data['CAN'][talon_id]['analog_in_position'] = hal_data['CAN'][talon_id]['value']  # noqa:E501


# pulled from master branch on pyfrc; remove when latest version actually works
def four_motor_swerve_drivetrain(
    lr_motor, rr_motor, lf_motor, rf_motor,
    lr_angle, rr_angle, lf_angle, rf_angle,
    x_wheelbase=2, y_wheelbase=2, speed=5
):
    # Calculate speed of each wheel
    lr = lr_motor * speed
    rr = rr_motor * speed
    lf = lf_motor * speed
    rf = rf_motor * speed

    # Calculate angle in radians
    lr_rad = math.radians(lr_angle)
    rr_rad = math.radians(rr_angle)
    lf_rad = math.radians(lf_angle)
    rf_rad = math.radians(rf_angle)

    # Calculate wheelbase radius
    wheelbase_radius = math.hypot(x_wheelbase / 2, y_wheelbase / 2)

    # Calculates the Vx and Vy components
    # Sin an Cos inverted because forward is 0 on swerve wheels
    Vx = (
        (math.sin(lr_rad) * lr)
        + (math.sin(rr_rad) * rr)
        + (math.sin(lf_rad) * lf)
        + (math.sin(rf_rad) * rf))
    Vy = (
        (math.cos(lr_rad) * lr)
        + (math.cos(rr_rad) * rr)
        + (math.cos(lf_rad) * lf)
        + (math.cos(rf_rad) * rf))

    # Adjusts the angle corresponding to a diameter
    # that is perpendicular to the radius (add or subtract 45deg)
    lr_rad = (lr_rad + (math.pi / 4)) % (2 * math.pi)
    rr_rad = (rr_rad - (math.pi / 4)) % (2 * math.pi)
    lf_rad = (lf_rad - (math.pi / 4)) % (2 * math.pi)
    rf_rad = (rf_rad + (math.pi / 4)) % (2 * math.pi)

    # Finds the rotational velocity by finding the torque and adding them up
    Vw = wheelbase_radius * (
            (math.cos(lr_rad) * lr)
            + (math.cos(rr_rad) * -rr)
            + (math.cos(lf_rad) * lf)
            + (math.cos(rf_rad) * -rf))

    Vx *= 0.25
    Vy *= 0.25
    Vw *= 0.25

    return Vx, Vy, Vw


class PhysicsEngine(object):
    #  mirrors swerve_config in robot.py (name, steer_id, drive_id)
    swerve_config = [
        ('Back Right', 15, 12),
        ('Back Left', 4, 10),
        ('Front Right', 1, 2),
        ('Front Left', 5, 13),
    ]

    chassis_length = 32
    chassis_width = 28

    def __init__(self, physics_controller):
        self.physics_controller = physics_controller

    def update_sim(self, hal_data, now, tm_diff):
        vx, vy, vw = four_motor_swerve_drivetrain(
            get_talon_target(hal_data, self.swerve_config[1][2]) / 1024,
            get_talon_target(hal_data, self.swerve_config[0][2]) / 1024,
            get_talon_target(hal_data, self.swerve_config[3][2]) / 1024,
            get_talon_target(hal_data, self.swerve_config[2][2]) / 1024,
            get_talon_target(hal_data, self.swerve_config[1][1]) * (180 / 512),
            get_talon_target(hal_data, self.swerve_config[0][1]) * (180 / 512),
            get_talon_target(hal_data, self.swerve_config[3][1]) * (180 / 512),
            get_talon_target(hal_data, self.swerve_config[2][1]) * (180 / 512),
            self.chassis_length / 12,
            self.chassis_width / 12,
            14
        )

        self.physics_controller.vector_drive(vx, vy, vw, tm_diff)

        x, y, angle_rad = self.physics_controller.get_position()

        copy_talon_analog_pos(hal_data, self.swerve_config[0][1])
        copy_talon_analog_pos(hal_data, self.swerve_config[1][1])
        copy_talon_analog_pos(hal_data, self.swerve_config[2][1])
        copy_talon_analog_pos(hal_data, self.swerve_config[3][1])

        hal_data['robot']['navxmxp_i2c_1_angle'] = angle_rad * (180 / math.pi)
        hal_data['robot']['navxmxp_spi_4_angle'] = angle_rad * (180 / math.pi)
