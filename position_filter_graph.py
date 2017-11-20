import math
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
import position_filtering


def gyro_noisy(pose, running_time):
    return normal(pose[2], running_time * (math.pi / 180) / 120)

def gyro_rate_noisy(pose):
    return normal(pose[5], 0.1 * (math.pi / 180))

def accelerometer_noisy(pose):
    return normal([pose[6] / 9.81, pose[7]  / 9.81], 0.008 * 9.81)

angle_sigma = 0.045056
speed_sigma = 0.1
movement_sigma = 0.1
movement_covariance = np.identity(9) * (movement_sigma ** 2)

chassis_length = 24.69
chassis_width = 22.61
dt = 0.1
t = 0

real_pose = np.expand_dims(np.array([0, 0, 0, 0, 0, 0, 5, 5, math.pi / 6]), 1)
pos_filter = position_filtering.PositionFilter(chassis_width, chassis_length)
#pos_filter.pose = real_pose


def timestep_update():
    d = position_filtering.swerve_encoder_model(chassis_width, chassis_length, real_pose)
    assert d.shape == (8,1)

    d = np.reshape(normal(d, [[[angle_sigma]]*4 + [[speed_sigma]]*4]), (8,1))

    pos_filter.swerve_encoder_update(
        angle_sigma ** 2, speed_sigma ** 2,
        None, None, measurement=d)

    pos_filter.ahrs_gyro_update(None, np.array([
        gyro_noisy(real_pose, t),
        gyro_rate_noisy(real_pose)
    ]))

    pos_filter.ahrs_accelerometer_update(None, accelerometer_noisy(real_pose))

def timestep_predict():
    global real_pose, t
    transition_matrix = position_filtering.state_transition(dt)
    real_pose = normal(transition_matrix @ real_pose, movement_sigma)

    t += dt

    pos_filter.predict(movement_covariance, dt)

ts = []

xs = []
ys = []
hdgs = []

accelero_x = []
accelero_y = []

real_xs = []
real_ys = []
real_hdgs = []

for step in range(int(15 / dt)):
    timestep_predict()
    timestep_update()

    ts.append(t)

    xs.append(pos_filter.pose[0][0])
    ys.append(pos_filter.pose[1][0])
    hdgs.append(pos_filter.pose[2][0])

    accelero = accelerometer_noisy(real_pose)
    accelero_x.append(accelero[0][0])
    accelero_y.append(accelero[1][0])


    real_xs.append(real_pose[0][0])
    real_ys.append(real_pose[1][0])
    real_hdgs.append(real_pose[2][0])

plt.subplot(211)
plt.plot(ts, xs, 'r-')
plt.plot(ts, ys, 'b-')
plt.plot(ts, hdgs, 'g-')

plt.plot(ts, real_xs, 'm--')
plt.plot(ts, real_ys, 'c--')
plt.plot(ts, real_hdgs, 'y--')

plt.subplot(212)
plt.plot(ts, accelero_x, 'm--')
plt.plot(ts, accelero_y, 'c--')

plt.show()
