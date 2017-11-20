"""
Implements the basic Kalman Filter equations.
"""
import numpy as np


def predict(
        current_state, current_covar,
        model, noise,
        control_input=None, control_model=None):
    new_state = (model @ current_state)
    if not ((control_input is None) or (control_model is None)):
        new_state += (control_model @ control_input)

    new_covar = (model @ current_covar @ model.transpose())
    new_covar += noise

    return new_state, new_covar


def update(
        current_state, current_covar,
        measurement, model, noise):
    if measurement.ndim == 1:
        measurement = np.expand_dims(measurement, axis=1)

    res = measurement - (model @ current_state)
    res_covar = (model @ current_covar @ model.transpose()) + noise
    gain = (current_covar @ model.transpose() @ np.linalg.inv(res_covar))

    new_state = current_state + (gain @ res)
    new_covar = (np.identity(gain.shape[0]) - (gain @ model)) @ current_covar

    return new_state, new_covar


def ekf_update(
        state, covar,
        measurement, measurement_model, linear_model,
        noise):
    if measurement.ndim == 1:
        measurement = np.expand_dims(measurement, axis=1)

    res = measurement - measurement_model(state)
    res_covar = (linear_model @ covar @ linear_model.transpose()) + noise
    gain = (covar @ linear_model.transpose() @ np.linalg.inv(res_covar))

    new_state = state + (gain @ res)
    new_covar = (np.identity(gain.shape[0]) - (gain @ linear_model)) @ covar

    return new_state, new_covar
