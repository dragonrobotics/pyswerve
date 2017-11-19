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
    res = measurement - (model @ current_state)
    res_covar = (model @ current_covar @ model.transpose()) + noise
    gain = (current_covar @ model.transpose() @ np.linalg.inv(res_covar))

    new_state = current_state + (gain @ res)
    new_covar = (np.identity(gain.shape[0]) - (gain @ model)) @ current_covar

    return new_state, new_covar
