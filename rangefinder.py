"""
Implements common logic for rangefinder sensors.

Common parameters:
`mix` is a vector of shape (4,) containing the inverse sensor model
parameters:
    [a_hit, a_unexp, a_max, a_rand]
where
    a_hit + a_unexp + a_max + a_rand = 1
and
    P(z_t | x_t, m)
    = [a_hit, a_unexp, a_max, a_rand] * [p_hit, p_unexp, p_max, p_rand]
in other words, `mix` defines the mixing parameters
for a mixed distribution composed of four other distributions:
    - p_hit is a Gaussian distribution centered around the expected
      distance, z_exp
    - p_unexp is an exponential distribution that is cut off at
      z=z_exp
    - p_max is a very narrow uniform distribution centered around the
      sensor's maximum range.
    - p_rand is a uniform distribution spread across the sensor's
      entire range.
Together, these four distributions comprise a beam-based sensor model.

`model` is a vector of shape (4,) containing other model parameters:
    [sigma_hit, lambda_short, z_small, max_range]
these parameters define the four distributions described above.
(z_small is the width of p_max)
"""

import math
import numpy as np


dist_cmp_thrs = 1  # distance comparison threshold


def normalize(p):
    """
    Normalize all values within p.

    Returns:
        ndarrray: An ``ndarray`` with the same size and shape as p, normalized
            such that all values fall within the range [0, 1] and sum to 1.
    """
    return p / np.sum(p)


def rangefinder_likelihoods(z, z_exp, model):
    """
    Compute likelihoods for each distribution in the rangefinder model.

    Args:
        z (number): A distance measurement from the rangefinder.
        z_exp (number): The expected distance measurement from e.g. a map.
        model: A model parameter vector.
    """
    v_hit = model[0]**2  # sigma_hit**2

    p = [0, 0, 0, 0]

    # p_hit
    if z - model[3] <= dist_cmp_thrs:
        p[0] = (np.exp(-0.5 * ((z - z_exp)**2) / v_hit)
                / np.sqrt(2 * math.pi * v_hit))

    # p_unexp
    if z - z_exp <= dist_cmp_thrs:
        p[1] = model[1] * np.exp(-model[1] * z)

    # p_max
    # if sensor reading is within z_small units of max range...
    if np.abs(z - model[3]) - model[2] <= dist_cmp_thrs:
        p[2] = 1 / model[2]

    # p_rand
    p[3] = 1 / model[3]

    # convert p into an ndarray and normalize
    return normalize(np.array(p, dtype=np.float32))


def log_lambda(z, z_exp, model):
    """
    Compute the relative log-likelihood of a map, given a rangefinder model.

    Specifically, this computes:
    .. math::
        \\log \\frac{p(r=z | m)}{p(r=z | \\neg m)}

    Args:
        z (number): A distance measurement from the rangefinder.
        z_exp (number): The expected distance measurement from e.g. a map.
        model: A model parameter vector.

    Returns:
        number: The relative log-likelihood of a map, given the passed
            measured and expected distances.
    """

    v_hit = model[0]**2  # sigma_hit**2

    log_p_hit = (
        (((z_exp - z)**2) * -0.5 / v_hit)
        - np.log(np.sqrt(2 * math.pi * v_hit))
    )

    p_free = (1 / model[3])  # p_rand

    # p_max
    # if sensor reading is within z_small units of max range...
    if np.abs(z - model[3]) - model[2] <= dist_cmp_thrs:
        p_free += (1 / model[2])

    return log_p_hit - np.log(p_free)


def rangefinder_model(z, z_exp, model, mix):
    """
    Calculate the likelihood of one sensor measurement,
    given model and mixing parameters, as well as the expected distance
    to an obstruction.

    Args:
        z (number): A distance measurement from the rangefinder.
        z_exp (number): The expected distance measurement from e.g. a map.
        model: A model parameter vector.
        mix: A model mixing vector.

    Returns:
        number: The probability of the sensor measurement, given the
        expected distance and model parameters.
    """

    p = rangefinder_likelihoods(z, z_exp, model)
    return np.sum(mix * p)


def learn_intrinsic_parameters(z, z_exp, model):
    """
    Attempt to calculate model and mixing parameters from actual data.

    This attempts to estimate model and mixing parameters using a maximum-
    likelihood estimator.

    Args:
        z (number ndarray): A vector containing sensor measurement data.
        z_exp (number ndarray): A vector containing expected distances for
            each measurement in `z`.
        model (number ndarray, shape (4,)): An initial estimate for all
            model parameters.

    Returns:
        A tuple containing both a model parameter vector and a mixing vector.
    """

    current_mix = np.zeros([4])
    current_model = np.copy(model)

    # NOTE: we can stop beforehand if the algorithm converges
    for step in range(12):
        # Calculate likelihoods for each distribution separately:
        e = []
        e_hit = []
        e_short = []

        for z_i, z_exp_i in zip(z, z_exp):
            p = rangefinder_likelihoods(z_i, z_exp_i, current_model)
            e.append(p)
            e_hit.append(p[0])
            e_short.append(p[1])

        e = np.array(e)
        e_hit = np.array(e_hit)
        e_short = np.array(e_short)

        current_mix = np.average(e, axis=0)

        # sigma_hit
        current_model[0] = np.sqrt(
            np.sum(e_hit * (z - z_exp)**2)
            / np.sum(e_hit)
        )

        # lambda_short
        current_model[1] = (
            np.sum(e_short)
            / np.sum(e_short * z)
        )

    return current_model, current_mix
