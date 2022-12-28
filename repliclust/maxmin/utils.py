""" 
This module provides utility functions for max-min sampling.
"""

import numpy as np
from scipy import stats
from repliclust import config

def sample_with_maxmin(
    n_samples, ref_val, min_val, maxmin_ratio, f_constraint
    ):
    """
    Generate samples around a reference value, with a fixed ratio 
    between the maximum and minimum sample. Sampling proceeds pairwise
    to enforce a desired constraint on the samples.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    ref_val : float, >0
        Reference value. Specifies the typical value for the samples.
    min_val : float, >0
        Minimum value across all the samples.
    maxmin_ratio : float
        Ratio between largest and smallest value across all samples.
    f_constraint : function
        Function to enforce a constraint.

    Returns
    -------
    samples : ndarray
        Sorted array containing the samples
    """
    if (maxmin_ratio == 1) or (min_val == 0):
        samples = np.full(n_samples, fill_value=ref_val)
        return samples

    max_val = min_val * maxmin_ratio
    
    if (n_samples > 2):
        # Besides min_val and max_val, only need n-2 samples.
        n_gotta_sample = n_samples-2 
        samples = np.full(n_gotta_sample, fill_value=float(ref_val))
        # Sample according to triangular distribution with endpoints 
        # given by min_val and max_val, and mode given by ref. 
        # Sample pairwise. The first sample in each pair is generated 
        # randomly; the second sample is calculated from the first using
        # the constraint function.
        while (n_gotta_sample >= 2):
            samples[n_gotta_sample-1] = config._rng.triangular(
                                            left = min_val,
                                            mode = ref_val,
                                            right = max_val 
                                            )
            samples[n_gotta_sample-2] = f_constraint(
                                            samples[n_gotta_sample-1]
                                            )
            n_gotta_sample -= 2
        samples = np.concatenate(
                    [[min_val], np.sort(samples), [max_val]]
                    )
    elif (n_samples == 2):
        samples = np.array([min_val, max_val])
    elif (n_samples == 1):
        samples = np.array([ref_val])
    elif (n_samples == 0):
        raise ValueError('number of samples must be greater than 0')

    return np.sort(samples)

# def sample_with_fixed_median(left, median, right):
#     """
#     Draw a random sample from a distribution with bounded support
#     and a fixed median.

#     Parameters
#     ----------
#     left : `float`
#         Left endpoint of the distribution's support.
#     median : `float`
#         Median of the distribution.
#     right : `float`
#         Right endpoint of the distribution's support.

#     Returns
#     -------
#     sample : `float`
#         A sample drawn from a distribution with the desired median
#         and bounded support.
#     """
#     beta_median = (median - left)/(right-left)
#     a_min = 1
#     a = 2
#     a_max = 3
#     b = (3*a - 1 + (2-3*a)*beta_median)/beta_median

#     # # optimize the parameters
#     # for i in range(10):
#     #     b = (3*a - 1 + (2-3*a)*beta_median)/beta_median
#     #     quantile_error = (stats.beta.ppf(0.84,a,b) 
#     #                         - stats.beta.ppf(0.16,a,b)
#     #                         - 0.68)
#     #     # print('median=',beta_median)
#     #     # print('b=',b)
#     #     # print('q1',stats.beta.ppf(0.84,a,b))
#     #     # print('q2',stats.beta.ppf(0.16,a,b))
#     #     if i==0: print('error',quantile_error)

#     #     if (quantile_error > 0): # too much spread, increase a
#     #         a_min = a
#     #         a = (a + a_max)/2
#     #     elif (quantile_error < 0): # too little spread, decrease a
#     #         a_max = a
#     #         a = (a + a_min)/2

#     # print(quantile_error)

#     return left + (config._rng.beta(a=a,b=b) * (right - left))
