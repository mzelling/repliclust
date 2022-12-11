""" 
This module provides utility functions for max-min sampling.
"""

import numpy as np
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
                                                        left=min_val, 
                                                        mode=ref_val, 
                                                        right=max_val
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