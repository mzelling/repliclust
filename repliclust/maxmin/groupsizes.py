"""
This module provides functionality for sampling the number of data
points in each cluster using a max-min approach.
"""

import numpy as np

from repliclust.maxmin.utils import sample_with_maxmin
from repliclust.base import GroupSizeSampler


def _float_to_int(groupsizes_float, total):
    """
    Convert fractional group sizes to integers while ensuring
    1) that each class size is >= 1 and 2) that all group sizes
    sum to the desired total.

    Parameters
    ----------
    groupsizes_float : ndarray, dtype=float
        Fractional group sizes that we wish to turn into integers.
    total : int
        The desired sum total of the group sizes.

    Returns
    -------
    out : ndarray, dtype=int
        Integral group sizes.
    """
    if len(groupsizes_float) > total:
        raise ValueError('number of group sizes must not exceed'
                + ' the provided total (' + str(total) + ').')
    elif np.any(groupsizes_float <= 0):
        raise ValueError('fractional group sizes must be strictly'
                + ' positive.')
    elif total < 1:
        raise ValueError('total must be >= 1.')

    # Add 1 to the rounded fractional group sizes, then sort.
    class_sz = (np.max([1, int(total - np.sum(groupsizes_float))]) 
                +  np.sort(np.round(groupsizes_float)))
    # Reduce group sizes until we hit the desired total.
    # Decrease the largest group sizes first.
    class2shrink_idx = len(class_sz) - 1
    while (np.sum(class_sz) > total):
        if (class_sz[class2shrink_idx] > 1):
            class_sz[class2shrink_idx] -= 1
            class2shrink_idx -= 1
        else:
            class2shrink_idx -= 1
        # Start from the beginning again.
        if (class2shrink_idx == -1):
            class2shrink_idx = len(class_sz) - 1
    return class_sz.astype(int)


class MaxMinGroupSizeSampler(GroupSizeSampler):
    """
    Sample the number of data points in each cluster using pairwise 
    max-min sampling.

    Attributes
    ----------
    imbalance_ratio : float, >=1
        The desired ratio between largest and smallest group size.

    Methods
    -------
    __init__(self, imbalance_ratio)
    make_group_sizes(self, clusterdata)
    """

    def __init__(self, imbalance_ratio=2):
        """ Instantiate a MaxMinGroupSizeSampler object. """
        if (imbalance_ratio < 1):
            raise ValueError('Imbalance ratio must be >=1')

        self.imbalance_ratio = imbalance_ratio


    def sample_group_sizes(self, archetype, total):
        """
        Sample the number of data points for each cluster using 
        pairwise max-min sampling. 

        Parameters
        ----------
        archetype : Blueprint
            Blueprint for a mixture model.
        total : int
            The total number of data points (sum of group sizes).

        Returns
        -------
        group_sizes : ndarray
            The number of data points for each cluster.

        """
        n_clusters = archetype.n_clusters

        if ((not isinstance(total, int)) 
                or (total <= 0)
                or (n_clusters <= 0)):
            raise ValueError('the number of samples and the number of'
                    + ' clusters must be positive integers.')
        elif n_clusters > total:
            raise ValueError('Number of clusters must not exceed number'
                    + ' of samples')

        # Set reference group size to be the average group size.
        ref_group_sz = total/n_clusters
        # Require that ref size equals avg of min and max group sizes.
        min_group_sz = 2*ref_group_sz/(1 + self.imbalance_ratio)
        # Constraint function ensures group sizes sum to n_samples.
        f = lambda s: (2*ref_group_sz - s)
        # Compute fractional group sizes.
        groupsizes_float = sample_with_maxmin(
                            n_clusters, ref_group_sz, min_group_sz, 
                            self.imbalance_ratio, f
                            )
        # Convert fractional group sizes into integers and return.
        return _float_to_int(groupsizes_float, total)