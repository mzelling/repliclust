"""
This module provides a class for sampling cluster centers uniformly
at random.
"""

import numpy as np

from repliclust import config

from repliclust.base import ClusterCenterSampler
from repliclust.utils import sample_unit_vectors
from repliclust.utils import log_volume, radius_from_log_volume


def validate_packing(packing: float):
    """ Validate a proposed value for the packing parameter. """
    if packing <= 0:
        raise ValueError('packing must be greater than zero.')
    else:
        return packing

def adjusted_log_packing(packing: float, dim: int):
    """
    Adjust the volume density of clusters within a bounded region for 
    higher dimensions, based on the density in 2D.

    Parameters
    ----------
    packing : float, >0
        The volume density of clusters within a bounded region in 2D.
    dim : int, >=2
        The actual number of dimensions.

    Returns
    -------
    log_packing : float, >0
        The logarithm of the adjusted cluster density.

    Notes
    -----
    This function is based on Ball's lower bound for the maximum density
    of a hard sphere packing in high dimensions.
    """
    validate_packing(packing)
    if dim < 2:
        raise ValueError('number of dimensions must be at least 2.')
    return np.log(packing) + np.log(dim-1) - (dim-2)*np.log(2)


class RandomCenters(ClusterCenterSampler):
    """
    Sample cluster centers uniformly at random within a bounded region.

    Parameters
    ----------
    packing : float, >0
        The volume density of clusters within the sampling box. Can be
        greater than 1, but then clusters are guaranteed to overlap.
    
    Methods
    -------
    sample_cluster_centers(archetype)
    """

    def __init__(self, packing=0.1):
        """
        Instantiate a RandomCenters object.
        """
        self.packing = validate_packing(packing)

    def sample_cluster_centers(
            self, archetype
        ):
        """
        Sample cluster centers uniformly at random within a bounded
        region.

        Parameters
        ----------
        archetype : Archetype
            A archetype for a mixture model.

        Returns
        -------
        centers : ndarray
            Cluster centers arranged as a matrix. Each row is a center.
        """
        k = archetype.n_clusters
        p = archetype.dim
        
        log_sampling_volume = (
            log_volume(archetype.scale, p)  # volume of single cluster
                + np.log(k)  # multiply by number of clusters
                - adjusted_log_packing(self.packing, p) # div by density
            )
        sampling_radius = radius_from_log_volume(log_sampling_volume, p)

        directions = sample_unit_vectors(k,p)
        radii = config._rng.uniform(low=0, high=sampling_radius, size=k)

        return directions * radii[:, np.newaxis]