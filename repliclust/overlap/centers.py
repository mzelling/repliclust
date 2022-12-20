"""
This module implements a ClusterCenterSampler based on achieving the
desired degree of pairwise overlap between clusters by minimizing an
objective function.
"""

import numpy as np

from repliclust import config
from repliclust.base import ClusterCenterSampler
from repliclust.utils import assemble_covariance_matrix
from repliclust.random_centers import RandomCenters
from repliclust.overlap import gradients

class ConstrainedOverlapCenters(ClusterCenterSampler):
    """
    This class provides an implementation for optimizing the location
    of cluster centers to achieve the desired degrees of overlap between
    pairs of clusters.

    Parameters
    ----------
    max_overlap : float between 0 and 1
        The maximum allowed overlap between two cluster centers, 
        measured as a fraction of cluster mass.
    min_overlap : float
        The minimum overlap each cluster needs to have with some other
        cluster, preventing it to be isolated. The overlap is measured
        as a fraction of cluster mass.
    packing : float
        Sets the ratio of total cluster volume to the sampling volume.
        Used when choosing random cluster centers for initializing the 
        optimization.
    learning_rate : float
        The rate at which cluster centers are optimized. If numerical
        instabilities appear, it is recommended to lower this number.
    max_epoch : int
        The maximum number of optimization epochs to run. Increasing
        this number may slow down the optimization.
    tol : float
        Numerical tolerance for achieving the desired overlap
        between pairs of clusters.
    """
    
    def __init__(self, max_overlap=0.1, min_overlap=0.09, packing=0.1, 
                    **optimization_args):
        """ Instantiate a ConstrainedOverlapCenters object. """
        self.max_overlap = max_overlap
        self.min_overlap = min_overlap
        self.packing = packing
        self.overlap_bounds = {'min': min_overlap, 'max': max_overlap}
        self.optimization_args = optimization_args


    def _print_optimization_progress(self, epoch_count: int, 
                                     max_epoch: int, pad_epoch: int, 
                                     loss: float):
        """ Print progress update while optimizing cluster centers. """
        print(" epoch " 
                + str(epoch_count).rjust(pad_epoch, " ")
                + "/" + str(max_epoch)
                + " ... loss ->"
                + np.format_float_scientific(
                    loss, precision=3).rjust(10, " "))

    
    def _print_optimization_result(self, epoch_count: int, 
                                   pad_epoch: int):
        """ Print the result of optimizing cluster centers. """
        pad_epoch_actual = int(np.floor(
                                1+np.log10(epoch_count)))
        print("\n[====== completed in "
                + str(epoch_count)
                + " "
                + ("epochs " if epoch_count > 1 else "epoch =")
                + ("="*(4+(pad_epoch-pad_epoch_actual)))
                + "]\n")


    def _optimize_centers(self, centers, cov_inv, 
                          max_epoch=100, learning_rate=0.1, tol=1e-10,
                          verbose=False):
        """
        Optimize the cluster centers to achieve the desired overlaps.

        Parameters
        ----------
        centers : ndarray
            The initial centers. Each row is a center. The i-th row
            gives the coordinates of the i-th center.
        cov_inv : list of ndarray
            The inverse covariance matrices of the clusters. The i-th
            element is the inverse covariance matrix of the i-th
            cluster.
        max_epoch : int
            The maximum number of epochs during optimization.
        learning_rate : float
            The learning rate to use during optimization. It is 
            recommended to to lower this value if optimization
            encounters numerical instability; conversely, if
            optimization progresses too slowly, it is recommended to
            increase this value.
        tol : float
            Numerical tolerance for achieving the desired overlap
            between pairs of clusters.
        verbose : bool
            If true, print the progress during optimization.

        Returns
        -------
        centers : ndarray
            The optimized centers.
        """
        print("\n[=== optimizing cluster overlaps ===]\n")
        pad_epoch = int(np.maximum(2, np.floor(1+np.log10(max_epoch))))

        epoch_count = 0
        keep_optimizing = (epoch_count < max_epoch)
        while keep_optimizing:
            epoch_order = config._rng.permutation(centers.shape[0])
            for i in epoch_order:
                gradients.update_centers(
                    i, centers, cov_inv, 
                    learning_rate=learning_rate, 
                    overlap_bounds=self.overlap_bounds
                    )
            epoch_count += 1
            keep_optimizing = (epoch_count < max_epoch)
            
            loss = gradients.total_loss(centers, cov_inv,
                                        self.overlap_bounds)
            if verbose:
                self._print_optimization_progress(
                        epoch_count, max_epoch, pad_epoch, loss)
            if np.allclose(loss, 0):
                if not verbose: print(" "*17 + "...")
                self._print_optimization_result(epoch_count, pad_epoch)
                return centers
            
        return centers
        

    def sample_cluster_centers(self, archetype, print_progress=False):
        """
        Sample cluster centers at random and iteratively adjust them
        until the desired degrees of overlap between clusters are
        satisfied.

        Parameters
        ----------
        archetype : Archetype
            Archetype conveying the desired number of clusters and other
            attributes.

        print_progress : bool
            If true, print the progress during optimization.

        Returns
        -------
        centers : ndarray
            The optimized cluster centers.
        """
        if (archetype.n_clusters == 1):
            return np.zeros(archetype.dim)[np.newaxis,:]

        cov_inv = [assemble_covariance_matrix(
                        archetype._axes[i], archetype._lengths[i],
                            inverse=True)
                        for i in range(archetype.n_clusters)]
        init_centers = (RandomCenters(packing=self.packing)
                            .sample_cluster_centers(archetype))
        centers = self._optimize_centers(init_centers, cov_inv,
                                         verbose=print_progress, 
                                         **self.optimization_args)
        return centers

