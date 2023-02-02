"""
This module implements a ClusterCenterSampler based on achieving the
desired degree of pairwise overlap between clusters by minimizing an
objective function.
"""

import numpy as np
from scipy.stats import norm

from repliclust import config
from repliclust.base import ClusterCenterSampler
from repliclust.utils import assemble_covariance_matrix
from repliclust.random_centers import RandomCenters
from repliclust.overlap import _gradients_marginal as _gradients
from repliclust.overlap import _gradients_marginal_lda as _gradients_lda


def overlap2quantile_vec(overlaps):
    """ Convert overlaps to the corresponding quantiles. """
    return norm.ppf(1-(overlaps/2))


def quantile2overlap_vec(quantiles):
    """ Convert quantiles to the corresponding overlaps. """
    return 2*(1-norm.cdf(quantiles))


class ConstrainedOverlapCenters(ClusterCenterSampler):
    """
    """
    def __init__(self, max_overlap=0.1, min_overlap=0.09, 
                    mode='lda', packing=0.1, learning_rate=0.1):
        """ Instantiate a ConstrainedOverlapCenters object. """
        self.mode = mode
        self.packing = packing
        self.overlap_bounds = {'min': min_overlap, 'max': max_overlap}

    def _optimize_centers(self, centers, cov, 
                          max_epoch=500, learning_rate=0.5, tol=1e-5,
                          verbose=False, quiet=False):
        
        epoch_count = 0
        loss = np.Inf
        while check_for_continuation():
            epoch_order = config._rng.permutation(centers.shape[0])
            for cluster_idx in epoch_order:
                _gradients.update_centers(
                    cluster_idx, centers, cov, 
                    learning_rate=learning_rate, 
                    overlap_bounds=self.overlap_bounds,
                )
            epoch_count += 1
            loss = _gradients.overlap_loss(centers, cov,
                                           self.overlap_bounds)

        




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

    
    def _print_optimization_victory(self, epoch_count: int, 
                                    pad_epoch: int, loss: float):
        """ Print result of successfully optimizing cluster centers. """
        pad_epoch_actual = int(np.floor(
                                1+np.log10(epoch_count)))
        print("\n[====== completed in "
                + str(epoch_count)
                + " "
                + ("epochs " if epoch_count > 1 else "epoch =")
                + ("="*(4+(pad_epoch-pad_epoch_actual)))
                + "]\n")

    def _print_optimization_defeat(self, epoch_count: int, 
                                   pad_epoch: int, loss: float):
        """ Print warning of failure to optimize cluster centers. """
        pad_epoch_actual = int(np.floor(
                                1+np.log10(epoch_count)))
        print("\n[=== unfinished after "
                + str(epoch_count)
                + " "
                + ("epochs " if epoch_count > 1 else "epoch =")
                + ("="*(pad_epoch-pad_epoch_actual))
                + "===]\n")
        print("\t^ LOSS = " + str(loss))


    def _optimize_centers(self, centers, cov, 
                          max_epoch=500, learning_rate=0.5, tol=1e-5,
                          verbose=False, quiet=False):
        """
        Optimize the cluster centers to achieve the desired overlaps.

        Parameters
        ----------
        centers : ndarray
            The initial centers. Each row is a center. The i-th row
            gives the coordinates of the i-th center.
        cov : list of ndarray
            The covariance matrices of the clusters. The i-th
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
            If true, print step-by-step progress updates during
            optimization. Even if ``verbose=False``, will still print
            summary of optimization status unless ``quiet=True``.
        quiet : bool
            If true, suppress all print output.

        Returns
        -------
        centers : ndarray
            The optimized centers.
        """
        print('eta=',learning_rate)
        if not quiet: print("\n[=== optimizing cluster overlaps ===]\n")
        pad_epoch = int(np.maximum(2, np.floor(1+np.log10(max_epoch))))

        epoch_count = 0
        keep_optimizing = (epoch_count < max_epoch)
        while keep_optimizing:
            epoch_order = config._rng.permutation(centers.shape[0])
            for i in epoch_order:
                _gradients.update_centers(
                    i, centers, cov, 
                    learning_rate=learning_rate, 
                    overlap_bounds=self.overlap_bounds,
                    )
            epoch_count += 1
            keep_optimizing = (epoch_count < max_epoch)
            
            loss = _gradients.overlap_loss(centers, cov,
                                           self.overlap_bounds)
            if verbose:
                self._print_optimization_progress(
                        epoch_count, max_epoch, pad_epoch, loss)
            if np.allclose(loss, 0, atol=tol):
                if not verbose and not quiet:
                    print(" "*17 + "...")
                if not quiet:
                    self._print_optimization_victory(epoch_count,
                                                     pad_epoch, loss)
                return centers

            if not keep_optimizing and not quiet:
                print(" "*17 + "...")
                self._print_optimization_defeat(epoch_count, pad_epoch,
                                                loss)
            
        return (centers, loss)
        

    def sample_cluster_centers(self, archetype, 
                               n_random_restarts=3,
                               print_progress=False,
                               quiet=False):
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
            If true, print step-by-step progress updates during the
            optimization. Even if ``print_progress=False``, will still
            print a summary of the optimization status unless
            ``quiet=True``.

        quiet : bool
            If true, suppress all print output.

        Returns
        -------
        centers : ndarray
            The optimized cluster centers.
        """
        if (archetype.n_clusters == 1):
            return np.zeros(archetype.dim)[np.newaxis,:]

        cov = [assemble_covariance_matrix(
                    archetype._axes[i], archetype._lengths[i],
                        inverse=False)
                    for i in range(archetype.n_clusters)]

        best_centers = None
        best_loss = np.Inf
        best_center_idx = None
        # Restart optimization several times and take best result
        print("\nINITIATING RESTARTS...")
        for random_restart in range(n_random_restarts):
            init_centers = (RandomCenters(packing=self.packing)
                                .sample_cluster_centers(archetype))
            centers, loss = self._optimize_centers(init_centers, cov,
                                verbose=False,
                                quiet=True,
                                **self.optimization_args)
            print("\tRestart " + str(random_restart) 
                    + ": loss=" + str(loss))
            if loss < best_loss:
                best_centers = centers
                best_loss = loss

        print("BEST RESTART: loss=" + str(best_loss))

        return best_centers


class LDAConstrainedOverlapCenters(ConstrainedOverlapCenters):
    """
    More precise but slower implementation.
    """

    def _optimize_centers(self, centers, cov, 
                          max_epoch=100, learning_rate=0.1,
                          penalty_coef=0.5, 
                          atol=1e-6, rtol=1e-6,
                          verbose=False, quiet=False):
        # compute the inverses of average pairwise covariances
        if not quiet:
            print("\nOptimizing cluster centers (learning rate="
                    + str(np.round(learning_rate,3)) + ", mode='precise') ...")
            print("Inverting covariance matrices " +
                  "(if this takes too long, set mode='fast') ...")
        k = centers.shape[0]
        ave_cov_inv = [np.linalg.inv((cov[i] + cov[j])/2)
                       for i in range(k) for j in range(i+1,k)]

        # print status update that optimization is starting
        if not quiet: print("\n[=== optimizing cluster overlaps ===]\n")
        pad_epoch = int(np.maximum(2, np.floor(1+np.log10(max_epoch))))

        epoch_count = 0
        keep_optimizing = (epoch_count < max_epoch)
        while keep_optimizing:
            epoch_order = config._rng.permutation(centers.shape[0])
            for i in epoch_order:
                _gradients_lda.update_centers(
                    i, centers, cov, ave_cov_inv,
                    learning_rate=learning_rate, 
                    penalty_coef=penalty_coef,
                    overlap_bounds=self.overlap_bounds,
                    )
            epoch_count += 1
            keep_optimizing = (epoch_count < max_epoch)
            
            loss = _gradients_lda.overlap_loss(
                        centers, cov, ave_cov_inv, self.overlap_bounds)

            if verbose:
                self._print_optimization_progress(
                        epoch_count, max_epoch, pad_epoch, loss)
            if np.allclose(loss, 0, atol=atol, rtol=rtol):
                if not verbose and not quiet:
                    print(" "*17 + "...")
                if not quiet:
                    self._print_optimization_victory(epoch_count,
                                                     pad_epoch, loss)
                return (centers, loss)

            if not keep_optimizing and not quiet:
                print(" "*17 + "...")
                self._print_optimization_defeat(epoch_count, pad_epoch,
                                                loss)
            
        return (centers, loss)