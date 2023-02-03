"""
This module implements a ClusterCenterSampler based on achieving the
desired degree of pairwise overlap between clusters by minimizing an
objective function.
"""

import numpy as np
from warnings import warn
from scipy.stats import norm
from tqdm import tqdm
from repliclust import config
from repliclust.base import ClusterCenterSampler
from repliclust.utils import assemble_covariance_matrix
from repliclust.random_centers import RandomCenters
from repliclust.overlap import _gradients


def overlap2quantile_vec(overlaps):
    """ Convert overlaps to the corresponding quantiles. """
    return norm.ppf(1-(overlaps/2))


def quantile2overlap_vec(quantiles):
    """ Convert quantiles to the corresponding overlaps. """
    return 2*(1-norm.cdf(quantiles))


def assess_obs_overlap(centers, cov_list, ave_cov_inv_list, mode='lda'):
    """
    Compute the observed minimum and maximum overlap between cluster
    centers.

    Parameters
    ----------
    centers : ndarray
        The cluster centers arranged as a matrix. Each row is a center.
    cov_list : list[ndarray]
        A list whose i-th entry is the covariance matrix of the i-th
        cluster.
    ave_cov_inv_list : list[ndarray]
        A list with k*(k-1)/2 entries that gives the inverses of the
        average covariance matrices between each distinct pair of 
        clusters.
    mode : {'lda', 'c2c'}
        The method for calculating cluster overlap.
    """
    obs_separation = _gradients.assess_obs_separation(
                            centers, cov_list, ave_cov_inv_list, mode)
    return {'min': quantile2overlap_vec(obs_separation['max']),
            'max': quantile2overlap_vec(obs_separation['min'])}


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
        Minimum degree of overlap each cluster should have with its 
        closest neighbor, measured as a fraction of cluster mass.
    packing : float
        Sets the ratio of total cluster volume to the sampling volume.
        Used when choosing random cluster centers for initializing the 
        optimization.
    learning_rate : float
        The rate at which cluster centers are optimized. If numerical
        instabilities appear, it is recommended to lower this number.
    linear_penalty_weight : float
        The weight for the linear penalty in the overlap loss. If zero,
        the overlap loss carries only a quadratic penalty and 
        minimization cannot make the overlap loss vanish exactly; in
        this case, minimization stops when the overlap loss is almost
        zero (within tolerance `ATOL`).
    overlap_mode : {'lda', 'c2c', 'exact'}
        Method for calculating cluster overlap.
    max_epoch : int
        The maximum number of optimization epochs to run. Increasing
        this number may slow down the optimization.
    n_restarts : int
        Number of times to repeat the optimization, each time with a 
        different random initialization. The final result is the best
        result attained among the `n_restarts` runs.
    ATOL : float
        Absolute numerical tolerance for optimization.
    RTOL : float
        Relative numerical tolerance for optimization.
    """
    def __init__(self, max_overlap=0.1, min_overlap=0.09, 
                    packing=0.1, learning_rate=0.1,
                    linear_penalty_weight=0.5, overlap_mode='lda',
                    max_epoch = 100, n_restarts = 3, ATOL = 1e-10, 
                    RTOL = 1e-3):
        """ Instantiate a ConstrainedOverlapCenters object. """
        # transform overlap constraints into separation quantiles
        if ((min_overlap < max_overlap)
                and (min_overlap > 0) 
                and (max_overlap < 1)):
            self.quantile_bounds = {
                'min': overlap2quantile_vec(max_overlap),
                'max': overlap2quantile_vec(min_overlap)
            }
        else:
            raise ValueError("invalid overlap constraints")

        # validate packing input
        if packing > 0 and packing < 1:
            self.packing = packing
        else: 
            raise ValueError("'packing' must be strictly"
                                + " between 0 and 1.")

        # validate linear penalty weight input
        if linear_penalty_weight >= 0 and linear_penalty_weight <= 1:
            self.linear_penalty_weight = linear_penalty_weight
        else:
            raise ValueError("'linear_penalty_weight' should be between"
                                + " 0 and 1.")
        
        # validate learning rate input
        if learning_rate <= 0: 
            raise ValueError("'learning_rate' must be positive.")
        elif learning_rate > 1:
            print("WARNING: 'learning_rate' exceeds 1!")
            self.learning_rate = learning_rate
        else:
            self.learning_rate = learning_rate

        # validate n_restarts
        if n_restarts >= 1:
            self.n_restarts = int(n_restarts)
        else: 
            raise ValueError("'n_restarts' should be an integer >= 1")

        # validate max_epoch
        if max_epoch >= 1:
            self.max_epoch = int(max_epoch)
        else: 
            raise ValueError("'max_epoch' should be an integer >= 1")

        # validate numerical tolerances
        if ATOL > 0 and RTOL > 0:
            self.ATOL = ATOL; self.RTOL = RTOL
        else:
            raise ValueError("'ATOL' and 'RTOL' should exceed 0")

        # validate overlap mode input
        if overlap_mode in ['lda','c2c', 'exact']:
            self.overlap_mode = overlap_mode
        else:
            raise ValueError("'overlap_mode' invalid")

        self.overlap_bounds = {'min': min_overlap, 'max': max_overlap}

    def _check_for_continuation(self, epoch, loss_curr,
                                loss_prev, grad_size):
        """
        Return TRUE if we should keep optimizing the cluster centers,
        and FALSE if we should stop. 

        The conditions for stopping are:
        1) Loss small enough (exactly zero when penalty is partly 
           linear and within absolute tolerance of zero for purely
           quadratic penalty).
        2) Reached maximum number of epochs.
        3) Norm of gradient is within absolute tolerance of zero.
        4) Make little progress (current loss is has decreased from
           previous loss by less than relative tolerance.)

        Parameters
        ----------
        epoch : int
            The running epoch count.
        loss_curr : float
            The loss obtained in the current iteration.
        loss_prev : float
            The loss obtained in the previous iteration.
        grad_size : float
            The size of the current gradient.

        Returns
        -------
        bool : TRUE if gradient descent should continue; FALSE otherwise.
        """
        # Stop if maximum number of epochs reached.
        if epoch > self.max_epoch-1:
            return False
        # Stop if penalty is partly linear and loss is exactly zero.
        elif self.linear_penalty_weight > 0 and loss_curr == 0:
              return False
        # Stop if penalty is quadratic and loss almost zero.
        elif (np.allclose(self.linear_penalty_weight, 0, atol=self.ATOL)
              and np.allclose(loss_curr, 0, atol=self.ATOL)):
              return False
        # Stop if gradient is too small.
        elif np.allclose(grad_size, 0, atol=self.ATOL):
            return False
        # Stop if progress is too slow.
        elif loss_curr/loss_prev >= 1-self.RTOL:
            return False
        # If none of the stopping conditions apply, return true
        else:
            return True
        

    def _optimize_centers(self, centers, cov_list=None, 
                          ave_cov_inv_list=None, axis_deriv_t_list=None, 
                          quiet=False, progress_bar=None):
        """
        Minimize overlap loss using stochastic gradient descent.

        Parameters
        ----------
        centers : ndarray
            The cluster centers arranged as a matrix. Each row is a center.
        cov_list : list[ndarray]
            A list whose `i`-th component is the covariance matrix of the
            `i`-th cluster center.
        ave_cov_inv_list : list[ndarray]
            A list that stores the inverse of the average covariance
            matrices between each distinct pair of clusters.
        axis_deriv_t_list : list[ndarray]
            A list that stores the transpose of the differential of the
            separation axis axis with respect to reference cluster's
            center, for each distinct pair of clusters.

        Returns
        -------
        loss : float
            The overlap loss attained by the optimized cluster centers.

        Side effects
        ------------
        The `centers` argument is modified in-place. At the end of the
        function call, `centers` stores the optimized cluster centers.
        """
        n_clusters = centers.shape[0]
        dim = centers.shape[1]

        # Initialize parameters for tracking optimization progress
        epoch = 0
        grad_size = np.Inf; loss_curr = np.Inf; loss_prev = np.Inf
        while self._check_for_continuation(epoch, loss_curr, loss_prev,
                                           grad_size):
            if progress_bar: progress_bar.update(n=1)
            epoch_queue= config._rng.permutation(n_clusters)
            grad_size = 0
            for cluster_idx in epoch_queue:
                grad, loss = _gradients.update_centers(
                    cluster_idx, centers, cov_list, 
                    ave_cov_inv_list, axis_deriv_t_list,
                    learning_rate=self.learning_rate,
                    linear_penalty_weight=self.linear_penalty_weight,
                    quantile_bounds=self.quantile_bounds, 
                    mode=self.overlap_mode,
                    )
                grad_size += np.linalg.norm(grad)/n_clusters
            # Increment parameters for tracking the optimization
            epoch += 1
            loss_prev = loss_curr
            loss_curr = loss

        # make progress bar appear to have gone for max_epoch iterations
        if progress_bar: progress_bar.update(n=self.max_epoch-epoch)
        return loss_curr

    
    def sample_cluster_centers(self, archetype, quiet=False):
        """
        Sample cluster centers for the given archetype.
        """
        n_clusters = archetype.n_clusters
        dim = archetype.dim

        # If there is only one cluster, use the origin as the center
        if (n_clusters == 1): return np.zeros(dim)[np.newaxis,:]
        
        # Compute list of covariance matrices
        cov_list = [assemble_covariance_matrix(
                            archetype._axes[i], archetype._lengths[i],
                            inverse=False)
                        for i in range(n_clusters)]

        # Compute method-specific lists of matrices
        if self.overlap_mode=='lda':
            ave_cov_inv_list = [np.linalg.inv((cov_list[i]
                                                + cov_list[j])/2)
                                    for i in range(n_clusters) 
                                        for j in range(i+1, n_clusters)]
            axis_deriv_t_list = ave_cov_inv_list
        elif self.overlap_mode=='c2c':
            ave_cov_inv_list = None
            axis_deriv_t_list = None
        elif self.overlap_mode=='exact':
            raise NotImplementedError("the 'exact' mode has not been" +
                                        " implemented yet.")

        # Repeat optimization from different initializations
        progress_bar = None if quiet else (
                        tqdm(total=int(self.n_restarts*self.max_epoch),
                             desc="Optimizing Cluster Centers")
                        )

        best_loss = np.Inf
        best_centers = None
        for optimization_restart in range(self.n_restarts):
            # Randomly initialize cluster centers
            centers = (RandomCenters(packing=self.packing)
                            .sample_cluster_centers(archetype))
            loss = self._optimize_centers(centers, cov_list, 
                                          ave_cov_inv_list,
                                          axis_deriv_t_list,
                                          progress_bar=progress_bar)
            if loss < best_loss:
                # Must copy current-best centers to prevent overwriting
                best_centers = np.copy(centers)
                best_loss = loss
                if np.allclose(best_loss, 0):
                    if not quiet:
                        progress_bar.update(
                            n=((self.n_restarts-optimization_restart-1)
                                *self.max_epoch)
                            )
                    break

        if np.allclose(best_loss,0):
            if not quiet: progress_bar.set_postfix({"Status": "SUCCESS"})
        else:
            # Assess attained overlaps
            obs_overlap = assess_obs_overlap(best_centers, cov_list, 
                                             ave_cov_inv_list,
                                             self.overlap_mode)
            if not quiet: progress_bar.set_postfix({"Status": "WARNING"})
            print("\tWARNING: Failed to converge!"
                    + " Attained overlaps:" 
                    + " min="
                    + np.format_float_scientific(obs_overlap['min'], 
                                                 precision=2)
                    + ", max="
                    + np.format_float_scientific(obs_overlap['max'],
                                                 precision=2),
            )

        # Shift centers to have zero-mean and return the result
        centers_mean = np.mean(centers, axis=0)
        centers_adjusted = centers - centers_mean[np.newaxis,:]
        return centers_adjusted