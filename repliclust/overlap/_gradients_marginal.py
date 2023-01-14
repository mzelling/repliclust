import numpy as np
from scipy.stats import chi2, norm


def compute_other_cluster_idx(cluster_idx,k):
    """
    Compute other cluster indices.

    Parameters
    ----------
    cluster_idx : int
        Cluster index to exclude.
    k : int
        Number of clusters.

    Returns
    -------
    out : list of int
        All cluster indices except for cluster_idx.
    """
    return [i for i in range(k) if i != cluster_idx]


def make_sd_term(diff_mat_unit, diff_tf_mat_unit):
    return np.sqrt(
                np.sum(diff_mat_unit * diff_tf_mat_unit, axis=0)
            )[np.newaxis,:]


def sd_term(diff_mat, diff_mat_tf):
    return np.sqrt(np.sum(diff_mat * diff_mat_tf,axis=0)[np.newaxis,:])


def diff_normsq_vec(diff_mat):
    return np.sum(diff_mat**2,axis=0)[np.newaxis,:]


def inv_quadratic_term(sd_sum):
    return -1/(sd_sum**2)


def summand_term(sd_term, diff_mat_tf):
    return (1/2) * (1/sd_term) * diff_mat_tf


def marginal_gradient_vec(diff_mat, diff_tf_mat_1, diff_tf_mat_2):
    """
    Compute the gradients.
    """
    sd_term_1 = sd_term(diff_mat, diff_tf_mat_1)
    sd_term_2 = sd_term(diff_mat, diff_tf_mat_2)
    sd_sum = sd_term_1 + sd_term_2

    return ((2 * diff_mat / sd_sum)
                + (diff_normsq_vec(diff_mat) 
                    * inv_quadratic_term(sd_sum)
                    * (summand_term(sd_term_1, diff_tf_mat_1) 
                        + summand_term(sd_term_2, diff_tf_mat_2))))


def make_marginal_args(cluster_idx, centers, cov):
    """
    Make the matrix of differences between centers, as well as the
    transformed differences, but use covariance matrices rather than
    inverse covariance matrices for the transformation.
    """
    other_cluster_idx = compute_other_cluster_idx(
                            cluster_idx, centers.shape[0])
    diff_mat = np.transpose(centers[cluster_idx,:][np.newaxis,:]
                                - centers[other_cluster_idx,:])
    assert(diff_mat.shape == (centers.shape[1],centers.shape[0]-1))

    diff_tf_mat_1 = cov[cluster_idx] @ diff_mat
    diff_tf_mat_2 = np.concatenate(
        list(map(lambda i: (cov[other_cluster_idx[i]] \
            @ diff_mat[:,i])[:,np.newaxis], 
            range(len(other_cluster_idx)))), axis=1)

    return {'diff_mat': diff_mat, 
            'diff_tf_mat_1': diff_tf_mat_1, 
            'diff_tf_mat_2': diff_tf_mat_2}


def make_quantile_vec(diff_mat, diff_tf_mat_1, diff_tf_mat_2):
    """ Make quantile for marginal overlap. """
    return (diff_normsq_vec(diff_mat) / 
                (sd_term(diff_mat, diff_tf_mat_1)
                 + sd_term(diff_mat, diff_tf_mat_2)))


def overlap2quantile_vec(overlaps):
    """ Convert overlaps to the corresponding quantiles. """
    return norm.ppf(1-(overlaps/2))


def quantile2overlap_vec(quantiles):
    """ Convert quantiles to the corresponding overlaps. """
    return 2*(1-norm.cdf(quantiles))


def update_centers(cluster_idx, centers, cov, learning_rate, 
                    overlap_bounds):
    """
    Perform an iteration of stochastic gradient descent on the cluster
    centers. 

    Parameters
    ----------
    cluster_idx : int
        Index of reference cluster (for stochastic gradient descent).
    centers : ndarray, shape (k, p)
        Matrix of all cluster centers. Each row is a center.
    cov : list of ndarray; length k, each ndarray of shape (p, p)
        List of covariance matrices.
    learning_rate : float
        Learning rate for gradient descent.
    overlap_bounds : dict with keys 'min' and 'max'
        Minimum and maximum allowed overlaps between clusters.
    cov : list of ndarray; length k, each ndarray of shape (p, p)
        List of covariance matrices.

    Side effects
    ------------
    Update centers by taking a stochastic gradient descent step.
    """
    # Make reusable arguments for the subsequent steps
    p = centers.shape[1]
    k = centers.shape[0]

    other_cluster_idx = compute_other_cluster_idx(cluster_idx,k)
    marginal_args = make_marginal_args(
                        cluster_idx, centers, cov
                    ) 
    quantiles = make_quantile_vec(**marginal_args)
    #print("quantiles", quantiles)
    overlaps = quantile2overlap_vec(quantiles)
    #print("overlaps", overlaps)

    # See if any clusters repel the reference cluster
    repel_mask = (overlaps >= overlap_bounds['max']).flatten()
    if np.any(repel_mask):
        # Select only the clusters that repel the reference cluster
        grad_args = {X_name: X[:,repel_mask] for X_name, X in \
                     marginal_args.items()}
        gradients = marginal_gradient_vec(**grad_args)
        #print('gradients', gradients)
        q_min = overlap2quantile_vec(overlap_bounds['max'])
        q = quantiles[:,repel_mask]
        #MSE_grad = -(H_vec(q_min - q) + 2*ReLU_vec(q_min - q))*gradients
        MSE_grad = -(2*ReLU_vec(q_min - q))*gradients
        #print('MSE grad', MSE_grad)
        # Update centers matrix with a gradient step
        repel_idx = np.array(other_cluster_idx)[repel_mask]
        centers[cluster_idx,:] -= (learning_rate 
                                    * (np.sum(MSE_grad, axis=1)/(k-1)))
        centers[repel_idx,:] -= (learning_rate
                                    * np.transpose(-MSE_grad))
        return "status: moving clusters farther away from each other"

    elif np.max(overlaps) <= overlap_bounds['min']:
        # Select only the cluster closest to the reference cluster
        attract_pre_idx = np.argmax(overlaps)
        grad_args = {X_name: X[:,[attract_pre_idx]] for X_name, X in \
                     marginal_args.items()}
        gradients = marginal_gradient_vec(**grad_args)
        #print('gradients', gradients)
        q_max = overlap2quantile_vec(overlap_bounds['min'])
        q = quantiles[:,[attract_pre_idx]]
        #MSE_grad = (H_vec(q - q_max) + 2*ReLU_vec(q - q_max))*gradients
        MSE_grad = (2*ReLU_vec(q - q_max))*gradients
        #print('MSE_grad', MSE_grad)
        # Update centers matrix with a gradient step
        attract_idx = other_cluster_idx[attract_pre_idx]
        centers[cluster_idx,:] -= learning_rate * MSE_grad.flatten()
        centers[attract_idx,:] -= learning_rate * (-MSE_grad).flatten()
        return "status: moving clusters closer to each other"

    else:
        return "status: doing nothing because overlap constraints are satisfied"


def ReLU_vec(x):
    """ Apply rectified linear unit x+ = max(x,0). """
    return np.maximum(x,0)
        

def poly_vec(x):
    """ Apply the polynomial p(x) = x + x**2. """
    return x + (x**2)


def H_vec(x):
    """ Apply the step (Heaviside) function. """
    return (np.sign(x) + 1)/2


def single_cluster_loss(cluster_idx, centers, cov, overlap_bounds):
    """
    Compute the marginal overlap loss for a reference cluster.
    """
    quantiles = make_quantile_vec(
                    **make_marginal_args(cluster_idx, centers, cov)
                    )
    overlaps = quantile2overlap_vec(quantiles)
    if (np.max(overlaps) > overlap_bounds['min']):
        # return the loss for repulsion
        q_min = overlap2quantile_vec(overlap_bounds['max'])
        return np.sum(
             poly_vec(ReLU_vec(q_min - quantiles))
            )
    else:
        # return the loss for attraction
        q_max = overlap2quantile_vec(overlap_bounds['min'])
        return poly_vec(ReLU_vec(np.min(quantiles) - q_max))


def overlap_loss(centers, cov, overlap_bounds):
    """
    Compute the total overlap loss.
    """
    k = centers.shape[0]
    return np.sum(list(
            map(lambda i: single_cluster_loss(
                            i, centers, cov, overlap_bounds
                          )/k, 
                range(k))))


def assess_obs_overlap(centers, cov):
    """
    Compute the observed overlap between cluster centers.
    """
    k = centers.shape[0]
    args_list = [make_marginal_args(i, centers, cov) for i in range(k)]
    max_overlaps = np.array([ np.max(quantile2overlap_vec(
                                make_quantile_vec(**args)))
                             for args in args_list ])
    return {'min': np.min(max_overlaps),
            'max': np.max(max_overlaps)}
