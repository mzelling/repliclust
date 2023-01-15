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


def l_dot_diff_vec(l_mat, diff_mat):
    """ BESPOKE """
    return np.sum(l_mat * diff_mat,axis=0)[np.newaxis,:]


def inv_quadratic_term(sd_sum):
    """ APPROVED """
    return -1/(sd_sum**2)


def summand_term(sd_term, diff_mat_tf):
    """ APPROVED """
    return (1/2) * (1/sd_term) * diff_mat_tf


def sd_term(diff_mat, diff_mat_tf):
    """ APPROVED """
    return np.sqrt(np.sum(diff_mat * diff_mat_tf,axis=0)[np.newaxis,:])


def marginal_gradient_vec(diff_mat, l_mat, l_tf_mat_1, l_tf_mat_2,
                          l_tf_plus_mat_1, l_tf_plus_mat_2):
    """
    diff_mat
    l_mat : 
    l_tf_mat_1 : 
    l_tf_mat_2 : 
    l_tf_plus_mat_1 : 
    l_tf_plus_mat_2 : 
    """
    l_sd_term_1 = sd_term(l_mat, l_tf_mat_1)
    l_sd_term_2 = sd_term(l_mat, l_tf_mat_2)
    l_sd_sum = l_sd_term_1 + l_sd_term_2

    first_term = 2 * l_mat / l_sd_sum
    second_term = (l_dot_diff_vec(l_mat, diff_mat)
                    * (inv_quadratic_term(l_sd_sum)
                        * (summand_term(l_sd_term_1, l_tf_plus_mat_1) 
                           + summand_term(l_sd_term_2, l_tf_plus_mat_2))
                      )
                  )
    return first_term + second_term

def linearized_idx(i,j,k):
    """
    Linearize the index assuming i < j.
    """
    assert i < j
    return int(i*(k-1) - ((i-1)*i/2) + (j-i-1))


def get_1d_idx(i,j_vec,k):
    """
    Compute the one-dimensional index corresponding to cluster i and
    cluster j. The j indices are a vector, and the output is vectorized.

    The inputs i, j_vec are numbers from 0 to k-1 as in a k-by-k matrix.
    The output is the linear index when saving the k-k matrix (with the
    diagonal removed) as a vector by sweeping across columns (j) for
    increasing row (i) index.
    """
    return [linearized_idx(i,j,k) if i<j else linearized_idx(j,i,k)
            for j in j_vec]


def matvecprod_vectorized(matrix_list, matrix_col_idx, vectors_mat):
    """
    matrix_list : list of (p,p) matrices
    matrix_col_idx : relates column indices of vectors_mat to the
                    indices of matrix_list
    vectors_mat : shape (p,k-1)
    """
    return np.concatenate(
                list(map(lambda j: (matrix_list[matrix_col_idx[j]] 
                                @ vectors_mat[:,j])[:,np.newaxis],
                    range(len(matrix_col_idx)))), 
                axis=1
            )


def make_marginal_args(i, centers, cov, ave_cov_inv):
    """
    Make the matrix of differences between centers, as well as the
    transformed differences, but use covariance matrices rather than
    inverse covariance matrices for the transformation.
    """
    k = centers.shape[0]
    other_cluster_idx = compute_other_cluster_idx(i, k)

    # compute the differences between cluster centers
    diff_mat = np.transpose(centers[i,:][np.newaxis,:]
                                - centers[other_cluster_idx,:])

    # compute the matrix of LDA axes
    l_mat = matvecprod_vectorized(ave_cov_inv, 
                                  get_1d_idx(i,other_cluster_idx,k),
                                  diff_mat)

    # compute other matrices needed for the gradient
    l_tf_mat_1 = cov[i] @ l_mat
    l_tf_mat_2 = matvecprod_vectorized(cov, other_cluster_idx, l_mat)
    l_tf_plus_mat_1 = matvecprod_vectorized(
                        ave_cov_inv, 
                        get_1d_idx(i,other_cluster_idx,k),
                        l_tf_mat_1
                        )
    l_tf_plus_mat_2 = matvecprod_vectorized(
                        ave_cov_inv,
                        get_1d_idx(i,other_cluster_idx,k),
                        l_tf_mat_2)

    return {'diff_mat': diff_mat, 
            'l_mat': l_mat,
            'l_tf_mat_1': l_tf_mat_1, 
            'l_tf_mat_2': l_tf_mat_2,
            'l_tf_plus_mat_1': l_tf_plus_mat_1,
            'l_tf_plus_mat_2': l_tf_plus_mat_2}


def make_quantile_vec(diff_mat, l_mat, l_tf_mat_1, l_tf_mat_2, 
                      l_tf_plus_mat_1=None, l_tf_plus_mat_2=None):
    """ Make quantile for LDA-based marginal overlap. """
    return (np.sum(l_mat * diff_mat,axis=0)[np.newaxis,:] / 
                (sd_term(l_mat, l_tf_mat_1)
                 + sd_term(l_mat, l_tf_mat_2)))


def overlap2quantile_vec(overlaps):
    """ Convert overlaps to the corresponding quantiles. """
    return norm.ppf(1-(overlaps/2))


def quantile2overlap_vec(quantiles):
    """ Convert quantiles to the corresponding overlaps. """
    return 2*(1-norm.cdf(quantiles))


def update_centers(cluster_idx, centers, cov, ave_cov_inv, 
                    learning_rate, overlap_bounds):
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
    ave_cov_inv : list of ndarray; length k*(k-1)/2
        List of inverses of the pairwise average covariance matrices.
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
                        cluster_idx, centers, cov, ave_cov_inv
                    ) 
    quantiles = make_quantile_vec(**marginal_args)
    overlaps = quantile2overlap_vec(quantiles)

    # See if any clusters repel the reference cluster
    repel_mask = (overlaps >= overlap_bounds['max']).flatten()
    if np.any(repel_mask):
        # Select only the clusters that repel the reference cluster
        grad_args = {X_name: X[:,repel_mask] for X_name, X in \
                     marginal_args.items()}
        gradients = marginal_gradient_vec(**grad_args)
        q_min = overlap2quantile_vec(overlap_bounds['max'])
        q = quantiles[:,repel_mask]
        MSE_grad = -(H_vec(q_min - q) + 2*ReLU_vec(q_min - q))*gradients
        #MSE_grad = -(2*ReLU_vec(q_min - q))*gradients

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
        q_max = overlap2quantile_vec(overlap_bounds['min'])
        q = quantiles[:,[attract_pre_idx]]
        MSE_grad = (H_vec(q - q_max) + 2*ReLU_vec(q - q_max))*gradients
        #MSE_grad = (2*ReLU_vec(q - q_max))*gradients

        # Update centers matrix with a gradient step
        attract_idx = other_cluster_idx[attract_pre_idx]
        centers[cluster_idx,:] -= learning_rate * MSE_grad.flatten()
        centers[attract_idx,:] -= learning_rate * (-MSE_grad).flatten()
        return "status: moving clusters closer to each other"

    else:
        return ("status: doing nothing because overlap "
                + "constraints are satisfied")


def ReLU_vec(x):
    """ Apply rectified linear unit x+ = max(x,0). """
    return np.maximum(x,0)
        

def poly_vec(x):
    """ Apply the polynomial p(x) = x + x**2. """
    return x + (x**2)


def H_vec(x):
    """ Apply the step (Heaviside) function. """
    return (np.sign(x) + 1)/2


def single_cluster_loss(cluster_idx, centers, cov, ave_cov_inv,
                        overlap_bounds):
    """
    Compute the marginal overlap loss for a reference cluster.
    """
    quantiles = make_quantile_vec(
                    **make_marginal_args(cluster_idx, centers, 
                                         cov, ave_cov_inv)
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


def overlap_loss(centers, cov, ave_cov_inv, overlap_bounds):
    """
    Compute the total overlap loss.
    """
    k = centers.shape[0]
    return np.sum(list(
            map(lambda i: single_cluster_loss(
                            i, centers, cov, ave_cov_inv, overlap_bounds
                          )/k, 
                range(k))))


def assess_obs_overlap(centers, cov, ave_cov_inv):
    """
    Compute the observed overlap between cluster centers.
    """
    k = centers.shape[0]
    args_list = [make_marginal_args(i, centers, cov, ave_cov_inv) 
                    for i in range(k)]
    max_overlaps = np.array([ np.max(quantile2overlap_vec(
                                make_quantile_vec(**args)))
                             for args in args_list ])
    return {'min': np.min(max_overlaps),
            'max': np.max(max_overlaps)}