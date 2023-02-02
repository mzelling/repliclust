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


def quantile_gradient_vec(diff_mat, l_mat, l_tf_mat_1, l_tf_mat_2,
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

def loss_gradient_vec(q, q_thold, q_grad, mode='repel', 
                      linear_weight=0.5):
    """
    Compute the gradient of the loss function with respect to the
    cluster centers.

    Parameters
    ----------
    q : ndarray
        The vector of separation quantiles.
    q_thold : float
        If mode='repel', the minimum separation quantile. If
        mode='attract', the maximum separation quantile.
    q_grad : ndarray
        Matrix of gradients with respect to the separation quantiles.
    linear_weight : float, between 0 and 1
        Weight of the linear penalty. If linear_weight=0 then the
        penalty is quadratic. If linear_weight=1 the penalty is linear.
        For intermediate values, the penalty is mixed.

    Returns
    -------
    gradient : ndarray
        The gradient with respect to the loss function.
    """
    if mode=='repel':
        return -(linear_weight*H_vec(q_thold - q)
                    + 2*(1-linear_weight)*ReLU_vec(q_thold - q))*q_grad
    elif mode=='attract':
        return (linear_weight*H_vec(q - q_thold)
                    + 2*(1-linear_weight)*ReLU_vec(q - q_thold))*q_grad


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


def subset_clusters_of_interest(data, subset):
    """ 
    Reduce precomputed data to columns corresponding to the
    clusters of interest. 

    Parameters
    ----------
    data : dict
        Precomputed data.
    subset : ndarray(dtype='bool') or list[int]
        Index into the columns of interest. Can be boolean array or
        a list of integer indices.

    Returns
    -------
    reduced_data : dict
        The input dictionary but with each item reduced to contain only
        the columns of interest. 
    """
    return {X_name: X[:,subset] for X_name, X in data.items()}


def apply_gradient_update(centers, loss_grad,
                          cluster_idx, other_cluster_idx, 
                          subset, learning_rate, mode='repel'):
    """
    Apply a gradient step to update the cluster centers. Helper function
    for update_centers.

    Parameters
    ----------
    centers : ndarray
        The current locations of the cluster centers.
    loss_grad : ndarray
        The gradients with respect to the loss function.
    cluster_idx : int
        The index of the reference cluster.
    other_cluster_idx : list[int]
        The indices for the other clusters.
    subset : ndarray (dtype=bool) or int
        Index into the columns of interest. If mode='repel', `subset` is
        a boolean array. If mode='attract', `subset` is an integer.
    learning_rate : float
        Learning rate for the gradient descent step.
    mode : {'repel', 'attract'}
        Select whether the gradient descent step moves clusters further
        apart ('repel') or closer together ('attract').

    Returns
    -------
    """
    if mode=='repel':
        repel_idx = np.array(other_cluster_idx)[subset]
        centers[cluster_idx,:] -= (learning_rate 
                                    * (np.sum(loss_grad, axis=1)))
        centers[repel_idx,:] -= (learning_rate
                                    * np.transpose(-loss_grad))
    elif mode=='attract':
        attract_idx = other_cluster_idx[subset]
        centers[cluster_idx,:] -= learning_rate * loss_grad.flatten()
        centers[attract_idx,:] -= learning_rate * (-loss_grad).flatten()


def update_centers(cluster_idx, centers, cov, ave_cov_inv, 
                   learning_rate, penalty_coef, overlap_bounds):
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
    penalty_coef : float, between 0 and 1
        Determines the penalty function for the loss. The penalty
        polynomial is p(x) = penalty_coef*x + (1-penalty_coef)*(x**2).
    overlap_bounds : dict with keys 'min' and 'max'
        Minimum and maximum allowed overlaps between clusters.
    cov : list of ndarray; length k, each ndarray of shape (p, p)
        List of covariance matrices.

    Side effects
    ------------
    Update centers by taking a stochastic gradient descent step.
    """
    other_cluster_idx = compute_other_cluster_idx(cluster_idx,
                                                  centers.shape[0])
    precomputed_data = make_marginal_args(
                            cluster_idx, centers, cov, ave_cov_inv) 
    quantiles = make_quantile_vec(**precomputed_data)
    overlaps = quantile2overlap_vec(quantiles)

    # Case 1: Clusters are too close to this cluster.
    which_clusters_repel = get_max_overlap_violators(
                                overlaps, overlap_bounds['max'])
    if np.any(which_clusters_repel):
        # Reduce data to all clusters that are too close
        precomputed_data = subset_clusters_of_interest(
                                precomputed_data, which_clusters_repel)
        q = quantiles[:,which_clusters_repel]
        q_min = overlap2quantile_vec(overlap_bounds['max'])
        # Compute gradients and update cluster centers
        quantile_grad = quantile_gradient_vec(**precomputed_data)
        loss_grad = loss_gradient_vec(q, q_min, quantile_grad, 
                                      mode='repel', 
                                      linear_weight=penalty_coef)
        apply_gradient_update(centers, loss_grad, cluster_idx, 
                              other_cluster_idx, which_clusters_repel, 
                              learning_rate, mode='repel')
    # Case 2: All clusters are too far away from this cluster.
    elif violate_min_overlap(overlaps, overlap_bounds['min']):
        # Reduce data to only the closest cluster
        closest_cluster_idx = np.argmax(overlaps)
        precomputed_data = subset_clusters_of_interest(
                                precomputed_data, 
                                subset=[closest_cluster_idx])
        q = quantiles[:,[closest_cluster_idx]]
        q_max = overlap2quantile_vec(overlap_bounds['min'])
        # Compute gradients and update cluster centers
        quantile_grad = quantile_gradient_vec(**precomputed_data)
        loss_grad = loss_gradient_vec(q, q_max, quantile_grad, 
                                      mode='attract', 
                                      linear_weight=penalty_coef)
        apply_gradient_update(centers, loss_grad, cluster_idx, 
                              other_cluster_idx, closest_cluster_idx, 
                              learning_rate, mode='attract')


def get_max_overlap_violators(overlaps, max_overlap):
    """ 
    Return boolean array whose entries are TRUE if the
    corresponding cluster violates the maximum overlap condition with
    respect to the reference cluster.
    """
    return (overlaps >= max_overlap).flatten()


def violate_min_overlap(overlaps, min_overlap):
    """ Return true if cluster violates minimum overlap condition."""
    return np.max(overlaps) < min_overlap


def ReLU_vec(x):
    """ 
    Apply rectified linear unit x+ = max(x,0).

    Parameters
    ----------
    x : ndarray
        Input array of numbers.

    Returns
    -------
    relu : ndarray
        Rectified linear unit applied to each entry of input array.
    """
    return np.maximum(x,0)
        

def poly_vec(x, linear_weight=0.5):
    """ 
    Apply a penalty polynomial p(x) = lmbda*x + (1-lmbda)*(x**2) in
    a vectorized fashion.

    Parameters
    ----------
    x : ndarray
        Input array of numbers.
    linear_weight : float, between 0 and 1 (inclusive)
        Determine the relative weighting of the linear penalty. If
        linear_weight=0, the penalty is quadratic. If linear_weight=1,
        the penalty is linear. For intermediate values, the penalty is
        mixed.

    Returns
    -------
    poly : ndarray, same shape as `x`
        Penalty polynomial evaluated at each element in `x`.
    """
    return linear_weight*np.abs(x) + (1-linear_weight)*(x**2)


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
    n_clusters = centers.shape[0]
    return np.sqrt(np.sum(list(
            map(lambda i: single_cluster_loss(
                            i, centers, cov, ave_cov_inv, overlap_bounds
                          )/n_clusters, 
                range(n_clusters))))
            )


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