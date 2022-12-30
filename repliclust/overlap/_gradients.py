import numpy as np
from scipy.stats import chi2

def mdist_vectorized(diff_mat, diff_tf_mat):
    """
    Compute Mahalanobis distance (vectorized).

    Parameters
    ----------
    diff_mat : ndarray, shape (p, k-1)
        Differences between pairs of distinct cluster centers,
        one vs all. The j-th column is the difference vector
        between the reference cluster center and the j-th other
        cluster center.
    diff_tf_mat : ndarray, shape (p, k-1)
        Differences between pairs of distinct cluster centers left
        multiplied by the appropriate inverse covariance matrices.

    Returns
    -------
    mdist : ndarray, shape (1, k-1)
        Mahalanobis distances between reference cluster and
        other clusters.
    """
    mdist = np.sqrt(
        np.sum(diff_mat * diff_tf_mat, axis=0)
        )[np.newaxis,:]
    return mdist

def harsum_vectorized(X, Y):
    """
    Compute harmonic sum (vectorized).

    Parameters
    ----------
    X : ndarray, shape (1, k-1)
        Input matrix.
    Y : ndarray, shape (1, k-1)
        Input matrix.

    Returns
    -------
    out : ndarray, shape (1, k-1)
        Harmonic sum of X and Y.
    """
    return 1/(1/X + 1/Y)

def chi2term_vectorized(mharsum_vec, p):
    """
    Compute the chi2 term of the overlap gradients with respect to a 
    reference cluster center vs all other centers (vectorized).
    Compute the harmonic sum of Mahalanobis distances (vectorized).

    Parameters
    ----------
    mharsum_vec : ndarray, shape (1, k-1)
        Harmonic sum of Mahalanobis distances (vectorized).
    p : int
        Dimensionality of the data (degrees of freedom for chi2).    

    Returns
    -------
    out : ndarray, shape (1, k-1)
        Chi2(p) density evaluated at the appropriate quantile cutoffs.
    """
    return -chi2.pdf(mharsum_vec**2, df=p)
                                    
def cubicterm_vectorized(mharsum_vec):
    """
    Compute the inverse cubic term of the overlap gradients with respect
    to a reference cluster center vs all other centers (vectorized).

    Parameters
    ----------
    mharsum_vec : ndarray, shape (1, k-1)
        Harmonic sum of Mahalanobis distances (vectorized).

    Returns
    -------
    out : ndarray, shape (1, k-1)
        Inverse cubic term of the overlap gradients.
    """
    return -2*(mharsum_vec**3)

def squareterm_vectorized(mharsum_vec):
    """
    Compute the inverse square term.
    """
    return -(mharsum_vec**2)

def summandterm_vectorized(mdist_vec, diff_tf_mat):
    """
    Compute summand term by broadcasting.

    Parameters
    ----------
    mdist_vec : ndarray, shape (1, k-1)
        The vectorized Mahalanobis distances.
    diff_tf_mat : ndarray, shape (p, k-1)
        Differences between pairs of distinct cluster centers left
        multiplied by the appropriate inverse covariance matrices.

    Returns
    -------
    out : ndarray, shape (p, k-1)
        Summand term involved in computing the overlap gradient.
    """
    return (-1/mdist_vec**3) * diff_tf_mat 

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

def make_premahalanobis_args(
        cluster_idx, other_cluster_idx, centers, cov_inv
        ):
    """
    Compute some quantities needed in other functions: differences 
    between cluster centers, differences transformed by the reference
    clusters inverse covariance matrix, and differences transformed by
    the corresponding clusters' covariance matrices.

    Parameters
    ----------
    cluster_idx : int
        Index of reference cluster.
    other_cluster_idx : list of int
        List of other cluster indices.
    centers : ndarray
        Matrix of cluster centers. Each row is a center.
    cov_inv : list of ndarray
        List of inverse covariance matrices.

    Returns
    -------
    out : dict with keys 'diff_mat', 'diff_tf_mat_1', 'diff_tf_mat_2'
        Provide quantities useful for downstream computations.
    """
    p = centers.shape[1]
    k = centers.shape[0]

    diff_mat = np.transpose(centers[cluster_idx,:][np.newaxis,:] \
        - centers[other_cluster_idx,:])
    diff_tf_mat_1 = cov_inv[cluster_idx] @ diff_mat
    diff_tf_mat_2 = np.concatenate(
        list(map(lambda i: (cov_inv[other_cluster_idx[i]] \
            @ diff_mat[:,i])[:,np.newaxis], 
            range(len(other_cluster_idx)))), axis=1)

    return {'diff_mat': diff_mat, 'diff_tf_mat_1': diff_tf_mat_1, 
            'diff_tf_mat_2': diff_tf_mat_2}

def make_mahalanobis_args(diff_mat, diff_tf_mat_1, diff_tf_mat_2):
    """
    Cmopute Mahalanobis quantities for use in other functions.
    """
    mdist_vec_1 = mdist_vectorized(diff_mat, diff_tf_mat_1)
    mdist_vec_2 = mdist_vectorized(diff_mat, diff_tf_mat_2)
    mharsum_vec = harsum_vectorized(mdist_vec_1, mdist_vec_2)
    return {'mdist_vec_1': mdist_vec_1, 'mdist_vec_2': mdist_vec_2, 
            'mharsum_vec': mharsum_vec}

def make_mharsum_vec(cluster_idx, centers, cov_inv):
    """
    Compute harmonic sum of Mahalanobis distances from centers and
    inverse covariance matrices.
    """
    k = centers.shape[0]
    premahal_args = make_premahalanobis_args(cluster_idx,
        compute_other_cluster_idx(cluster_idx, k), centers, cov_inv)
    mahal_args = make_mahalanobis_args(**premahal_args)
    return mahal_args['mharsum_vec']

def gradient_vectorized(
        diff_mat=None, diff_tf_mat_1=None, diff_tf_mat_2=None, 
        mdist_vec_1=None, mdist_vec_2=None, mharsum_vec=None,
        mode="overlap"
        ):
    """
    Compute the gradient of overlaps of a reference cluster with
    all other k-1 clusters.

    Parameters
    ----------
    diff_mat : ndarray, shape (p, k-1)
        Matrix of differences between reference cluster center and
        the other k-1 cluster centers.
    diff_tf_mat_1 : ndarray, shape (p, k-1)
        Same as diff_mat, except each column is left-multiplied by
        inverse covariance matrix of reference cluster.
    diff_tf_mat_2 : ndarray, shape (p, k-1)
        Same as diff_mat, except each column is left-multiplied by
        inverse covariance matrix of corresponding OTHER cluster.

    Returns
    -------
    out : ndarray, shape (p, k-1)
        Gradient vectors of the reference cluster's overlap with
        the other clusters, with respect to the reference center.
        The j-th column of this matrix is the derivative of the
        overlap between the reference cluster and the j-th other
        cluster with respect to the reference center. To get the
        derivative of the same quantity with respect to the centers
        of the OTHER clusters, simply multiply the output by -1.
    """
    p = diff_mat.shape[0]
    if (mode == 'overlap'):
        return np.multiply(
            chi2term_vectorized(mharsum_vec, p), 
            cubicterm_vectorized(mharsum_vec)) * \
                np.add(
                    summandterm_vectorized(mdist_vec_1, 
                                           diff_tf_mat_1),
                    summandterm_vectorized(mdist_vec_2, diff_tf_mat_2))
    elif (mode == 'mharsum'):
        return squareterm_vectorized(mharsum_vec) * \
            np.add(summandterm_vectorized(mdist_vec_1, diff_tf_mat_1),
                   summandterm_vectorized(mdist_vec_2, diff_tf_mat_2))
    else:
        raise ValueError("the specified mode does not exist. Choose "
            + "'overlap' or 'mharsum'.")


def compute_overlaps_vectorized(mharsum_vec, p):
    """
    Compute overlaps between a reference cluster and all other
    clusters.

    Parameters
    ----------
    mharsum_vec : ndarray, shape (1, k-1)
        Harmonic sum of Mahalanobis distances.
    p : int
        Dimensionality of the clusters / degrees of freedom for the
        chi-square distribution.

    Returns
    -------
    out : ndarray, shape (1, k-1)
        Overlaps between reference cluster and all other clusters.
    """
    return 1 - chi2.cdf(mharsum_vec**2, df=p)


def update_centers(cluster_idx, centers, cov_inv, learning_rate, 
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
    cov_inv : list of ndarray; length k, each ndarray of shape (p, p)
        List of inverse covariance matrices.
    learning_rate : float
        Learning rate for gradient descent.
    overlap_bounds : dict with keys 'min' and 'max'
        Minimum and maximum allowed overlaps between clusters.

    Side effects
    ------------
    Update centers by taking a stochastic gradient descent step.
    """
    # Make reusable arguments for the subsequent steps
    p = centers.shape[1]
    k = centers.shape[0]
    other_cluster_idx = compute_other_cluster_idx(cluster_idx, k)
    premahal_args = make_premahalanobis_args(
                        cluster_idx, other_cluster_idx, centers, cov_inv
                        )
    mahal_args = make_mahalanobis_args(**premahal_args)
    mharsum_vec = mahal_args['mharsum_vec']

    # Compute overlaps between chosen cluster and other clusters
    overlaps = compute_overlaps_vectorized(mahal_args['mharsum_vec'], p)

    # See if any clusters repel the reference cluster
    repel_mask = (overlaps >= overlap_bounds['max']).flatten()
    if np.any(repel_mask):
        # Select only the clusters that repel the reference cluster
        grad_args = {X_name: X[:,repel_mask] for X_name, X in \
                    (premahal_args | mahal_args).items()}
        #gradients = gradient_vectorized(**grad_args, mode='overlap')
        gradients = gradient_vectorized(**grad_args, mode='mharsum')
        q_min = np.sqrt(chi2.ppf(1-overlap_bounds['max'], df=p))
        q = mharsum_vec[:,repel_mask]

        #MSE_grad = 2*(overlaps[:,repel_mask] - overlap_bounds['max'])*gradients
        #MSE_grad = (1 + 2*(overlaps[:,repel_mask] - overlap_bounds['max']))*gradients
        MSE_grad = -(1 + 2*(q_min - q))*gradients
        # Update centers matrix with a gradient step
        repel_idx = np.array(other_cluster_idx)[repel_mask]
        centers[cluster_idx,:] -= learning_rate * (np.sum(MSE_grad, axis=1)/(k-1))
        centers[repel_idx,:] -= learning_rate * np.transpose(-MSE_grad)
        return MSE_grad # "status: moving clusters further away from each other"

    elif np.max(overlaps) <= overlap_bounds['min']:
        # Select only the cluster closest to the reference cluster
        attract_pre_idx = np.argmax(overlaps)
        grad_args = {X_name: X[:,[attract_pre_idx]] for X_name, X in \
                    (premahal_args | mahal_args).items()}
        gradients = gradient_vectorized(**grad_args, mode='mharsum')
        q_max = np.sqrt(chi2.ppf(1-overlap_bounds['min'], df=p))
        q = mharsum_vec[:,[attract_pre_idx]]
        #MSE_grad = 2*(q - q_max)*gradients
        MSE_grad = (1 + 2*(q - q_max))*gradients
        # Update centers matrix with a gradient step
        attract_idx = other_cluster_idx[attract_pre_idx]
        centers[cluster_idx,:] -= learning_rate * MSE_grad.flatten()
        centers[attract_idx,:] -= learning_rate * (-MSE_grad).flatten()
        return "status: moving clusters closer to each other"

    else:
        return "status: doing nothing because overlap constraints are satisfied"
        

def cluster_loss(cluster_idx, centers, cov_inv, overlap_bounds):
    """
    Compute the overlap loss for a reference cluster.
    """
    k = centers.shape[0]
    p = centers.shape[1]
    mharsum_vec = make_mharsum_vec(cluster_idx, centers, cov_inv)
    overlaps = compute_overlaps_vectorized(mharsum_vec, p)

    if (np.max(overlaps) > overlap_bounds['min']):
        # return the loss for repulsion
        return np.sum(np.maximum(overlaps - overlap_bounds['max'],0)**2)
    else:
        # return the loss for attraction
        q_max = np.sqrt(chi2.ppf(1-overlap_bounds['min'], df=p))
        q = np.min(mharsum_vec)
        return 0.1*(q - q_max)**2


def assess_obs_overlap(centers, cov_inv):
    """
    Assess the observed min and maximum overlap between cluster centers.
    """
    k = centers.shape[0]
    p = centers.shape[1]

    max_obs_overlaps = list(
        map(lambda i: np.max(compute_overlaps_vectorized(
                make_mharsum_vec(i, centers, cov_inv), p)), 
            range(k)))
    obs_overlap = {'min': np.min(max_obs_overlaps), 
                   'max': np.max(max_obs_overlaps)}
    return obs_overlap
    

def total_loss(centers, cov_inv, overlap_bounds):
    """
    Compute the total overlap loss.
    """
    k = centers.shape[0]
    return np.sum(list(map(lambda i: cluster_loss(i, centers, cov_inv, 
        overlap_bounds)/k, range(k))))