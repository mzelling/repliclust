import numpy as np
from scipy.stats import chi2, norm

def update_centers(ref_cluster_idx, centers, cov_list, 
                   ave_cov_inv_list, axis_deriv_t_list,
                   learning_rate, linear_penalty_weight, 
                   quantile_bounds,
                   mode='lda'):
    """
    Perform an iteration of stochastic gradient descent on the cluster
    centers. 

    Parameters
    ----------
    ref_cluster_idx : int
        Index of reference cluster (for stochastic gradient descent).
    centers : ndarray, shape (k, p)
        Matrix of all cluster centers. Each row is a center.
    cov_list : list of ndarray; length k, each ndarray of shape (p, p)
        List of covariance matrices.
    axis_deriv_t_list : list of ndarray; length k*(k-1)/2
        List of the transposed differentials of the axis with respect
        to the reference cluster's center.
    ave_cov_inv_list : list of ndarray; length k*(k-1)/2
        List of inverses of the pairwise average covariance matrices.
    learning_rate : float
        Learning rate for gradient descent.
    linear_penalty_weight : float, between 0 and 1
        Determines the penalty function for the loss. The penalty
        polynomial is p(x) = linear_penalty_weight*x + \
            (1-linear_penalty_weight)*(x**2).
    quantile_bounds : dict with keys 'min' and 'max'
        Minimum and maximum allowed separation between clusters.

    Returns
    -------
    (grad, loss) : tuple
        Tuple whose first component is the loss gradient and whose 
        second component is the loss.

    Side effects
    ------------
    Update centers by taking a stochastic gradient descent step.
    """ 
    other_cluster_idx = np.array([idx for idx in range(centers.shape[0])
                                  if not idx == ref_cluster_idx],
                                  dtype='int')
    q = compute_quantiles(ref_cluster_idx, other_cluster_idx, centers,
                          cov_list=cov_list, 
                          ave_cov_inv_list=ave_cov_inv_list, mode=mode)
    matrix_lists = {'cov_list': cov_list, 
                    'ave_cov_inv_list': ave_cov_inv_list,
                    'axis_deriv_t_list': axis_deriv_t_list}

    repulsive_clusters_idx = other_cluster_idx[mask_repulsive(
                                    q=q, min_q=quantile_bounds['min'])
                                ]
    # Case 1: Clusters are too close to this cluster.
    if (repulsive_clusters_idx.size > 0):
        q, q_grad = quantile_gradients(
                       centers[ref_cluster_idx,:],
                       centers[repulsive_clusters_idx,:], 
                       ref_cluster_idx, repulsive_clusters_idx, 
                       mode=mode, **matrix_lists)
        loss_grad = loss_gradient_vec(q, quantile_bounds['min'], 
                                      q_grad, mode='repel', 
                                      linear_weight=linear_penalty_weight)
        apply_gradient_update(centers, loss_grad, ref_cluster_idx, 
                              repulsive_clusters_idx, 
                              learning_rate, mode='repel')
        loss = overlap_loss(centers, quantile_bounds, 
                            linear_penalty_weight,cov_list=cov_list,
                            ave_cov_inv_list=ave_cov_inv_list, mode=mode)
    # Case 2: All clusters are too far away from this cluster.
    elif check_for_attraction(q=q, max_q=quantile_bounds['max']):
        # Reduce data to only the closest cluster
        closest_cluster_idx = other_cluster_idx[np.argmin(q)]
        q, q_grad = quantile_gradients(
                       centers[ref_cluster_idx,:],
                       centers[[closest_cluster_idx],:], 
                       ref_cluster_idx, [closest_cluster_idx], 
                       mode=mode, **matrix_lists)
        loss_grad = loss_gradient_vec(q, quantile_bounds['max'], 
                                      q_grad, mode='attract', 
                                      linear_weight=linear_penalty_weight)
        apply_gradient_update(centers, loss_grad, ref_cluster_idx, 
                              [closest_cluster_idx], learning_rate, 
                              mode='attract')
        loss = overlap_loss(centers, quantile_bounds,
                            linear_penalty_weight,cov_list=cov_list,
                            ave_cov_inv_list=ave_cov_inv_list, mode=mode)
    # Case 3: Overlap constraints are satisfied -- do nothing
    else:
        loss = overlap_loss(centers, quantile_bounds,
                            linear_penalty_weight,cov_list=cov_list,
                            ave_cov_inv_list=ave_cov_inv_list, 
                            mode=mode)
        loss_grad = np.zeros(shape=(centers.shape[1],
                                    centers.shape[0]-1))

    return (loss_grad, loss)


def quantile_gradients(ref_center, other_centers, ref_cluster_idx, 
                       other_cluster_idx,
                       cov_list=None, ave_cov_inv_list=None,
                       axis_deriv_t_list=None,
                       mode='lda'):
    """
    Compute the quantile gradients with respect to the center of the
    reference cluster.

    Specifically, this function computes the gradient of the separation
    quantile q_ij (between the `i`-th and `j`-th clusters) with respect
    to the `i`-th ("reference") cluster center. The resulting gradient
    vectors are arranged as a p-by-(k-1) matrix whose `j`-th column is
    the gradient of q_ij with respect to the `i`-th cluster center. The
    analogous gradients with respect to the `j`-th ("other") cluster are
    simply the negative of the "reference" gradients. Finally, it is
    not necessary to compute the gradient of q_ij with respect to cluster
    centers besides `i`, `j` since such cluster centers to not influence
    q_ij.

    Parameters
    ----------
    ref_center : ndarray, shape (p,)
        The center of the reference cluster.
    other_centers : ndarray, shape (k-1,p)
        The other cluster centers (not necessarily all of them).
    ref_cluster_idx : int
        The absolute index of the reference cluster (when considering
        ALL clusters).
    other_cluster_idx : list[int]
        The absolute indices of the other clusters (when considering
        ALL clusters).

    Returns
    -------
    (q, q_grad) : tuple[ndarray]
        The first component is an array with shape (1,k-1) that reports
        the separation quantiles for each cluster. The second component
        is an array of shape (p,k-1) whose columns are the gradients of
        the quantiles q_ij (separation quantile between cluster centers
        `i` and `j`) with respect to the "reference" cluster center `i`,
        where `j` corresponds to the "other" cluster center and indexes
        the columns of the gradient matrix.
    """
    # Get delta matrix (difference between cluster centers)
    delta_mat = np.transpose(ref_center[np.newaxis,:]
                                - other_centers)

    # Get axis matrix and list of axis differentials
    if mode=='lda':
        axis_mat = matvecprod_vectorized(
                        ave_cov_inv_list, 
                        get_1d_idx(ref_cluster_idx, other_cluster_idx,
                                   len(cov_list)),
                        delta_mat)
    elif mode=='c2c':
        axis_mat = delta_mat
    elif mode=='exact':
        raise NotImplementedError('the exact method is currently not'
                                   + ' implemented.')

    # Get cov-transformed axis matrices
    ref_tf, other_tf = cov_transform(axis_mat, cov_list, 
                                     ref_cluster_idx, other_cluster_idx)

    # Get marginal standard deviations
    ref_std, other_std = marginal_std(axis_mat, ref_tf, other_tf)

    # put the quantile gradients together using the product rule; upon
    # writing the quantile as A/B, we get dA*B + A*d(1/B)
    marginal_std_sum = ref_std + other_std
    axis_dot_delta = np.sum(axis_mat * delta_mat, axis=0)[np.newaxis,:]
    if mode=='lda':
        numerator_term = ((axis_mat + matvecprod_vectorized(
                                        axis_deriv_t_list, 
                                        get_1d_idx(ref_cluster_idx, 
                                            other_cluster_idx,
                                            len(cov_list)), 
                                        delta_mat))
                            / marginal_std_sum)
        denominator_term = (
            (1/2)*(-axis_dot_delta)/(marginal_std_sum**2)
                * matvecprod_vectorized(
                    axis_deriv_t_list, 
                    get_1d_idx(ref_cluster_idx, other_cluster_idx,
                            len(cov_list)),
                    ref_tf/ref_std + other_tf/other_std))
    elif mode=='c2c':
        numerator_term = (axis_mat + delta_mat) / marginal_std_sum
        denominator_term = (
            (1/2)*(-axis_dot_delta)/(marginal_std_sum**2)
                * (ref_tf/ref_std + other_tf/other_std)
        )

    q_grad = numerator_term + denominator_term
    q = axis_dot_delta/marginal_std_sum
    return (q, q_grad)


def compute_quantiles(ref_cluster_idx, other_cluster_idx, 
                      centers, cov_list=None, ave_cov_inv_list=None, 
                      mode='lda'):
    """
    Compute the separation quantiles for the other clusters.

    Parameters
    ----------
    ref_cluster_idx : int
        The index of the reference cluster.
    other_cluster_idx : list[int] or ndarray, dtype='int'
        Indices of some other clusters.
    centers : ndarray
        Array of ALL other cluster centers.
    cov_list : list[ndarray]
        List of ALL cluster covariance matrices.
    ave_cov_inv_list : list[ndarray]
        List of the inverse of pairwise average covariance matrices
        for ALL pairs of clusters.
    mode : {'lda','c2c','exact'}
        Method for computing cluster overlap.

    Returns
    -------
    quantiles : ndarray
        Separation quantiles for the other clusters, formatted as a
        vector.
    """
    delta_mat = np.transpose(centers[ref_cluster_idx,:][np.newaxis,:]
                             - centers[other_cluster_idx,:])
    if mode=='lda':
        axis_mat = matvecprod_vectorized(
                        ave_cov_inv_list, 
                        get_1d_idx(ref_cluster_idx, other_cluster_idx,
                                   len(cov_list)),
                        delta_mat)
    elif mode=='c2c':
        axis_mat = delta_mat
    elif mode=='exact':
        raise NotImplementedError("the 'exact' method has not been "
                                    + "implemented yet.")

    axis_dot_delta = np.sum(axis_mat * delta_mat, axis=0)[np.newaxis,:]
    tf_ref, tf_other = cov_transform(
                                axis_mat, cov_list, ref_cluster_idx, 
                                other_cluster_idx)
    std_ref, std_other = marginal_std(axis_mat, tf_ref, tf_other)

    return (axis_dot_delta/(std_ref + std_other)).flatten()


def cov_transform(axis_mat, cov_list, ref_cluster_idx, 
                  other_cluster_idx):
    """
    Transform input axes by left-multiplying with covariance matrices.

    Output is a tuple of two matrices. The first component is the result
    of transforming all clusters' axes by the reference covariance
    matrix; the second component is the result of transforming by the
    cluster-specific covariance matrix.

    Parameters
    ----------
    axis_mat : ndarray, shape (p,k-1)
        Matrix whose j-th column is the axis corresponding to the j-th
        other cluster.
    cov_list : list[ndarray, shape (p,p)]
        List of all cluster covariance matrices.
    ref_cluster_idx : int
        Index of reference cluster within `cov_list`.
    other_cluster_idx : list[int]
        Indices of other clusters within `cov_list`.

    Returns
    -------
    (ref,specific) : tuple[ndarray, shape (p,k-1)]
        Tuple with two components, both of which are matrices of shape
        (p,k-1).
    """
    ref_transform = cov_list[ref_cluster_idx] @ axis_mat
    specific_transform = matvecprod_vectorized(cov_list, 
                                               other_cluster_idx, 
                                               axis_mat)
    return (ref_transform, specific_transform)


def marginal_std(axis_mat, ref_transform, specific_transform):
    """
    Compute the marginal standard deviations with respect to the
    reference cluster and other clusters.

    Parameters
    ----------
    axis_mat : ndarray, shape (p,k-1)
        Matrix of axes (each axis is a column).
    ref_transform : ndarray, shape (p,k-1)
        Product of reference cluster's covariance matrix and axes
        corresponding to the other clusters.
    specific_transform : ndarray, shape (p,k-1)
        Product of other clusters' covariance matrices and the
        corresponding axes.

    Returns
    -------
    (std_ref, std_other) : tuple[ndarray, shape (1,k-1)]
        Tuple with two components. The first component is the marginal
        standard deviation along each axis, computed with the reference
        cluster's covariance matrix; the second component consists of
        the same marginal standard deviations, but computed with the
        respective other cluster's covariance matrix.
    """
    reference_std = np.sqrt(np.sum(axis_mat * ref_transform, 
                            axis=0)[np.newaxis,:])
    other_std = np.sqrt(np.sum(axis_mat * specific_transform, 
                            axis=0)[np.newaxis,:])
    return (reference_std, other_std)


def loss_gradient_vec(q, q_thold, q_grad, mode='repel', 
                      linear_weight=0.5):
    """
    Compute the gradient of the loss function with respect to the
    cluster centers in the repulsive and attractive cases.

    Parameters
    ----------
    q : ndarray, shape (1,?)
        Matrix of separation quantiles.
    q_thold : float
        If mode='repel', the minimum separation quantile. If
        mode='attract', the maximum separation quantile.
    q_grad : ndarray, shape (*,?)
        Matrix of gradients of the separation quantiles with respect
        to the `i`-th cluster center.
    linear_weight : float, between 0 and 1
        Weight of the linear penalty. If linear_weight=0 then the
        penalty is quadratic. If linear_weight=1 the penalty is linear.
        For intermediate values, the penalty is mixed.

    Returns
    -------
    gradient : ndarray, shape (p,k-1)
        The gradients of all summands in the loss function (relative to
        the reference cluster) with respect to the reference cluster's
        center.
    """
    if mode=='repel':
        penalty_grad = -(linear_weight*H_vec(q_thold - q)
                            + 2*(1-linear_weight)*ReLU_vec(q_thold - q))
        return penalty_grad * q_grad
    elif mode=='attract':
        penalty_grad = (linear_weight*H_vec(q - q_thold)
                        + 2*(1-linear_weight)*ReLU_vec(q - q_thold))
        return penalty_grad * q_grad


def get_1d_idx(i,j_vec,k):
    """
    Compute the one-dimensional index corresponding to cluster i and
    cluster j. The j indices are a vector, and the output is vectorized.

    The inputs i, j_vec are numbers from 0 to k-1 as in a k-by-k matrix.
    The output is the linear index when saving the k-k matrix (with the
    diagonal removed) as a vector by sweeping across columns (j) for
    increasing row (i) index.
    """
    def linearized_idx(i,j,k):
        """ Linearize the index assuming i < j. """
        assert i < j
        return int(i*(k-1) - ((i-1)*i/2) + (j-i-1))

    return [linearized_idx(i,j,k) if i<j else linearized_idx(j,i,k)
            for j in j_vec]


def matvecprod_vectorized(matrix_list, matrix_col_idx, vectors_mat):
    """
    Compute matrix-vector products and form a new matrix with the
    results as columns.

    Parameters
    ----------
    matrix_list : list[ndarray of shape (p,p)]
        List of matrices.
    matrix_col_idx : list[int] or ndarray, dtype='int'
        Maps column indices of `vectors_mat` to corresponding indices
        of `matrix_list`.
    vectors_mat : ndarray, shape (p,k-1)
        Collection of vectors arranged as the columns of a matrix.

    Returns
    -------
    vector_list : list[ndarray]
        List whose `i`-th entry stores the result of right-multiplying
        the `i`-th matrix in `matrix_list` by the `i`-th column of 
        `vectors_mat`. Each entry is a ndarray with shape (p,1).
    """
    return np.concatenate(
                list(map(lambda j: (matrix_list[matrix_col_idx[j]] 
                                @ vectors_mat[:,j])[:,np.newaxis],
                    range(len(matrix_col_idx)))), 
                axis=1
            )


def apply_gradient_update(centers, loss_grad,
                          ref_cluster_idx, other_cluster_idx, 
                          learning_rate, mode='repel'):
    """
    Apply a gradient step to update the cluster centers. Helper function
    for update_centers.

    Parameters
    ----------
    centers : ndarray
        The current locations of the cluster centers.
    loss_grad : ndarray
        The gradients with respect to the loss function.
    ref_cluster_idx : int
        The index of the reference cluster.
    other_cluster_idx : list[int]
        The indices for the other clusters.
    learning_rate : float
        Learning rate for the gradient descent step.
    mode : {'repel', 'attract'}
        Select whether the gradient descent step moves clusters further
        apart ('repel') or closer together ('attract').

    Returns
    -------
    """
    if mode=='repel':
        centers[ref_cluster_idx,:] -= (learning_rate 
                                        * np.sum(loss_grad, axis=1))
        centers[other_cluster_idx,:] -= (learning_rate
                                        * np.transpose(-loss_grad))
    elif mode=='attract':
        centers[ref_cluster_idx,:] -= (learning_rate 
                                        * np.sum(loss_grad, axis=1))
        centers[other_cluster_idx,:] -= (learning_rate
                                            * np.transpose(-loss_grad))


def mask_repulsive(q=None, min_q=None, o=None, max_o=None):
    """
    Return a boolean array in which each entry is TRUE if the
    corresponding cluster is a repulsive cluster, i.e., violates
    the max overlap (minimum separation) constraint.
    """
    if (q is not None) and min_q:
        return (q < min_q).flatten()
    elif (o is not None) and max_o:
        return (o >= max_o).flatten()


def check_for_attraction(q=None, max_q=None, o=None, min_o=None):
    """
    Return TRUE if cluster attracts its closest neighbor, i.e., all
    other clusters are further away than allowed by maximum separation.
    """
    if (q is not None) and max_q:
        return np.min(q) >= max_q
    elif (o is not None) and min_o:
        return np.max(o) < min_o


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


def single_cluster_loss(q, q_bounds, linear_penalty_weight):
    """
    Compute the marginal overlap loss for a reference cluster.
    """
    if not check_for_attraction(q=q, max_q=q_bounds['max']):
        # return the loss for repulsion
        out = np.sum(
                     poly_vec(ReLU_vec(q_bounds['min'] - q), 
                              linear_weight=linear_penalty_weight)
                    )
        return out
    else:
        # return the loss for attraction
        out = poly_vec(ReLU_vec(np.min(q) - q_bounds['max']),
                       linear_weight=linear_penalty_weight)
        return out


def overlap_loss(centers, q_bounds, linear_penalty_weight,
                 mode=None, cov_list=None, ave_cov_inv_list=None):
    """
    Compute the total overlap loss.

    Parameters
    ----------
    centers : ndarray
    q_bounds : dict
    linear_penalty_weight : float
    mode : {}
    cov_list : list[ndarray]
    ave_cov_inv : list[ndarray]

    Returns
    -------
    loss : float
        The overlap loss for the cluster ensemble.
    """
    n_clusters = centers.shape[0]
    return np.sqrt(np.sum(list(
            map(lambda i: single_cluster_loss(
                    compute_quantiles(
                        i, [j for j in range(n_clusters) if j != i], 
                        centers, cov_list=cov_list, 
                        ave_cov_inv_list=ave_cov_inv_list, mode=mode),
                    q_bounds, linear_penalty_weight)
                    /n_clusters, 
                range(n_clusters))))
            )


def assess_obs_separation(centers, cov_list, ave_cov_inv_list, 
                          mode='lda'):
    """
    Compute the observed separation between cluster centers.
    """
    n_clusters = centers.shape[0]
    min_separations = np.array(
                        [ np.min(compute_quantiles(
                            i, [j for j in range(n_clusters) if j != i], 
                            centers, cov_list, ave_cov_inv_list, 
                            mode=mode
                          )) for i in range(n_clusters) ])
    return {'max': np.max(min_separations),
            'min': np.min(min_separations)}