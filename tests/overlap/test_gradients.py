import pytest
import numpy as np
from scipy.stats import chi2

from repliclust.overlap import gradients
from repliclust.maxmin.archetype import MaxMinArchetype

@pytest.fixture
def raw_cluster_input():
    k = 3
    p = 2

    cov_inv_0 = np.eye(2)
    cov_inv_1 = 17*np.eye(2)
    cov_inv_2 = (1e-6)*np.eye(2)
    cov_inv = [cov_inv_0, cov_inv_1, cov_inv_2]

    center_0 = np.zeros(shape=(1,2))
    center_1 = np.array([[0,1]])
    center_2 = np.array([[1,0]])
    centers = np.concatenate([center_0,center_1,center_2],axis=0)

    return {'centers': centers, 'cov_inv': cov_inv, 'p': p, 'k': k}


@pytest.fixture
def archetype():
    bp = MaxMinArchetype()
    return bp


@pytest.fixture
def array_shapes_broadcast():
    p = 1000
    k = 100
    a = np.arange(1,(k-1)+1)[np.newaxis,:]
    A = np.repeat(a[:,::-1], axis=0, repeats=p)
    A[100,:] = 17
    return {'rowmat': a, 'bigmat': A, 'p': p, 'k': k}


@pytest.fixture
def array_shapes_big():
    p = 1000
    k = 100
    A = (2)*np.ones(shape=(p,k-1))
    B = (13/2)*np.ones(shape=(p,k-1))
    return {'mat1': A, 'mat2': B, 
            'matprod': 13*np.ones(shape=(p,k-1)), 
            'p': p, 'k': k}


@pytest.fixture
def array_shapes_small():
    k = 20
    p = 4
    a = 2*np.ones(shape=(1,k-1))
    b = 3*np.ones(shape=(1,k-1))
    return {'mat1': a, 'mat2': b, 'k': k, 'p': p}


def test_mdist_vectorized(array_shapes_big):
    mtx = array_shapes_big
    result = gradients.mdist_vectorized(mtx['mat1'], mtx['mat2'])
    assert result.shape == (1,mtx['k']-1)
    assert np.allclose(result, 
        np.sqrt(13*mtx['p']) * np.ones(shape=(1,mtx['k']-1)))


def test_harsum_vectorized(array_shapes_small):
    mtx = array_shapes_small
    result = gradients.harsum_vectorized(mtx['mat1'], mtx['mat2'])
    assert result.shape == (1,mtx['k']-1)
    assert np.allclose(result, (6/5)*np.ones(shape=(1,mtx['k']-1)))


def test_chi2term_vectorized(array_shapes_small):
    mtx = array_shapes_small
    result = gradients.chi2term_vectorized(mtx['mat1'], mtx['p'])
    assert result.shape == (1,mtx['k']-1)
    assert np.allclose(result, -chi2.pdf(mtx['mat1']**2, df=mtx['p']))


def test_cubicterm_vectorized(array_shapes_small):
    mtx = array_shapes_small
    result = gradients.cubicterm_vectorized(mtx['mat1'])
    assert result.shape == (1,mtx['k']-1)
    assert np.allclose(result, -16*np.ones(shape=(1,mtx['k']-1)))


def test_squareterm_vectorized(array_shapes_small):
    mtx = array_shapes_small
    result = gradients.squareterm_vectorized(mtx['mat1'])
    assert result.shape == (1,mtx['k']-1)
    assert np.allclose(result, -4*np.ones(shape=(1,mtx['k']-1)))


def test_summandterm_vectorized(array_shapes_broadcast):
    mtx = array_shapes_broadcast
    result = gradients.summandterm_vectorized(
        mtx['rowmat'], mtx['bigmat']
        )
    assert result.shape == (mtx['p'], mtx['k']-1)
    desired_result = np.ones(shape=(mtx['p'], mtx['k']-1))
    for i in range(mtx['p']):
        for j in range(mtx['k']-1):
            if (i == 100):
                desired_result[i,j] = (-1/((j+1)**3))*17
            else:
                desired_result[i,j] = (-1/((j+1)**3))*((mtx['k']-1) - j)
    assert np.allclose(result, desired_result)


def test_compute_other_cluster_idx():
    i = 1
    k = 3
    test = gradients.compute_other_cluster_idx(i,k)
    assert test == [0,2]


def test_make_gradient_arguments(raw_cluster_input):
    data = raw_cluster_input
    # take cluster 1 as the reference
    result1 = gradients.make_premahalanobis_args(1, 
                gradients.compute_other_cluster_idx(1, data['k']), 
                    data['centers'], data['cov_inv'])

    assert result1['diff_mat'].shape == (data['p'], data['k']-1)
    assert result1['diff_tf_mat_1'].shape == (data['p'], data['k']-1)
    assert result1['diff_tf_mat_2'].shape == (data['p'], data['k']-1)

    assert np.allclose(
        result1['diff_mat'], 
        np.transpose(np.array([[0,1],[-1,1]]))
        )
    assert np.allclose(
        result1['diff_tf_mat_1'], 
        np.transpose(17 * np.array([[0,1],[-1,1]]))
        )
    assert np.allclose(
        result1['diff_tf_mat_2'], 
        np.transpose(np.array([[0,1],[-1e-6,1e-6]]))
        )


def test_gradient_vectorized(archetype):
    # extract the cluster centers and covariance structures
    p = archetype.dim
    k = archetype.n_clusters

    # test if gradient produces output in right format
    centers = np.random.multivariate_normal(
                mean=np.zeros(p), cov=(k**2)*np.eye(p), size=k
                )
    cov_inv = np.array([np.eye(p) for i in range(k)])
    premahal_args = gradients.make_premahalanobis_args(1, 
                        gradients.compute_other_cluster_idx(1, k), 
                            centers, cov_inv)
    mahal_args = gradients.make_mahalanobis_args(**premahal_args)
    result = gradients.gradient_vectorized(**premahal_args, 
                                           **mahal_args)
    assert result.shape == (p, k-1)

    # check that sign of gradient is right when clusters too close
    centers = np.array([[0,2],[0,-1]])
    cov_inv = [(1/4)*np.eye(2), np.eye(2)]
    premahal_args = gradients.make_premahalanobis_args(0, 
                        gradients.compute_other_cluster_idx(0, 2), 
                            centers, cov_inv)
    mahal_args = gradients.make_mahalanobis_args(**premahal_args)
    result = gradients.gradient_vectorized(
        **(premahal_args | mahal_args), mode='overlap'
        )
    assert np.allclose(result[0,0], 0)
    assert result[1,0] < 0
    # check numeric value of the gradient (computed by hand)
    assert np.all(
        np.allclose(result, chi2.pdf(1,df=2)*np.array([[0.],[-2/3]]))
        )

    # check that sign of gradient is right when clusters too far
    centers = np.array([[0,100],[0,-100]])
    cov_inv = [np.eye(2), np.eye(2)]
    premahal_args = gradients.make_premahalanobis_args(0, 
                        gradients.compute_other_cluster_idx(0, 2), 
                            centers, cov_inv)
    mahal_args = gradients.make_mahalanobis_args(**premahal_args)
    result = gradients.gradient_vectorized(
        **(premahal_args | mahal_args), mode='mharsum'
        )
    assert np.allclose(result[0,0], 0)
    assert result[1,0] > 0
    # check numeric value of the gradient (computed by hand)
    assert np.all(np.allclose(result, np.array([[0.],[0.5]])))


def test_compute_overlaps_vectorized():
    pass


def test_update_centers():
    # set some common parameters
    overlap_bounds = {'min': 0.001, 'max': 0.05}
    learning_rate = 0.1

    # two centers that overlap too much -> should be separated
    centers = np.array([[0.,1.],[0.,-1.]])
    cov_inv = [(1/4)*np.eye(2), np.eye(2)]
    premahal_args = gradients.make_premahalanobis_args(0, 
                        gradients.compute_other_cluster_idx(0, 2), 
                            centers, cov_inv)
    mahal_args = gradients.make_mahalanobis_args(**premahal_args)
    status = gradients.update_centers(
        0, centers, cov_inv, learning_rate, overlap_bounds
        )
    assert np.allclose(centers[0,0] - centers[1,0], 0)
    assert np.abs(centers[0,1] - centers[1,1]) > 2

    # two centers that are far away -> should be moved closer together
    centers = np.array([[0.,100.],[0.,-100.]])
    cov_inv = [np.eye(2), np.eye(2)]
    premahal_args = gradients.make_premahalanobis_args(0, 
                        gradients.compute_other_cluster_idx(0, 2), 
                            centers, cov_inv)
    mahal_args = gradients.make_mahalanobis_args(**premahal_args)
    status = gradients.update_centers(
        0, centers, cov_inv, learning_rate, overlap_bounds
        )
    assert np.allclose(centers[0,0] - centers[1,0], 0)
    assert np.abs(centers[0,1] - centers[1,1]) < 200.0


def test_assess_obs_overlap():
    centers = np.array([[0.,1.], [0.,0.], [0.,-10.]])
    cov_inv = [np.eye(2), np.eye(2), np.eye(2)]
    result = gradients.assess_obs_overlap(centers, cov_inv)
    assert result['min'] == 1 - chi2.cdf(25, df=2)
    assert result['max'] == 1 - chi2.cdf(1/4, df=2)


def test_cluster_loss():
    pass


def test_total_loss():
    pass