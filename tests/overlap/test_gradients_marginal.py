import pytest
import numpy as np
import repliclust.overlap._gradients_marginal as me
from repliclust.maxmin.archetype import MaxMinArchetype


@pytest.fixture
def archetype():
    arch = MaxMinArchetype()
    return arch


def test_make_marginal_args():
    cluster_idx = 0
    other_cluster_idx = 1
    centers = np.array([[1,0],[-1,0]])
    cov = [np.eye(2), np.eye(2)]

    p = centers.shape[1]
    k = centers.shape[0]

    diff_mat = np.transpose(centers[cluster_idx,:][np.newaxis,:]
                                - centers[other_cluster_idx,:])

    print(diff_mat)
    print(np.sum(diff_mat ** 2, axis=0))

    diff_mat_norms = np.sqrt(
                        np.sum(diff_mat ** 2, axis=0)
                     )[np.newaxis,:]
    assert np.all(diff_mat_norms > 0)
    assert(diff_mat_norms.shape == (1,k-1))
    diff_mat_unit = diff_mat / diff_mat_norms
    assert(diff_mat_unit.shape == (p,k-1))



def test_make_marginal_gradient(archetype):
    # extract the cluster centers and covariance structures
    p = archetype.dim
    k = archetype.n_clusters

    # test if gradient produces output in right format
    centers = np.random.multivariate_normal(
                mean=np.zeros(p), cov=(k**2)*np.eye(p), size=k
                )
    cov = np.array([np.eye(p) for i in range(k)])
    marginal_args = me.make_marginal_args(1, centers, cov)
    result = me.marginal_gradient_vec(**marginal_args)
    assert result.shape == (p, k-1)

    # check that sign of gradient is right when clusters too close
    centers = np.array([[0,2],[0,-1]])
    cov = [4*np.eye(2), np.eye(2)]
    marginal_args = me.make_marginal_args(0, centers, cov)
    result = me.marginal_gradient_vec(**marginal_args)

    assert np.allclose(result[0,0], 0)
    assert result[1,0] > 0
    # check numeric value of the gradient (computed by hand)
    assert np.all(
        np.allclose(result, np.array([[0.],[0.5]]))
        )

    # check that sign of gradient is right when clusters too far
    centers = np.array([[0,100],[0,-100]])
    cov_inv = [np.eye(2), np.eye(2)]
    marginal_args = me.make_marginal_args(0, centers, cov_inv)
    result = me.marginal_gradient_vec(**marginal_args)
    assert np.allclose(result[0,0], 0)
    assert result[1,0] > 0
    # check numeric value of the gradient (computed by hand)
    assert np.all(np.allclose(result, np.array([[0.],[0.75]])))


def test_update_centers():
    # set some common parameters
    overlap_bounds = {'min': 0.001, 'max': 0.05}
    learning_rate = 0.1

    # two centers that overlap too much -> should be separated
    centers = np.array([[0.,1.],[0.,-1.]])
    cov = [4*np.eye(2), np.eye(2)]
    marginal_args = me.make_marginal_args(0, centers, cov)
    status = me.update_centers(
        0, centers, cov, learning_rate, overlap_bounds
        )
    assert np.allclose(centers[0,0] - centers[1,0], 0)
    assert np.abs(centers[0,1] - centers[1,1]) > 2

    # two centers that are far away -> should be moved closer together
    centers = np.array([[0.,100.],[0.,-100.]])
    cov_inv = [np.eye(2), np.eye(2)]
    marginal_args = me.make_marginal_args(0, centers, cov)
    status = me.update_centers(
        0, centers, cov, learning_rate, overlap_bounds
        )
    assert np.allclose(centers[0,0] - centers[1,0], 0)
    assert np.abs(centers[0,1] - centers[1,1]) < 200.0



def test_ReLU_vec():
    """ Apply rectified linear unit x+ = max(x,0). """
    assert me.ReLU_vec(-1) == 0
    assert me.ReLU_vec(0) == 0
    assert me.ReLU_vec(1) == 1
    rand_vec = np.random.normal(size=50)
    assert np.allclose(me.ReLU_vec(rand_vec),
                       np.array([me.ReLU_vec(rand_vec[i]) 
                                 for i in range(50)]))
        

def test_poly_vec():
    """ Apply the polynomial p(x) = x + x**2. """
    assert me.poly_vec(-1) == 0
    assert me.poly_vec(0) == 0
    assert me.poly_vec(1) == 2
    assert me.poly_vec(3) == 12
    rand_vec = np.random.normal(size=50)
    assert np.allclose(me.poly_vec(rand_vec), 
                       np.array([me.poly_vec(rand_vec[i]) 
                                 for i in range(50)]))

def test_single_cluster_loss():
    """ Compute overlap loss for a reference cluster. """

    centers = np.array([[-4,0],
                        [0,0],
                        [2,0]])
    cov = [np.eye(2) for i in range(3)]

    overlap_bounds = {'min': 0.16, 'max': 0.32}
    assert me.single_cluster_loss(0, centers, cov, overlap_bounds) > 0
    assert me.single_cluster_loss(1, centers, cov, overlap_bounds) == 0
    assert me.single_cluster_loss(1, centers, cov, overlap_bounds) == 0

    overlap_bounds = {'min': 0.01, 'max': 0.10}
    assert me.single_cluster_loss(0, centers, cov, overlap_bounds) == 0
    assert me.single_cluster_loss(1, centers, cov, overlap_bounds) > 0
    assert me.single_cluster_loss(1, centers, cov, overlap_bounds) > 0


def test_overlap_loss():
    pass
