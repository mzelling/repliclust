import pytest

import numpy as np

import repliclust
from repliclust.utils import log_volume, radius_from_log_volume
from repliclust.utils import sample_unit_vectors, make_orthonormal_axes
from repliclust.utils import assemble_covariance_matrix

def check_seed(seed, fn, **kwargs):
        repliclust.set_seed(seed)
        out1 = fn(**kwargs)
        out2 = fn(**kwargs)
        repliclust.set_seed(seed)
        out3 = fn(**kwargs)
        assert ((not np.allclose(out1, out2)) 
                    and np.allclose(out1, out3))

def test_log_volume():
    assert np.allclose(log_volume(1,2), np.log(np.pi))
    assert np.allclose(log_volume(4,3), np.log((4/3)*np.pi*(4**3)))
    assert np.allclose(log_volume(0,10), -np.Inf)
    assert np.allclose(log_volume(1,1), np.log(2))

def test_radius_from_log_volume():
    assert np.allclose(radius_from_log_volume(np.log(np.pi), 2), 1)
    assert np.allclose(radius_from_log_volume(
                np.log((4/3)*np.pi*(4**3)), 3
                ), 4)
    assert np.allclose(radius_from_log_volume(-np.Inf, 10), 0)
    assert np.allclose(radius_from_log_volume(np.log(2), 1), 1)

def test_sample_unit_vectors():
    # test shape
    assert sample_unit_vectors(10,3).shape == (10,3)
    assert sample_unit_vectors(1,2).shape == (1,2)

    # test random seed
    repliclust.set_seed(1)
    out1 = sample_unit_vectors(20,20)
    out2 = sample_unit_vectors(20,20)
    repliclust.set_seed(1)
    out3 = sample_unit_vectors(20,20)
    assert (not np.allclose(out1, out2)) and np.allclose(out1, out3)

    # test that vectors have unit length
    out = sample_unit_vectors(50,100)
    for i in range(50):
        assert np.allclose(np.linalg.norm(out[i,:]), 1)

def test_assemble_covariance_matrix():
    with pytest.raises(ValueError):
        # axes are not square
        assemble_covariance_matrix(axes=np.array([2, 1]), 
                                   axis_lengths=np.array([1]))
        # mismatch between dimension of axes and axis_lengths
        assemble_covariance_matrix(axes=np.eye(4),
                                   axis_lengths=np.array([1,2]))
    axes = np.eye(3)
    axis_lengths = np.array([1,2,3])
    cov_inv = assemble_covariance_matrix(
                axes, axis_lengths, inverse=True)
    cov = assemble_covariance_matrix(
                axes, axis_lengths, inverse=False)
    # test that axes are eigenvectors of cov
    # test that eigenvalues of axes are the squares of axis_lengths
    assert np.allclose(cov @ np.transpose(axes), 
                       np.diag(axis_lengths**2) @ np.transpose(axes))
    assert np.allclose(
                cov_inv @ np.transpose(axes), 
                np.diag(1/(axis_lengths**2)) @ np.transpose(axes)           
                )

def test_make_orthonormal_axes():
    # test exceptions
    with pytest.raises(ValueError):
        # attempt to make more axes than there are dimensions
        make_orthonormal_axes(10,2)

    # test shape
    assert make_orthonormal_axes(5,10).shape == (5,10)

    # test random seed
    check_seed(76, make_orthonormal_axes, n=10, dim=200)

    # test that the output matrix is actually orthonormal
    def check_orthonormal(matrix):
        assert np.allclose(matrix @ np.transpose(matrix), 
                           np.eye(matrix.shape[0]))
        if (matrix.shape[0] < matrix.shape[1]):
            assert not np.allclose(np.transpose(matrix) @ matrix, 
                                np.eye(matrix.shape[1]))

    check_orthonormal(make_orthonormal_axes(2,3))
    check_orthonormal(make_orthonormal_axes(3,3))
    check_orthonormal(make_orthonormal_axes(1,10))

