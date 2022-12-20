""" This module provides utility functions for repliclust. """

import numpy as np
from scipy.special import loggamma
from scipy.stats import ortho_group

from repliclust import config

def log_volume(radius: float, dim: int):
    """
    Compute the logarithm of a sphere's volume.

    Parameters
    ----------
    radius : float
        The radius of the sphere.
    dim : int
        The number of dimensions.

    Returns
    -------
    float
        The natural log of the volume of the sphere.
    """
    return dim*np.log(radius*np.sqrt(np.pi)) - loggamma((dim/2) + 1)


def radius_from_log_volume(log_volume: float, dim: int):
    """ 
    Compute the radius of a sphere for which the logarithm of volume
    is given.

    Parameters
    ----------
    log_volume : float
        The natural log volume of the sphere's volume.
    dim : int
        The number of dimensions.

    Returns
    -------
    float
        The radius of the sphere.
    """
    return np.exp(
        (loggamma(1 + (dim/2)) + log_volume) / dim
        ) / np.sqrt(np.pi)


def sample_unit_vectors(n: int, dim: int):
    """ 
    Sample `n` unit vectors in `dim` dimensions.

    Parameters
    ----------
    n : int
        The number of unit vectors to sample.
    dim : int
        The number of dimensions for each unit vector.

    Returns
    -------
    ndarray
        The unit vectors arranged as a matrix, where each row is a unit 
        vector. The output shape is `n` by `dim`.
    """
    # Sample Gaussian vectors and then normalize to unit length
    gaussian_vectors = config._rng.multivariate_normal(
        mean=np.zeros(dim), cov=np.eye(dim), size=n
        )
    return np.divide(
        gaussian_vectors, 
        np.sqrt(np.sum(gaussian_vectors**2, axis=1))[:,np.newaxis]
    )


def assemble_covariance_matrix(axes, axis_lengths, inverse=False):
    """
    Compute the covariance matrix or inverse covariance matrix
    corresponding to the given axes and their lengths.

    Parameters
    ----------
    axes : ndarray (p, p)
        A full set of orthonormal vectors (in an arbitrary number of
        dimensions p). Each row is a vector.
    axis_lengths : ndarray (p, )
        Desired lengths for the vectors in axes. The i-th element is the
        length of axes[i,:].
    inverse : bool (default=False)
        If True, output the inverse of the covariance matrix.
        Otherwise, output the covariance matrix.
    Returns
    -------
    cov : ndarray
        The covariance matrix with given axes (eigenvectors) and
        axis lengths (standard deviations, square root  of eigenvalues).
    """
    if (len(axes.shape) <= 1) or (axes.shape[0] != axes.shape[1]):
        raise ValueError("the desired axes must be a square matrix.")

    if ((len(axis_lengths.shape) != 1) 
            or (axis_lengths.shape[0] != axes.shape[0])):
        raise ValueError("length of axis_lengths must match the row/"
                + "column dimension of axes.")

    if inverse:
        return np.transpose(axes) @ np.diag((1/axis_lengths)**2) @ axes
    else:
        return np.transpose(axes) @ np.diag(axis_lengths**2) @ axes


def make_orthonormal_axes(n, dim):
    """
    Sample n_axes orthonormal axes in n_dim dimensions.

    Parameters
    ----------
    n : int
        Number of axes
    dim : int
        Number of dimensions of each axis

    Returns
    -------
    out : (n, dim) ndarray
        Orthonormal axes arranged as a matrix. Each row is a matrix.
    """
    if (n > dim):
        raise ValueError("number of orthonormal axes must not"
                + " be greater than the number of dimensions.")
    return ortho_group.rvs(dim,random_state=config._rng)[:n, :]
    