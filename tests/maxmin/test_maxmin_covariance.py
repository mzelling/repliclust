import pytest

import numpy as np

import repliclust
from repliclust.base import Archetype
from repliclust.maxmin.covariance import MaxMinCovarianceSampler

def test_init_MaxMinCovarianceSampler():
    """
    Make sure to catch illicit parameter values during instantiation.
    """

    # Try appropriate values for the parameters
    interior_cases = np.random.uniform(1,10,size=(100,3))
    edge_cases = np.concatenate([2-np.eye(3), np.ones(3)[np.newaxis,:]],
                                axis=0)
    Z_appropriate = np.concatenate([interior_cases,edge_cases], axis=0)
    args_appropriate = [{'aspect_ref': z[0], 'aspect_maxmin': z[1], 
                        'radius_maxmin': z[2]} for z in Z_appropriate]
    for args in args_appropriate:
        my_cov_sampler = MaxMinCovarianceSampler(**args)
        for attr in ['aspect_ref','aspect_maxmin','radius_maxmin']:
            assert hasattr(my_cov_sampler, attr)

    # Try invalid values for the parameters
    Z_inappropriate = np.concatenate(
                            [np.ones(3) - 0.5*np.eye(3), 
                             (1-0.01)*np.ones(3)[np.newaxis,:]]
                            )
    args_inappropriate = [{'aspect_ref': z[0], 
                           'aspect_maxmin': z[1], 
                           'radius_maxmin': z[2]} 
                           for z in Z_inappropriate]
    with pytest.raises(ValueError):
        for args in args_inappropriate:
            MaxMinCovarianceSampler(**args)


@pytest.fixture()
def setup_cov_sampler():
    """
    Initialize a valid MaxMinCov instance to test its methods.
    """
    cov_sampler = MaxMinCovarianceSampler(aspect_ref=1.5,
                                        aspect_maxmin=1.5,
                                        radius_maxmin=1.5)
    yield cov_sampler


def test_make_cluster_aspects(setup_cov_sampler):
    """
    Make sure that valid cluster aspect ratios are sampled.

    Test the range of acceptable numbers of clusters, and
    make sure setting a seed works.
    """
    cov_sampler = setup_cov_sampler

    with pytest.raises(ValueError):
        cov_sampler.make_cluster_aspect_ratios(0)
        cov_sampler.make_cluster_aspect_ratios(0.99)

    # test different numbers of clusters
    for n_clusters in range(1,100):
        cluster_aspects = cov_sampler.make_cluster_aspect_ratios(
                            n_clusters)
        assert np.all(cluster_aspects >= 1)
        assert np.max(cluster_aspects) >= cov_sampler.aspect_ref
        assert np.min(cluster_aspects) <= cov_sampler.aspect_ref

    # test seed
    seed = 23
    for i in range(10):
        repliclust.set_seed(23)
        cluster_aspects_new = cov_sampler.make_cluster_aspect_ratios(2)
        # make sure that each output is the same as previous output
        if i >= 1:
            assert np.all(cluster_aspects_new == cluster_aspects_prev)
        cluster_aspects_prev = cluster_aspects_new


def test_make_cluster_radii(setup_cov_sampler):
    """
    Make sure that the output gives valid cluster radii.
    Test the range of acceptable inputs.
    Make sure setting a seed works.
    """
    cov_sampler = setup_cov_sampler

    # Test appropriate inputs.
    interior_cases = np.concatenate(
                        [np.arange(1,20+1)[:,np.newaxis], 
                         np.random.uniform(0,10,size=20)[:,np.newaxis], 
                         np.random.choice(np.arange(2,100),
                            size=20)[:,np.newaxis]], 
                        axis=1)
    edge_cases = np.array([[1,1e-3,2], [1,1e-3,1],[2,100,1]])
    Z_appropriate = np.concatenate([interior_cases, edge_cases],axis=0)
    args_appropriate = [{'n_clusters': z[0], 
                         'ref_radius': z[1], 
                         'dim': z[2]} for z in Z_appropriate]

    for args in args_appropriate:
        tol = 1e-12
        print(args)
        cluster_radii = cov_sampler.make_cluster_radii(**args)
        print(cluster_radii)
        assert np.all(cluster_radii > 0)
        assert ((np.min(cluster_radii) <= args['ref_radius'] + tol)
                and (np.max(cluster_radii) >= args['ref_radius'] - tol))

    # Test inappropriate inputs.
    with pytest.raises(ValueError):
        cov_sampler.make_cluster_radii(n_clusters=0, ref_radius=1, 
                                        dim=10)
        cov_sampler.make_cluster_radii(n_clusters=1, ref_radius=0, 
                                        dim=10)
        cov_sampler.make_cluster_radii(n_clusters=1, ref_radius=1, 
                                        dim=0)

    # Test setting random seeds.
    seed = 717
    for i in range(10):
        repliclust.set_seed(717)
        cluster_radii_new = cov_sampler.make_cluster_radii(
            n_clusters=5,ref_radius=4, dim=25)
        if (i >= 1):
            assert np.all(cluster_radii_new == cluster_radii_prev)
        cluster_radii_prev = cluster_radii_new


def test_make_axis_lengths(setup_cov_sampler):
    """
    Make sure that the output gives valid axis lengths (>0).
    Ensure sure reference axis length lies between min and max
    Verify that the maxmin ratio equals the desired aspect ratio.
    """
    cov_sampler = setup_cov_sampler

    # Test appropriate inputs.
    interior_cases = np.concatenate(
                        [np.arange(2,50+2)[:,np.newaxis], 
                         np.random.uniform(0,10,size=50)[:,np.newaxis], 
                         np.random.uniform(1,10,size=50)[:,np.newaxis]], 
                        axis=1
                        )
    edge_cases = np.array([[1,0.5,1.5], [1,0.5,1], [2,0.1,1]])
    Z_appropriate = np.concatenate([interior_cases, edge_cases], axis=0)
    args_appropriate = [{'n_axes': z[0],
                         'reference_length': z[1],
                         'aspect_ratio': z[2]} 
                            for z in Z_appropriate]

    for args in args_appropriate:
        out = cov_sampler.make_axis_lengths(**args)
        assert ((np.min(out) <= args['reference_length']) 
                 and (np.max(out) >= args['reference_length']))

    # Test inappropriate inputs.
    with pytest.raises(ValueError):
        cov_sampler.make_axis_lengths(
                        n_axes=0, reference_length=1, aspect_ratio=2)
        cov_sampler.make_axis_lengths(
                        n_axes=0.5, reference_length=0, aspect_ratio=2)
        cov_sampler.make_axis_lengths(
                        n_axes=1, reference_length=1, aspect_ratio=0.5)
        cov_sampler.make_axis_lengths(
                        n_axes=2, reference_length=1, aspect_ratio=-2)
        cov_sampler.make_axis_lengths(
                        n_axes=2, reference_length=-1, aspect_ratio=2)

    # Test setting a random seed.
    seed = 123
    for i in range(10):
        repliclust.set_seed(seed)
        axis_lengths_new = cov_sampler.make_axis_lengths(
                                        n_axes=5, reference_length=4, 
                                        aspect_ratio=25
                                        )
        if (i >= 1):
            assert np.all(axis_lengths_new == axis_lengths_prev)
        axis_lengths_prev = axis_lengths_new


def test_sample_covariances(setup_cov_sampler):
    """
    Make sure the principal axes are orthogonal.
    """
    cov_sampler = setup_cov_sampler
    archetype = Archetype(n_clusters=52, dim=10)

    # Ensure output makes mathematical sense.
    for i in range(10):
        (axes_list, axis_lengths_list) = (cov_sampler
                                         .sample_covariances(archetype))

        for cluster_idx in range(archetype.n_clusters):
            # Test orthogonality of cluster axes
            assert np.all(
                np.allclose(axes_list[cluster_idx] 
                                @ np.transpose(axes_list[cluster_idx]),
                            np.eye(axes_list[cluster_idx].shape[0])))

    # Test setting random seeds.
    seed = 123
    for i in range(10):
        repliclust.set_seed(seed)
        cov_structure_new = cov_sampler.sample_covariances(archetype)
        if (i >= 1):
            for cluster_idx in range(archetype.n_clusters):
                # Iterate through axes_list and axis_lengths_list
                for j in range(2): 
                    assert np.all(
                        np.allclose(cov_structure_prev[j][cluster_idx],
                                    cov_structure_new[j][cluster_idx]))
        cov_structure_prev = cov_structure_new