import pytest
import numpy as np

import repliclust
from repliclust.config import _rng
from repliclust import base, distributions
from repliclust.maxmin.covariance import MaxMinCovarianceSampler
from repliclust.maxmin.groupsizes import MaxMinGroupSizeSampler
from repliclust.maxmin.archetype import MaxMinArchetype
from repliclust.distributions import FixedProportionMix
from repliclust.overlap.centers import \
    ConstrainedOverlapCenters


def test_set_seed():
    arch = base.Archetype(n_clusters=40, dim=13, scale=2)
    cov_sampler = MaxMinCovarianceSampler(aspect_ref=1.5, aspect_maxmin=3, 
                                          radius_maxmin=5)
    center_sampler = ConstrainedOverlapCenters(max_overlap=0.2, 
                                               min_overlap=0.19,
                                               packing=0.1)
    base.set_seed(123)
    # First sample
    cov_out_1 = cov_sampler.sample_covariances(arch)
    arch._axes = cov_out_1[0]; arch._lengths = cov_out_1[1]
    centers_out_1 = center_sampler.sample_cluster_centers(arch)
    # Second sample
    cov_out_2 = cov_sampler.sample_covariances(arch)
    arch._axes = cov_out_2[0]; arch._lengths = cov_out_2[1]
    centers_out_2 = center_sampler.sample_cluster_centers(arch)
    base.set_seed(123)
    # Third sample
    cov_out_3 = cov_sampler.sample_covariances(arch)
    arch._axes = cov_out_3[0]; arch._lengths = cov_out_3[1]
    centers_out_3 = center_sampler.sample_cluster_centers(arch)

    assert not np.any(centers_out_1 == centers_out_2)
    assert np.all(centers_out_1 == centers_out_3)

    for i in range(arch.n_clusters):
        assert not np.any(cov_out_1[0][i] == cov_out_2[0][i])
        assert np.all(cov_out_1[1][i] == cov_out_3[1][i])


def test_get_supported_distributions():
    # test that we can call NumPy distributions with the stored
    # names and parameter values
    distr = base.get_supported_distributions()
    for distr_name, distr_params in distr.items():
        assert hasattr(_rng, distr_name)
        assert np.abs(getattr(_rng, distr_name)(**distr_params)) > 0


def test_CovarianceSampler():
    with pytest.raises(NotImplementedError):
        cov_sampler = base.CovarianceSampler()


def test_ClusterCenterSampler():
    with pytest.raises(NotImplementedError):
        center_sampler = base.ClusterCenterSampler()


def test_GroupSizeSampler():
    with pytest.raises(NotImplementedError):
        groupsize_sampler = base.GroupSizeSampler()


def test_SingleClusterDistribution():
    scdist = base.SingleClusterDistribution()
    with pytest.raises(NotImplementedError):
        scdist._sample_1d(10, 2)
    # Use subclass to test sample_cluster functionality. This test
    # assumes a correct implementation of _sample_1d in the subclass.
    center = np.array([10,1])
    axes = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    axis_lengths = np.array([10,0.1])
    scdist_normal = distributions.Normal()
    n_samples = 1000
    samples = base.SingleClusterDistribution.sample_cluster(
                scdist_normal, n_samples, 
                center=center, axes=axes, axis_lengths=axis_lengths
                )
    assert samples.shape == (n_samples, 2)
    assert np.allclose(np.mean(samples, axis=0), center, 
                       atol=10*10/np.sqrt(n_samples))

    with pytest.raises(ValueError):
        # provide cluster center as a matrix
        base.SingleClusterDistribution.sample_cluster(
            scdist_normal, 100, 
                center=center[np.newaxis,:], axes=axes, 
                axis_lengths=axis_lengths
                )
    with pytest.raises(ValueError):
        # provide too many axis lengths
        base.SingleClusterDistribution.sample_cluster(
            scdist_normal, 100, 
                center=center, axes=axes, 
                axis_lengths=np.array([0,1,2])
                )

    
def test_DistributionMix():
    distr_mix = base.DistributionMix()
    with pytest.raises(NotImplementedError):
        distr_mix.assign_distributions(10)

def test_MixtureModel():
    k = 7
    p = 3
    centers = np.random.multivariate_normal(mean=np.zeros(p), 
                                            cov=10*np.eye(p), size=7)
    axes_list = [np.eye(p) for i in range(k)]
    axis_lengths_list = [np.ones(p) for i in range(k)]
    distr_list = (distributions.FixedProportionMix()
                    .assign_distributions(k))
    mixture_model = base.MixtureModel(
                        centers=centers, axes_list=axes_list,
                        axis_lengths_list=axis_lengths_list,
                        distributions_list=distr_list
                        )
    group_sizes = np.array(10*np.arange(0,k+1))
    with pytest.raises(ValueError):
        # group sizes has k+1 elements rather than k
        mixture_model.sample_data(group_sizes=group_sizes)
        
    group_sizes = np.array(10*np.arange(1,k+1))
    sampled_data, sampled_data_labels = mixture_model.sample_data(
        group_sizes=group_sizes
        )
    assert sampled_data.shape == (np.sum(group_sizes), p)
    assert sampled_data_labels.shape == (np.sum(group_sizes),)
    assert np.allclose(sampled_data_labels, 
                        np.repeat(np.arange(k), repeats=group_sizes))


def test_Archetype():
    # test making archetype with 1 cluster
    arch = base.Archetype(n_clusters=1,dim=10,scale=1,
                        covariance_sampler=MaxMinCovarianceSampler(),
                        center_sampler=ConstrainedOverlapCenters(),
                        groupsize_sampler=MaxMinGroupSizeSampler(),
                        distribution_mix=FixedProportionMix())
    mixture_model = arch.sample_mixture_model()

    # test making archetype with more clusters
    arch = base.Archetype(n_clusters=17,dim=241,scale=2,
                        covariance_sampler=MaxMinCovarianceSampler(),
                        center_sampler=ConstrainedOverlapCenters(),
                        groupsize_sampler=MaxMinGroupSizeSampler(),
                        distribution_mix=FixedProportionMix())
    mixture_model = arch.sample_mixture_model()

    # test that archetype properly adds custom attributes from children
    arch = base.Archetype(n_clusters=2,dim=3,scale=1,my_special_arg=1337)
    assert hasattr(arch, 'my_special_arg')
    assert arch.my_special_arg == 1337


def test_DataGenerator():
    # construct a data generator
    dg = base.DataGenerator(archetype = MaxMinArchetype())
    assert hasattr(dg, '_archetypes')
    assert isinstance(dg._archetypes, list)

    # test for catching invalid arguments


    # test the .synthesize interface for one archetype

    X, y, arch = dg.synthesize()
    assert X.shape[0] == MaxMinArchetype().n_samples
    assert X.shape[0] == y.shape[0]
    assert (len(X.shape) == 2) and (len(y.shape) == 1)
    assert arch.name == 'archetype0'

    X, y, arch = dg.synthesize(n_samples=101)
    assert X.shape[0] == 101
    assert X.shape[0] == y.shape[0]
    assert (len(X.shape) == 2) and (len(y.shape) == 1)
    assert arch.name == 'archetype0'


    # test the callable generator interface for one archetype
    count = dg._n_datasets
    for X, y, arch in dg():
        assert X.shape[0] == MaxMinArchetype().n_samples
        assert X.shape[0] == y.shape[0]
        assert (len(X.shape) == 2) and (len(y.shape) == 1)
        assert arch.name == 'archetype0'
        count -= 1
    assert count == 0

    count = 5
    for X, y, arch in dg(n_datasets=5):
        assert X.shape[0] == MaxMinArchetype().n_samples
        assert X.shape[0] == y.shape[0]
        assert (len(X.shape) == 2) and (len(y.shape) == 1)
        assert arch.name == 'archetype0'
        count -= 1
    assert count == 0

    count = 13
    for X, y, arch in dg(n_datasets=13, n_samples=77):
        assert X.shape[0] == 77
        assert X.shape[0] == y.shape[0]
        assert (len(X.shape) == 2) and (len(y.shape) == 1)
        assert arch.name == 'archetype0'
        count -= 1
    assert count == 0

    # test the iterator interface for a single archetype
    count = dg._n_datasets
    for X, y, arch in dg:
        assert X.shape[0] == MaxMinArchetype().n_samples
        assert X.shape[0] == y.shape[0]
        assert (len(X.shape) == 2) and (len(y.shape) == 1)
        assert arch.name == 'archetype0'
        count -= 1
    assert count == 0

    dg._n_datasets = 33
    count = 33
    for X, y, arch in dg:
        assert X.shape[0] == MaxMinArchetype().n_samples
        assert X.shape[0] == y.shape[0]
        assert (len(X.shape) == 2) and (len(y.shape) == 1)
        assert arch.name == 'archetype0'
        count -= 1
    assert count == 0
    
    # SETUP TESTS FOR MULTIPLE ARCHETYPES
    arch_dict = {'oblong':MaxMinArchetype(aspect_ref=5, n_samples=111),
                 'spherical':MaxMinArchetype(aspect_ref=1,
                                             n_samples=222)}
    arch_list = [MaxMinArchetype(aspect_ref=5, n_samples=333),
                 MaxMinArchetype(aspect_ref=1, n_samples=444)]
    dg_multi_dict = base.DataGenerator(archetype=arch_dict,
                                       n_datasets=13)
    dg_multi_list = base.DataGenerator(archetype=arch_list)


    # test the .synthesize interface for multiple archetypes
    for i in range(2):
        X, y, arch = dg_multi_dict.synthesize()
        if (arch.name == "oblong"):
            assert X.shape[0] == 111
            assert X.shape[0] == y.shape[0]
        elif (arch.name == "spherical"):
            assert X.shape[0] == 222
            assert X.shape[0] == y.shape[0]
        else:
            raise ValueError("archetype name incorrect")
        assert (len(X.shape) == 2) and (len(y.shape) == 1)
    
    for i in range(2):
        X, y, arch = dg_multi_list.synthesize()
        if (arch.name == "archetype0"):
            assert X.shape[0] == 333
            assert X.shape[0] == y.shape[0]
        elif (arch.name == "archetype1"):
            assert X.shape[0] == 444
            assert X.shape[0] == y.shape[0]
        else:
            raise ValueError("archetype name incorrect")
        assert (len(X.shape) == 2) and (len(y.shape) == 1)

    # now override the number of samples for all archetypes

    n_samples = 717
    for i in range(2):
        X, y, arch = dg_multi_dict.synthesize(n_samples=n_samples)
        if (arch.name == "oblong"):
            assert X.shape[0] == n_samples
            assert X.shape[0] == y.shape[0]
        elif (arch.name == "spherical"):
            assert X.shape[0] == n_samples
            assert X.shape[0] == y.shape[0]
        else:
            raise ValueError("archetype name incorrect")
        assert (len(X.shape) == 2) and (len(y.shape) == 1)
    
    for i in range(2):
        X, y, arch = dg_multi_list.synthesize(n_samples=n_samples)
        if (arch.name == "archetype0"):
            assert X.shape[0] == n_samples
            assert X.shape[0] == y.shape[0]
        elif (arch.name == "archetype1"):
            assert X.shape[0] == n_samples
            assert X.shape[0] == y.shape[0]
        else:
            raise ValueError("archetype name incorrect")
        assert (len(X.shape) == 2) and (len(y.shape) == 1)

    # Test that the underlying representation (after construction) is
    # the same for list-specified and dict-specified archetype 
    # collections. Because of that, subsequently test only one.
    assert (sorted([arch_name for arch_name, _ \
                    in dg_multi_list._archetypes])
                == ['archetype0', 'archetype1'])
    assert isinstance(dg_multi_dict._archetypes, list)
    assert isinstance(dg_multi_list._archetypes, list)
    assert len(dg_multi_list._archetypes) == 2
    assert len(dg_multi_dict._archetypes) == 2
    assert isinstance(dg_multi_list._archetypes[0], tuple)
    assert isinstance(dg_multi_dict._archetypes[0], tuple)
    assert len(dg_multi_list._archetypes[0]) == 2
    assert len(dg_multi_dict._archetypes[0]) == 2

    # Test the callable generator interface for multiple archetypes

    n_datasets = dg_multi_dict._n_datasets
    count = n_datasets
    for X, y, arch in dg_multi_dict(n_samples=123):
        assert X.shape[0] == 123
        assert X.shape[0] == y.shape[0]
        assert (len(X.shape) == 2) and (len(y.shape) == 1)
        assert arch.name == ("oblong" if (count % 2) == (n_datasets % 2)
                                else "spherical")
        count -= 1
    assert count == 0

    n_datasets = dg_multi_dict._n_datasets
    count = n_datasets
    n_samples_list = [10, 20]*int(np.floor(count/2))
    if (count % 2 == 1):
        n_samples_list.append(10)

    for X, y, arch in dg_multi_dict(n_samples=n_samples_list):
        assert X.shape[0] == n_samples_list[int(n_datasets-count)]
        assert X.shape[0] == y.shape[0]
        assert (len(X.shape) == 2) and (len(y.shape) == 1)
        assert arch.name == ("oblong" if (count % 2) == (n_datasets % 2)
                                else "spherical")
        count -= 1
    assert count == 0

    count = 5
    for X, y, arch in dg_multi_dict(n_datasets=5):
        if arch.name == 'oblong':
            assert X.shape[0] == 111
            assert X.shape[0] == y.shape[0]
            assert (len(X.shape) == 2) and (len(y.shape) == 1)
        elif arch.name == 'spherical':
            assert X.shape[0] == 222
            assert X.shape[0] == y.shape[0]
            assert (len(X.shape) == 2) and (len(y.shape) == 1)  
        count -= 1
    assert count == 0

    dg_multi_dict._next_archetype_idx == 0
    count = 3
    for X, y, arch in dg_multi_dict(n_datasets=3, n_samples=33):
        assert X.shape[0] == 33
        assert X.shape[0] == y.shape[0]
        assert (len(X.shape) == 2) and (len(y.shape) == 1)
        assert arch.name == "oblong" if (count % 2 == 1) else "spherical"
        count -= 1
    assert count == 0

    # test the iterator interface for multiple archetypes
    count = dg_multi_dict._n_datasets
    dg_multi_dict._next_archetype_idx = 0
    for X, y, arch in dg_multi_dict:
        assert X.shape[0] == (111 if
            (count - dg_multi_dict._n_datasets) % 2 == 0 else 222)
        assert X.shape[0] == y.shape[0]
        assert (len(X.shape) == 2) and (len(y.shape) == 1)
        assert arch.name == ('oblong' if 
            (count - dg_multi_dict._n_datasets) % 2 == 0 
                else 'spherical')
        count -= 1
    assert count == 0

    dg_multi_dict._n_datasets = 33
    dg_multi_dict._next_archetype_idx = 0
    count = 33
    for X, y, arch in dg_multi_dict:
        assert X.shape[0] == (111 if
            (count - dg_multi_dict._n_datasets) % 2 == 0 else 222)
        assert X.shape[0] == y.shape[0]
        assert (len(X.shape) == 2) and (len(y.shape) == 1)
        assert arch.name == ('oblong' if 
            (count - dg_multi_dict._n_datasets) % 2 == 0
                else 'spherical')
        count -= 1
    assert count == 0