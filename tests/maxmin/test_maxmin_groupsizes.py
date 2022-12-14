import pytest
import numpy as np

from repliclust.base import Archetype
from repliclust.maxmin.groupsizes import MaxMinGroupSizeSampler
from repliclust.maxmin.groupsizes import _float_to_int

@pytest.fixture(params = np.linspace(1,10,10))
def setup_groupsize_sampler(request):
    return MaxMinGroupSizeSampler(request.param)


def test_init_MaxMinGroupSizeSampler():
    """
    Ensure imbalance ratio is properly specified.
    """
    groupsize_sampler = MaxMinGroupSizeSampler(imbalance_ratio=1)
    assert groupsize_sampler.imbalance_ratio >= 1

    # test input check for inappropriate arguments
    with pytest.raises(ValueError):
        MaxMinGroupSizeSampler(imbalance_ratio = 0.5)
        MaxMinGroupSizeSampler(imbalance_ratio = -2)


def test__float_to_int_1():
    """ 
    Make sure output sums to desired total. 
    Ensure output group sizes are integers.
    """
    # Provide some appropriate inputs.
    for total in [3,50]:
        grpsizes1 = _float_to_int(np.array([0.5, 0.3, 0.4]), total)
        grpsizes2 = _float_to_int(np.array([0.5, 51.5, 0.4]), total)
        grpsizes3 = _float_to_int(np.array([23, 0.7, 3.3]), total)
        grpsizes4 = _float_to_int(np.array([23, 0.1, 3.3]), total)
    for grpsizes in [grpsizes1, grpsizes2, grpsizes3, grpsizes4]:
        assert grpsizes.dtype == 'int'
        assert np.sum(grpsizes) == total

    # Provide total smaller than length of approximate group sizes.
    with pytest.raises(ValueError):
        _float_to_int(np.array([0.1, 0.3, 0.4]), 2)
    with pytest.raises(ValueError):
        _float_to_int(np.array([10, 50.1, 30]), 2)

    # Provide an approximate group size of zero.
    with pytest.raises(ValueError):
        _float_to_int(np.array([10, 0, 30]), 100)


def test__float_to_int_2():
    """
    More tests for _float_to_int.
    """
    # Test appropriate inputs.
    for (float_group_sz, total) in [
            (np.array([23.2, 254.7, 0.1, 35.6]), 100),
            (np.array([0.2, 0.7, 0.1, 0.5]), 10),
            (np.array([2.5,1.5,5.2]), 3),
            (np.array([0.5]), 1)
            ]:
        out = _float_to_int(float_group_sz, total)
        assert ((np.sum(out) == total) and (np.all(out >= 1))
                    and np.issubdtype(out.dtype,np.integer))

    # Test inputs that should be left unchanged.

    assert np.allclose(
             _float_to_int(np.array([5,10,25,7]), 5+10+25+7),
             np.sort(np.array([5,10,25,7]))
             )

    # Test inappropriate inputs.
    for float_group_sz, total in [(np.array([0.5,1.5]), 1),
                                      (np.array([0.5,1.5]), 0),
                                      (np.array([2.5,1.5,5.2]), 2)]:
        with pytest.raises(ValueError):
            _float_to_int(float_group_sz, total)


def test_sample_group_sizes():
    archetype_sm = Archetype(n_clusters=2, dim=10)
    archetype_lg = Archetype(n_clusters=111, dim=4)
    archetypes = [archetype_sm, archetype_lg]

    groupsize_sampler = MaxMinGroupSizeSampler(imbalance_ratio=1.5)

    # Test with appropriate input.
    Z_appropriate = [[500,5], [200,1], [100,2], 
                     [1000,10], [1500,3], [100,100]]
    args_appropriate = [{'total': z[0], 'n_clusters': z[1]} 
                            for z in Z_appropriate]
    for args in args_appropriate:
        for bp in archetypes:
            bp.n_clusters = args['n_clusters']
            out = groupsize_sampler.sample_group_sizes(bp, 
                                                        args['total'])
            assert (np.issubdtype(out.dtype, np.integer) 
                    and np.all(out >= 1) 
                    and (np.sum(out) == args['total']))

    # Test with inappropriate input.
    Z_inappropriate = [[500,0],[0,10],[100,-1],[-0.5,5],[10,11]]
    args_inappropriate = [{'total': z[0], 'n_clusters': z[1]} 
                            for z in Z_inappropriate]
    for args in args_inappropriate:
        for bp in archetypes:
            with pytest.raises(ValueError):
                bp.n_clusters = args['n_clusters']
                groupsize_sampler.sample_group_sizes(bp, args['total'])
