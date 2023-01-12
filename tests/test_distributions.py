import pytest

import numpy as np
from scipy.stats import t

import repliclust
from repliclust import config, distributions
from repliclust.base import SingleClusterDistribution
from repliclust.distributions import QUANTILE_LEVEL
from repliclust.distributions import SUPPORTED_DISTRIBUTION_NAMES


# store example parameters for testing all supported distributions
DISTRIBUTION_TEST_PARAMS = {
    "normal": {},
    "standard_t": {'df': 3},
    "exponential": {},
    "weibull": {'a': 2},
    "gamma": {'scale': 2, 'shape': 5},
    "gumbel": {'loc': 5, 'scale': 2},
    "beta": {'a': 0.2, 'b': 0.7},
    "lognormal": {'mean': 1, 'sigma': 1},
    "chisquare": {'df': 10},
    "f": {'dfnum': 5, 'dfden': 10},
    "pareto": {'a': 0.1}
}


def make_distribution_test(
        distribution_name: str, n_sample_basic: int, 
        n_sample_quantile: int, **params
    ):
    # test sampling
    distr = distributions.DistributionFromNumPy(distribution_name, 
                                                **params)
    sample = distr._sample_1d(n_sample_basic,10)
    assert sample.shape == (n_sample_basic,)
    
    # test that seed works
    repliclust.set_seed(777)
    sample1 = distr._sample_1d(n_sample_basic, 100)
    sample2 = distr._sample_1d(n_sample_basic, 100)
    repliclust.set_seed(777)
    sample3 = distr._sample_1d(n_sample_basic, 100)
    assert not np.all(sample1 == sample2)
    assert np.all(sample1 == sample3)

    # test that the QUANTILE_LEVEL is 1
    sample = distr._sample_1d(n_sample_quantile, 1)
    assert np.allclose(np.quantile(sample, q=QUANTILE_LEVEL), 1,
                        rtol=1e-1)


def test_DistributionFromNumPy():
    n_sample_quantile = 1234567
    n_sample_basic = 100
    # imitate StandardT
    df = 5
    t_imitation = distributions.DistributionFromNumPy(
        'standard_t', df=df)
    print(t_imitation)
    t_real = distributions.StandardT(df = df)
    test_quantiles = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    assert np.allclose(
        np.quantile(t_imitation._sample_1d(n_sample_quantile,10),
                    q=test_quantiles),
        np.quantile(t_real._sample_1d(n_sample_quantile,10),
                    q=test_quantiles),
        rtol=1e-2
    )

    # test all supported distributions
    for distr in SUPPORTED_DISTRIBUTION_NAMES:
        make_distribution_test(distribution_name=distr, 
                                n_sample_basic=123,
                                n_sample_quantile=1234567,
                                **(DISTRIBUTION_TEST_PARAMS[distr]))


def test_Normal():
    # test sampling
    mvr = distributions.Normal()
    sample = mvr._sample_1d(100,1)
    assert sample.shape == (100,)
    assert np.abs(np.mean(sample)) < 10/np.sqrt(len(sample))

    # test that seed works
    repliclust.set_seed(123)
    sample1 = mvr._sample_1d(100,2)
    sample2 = mvr._sample_1d(100,2)
    repliclust.set_seed(123)
    sample3 = mvr._sample_1d(100,2)
    assert not np.all(sample1 == sample2)
    assert np.all(sample1 == sample3)


def test_Exponential():
    # test sampling
    exp = distributions.Exponential()
    sample = exp._sample_1d(100,1)
    assert sample.shape == (100,)
    assert np.abs(np.mean(sample) - 1) < 10

    # test that seed works
    repliclust.set_seed(123)
    sample1 = exp._sample_1d(100, 3)
    sample2 = exp._sample_1d(100, 3)
    repliclust.set_seed(123)
    sample3 = exp._sample_1d(100, 3)
    assert not np.all(sample1 == sample2)
    assert np.all(sample1 == sample3)


def test_StandardT():
    # test sampling
    repliclust.set_seed(None)
    t_1 = distributions.StandardT(df=1)
    t_10 = distributions.StandardT(df=10)
    sample_1 = t_1._sample_1d(123456, 1)
    sample_10 = t_10._sample_1d(123456, 1)
    assert np.allclose(np.quantile(sample_1, q=QUANTILE_LEVEL), 1, 
                                   rtol=1e-1)
    assert np.allclose(np.quantile(sample_10, q=QUANTILE_LEVEL), 1,
                                   rtol=1e-1)
    assert (sample_1.shape == (123456,) and
            sample_10.shape == (123456,))
    assert np.max(sample_1) > np.max(sample_10)

    # test that seed works
    repliclust.set_seed(123)
    sample1 = t_1._sample_1d(100, 2)
    sample2 = t_1._sample_1d(100, 2)
    repliclust.set_seed(123)
    sample3 = t_1._sample_1d(100, 2)
    assert not np.all(sample1 == sample2)
    assert np.all(sample1 == sample3)


def test_parse_distribution():
    # test error handling when distribution is invalid
    distr = "invalid_distribution"
    with pytest.raises(ValueError):
        distributions.parse_distribution(
            distr_name = distr, params = {})
    
    # test Normal, Exponential, StandardT
    assert isinstance(distributions.parse_distribution("normal"),
                      distributions.Normal)
    assert isinstance(distributions.parse_distribution("exponential"),
                      distributions.Exponential)
    t_distr = distributions.parse_distribution(
                        "standard_t", params={'df': 173})
    assert isinstance(t_distr, distributions.StandardT)
    assert t_distr.params['df'] == 173

    # test distributions from NumPy
    for distr in SUPPORTED_DISTRIBUTION_NAMES:
        my_distr = distributions.parse_distribution(
                            distr, DISTRIBUTION_TEST_PARAMS[distr])
        # exclude the special case distributions from isinstance test
        if distr not in {'normal', 'exponential', 'standard_t'}:
            assert isinstance(my_distr, 
                              distributions.DistributionFromNumPy)
        assert my_distr.params == DISTRIBUTION_TEST_PARAMS[distr]


def test_FixedProportionMix():
    with pytest.raises(ValueError):
        # provide empty list
        distr = distributions.FixedProportionMix(distributions=[])
    with pytest.raises(ValueError):
        # not provide list
        distr = distributions.FixedProportionMix(
            distributions=distributions.Normal())
            
    my_fix_solo_wo_param = distributions.FixedProportionMix(
        [('normal', 1.0, {})])
    my_fix_solo_w_param = distributions.FixedProportionMix(
        [('standard_t', 1.0, {'df': 1})])
    my_fix_multiple = distributions.FixedProportionMix(
        [('normal', 7, {}), ('exponential', 2, {}),
         ('standard_t', 1, {'df': 3})]
    )
    out1 = my_fix_solo_wo_param.assign_distributions(n_clusters=3)
    out2 = my_fix_solo_w_param.assign_distributions(n_clusters=2)
    out3 = my_fix_multiple.assign_distributions(n_clusters=10)

    # test that the output of assign_distributions is a list whose
    # elements are of class SingleClusterDistribution
    for output in [out1,out2,out3]:
        assert isinstance(out1, list)
        for elem in output:
            assert isinstance(elem, SingleClusterDistribution)

    # test that outputs are correct
    assert len(out1) == 3
    assert len(out2) == 2
    assert len(out3) == 10

    # test that distributions in outputs are correct
    assert isinstance(out1[0], distributions.Normal)
    assert isinstance(out2[0], distributions.StandardT)

    # test proportions in out3 are correct
    def make_count(output):
        standard_t_count = 0
        normal_count = 0
        exponential_count = 0
        for elem in output:
            if isinstance(elem, distributions.Normal):
                normal_count += 1
            elif isinstance(elem, distributions.Exponential):
                exponential_count += 1
            elif isinstance(elem, distributions.StandardT):
                standard_t_count += 1
        return (normal_count, exponential_count, standard_t_count)

    normal_count, exponential_count, standard_t_count = make_count(out3)
    assert ((normal_count, exponential_count, standard_t_count) 
                == (7,2,1))

    # test proportions that cannot be turned into integer counts exactly
    out4 = my_fix_multiple.assign_distributions(n_clusters=11)
    normal_count, exponential_count, standard_t_count = make_count(out4)
    assert ((normal_count >= 7) and (exponential_count >= 2) 
        and (standard_t_count >= 1))
    assert ((normal_count > 7) or (exponential_count > 2) 
        or (standard_t_count > 1))
    assert normal_count + exponential_count + standard_t_count == 11
