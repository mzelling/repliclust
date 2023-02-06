"""
This module implements probability distributions for individual
clusters, as well as distribution mixes that assign probability 
distributions to the clusters in a mixture model.
"""

import numpy as np
from repliclust import config

from repliclust import SUPPORTED_DISTRIBUTIONS
from repliclust.utils import assemble_covariance_matrix
from repliclust.base import SingleClusterDistribution, DistributionMix


N_EMPIRICAL_QUANTILE = int(1e+6)
QUANTILE_LEVEL = 0.682 #0.477 # 0.3415 # refers to a distr taking only positive vals

SUPPORTED_DISTRIBUTION_NAMES = list(
    SUPPORTED_DISTRIBUTIONS.keys())


class DistributionFromNumPy(SingleClusterDistribution):
    """
    Allows arbitrary method from Generator class in numpy.random.
    The method must take the argument 'size' for selecting the number
    of samples.
    """
    def __init__(self, name, **params):
        self.name = name
        self.params = params
        # compute empirical quantile for normalization
        sample = np.abs(getattr(config._rng, self.name)(
                    size=N_EMPIRICAL_QUANTILE, **(self.params)
                    ))
        self._empirical_quantile = np.quantile(np.abs(sample), 
                                               q=QUANTILE_LEVEL)

    def __repr__(self):    
       return "DistributionFromNumPy('" + self.name + "')"

    def _sample_1d(self, n, dim):
        return np.sqrt(np.sum(((np.abs(
                                    getattr(config._rng, self.name)
                                        (size=(n,dim), **(self.params))     
                                    ) / self._empirical_quantile)
                                **2), 
                              axis=1))


class DistributionFromPDF(SingleClusterDistribution):
    """
    Sample from arbitrary probability density function.
    """
    def __init__():
        raise NotImplementedError("this feature will be part of a"
                + " future release. Stay tuned!")


class Normal(DistributionFromNumPy):
    """
    Draw multivariate normal data for a single cluster.
    """
    def __init__(self):
        DistributionFromNumPy.__init__(self, "normal", loc=0, scale=1)
        self.params = {}

    # def _sample_1d(self, n, dim):
    #     return np.sqrt(config._rng.chisquare(df=dim,size=n))
#       return config._rng.normal(loc=0.0, scale=1.0, size=n)


class Exponential(DistributionFromNumPy):
    """
    Draw exponentially distributed data for a single cluster.
    """
    def __init__(self):
        DistributionFromNumPy.__init__(self, "exponential", scale=1)
        self.params = {}
    # def _sample_1d(self, n, dim):
    #     return np.sqrt(np.sum(config._rng.exponential(
    #                             scale=1, size=(n, dim)
    #                             )**2,
    #                           axis=1))

class StandardT(SingleClusterDistribution):
    """
    Draw t-distributed data for a single cluster.
    """
    def __init__(self, df=1):
        self._empirical_quantile = np.quantile(
            np.abs(config._rng.standard_t(
                df=df, size=N_EMPIRICAL_QUANTILE)), 
            q=QUANTILE_LEVEL
            )
        SingleClusterDistribution.__init__(self, df=df)

    def _sample_1d(self, n, dim):
        return np.sqrt(np.sum((np.abs(config._rng.standard_t(
                                         size=(n,dim), **(self.params)))
                                    / self._empirical_quantile)
                               **2, axis=1))


def parse_distribution(distr_name: str, params: dict = {}):
    """
    Return the :py:class:`SingleClusterDistribution` object 
    corresponding to the probability distribution with name 
    `distr_name`.
    """
    if distr_name not in SUPPORTED_DISTRIBUTION_NAMES:
        raise ValueError("distribution '" + distr_name + "' is" 
                + " currently not supported. Please check for"
                + " misspellings. The list"
                + " of supported distributions is " 
                + str(sorted(SUPPORTED_DISTRIBUTION_NAMES)))
    else:
        if distr_name == "normal":
            return Normal()
        elif distr_name == "exponential":
            return Exponential()
        elif distr_name == "standard_t":
            return StandardT(**params)
        else:
            if params:
                return DistributionFromNumPy(distr_name, **params)
            else:
                default_params = SUPPORTED_DISTRIBUTIONS[distr_name]
                return DistributionFromNumPy(distr_name, 
                                             **default_params)


class FixedProportionMix(DistributionMix):
    """
    Assign probability distributions to clusters according to fixed
    proportions. For example, you may choose that 50% of clusters have
    a multivariate normal distribution and 50% have an exponential 
    distribution.

    Parameters
    ----------
    distributions : list of tuple
        List of distributions to mix. Each distribution appears as
        a tuple (name, proportion, params), where name is a string
        giving the distribution name; proportion is a number giving
        the desired proportion of clusters with the named 
        distribution, and params is a dict whose (key, value) pairs
        are the names and values of distributional parameters.

    Methods
    -------
    assign_distributions(self, n_clusters):
        Assign probability distributions to clusters.

    Attributes
    ----------
    _distributions : list
        List of probability distributions.
    _proportions : :py:class:`ndarray <numpy.ndarray>`
        Desired proportion of clusters having the corresponding
        distribution. The i-th entry corresponds to the i-th element
        in _distribution.

    """
    
    def __init__(self, distributions=[('normal', 1.0, {})]):
        """ Instantiate a FixedProportionMix object. """
        if (not isinstance(distributions, list)) or not distributions:
            raise ValueError("argument 'distributions' should be" +
                    " a non-empty list.")
        
        distr, prop, params = zip(*distributions)
        self._proportions = np.array(prop)/np.sum(prop)
        self._distributions = [ parse_distribution(distr_name, params) 
            for distr_name, params in zip(distr, params) ]

    def assign_distributions(self, n_clusters):
        """ 
        Assign probability distributions to all the clusters of a 
        probabilistic mixture model.

        Parameters
        ----------
        n_clusters : int
            The number of clusters for which to assign probability
            distributions.

        Returns
        -------
        distributions : list of :py:class:`SingleClusterDistribution`
            Probability distributions for the clusters of a 
            probabilistic mixture model.
        """
        mult_raw = np.floor(
            self._proportions * n_clusters
            ).astype('int')

        distr_out = []
        for idx, mult in enumerate(mult_raw):
            distr_out += [self._distributions[idx]] * mult
        # add more distributions in case of rounding errors 
        while (len(distr_out) < n_clusters):
            distr_out.append(self._distributions[
                config._rng.choice(len(self._distributions))
                ])
        # shuffle the distributions among clusters
        config._rng.shuffle(distr_out)
        
        return list(distr_out)
            
