"""
Base classes for data generators, archetypes, mixture models, and 
related objects.
"""

import numpy as np

from repliclust import config
from repliclust.utils import sample_unit_vectors

# names and default parameters of supported probability distribution
SUPPORTED_DISTRIBUTIONS = {
    'normal': {},
    'standard_t': {'df': 1},
    'exponential': {},
    'beta': {'a': 0.3, 'b': 0.5},
    'chisquare': {'df': 1},
    'gumbel': {'scale': 1.0},
    'weibull': {'a': 2},
    'gamma': {'shape': 0.5, 'scale': 1.0},
    'pareto': {'a': 1},
    'f': {'dfnum': 1, 'dfden': 1},
    'lognormal': {'sigma': 1.0},
}


def set_seed(seed):
    """
    Set a programwide seed for repliclust.
    """
    config._rng = np.random.default_rng(seed)


def get_supported_distributions():
    """
    Print the names of currently supported probability distributions,
    as well as their default parameters. These distribution names are
    adopted from the numpy.random module.
    """
    return SUPPORTED_DISTRIBUTIONS


class CovarianceSampler():
    """
    Base class for sampling covariances (principal axes and their 
    lengths) of all clusters in a mixture model.

    Methods
    -------
    sample_covariances(archetype)

    """
    def __init__(self):
        raise NotImplementedError('this method is abstract. Instantiate'
                ' objects from subclasses of CovarianceSampler, such' +
                ' as MaxMinCovarianceSampler.')

    def sample_covariances(self, archetype):
        raise NotImplementedError('this method is abstract. To sample'
                + ' covariances, call this method from an instance of a'
                + ' subclass of CovarianceSampler, such as' 
                + ' MaxMinCovarianceSampler.')


class ClusterCenterSampler():
    """
    Base class for sampling the locations of all cluster centers in 
    a mixture model. 

    Methods
    -------
    sample_cluster_centers(archetype)
    
    """
    def __init__(self):
        raise NotImplementedError('this method is abstract. Instantiate'
                + ' objects from subclasses of ClusterCenterSampler,'
                + ' such as ConstrainedOverlapCenters.')

    def sample_cluster_centers(self, archetype):
        raise NotImplementedError('this method is abstract. To sample'
                + ' cluster centers, call this method from an instance '
                + ' of a subclass of ClusterCenterSampler, such as ' 
                + ' ConstrainedOverlapCenters.')

class GroupSizeSampler():
    """
    Base class for sampling group sizes (the number of data points in
    each cluster) for all clusters in a data set.

    Methods
    -------
    sample_group_sizes(archetype)

    """
    def __init__(self):
        raise NotImplementedError('this class is abstract. Instantiate'
            + ' objects from subclasses of GroupSizeSampler, such as'
            + " MaxMinGroupSizeSampler.")

    def sample_group_sizes(self, archetype, total):
        raise NotImplementedError('this method is abstract. To sample' 
            + ' group sizes, run this method from a subclass of' +
            + ' GroupSizeSampler, such as MaxMinGroupSizeSampler.')


class SingleClusterDistribution():
    """
    Base class for specifying the probability distribution of a
    single cluster in a mixture model.

    Methods
    -------
    """
    def __init__(self, **params):
        self.params = params

    def _sample_1d(self, n, **params):
        """ 
        Sample one-dimensional data. 

        Parameters
        ----------
        n : int
            The number of samples to generate.

        Returns
        -------
        samples : ndarray
            A vector of samples.
        
        """
        raise NotImplementedError("method 'sampler_1d' is abstract for"
            + " class 'SingleClusterDistribution'. Its subclasses, such"
            + " as 'MultivariateNormal' provide a concrete "
            + " implementation by"
            + " overriding this function.")

    def sample_cluster(self, n: int, center: np.ndarray, 
                       axes: np.ndarray, axis_lengths: np.ndarray):
        """ 
        Sample data for a single cluster. 

        Parameters
        ----------
        n : int
            The number of samples to generate.
        center : ndarray
            The cluster center.
        

        Returns
        -------
        """
        if not ((len(center.shape) == 1) 
                    and (axis_lengths.shape == center.shape)
                    and (center.shape[0] == axes.shape[1])
                    and (center.shape[0] == axis_lengths.shape[0])):
            raise ValueError('cluster center and axis lengths must be'
                    + ' vectors; axes must be square matrix.')
        dim = axes.shape[1]
        directions = sample_unit_vectors(n, dim)
        scaling = self._sample_1d(n)
        return (center[np.newaxis,:]
                + ((directions * scaling[:,np.newaxis])
                    @ np.diag(axis_lengths)
                    @ axes)
        )



class DistributionMix():
    """
    Base class for assigning probability distributions to clusters
    in a mixture model.

    Methods
    -------
    assign_distributions(n_clusters)

    """

    def assign_distributions(self, n_clusters):
        """
        Assign probability distributions to all clusters in a mixture
        model.

        Parameters
        ----------
        n_clusters : int
            The number of clusters in the mixture model.

        Returns
        -------
        distributions : list of SingleClusterDistribution
            A list whose i-th element is the probability distribution
            assigned to the i-th cluster.
        """
        raise NotImplementedError("this method is abstract. Please run"
            + " assign_distributions from a subclass of"
            + " DistributionMix, such as FixedProportionMix.")


class MixtureModel():
    """
    Base class for a probabilistic mixture model. Instances of this
    class sample data from the mixture distribution.

    Parameters
    ----------
    centers : ndarray
        The locations of the cluster centers in this mixture model,
        arranged as a matrix. The i-th row of this matrix stores the 
        i-th cluster center.
    axes_list : list of ndarray
        A list of the principal axes of each cluster. The i-th element
        is a matrix whose rows are the orthonormal axes of the i-th
        cluster.
    axis_lengths_list : list of ndarray
        A list containing the lengths of the principal axes of each 
        cluster. The i-th element is a vector whose j-th entry is the 
        length of the j-th principal axis of cluster i.
    distributions_list : list of SingleClusterDistribution
        A list assigning a probability distribution to each cluster
        in this mixture model. The i-th element is the probability
        distribution of the i-th cluster.

    Methods
    -------
    sample_data(group_sizes)
    
    """
    
    def __init__(
        self, centers, axes_list, axis_lengths_list, distributions_list
        ):
        """ Instantiate a MixtureModel. """
        self._centers = centers
        self._axes_list = axes_list
        self._axis_lengths_list = axis_lengths_list
        self._distributions_list = distributions_list

    def sample_data(self, group_sizes) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample a data set from the mixture distribution.

        Parameters
        ----------
        group_sizes : ndarray, shape (k,)
            The number of data points to sample for each of k clusters.

        Returns
        -------
        (X, y) : tuple[ndarray, ndarray]
            The matrix X represents the sampled data points, while the
            vector y stores the cluster labels as integers ranging
            from zero to the number of clusters minus one.
        """
        n = np.sum(group_sizes) # compute total number of samples
        k = self._centers.shape[0] # extract number of clusters
        dim = self._centers.shape[1] # extract number of dimensions
        X = np.full(shape=(n, dim), fill_value=np.nan)
        y = np.full(n, fill_value=np.nan).astype(int)

        if (group_sizes.shape != (k,)):
            raise ValueError('group_sizes must be a vector whose length'
                    + ' equals the number of clusters.')

        start = 0
        for i in range(k):
            end = start + group_sizes[i]
            y[start:end] = i
            X[start:end,:] = self._distributions_list[i].sample_cluster(
                n=group_sizes[i], center=self._centers[i,:], 
                axes=self._axes_list[i], 
                axis_lengths = self._axis_lengths_list[i]
                )
            start = end

        return (X, y)

class Archetype():
    """
    Base class for a data set archetype. Instances of this class
    sample probabilistic mixture models with specified geometry.

    Parameters
    ----------
    n_clusters : int
        The desired number of clusters in each mixture model.
    dim : int
        The number of dimensions of each mixture model.
    n_samples : int
        The desired total number of data points.
    scale : float
        The typical length scale for clusters in each mixture model.
        Increasing this parameters makes the values of all coordinates 
        bigger (but the underlying geometry stays the same).
    covariance_sampler : CovarianceSampler
        Sampler for cluster covariances.
    center_sampler : ClusterCenterSampler
        Sampler for the locations of cluster centers.
    groupsize_sampler : GroupSizeSampler
        Sampler for the number of data points in each cluster.
    distribution_mix : DistributionMix
        Object that assigns probability distributions to all clusters
        in a mixture model.

    Methods
    -------
    sample_mixture_model()

    """
    def __init__(
            self,
            n_clusters: int,
            dim: int,
            n_samples: int = 500,
            name = None,
            scale: float = 1.0,
            covariance_sampler: CovarianceSampler = None, 
            center_sampler: ClusterCenterSampler = None,
            groupsize_sampler: GroupSizeSampler = None,
            distribution_mix: DistributionMix = None,
            **kwargs,
        ):
        
        self.n_clusters = int(n_clusters)
        self.dim = dim
        self.n_samples = n_samples
        self.name = name
        self.scale = scale

        self.covariance_sampler = covariance_sampler
        self.center_sampler = center_sampler
        self.groupsize_sampler = groupsize_sampler
        self.distribution_mix = distribution_mix

        for key, val in kwargs.items():
            setattr(self, key, val)
    
    def sample_mixture_model(self):
        """
        Sample a probabilistic mixture model according to this 
        archetype.

        Returns
        -------
        out : MixtureModel
            A probabilistic mixture model satisfying the geometric
            constraints imposed by this archetype.
        """
        self._axes, self._lengths = \
            self.covariance_sampler.sample_covariances(self)
        if self.n_clusters >= 2:
            self._centers = (self.center_sampler
                                .sample_cluster_centers(self))
        elif self.n_clusters == 1:
            self._centers = np.zeros(self.dim)[np.newaxis,:]
        else:
            raise ValueError('the number of clusters must be'
                                + ' at least 1.')
            
        self._distributions = \
            self.distribution_mix.assign_distributions(self.n_clusters)
        
        # construct the mixture model
        mixture_model = MixtureModel(
                            self._centers, 
                            self._axes, self._lengths, 
                            self._distributions
                            )
        # # delete the mixture model data from this archetype object 
        # self._axes = None
        # self._lengths = None
        # self._centers = None

        return mixture_model


class DataGenerator():
    """
    Base class for a data generator. Instances of this class generate
    synthetic data sets based on archetypes indicating their desired
    geometries.


    Note
    ----
    There are three different ways to generate data sets with a 
    DataGenerator. After constructing a DataGenerator dg, you can 
    write...
        ___________________________________________________
    1) | X, y, archetype_name = dg.synthesize(?n_samples)  |
        `-------------------------------------------------'
        Generate a single data set with the desired number of samples.
        ______________________________________
    2) | for X, y, archetype_name in dg: ...  |                         
        `------------------------------------'
        Iterate over dg and generate dg.n_datasets datasets, each with
        the number of samples specified by the corresponding archetype.
        _______________________________________________________________
    3) | for X, y, archetype_name in dg(?n_datasets, ?n_samples): ...  |
        `-------------------------------------------------------------'
        Iterate over dg and generate n_datasets datasets, each with
        n_samples data points if n_samples is a number; if n_samples
        is a list of n_datasets numbers, the i-th dataset will have
        n_samples[i] data points. If either n_datasets or n_samples
        are not specified, use n_datasets=dg.n_datasets and the number
        of data points specified by each archetype.

    In each case, the output format is as follows: X is a matrix-shaped
    NumPy array containing the data points (samples by variables) and
    y is a vector-shaped NumPy array containing the cluster labels. 
    Finally, archetype_name is the name of the archetype that was used
    to construct the dataset.


    Parameters
    ----------
    archetype : Archetype
        A archetype for generating synthetic data sets.

    Methods
    -------
    synthesize(n_samples)

    """

    def __init__(self, archetype, n_datasets=10, 
                 prefix='archetype'):
        """
        Instantiate a DataGenerator object.
        """
        if isinstance(archetype, Archetype):
            bp_name = (archetype.name if archetype.name 
                            else prefix+str(0))
            self._archetypes = [(bp_name, archetype)]
        elif isinstance(archetype, list):
            bp_name = (archetype.name if archetype.name 
                else prefix+str(0))
            self._archetypes = [(bp.name if bp.name else prefix+str(0),
                                    bp) 
                                 for i, bp in enumerate(archetype)]
        elif isinstance(archetype, dict):
            self._archetypes = [(bp_name, bp) for bp_name, bp 
                                    in archetype.items()]
        else:
            raise ValueError('archetypes should be a Archetype, list of'
                                + ' archetypes, or a dictionary whose'
                                + ' values are archetypes.')
        self._next_archetype_idx = 0
        self._n_datasets = n_datasets


    def __iter__(self):
        # reliably start with the first archetype in the list
        self._next_archetype_idx = 0
        self._iter_count = 0
        return self


    def __next__(self):
        """
        Fetch the next data set from this data generator.
        """
        if self._iter_count >= self._n_datasets:
            raise StopIteration

        bp_name, bp = self._archetypes[self._next_archetype_idx]
        group_sizes = (bp.groupsize_sampler
                         .sample_group_sizes(bp, bp.n_samples))
        X, y = bp.sample_mixture_model().sample_data(group_sizes) 

        self._next_archetype_idx = ((self._next_archetype_idx + 1) %
                                        len(self._archetypes))
        self._iter_count += 1
        return (X, y, bp_name)


    def __call__(self, n_datasets=None, n_samples=None):
        """
        Generate n_datasets from this data generator, where n_samples
        determines the number of samples in each dataset.

        Parameters
        ----------
        n_datasets : int (optional)
            Set the number of datasets to generate. If not specified,
            iterate over self.n_datasets datasets.
        n_samples : int or list[int] (optional)
            Set the number of samples for each data set. If n_samples
            is an int, each data set will consist of n_samples samples.
            If n_samples is a list of n_dataset integers, the i-th
            dataset will consist of n_samples[i] data points. If
            n_samples is not specified, use the value specified by
            each archetype instead.

        Yields
        ------
        (X, y, archetype_name) : tuple[ndarray, ndarray, str]
            Data set X, cluster labels y, and name of the archetype
            according to which X, y were generated. The data X is a
            matrix (samples by coordinates) and the cluster labels y are
            a vector.

        """
        if not n_datasets:
            n_datasets = self._n_datasets
        if not n_samples:
            _n_samples = [bp.n_samples for _, bp in self._archetypes]
        elif isinstance(n_samples, (int, float)):
            _n_samples = [n_samples]
        elif (isinstance(n_samples, list) and 
                (len(n_samples) == n_datasets)):
            _n_samples = n_samples
        else:
            raise ValueError('if you wish to override the number of'
                                + ' samples specified by the'
                                + ' archetype(s), n_samples should be'
                                + ' a number or a list of n_datasets'
                                + ' numbers.')


        for i in range(n_datasets):
            bp_name, bp = self._archetypes[i % len(self._archetypes)]
            group_sizes = (bp.groupsize_sampler
                             .sample_group_sizes(
                                bp, 
                                _n_samples[i % len(_n_samples)]))
            X, y = bp.sample_mixture_model().sample_data(group_sizes)
            yield (X, y, bp_name)


    def synthesize(self, n_samples=None):
        """
        Synthesize a data set according to the specified archetype(s).
        If this data generator consists of more than one archetype, this
        function cycles through the given archetypes.

        Parameters
        ----------
        n_samples : int
            Desired total number of data points to sample. Optional.
            If specified, overrides the number of samples specified
            by the archetype object.

        Returns
        -------
        (X, y, bp_name) : tuple[ ndarray(matrix), ndarray(vector), str ]
            The new data set, the cluster labels, and name of the 
            archetype.
        """
        bp_name, bp = self._archetypes[self._next_archetype_idx]
        group_sizes = (bp.groupsize_sampler
                         .sample_group_sizes(
                            bp, 
                            n_samples if n_samples else bp.n_samples))
        X, y = bp.sample_mixture_model().sample_data(group_sizes)
        # increment the index for the next archetype
        self._next_archetype_idx = ((self._next_archetype_idx + 1) %
                                        len(self._archetypes))
        return (X, y, bp_name)

    def __repr__(self):
        """
        Construct string representation of this DataGenerator.
        """
        return ("DataGenerator" 
                    + "\n\t- archetype(s) : " + str(1)
                    + "\n\t- n_datasets (default) : " + str(2))