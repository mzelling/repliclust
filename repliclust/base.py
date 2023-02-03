"""
Provides the core framework of `repliclust`.

An Archetype defines the overall geometry of a synthetic data set.
Feeding one or several Archetypes into a DataGenerator allows you to 
sample synthetic data sets with the desired geometries.

**Functions**:
    :py:func:`set_seed`
        Set a random seed for reproducibility.
    :py:func:`get_supported_distributions`
        Obtain a dictionary of supported probability distributions.

**Classes**:
    :py:class:`DataGenerator`
        Sample synthetic data sets based on data set archetypes.
    :py:class:`Archetype`
        Sample probabilistic mixture models with a desired overall
        geometric structure.
    :py:class:`MixtureModel`
        Probabilistic mixture model with defined cluster shapes, 
        locations, and probability distributions.
    :py:class:`DistributionMix`
        Mechanism for assigning probability distributions to clusters
        when sampling a :py:class:`MixtureModel` via an \
        :py:class:`Archetype`.
    :py:class:`SingleClusterDistribution`
        Define the probability distribution for a single cluster in 
        a `MixtureModel`.
    :py:class:`GroupSizeSampler`
        Sample the number of data points for each cluster.
    :py:class:`ClusterCenterSampler`
        Sample the locations of cluster centers for a \
        :py:class:`MixtureModel`.
    :py:class:`CovarianceSampler`
        Sample cluster shapes for a :py:class:`MixtureModel`.
"""

import numpy as np

from repliclust import config
from repliclust.utils import sample_unit_vectors

# names and default parameters of supported probability distribution
SUPPORTED_DISTRIBUTIONS = {
    'normal': {},
    'standard_t': {'df': 5},
    'exponential': {},
    'beta': {'a': 2.5, 'b': 8.5},
    'chisquare': {'df': 5},
    'gumbel': {},
    'weibull': {'a': 1.5},
    'gamma': {'shape': 3},
    'pareto': {'a': 10},
    'f': {'dfnum': 7, 'dfden': 10},
    'lognormal': {'sigma': 0.75},
}


def set_seed(seed):
    """
    Set a program-wide seed for `repliclust`.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    config._rng = np.random.default_rng(seed)


def get_supported_distributions():
    """
    Get a dictionary of the currently supported probability 
    distributions, as well as their default parameters. The names
    agree with the class names in the :py:class:`numpy.random.Generator`
    module.
    """
    return SUPPORTED_DISTRIBUTIONS


class CovarianceSampler():
    """
    Base class for sampling the shapes of all clusters in a 
    `MixtureModel`.

    Subclasses implement a concrete way of sampling cluster shapes
    by overriding the :py:meth:`sample_covariances` method, which 
    specifies a call signature that should be followed. By contrast, 
    subclasses define their own attributes without restriction. 
    
    See also
    --------
    :py:class:`MaxMinCovarianceSampler\
     <repliclust.maxmin.covariance.MaxMinCovarianceSampler>`
    """

    def __init__(self):
        raise NotImplementedError('this method is abstract. Instantiate'
                ' objects from subclasses of CovarianceSampler, such' +
                ' as MaxMinCovarianceSampler.')

    def sample_covariances(self, archetype):
        """
        Sample cluster shapes for all clusters in a `MixtureModel`.

        **Subclasses overriding this method should follow the call 
        signature below**.

        Parameters
        ----------
        archetype : :py:class:`Archetype <repliclust.base.Archetype>`
            Data set archetype specifying the desired overall geometry
            of a probabilistic mixture model.

        Returns
        --------
        (axes_list, axis_lengths_list): tuple[list[:py:class:\
            `numpy.ndarray`], list[:py:class:`numpy.ndarray`]]
            A tuple with two components. The first component, 
            `axes_list`, is a list whose `i`-th entry stores the 
            principal axes of cluster `i` as a matrix (each row is an 
            axis). The second component, `axis_lengths_list`, is a list 
            whose `i`-th entry stores the lengths of the `i`-th clusters
            principal axes as a vector (the `j`-th entry is the length 
            of the principal axis stored in the `j`-th row of 
            `axes[i]`).
        """
        raise NotImplementedError('this method is abstract. To sample'
                + ' covariances, call this method from an instance of a'
                + ' subclass of CovarianceSampler, such as' 
                + ' MaxMinCovarianceSampler.')


class ClusterCenterSampler():
    """
    Base class for sampling the locations of all cluster centers in 
    a `MixtureModel`.

    Subclasses implement a concrete way of sampling cluster centers by
    overriding the :py:meth:`sample_cluster_centers` method, which 
    specifies a call signature that should be followed. By contrast, 
    subclasses define their own attributes without restriction. 
    
    See also
    --------
    :py:class:`ConstrainedOverlapCenters <repliclust.overlap.centers.ConstrainedOverlapCenters>`
    """

    def __init__(self):
        raise NotImplementedError('this method is abstract. Instantiate'
                + ' objects from subclasses of ClusterCenterSampler,'
                + ' such as ConstrainedOverlapCenters.')

    def sample_cluster_centers(self, archetype):
        """
        Sample the locations of all clusters in a `MixtureModel`.

        **Subclasses overriding this method should follow the call 
        signature below**.

        Parameters
        ----------
        archetype : :py:class:`Archetype`
            Data set archetype specifying the desired overall geometry
            of a probabilistic mixture model.

        Returns
        --------
        centers: :py:class:`ndarray <numpy.ndarray>`
            A matrix whose `i`-th row gives the location of the `i`-th
            cluster in the mixture model.
        """
        raise NotImplementedError('this method is abstract. To sample'
                + ' cluster centers, call this method from an instance '
                + ' of a subclass of ClusterCenterSampler, such as ' 
                + ' ConstrainedOverlapCenters.')

class GroupSizeSampler():
    """
    Base class for sampling the number of data points for each cluster
    in a `MixtureModel`.

    Subclasses implement a concrete way of sampling group sizes by
    overriding the :py:meth:`sample_group_sizes` method, which specifies
    a call signature that should be followed. By contrast, subclasses 
    define their own attributes without restriction. 

    See also
    --------
    :py:class:`MaxMinGroupSizeSampler <repliclust.maxmin.groupsizes.MaxMinGroupSizeSampler>`
    """

    def __init__(self):
        raise NotImplementedError('this class is abstract. Instantiate'
            + ' objects from subclasses of GroupSizeSampler, such as'
            + " MaxMinGroupSizeSampler.")

    def sample_group_sizes(self, archetype, total):
        """
        Sample the number of data points for each cluster in a
        `MixtureModel`.

        **Subclasses overriding this method should follow the call 
        signature below**.

        Parameters
        ----------
        archetype : :py:class:`Archetype`
            Data set archetype specifying the desired overall geometry
            of a probabilistic mixture model. 
        total : `int`
            The total number of samples (sum of all group sizes).

        Returns
        -------
        group_sizes : :py:class:`ndarray <numpy.ndarray>`
            A vector whose `i`-th entry is the number of data points
            for the `i`-th cluster.
        """
        raise NotImplementedError('this method is abstract. To sample' 
            + ' group sizes, run this method from a subclass of' +
            + ' GroupSizeSampler, such as MaxMinGroupSizeSampler.')


class SingleClusterDistribution():
    """
    Base class for specifying the probability distribution of a
    single cluster in a `MixtureModel`.

    Subclasses implement a probability distribution by overriding the
    :py:meth:`_sample_1d` method, which specifies a call signature that
    should be followed. By contrast, subclasses 
    define their own attributes without restriction. 

    See also
    --------

    :py:class:`MultivariateNormal \
        <repliclust.distributions.MultivariateNormal>`
        Multivariate normal probability distribution for a single cluster.
    :py:class:`Exponential <repliclust.distributions.Exponential>`
        Exponential probability distribution for a single cluster.
    :py:class:`DistributionFromNumPy \
        <repliclust.distributions.DistributionFromNumPy>`
        Arbitrary probability distribution from \
        :py:class:`numpy <numpy.random.Generator>` for a single cluster.
    """

    def __init__(self, **params):
        self.params = params

    def _sample_1d(self, n, dim):
        """ 
        Sample one-dimensional data.

        **Subclasses overriding this method should follow the call 
        signature below**.

        Parameters
        ----------
        n : int
            The number of samples to generate.
        dim : int
            The number of dimensions of the cluster.

        Returns
        -------
        samples_1d : :py:class:`ndarray <numpy.ndarray>`
            `n` random samples arranged as a vector of length `n`.
        """
        raise NotImplementedError("method '_sample_1d' is an abstract"
            + " method for"
            + " class 'SingleClusterDistribution'. Its subclasses, such"
            + " as 'MultivariateNormal' provide a concrete "
            + " implementation by"
            + " overriding this function.")

    def sample_cluster(self, n: int, center: np.ndarray, 
                       axes: np.ndarray, axis_lengths: np.ndarray):
        """ 
        Sample data points for a single cluster. 

        Parameters
        ----------
        n : `int`
            The number of data points to generate.
        center : :py:class:`ndarray <numpy.ndarray>`
            The cluster center.
        
        Returns
        -------
        X : :py:class:`ndarray <numpy.ndarray>`
            Data points for a single cluster, arranged as a matrix with
            `n` rows (each row is a single data point).
        """
        if not ((len(center.shape) == 1) 
                    and (axis_lengths.shape == center.shape)
                    and (center.shape[0] == axes.shape[1])
                    and (center.shape[0] == axis_lengths.shape[0])):
            raise ValueError('cluster center and axis lengths must be'
                    + ' vectors; axes must be square matrix.')
        dim = axes.shape[1]
        directions = sample_unit_vectors(n, dim)
        scaling = self._sample_1d(n, dim)
        return (center[np.newaxis,:]
                + ((directions * scaling[:,np.newaxis])
                    @ np.diag(axis_lengths)
                    @ axes)
        )



class DistributionMix():
    """
    Base class for assigning probability distributions to all clusters
    in a `MixtureModel`.

    Subclasses implement a concrete assignment mechanism by overriding
    the :py:meth:`assign_distributions` method, which specifies a call
    signature that should be followed. By contrast, subclasses 
    define their own attributes without restriction. 

    See also
    --------
    :py:class:`FixedProportionMix <repliclust.distributions.FixedProportionMix>`
    """

    def assign_distributions(self, n_clusters):
        """
        Assign probability distributions to all clusters in a 
        `MixtureModel`.

        **Subclasses overriding this method should follow the call 
        signature below**.

        Parameters
        ----------
        n_clusters : int
            The number of clusters in the mixture model.

        Returns
        -------
        distributions : list[\
            :py:class:`SingleClusterDistribution`]
            A list whose `i`-th element represents the probability 
            distribution assigned to the `i`-th cluster.
        """
        raise NotImplementedError("this method is abstract. Please run"
            + " assign_distributions from a subclass of"
            + " DistributionMix, such as FixedProportionMix.")



class MixtureModel():
    """
    Represents a probabilistic mixture model from which you can 
    draw samples.

    Parameters
    ----------
    centers : :py:class:`ndarray <numpy.ndarray>`
        The locations of the cluster centers in this mixture model,
        arranged as a matrix. The i-th row of this matrix stores the 
        `i`-th cluster center.
    axes : list[:py:class:`ndarray <numpy.ndarray>`]
        A list of the principal axes of each cluster. The `i`-th element
        is a matrix whose rows are the orthonormal axes of the `i`-th
        cluster.
    axis_lengths : list[:py:class:`ndarray <numpy.ndarray>`]
        A list containing the lengths of the principal axes of each 
        cluster. The `i`-th element is a vector whose `j`-th entry is 
        the length of the `j`-th principal axis of cluster `i`.
    distributions : list[:py:class:`SingleClusterDistribution`]
        A list assigning a probability distribution to each cluster
        in this mixture model. The `i`-th element is the probability
        distribution of the `i`-th cluster.
    """
    
    def __init__(
        self, centers, axes_list, axis_lengths_list, distributions_list
        ):
        """ Instantiate a `MixtureModel`. """
        self.centers = centers
        self.axes_list = axes_list
        self.axis_lengths_list = axis_lengths_list
        self.distributions_list = distributions_list

    def sample_data(self, group_sizes):
        """
        Sample a data set from this :py:class:`MixtureModel`. 

        Parameters
        ----------
        group_sizes : :py:class:`ndarray <numpy.ndarray>`
            The number of data points to sample for each cluster,
            formatted as a vector whose length is the number of clusters
            in this :py:class:`MixtureModel`.

        Returns
        -------
        (X, y) : tuple[:py:class:`ndarray <numpy.ndarray>`, \
            :py:class:`ndarray <numpy.ndarray>`]
            Tuple with two components. The first component, `X' is a 
            matrix that stores the sampled data points (the `i`-th row
            is the `i`-th data point), while the second component, `y`,
            is a vector that stores the cluster labels as integers 
            ranging from zero to the number of clusters minus one. 
        """
        n = np.sum(group_sizes) # compute total number of samples
        k = self.centers.shape[0] # extract number of clusters
        dim = self.centers.shape[1] # extract number of dimensions
        X = np.full(shape=(n, dim), fill_value=np.nan)
        y = np.full(n, fill_value=np.nan).astype(int)

        if (group_sizes.shape != (k,)):
            raise ValueError('group_sizes must be a vector whose length'
                    + ' equals the number of clusters.')

        start = 0
        for i in range(k):
            end = start + group_sizes[i]
            y[start:end] = i
            X[start:end,:] = self.distributions_list[i].sample_cluster(
                n=group_sizes[i], center=self.centers[i,:], 
                axes=self.axes_list[i], 
                axis_lengths = self.axis_lengths_list[i]
                )
            start = end

        return (X, y)



class Archetype():
    """
    Base class for a data set archetype.

    Objects of this class sample probabilistic mixture models by
    first sampling cluster shapes, then sampling the locations for 
    all cluster centers, and finally assigning a probability 
    distribution to each cluster.
    
    Subclasses implement concrete ways of sampling probabilistic mixture
    models by providing a wrapper that runs this class's constructor 
    with certain choices for the `covariance_sampler`, `center_sampler`,
    `groupsize_sampler`, and `distribution_mix` parameters. 
    Alternatively, it is possible to directly construct an `Archetype`
    object by manually specifying these parameters.

    Parameters
    ----------
    n_clusters : int
        The desired number of clusters.
    dim : int
        The desired number of dimensions.
    n_samples : int, default=500
        The desired total number of data points.
    name : str, optional
        The name of this archetype.
    scale : float, default=1
        The typical length scale for clusters. Increasing this 
        parameter makes all clusters bigger without changing their
        relatives sizes and positions. The default is 1.
    covariance_sampler : :py:class:`CovarianceSampler`
        Sampler for cluster covariances.
    center_sampler : :py:class:`ClusterCenterSampler`
        Sampler for the locations of cluster centers.
    groupsize_sampler : :py:class:`GroupSizeSampler`
        Sampler for the number of data points in each cluster.
    distribution_mix : :py:class:`DistributionMix`
        Assigns probability distributions to clusters.
    **kwargs : `dict`, optional
        Extra arguments used by subclasses of :py:class:`Archetype` to
        store additional attributes.

    See also
    --------
    :py:class:`MaxMinArchetype <repliclust.maxmin.archetype.MaxMinArchetype>` :
        The default implementation for a dataset archetype.

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
        self.n_samples = int(n_samples)
        self.name = name
        self.scale = scale

        self.covariance_sampler = covariance_sampler
        self.center_sampler = center_sampler
        self.groupsize_sampler = groupsize_sampler
        self.distribution_mix = distribution_mix

        for key, val in kwargs.items():
            setattr(self, key, val)
    
    def sample_mixture_model(self, quiet=False):
        """
        Sample a probabilistic mixture model according to this 
        archetype.

        Returns
        -------
        mixture_model : :py:class:MixtureModel 
            A probabilistic mixture model with the overall geometric
            structure specified by this archetype.
        """
        if not (self.covariance_sampler and self.groupsize_sampler
                    and self.groupsize_sampler and self.center_sampler):
            raise Exception("sampling a mixture model requires"
                            + " a covariance sampler, center sampler,"
                            + " group size sampler, and distribution"
                            + " mix")

        self._axes, self._lengths = \
            self.covariance_sampler.sample_covariances(self)
        if self.n_clusters >= 2:
            self._centers = (self.center_sampler
                                .sample_cluster_centers(
                                    self,
                                    quiet=quiet))
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
    Data generator based on data set archetypes.

    Base class for a data generator. Instances of this class generate
    synthetic data sets based on archetypes indicating their desired
    geometries.

    There are three different ways to generate synthetic data sets with
    a DataGenerator. After constructing a DataGenerator `dg`, you can 
    write:

    1) ``X, y, archetype = dg.synthesize(n_samples)``
        Generate a single data set with the desired number of samples.

    2) ``for X, y, archetype in dg: ...``                        
        Iterate over `dg` and generate `dg._n_datasets` datasets, each 
        with the number of samples specified by the corresponding 
        archetype.

    3) ``for X, y, archetype in dg(n_datasets, n_samples): ...``
        Iterate over `dg` and generate `n_datasets` datasets, each with
        `n_samples` data points if `n_samples` is a number; if
        `n_samples` is a list of n_datasets numbers, the i-th dataset 
        will have `n_samples[i]` data points. If either `n_datasets` or 
        `n_samples` are not specified, use `n_datasets = dg._n_datasets`
        and the number of data points specified by each archetype.

    In each case, the output format is as follows: X is a matrix-shaped
    :py:class:`ndarray <numpy.ndarray>` containing the data points 
    (samples by variables) and y is a vector-shaped :py:class:`ndarray \
    <numpy.ndarray>` containing the cluster labels. 
    Finally, `archetype` is the data set archetype from which the data
    set was generated.


    Parameters
    ----------
    archetype : Archetype or list[Archetype] or dict[str, Archetype]
        One or several archetypes specifying the desired overall 
        geometry of synthetic data sets.

    """

    def __init__(self, archetype, n_datasets=10, 
                 quiet=False, prefix='archetype'):
        """
        Instantiate a DataGenerator object.
        """
        if isinstance(archetype, Archetype):
            arch_name = (archetype.name if archetype.name 
                            else prefix+str(0))
            self._archetypes = [(arch_name, archetype)]
        elif isinstance(archetype, list):
            self._archetypes = [(arch.name if arch.name \
                                    else prefix+str(i),
                                 arch) for i, arch in 
                                    enumerate(archetype)]
        elif isinstance(archetype, dict):
            self._archetypes = [(arch_name, arch) for arch_name, arch 
                                    in archetype.items()]
        else:
            raise ValueError('archetypes should be a Archetype, list of'
                                + ' archetypes, or a dictionary whose'
                                + ' values are archetypes.')
        self._next_archetype_idx = 0
        self._n_datasets = n_datasets
        self._quiet = quiet


    def __iter__(self):
        # reliably start with the first archetype in the list
        self._next_archetype_idx = 0
        self._iter_count = 0
        return self


    def __next__(self):
        """
        Fetch the next data set from this data generator.
        """
        quiet = self._quiet

        if self._iter_count >= self._n_datasets:
            raise StopIteration

        arch_name, arch = self._archetypes[self._next_archetype_idx]
        group_sizes = (arch.groupsize_sampler
                         .sample_group_sizes(arch, arch.n_samples))
        X, y = (arch.sample_mixture_model(quiet=quiet)
                    .sample_data(group_sizes))

        self._next_archetype_idx = ((self._next_archetype_idx + 1) %
                                        len(self._archetypes))
        self._iter_count += 1
        arch.name = arch_name # make sure the archetype stores its name
        return (X, y, arch)


    def __call__(self, n_datasets=None, n_samples=None, 
                 quiet=False):
        """
        Set up a generator to yield `n_datasets` data sets, where 
        `n_samples` determines the number of samples in each dataset.

        Parameters
        ----------
        n_datasets : int, optional
            Set the number of datasets to generate. If not specified,
            iterate over self.n_datasets datasets.
        n_samples : int or list[int], optional
            Set the number of samples for each data set. If n_samples
            is an int, each data set will consist of n_samples samples.
            If n_samples is a list of n_dataset integers, the i-th
            dataset will consist of n_samples[i] data points. If
            n_samples is not specified, use the value specified by
            each archetype instead.
        quiet : bool
            If true, suppress all print output.

        Yields
        ------
        (X, y, archetype) : tuple[:py:class:`ndarray \
            <numpy.ndarray>`, :py:class:`ndarray <numpy.ndarray>`, \
                :py:class:`Archetype <repliclust.base.Archetype>`]
            Tuple with three components. The first component, `X`, 
            stores the new data set as a matrix (each row is a data
            point). The second component, `y`, stores the cluster labels
            (`y[i]` is the label of data point `X[i,:]`). The third
            component, `archetype`, is the data set archetype that
            was used to create `X` and `y`.
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
                                + ' a number or a list whose length'
                                + ' matches the number of datasets to'
                                + ' generate.')


        for i in range(n_datasets):
            arch_name, arch = self._archetypes[i % len(self._archetypes)]
            group_sizes = (arch.groupsize_sampler
                             .sample_group_sizes(
                                arch, 
                                _n_samples[i % len(_n_samples)]))
            X, y = (arch.sample_mixture_model(
                            quiet=quiet)
                        .sample_data(group_sizes))
            arch.name = arch_name # make sure archetype stores its name
            yield (X, y, arch)


    def synthesize(self, n_samples=None, quiet=False):
        """
        Synthesize a data set according to the specified archetype(s).
        If this data generator consists of more than one archetype, this
        function cycles through the given archetypes.

        Parameters
        ----------
        n_samples : int
            Desired total number of data points to sample. Optional.
            If specified, overrides the number of samples specified
            by an archetype object.
        quiet : bool
            If true, suppress all print output. This option is useful
            when placing many successive calls to `synthesize`.

        Returns
        -------
        (X, y, archetype) : tuple[:py:class:`ndarray \
            <numpy.ndarray>`, :py:class:`ndarray <numpy.ndarray>`, \
                :py:class:`Archetype <repliclust.base.Archetype>`]
            Tuple with three components. The first component, `X`, 
            stores the new data set as a matrix (each row is a data
            point). The second component, `y`, stores the cluster labels
            (`y[i]` is the label of data point `X[i,:]`). The third
            component, `archetype`, is the data set archetype that
            was used to create `X` and `y`.
        """
        arch_name, arch = self._archetypes[self._next_archetype_idx]
        group_sizes = (arch.groupsize_sampler
                         .sample_group_sizes(
                            arch, 
                            n_samples if n_samples else arch.n_samples))
        X, y = (arch.sample_mixture_model(quiet=quiet)
                    .sample_data(group_sizes))
        # increment the index for the next archetype
        self._next_archetype_idx = ((self._next_archetype_idx + 1) %
                                        len(self._archetypes))
        arch.name = arch_name # make sure archetype stores its name
        return (X, y, arch)

    def __repr__(self):
        """
        Construct string representation of this DataGenerator.
        """
        return ("DataGenerator" 
                    + "\n\t- archetype(s) : "
                    + str(len(self._archetypes))
                    + "\n\t- n_datasets : "
                    + str(self._n_datasets))