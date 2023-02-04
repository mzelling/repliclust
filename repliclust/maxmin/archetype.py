"""
This module implements a archetype for mixture models. The user chooses
the desired geometry by setting the ratios between largest and smallest
values of various geometric parameters.
"""

import numpy as np

from repliclust.base import Archetype
from repliclust.overlap.centers import ConstrainedOverlapCenters

from repliclust.maxmin.covariance import MaxMinCovarianceSampler
from repliclust.maxmin.groupsizes import MaxMinGroupSizeSampler
from repliclust.distributions import FixedProportionMix



def validate_overlaps(max_overlap=0.05,min_overlap=1e-3):
    """ 
    Note that we allow max_overlap=1 and min_overlap=0, which
    should have the effect of removing one or both overlap constraints.
    """
    if (max_overlap <= 0):
        raise ValueError('max_overlap must be greater than zero because'
            + ' clusters are based on probability distributions that '
            + ' always overlap a small -- potentially tiny -- amount.'
            + ' Try setting max_overlap close to zero to achieve large'
            + ' separation between clusters.')
    elif (min_overlap >= 1):
        raise ValueError('min_overlap must be smaller than 1 because we'
            + ' cannot force all clusters to overlap COMPLETELY. Try'
            + ' setting min_overlap to a value close to 1 instead.')
    elif (max_overlap > 1) or (min_overlap < 0):
        raise ValueError("max_overlap and min_overlap should be" +
                         " between 0 and 1.")
    elif max_overlap <= min_overlap:
        raise ValueError("max_overlap must exceed min_overlap.")

def validate_maxmin_ratios(maxmin_ratio=2, arg_name="aspect_maxmin",
                            underlying_param="aspect ratio"):
    """ Check that a max-min ratio is >= 1. """
    if (maxmin_ratio < 1):
        raise ValueError("the parameter " + arg_name 
                + " must be >= 1, as it is the ratio"
                + " of the maximum " + underlying_param
                + " to the minimum " + underlying_param)

def validate_reference_quantity(ref_qty=1.5, min_allowed_value=1, 
                                name="aspect_ref"):
    """ 
    Check that a reference value exceeds its minimum allowed value. 
    """
    if (ref_qty < min_allowed_value):
        raise ValueError("the parameter " + name
                            + " should be greater than "
                            + str(min_allowed_value))
    

def validate_archetype_args(**args):
    """ Validate all provided arguments for a MaxMinArchetype. """
    validate_overlaps(args['max_overlap'], args['min_overlap'])

    maxmin_args = [
        ('aspect_maxmin', 
            args['aspect_maxmin'], 'cluster aspect ratio'),
        ('radius_maxmin', 
            args['radius_maxmin'], 'cluster radius'),
        ('imbalance_ratio', 
            args['imbalance_ratio'], 'cluster group size'),
        ]
    for arg_name, arg_val, underlying_param in maxmin_args:
        validate_maxmin_ratios(maxmin_ratio=arg_val, arg_name=arg_name,
                               underlying_param=underlying_param)

    ref_args = [
        ("aspect_ref", args['aspect_ref'], 1),
        # ("scale", args['scale'], 0),
    ]
    for arg_name, arg_val, min_allowed_val in ref_args:
        validate_reference_quantity(ref_qty=arg_val, 
                                    min_allowed_value=min_allowed_val,
                                    name=arg_name)


def parse_distribution_selection(distributions: list, proportions=None):
    """
    Parse user selection of probability distributions and reformat it
    as an input for constructing a FixedProportionMix object.

    Parameters
    ----------
    distributions : list of [ str | tuple[str, dict] ]
        Selection of probability distributions that should appear in
        each mixture model. Format is a list in which each element is 
        either the name of the probability distribution OR a tuple whose
        first entry is the name and the second entry is a dictionary of 
        distributional parameters. To print all valid distribution 
        names, call the function 
        repliclust.print_supported_distributions().

    Returns
    -------
    distributions_parsed : 
        Input for constructing a FixedProportionMix object.
    """
    if not proportions:
        proportions = np.full(len(distributions), 
                                fill_value=1/len(distributions))

    distr_parsed = []
    for i, distr in enumerate(distributions):
        distr_prop = proportions[i]
        if isinstance(distr, tuple) and isinstance(distr[1], dict):
            distr_name = distr[0]
            distr_params = distr[1]
            distr_parsed.append((distr_name, distr_prop, distr_params))
        elif isinstance(distr, str):
            distr_parsed.append((distr, distr_prop, {}))
        else:
            raise ValueError("distributions should be provided as a"
                    + " list of which each element is either a string"
                    + " or a two-element tuple whose first element is"
                    + " a string and whose second element is a"
                    + " dictionary")
    return distr_parsed




class MaxMinArchetype(Archetype):
    """
    A data set archetype that defines the overall geometry of a data
    set using max-min ratios.
    
    The user sets the ratios between largest and smallest values 
    of various geometric parameters.


    Parameters
    ----------
    n_clusters : int
        The desired number of clusters.
    dim : int
        The desired number of dimensions.
    radius_maxmin : float, >=1
        Ratio between the maximum and minimum radii among all clusters
        in a mixture model.
    aspect_maxmin : float, >=1
        Ratio between the maximum and minimum aspect ratios among all 
        clusters in a mixture model.
    aspect_ref : float, >=1
        Typical aspect ratio for the clusters in a mixture model.
        For example, if aspect_ref = 10, we expect that all clusters
        in the mixture model are strongly elongated.
    imbalance_maxmin : float, >=1
        Ratio between the greatest and smallest group sizes among all
        clusters in the mixture model.
    min_overlap : float in (0,1)
        The minimum required overlap between a cluster and *some* other
        cluster. This minimum overlap allows you to guarantee that no
        cluster will be isolated from all other clusters.
    max_overlap : float in (0,1)
        The maximum allowed level of overlap between any two clusters. 
        Measured as the fraction of cluster volume that overlaps.
    scale : float
        Reference length scale for generated data
    distributions : list of [str | tuple[str, dict]]
        Selection of probability distributions that should appear in
        each mixture model. Format is a list in which each element is 
        either the name of the probability distribution OR a tuple whose
        first entry is the name and the second entry is a dictionary of 
        distributional parameters. To print the names of all supported
        distributions and their parameters (along with default values), 
        print the output of repliclust.get_supported_distributions().
    distributions_proportions :
        The proportions of clusters that have each distribution listed
        in `distributions`.
    mode : {"auto", "lda", "c2c"}
        Select the degree of precision when computing cluster overlaps.
    
    Notes
    -----

    Below is a short glossary of some geometric terms used above.

    Group size : int
        The number of data points in a cluster.
    Cluster radius : float
        Geometric mean of the standard deviations along a cluster's 
        principal axes (eigenvectors of covariance matrix).
    Cluster aspect ratio : float
        Ratio between the lengths of a cluster's longest and shortest
        principal axes (eigenvectors of covariance matrix). 
        This value equals 1 for a spherical cluster and exceeds 1 for 
        an oblong cluster.
    """

    def guess_learning_rate(self, dim):
        """
        Guess the appropriate learning rate as a function of dimension.
        """
        return 0.5 #0.5*(1/np.log10(10+dim))

    def __init__(
            self, 
            n_clusters=6, dim=2, n_samples=500,
            max_overlap=0.05, min_overlap=1e-3, 
            imbalance_ratio=2, aspect_maxmin=2, radius_maxmin=3,
            aspect_ref=1.5, name=None, scale=1.0, packing=0.1,
            distributions=['normal', 'exponential'],
            distribution_proportions=None,
            overlap_mode='auto', linear_penalty_weight=0.01, 
            learning_rate='auto',
            ):
        """ Instantiate a MaxMinArchetype object. """
        covariance_args = {'aspect_ref': aspect_ref,
                           'aspect_maxmin': aspect_maxmin,
                           'radius_maxmin': radius_maxmin}
        groupsize_args = {'imbalance_ratio': imbalance_ratio}

        if learning_rate=='auto':
            learning_rate = self.guess_learning_rate(dim)
        elif not isinstance(learning_rate, float):
            raise ValueError("learning_rate should be 'auto' or a "
                                + "float between 0 and 1")
        
        center_args = {'max_overlap': max_overlap, 
                       'min_overlap': min_overlap,
                       'packing': packing,
                       'learning_rate': learning_rate,
                       'linear_penalty_weight': linear_penalty_weight}
        distributions_parsed = parse_distribution_selection(
                                    distributions, 
                                    distribution_proportions)

        validate_archetype_args(**(covariance_args | groupsize_args 
                                                | center_args))

        # choose cluster center sampler
        if overlap_mode=='auto':
            overlap_mode = 'lda' if n_clusters*dim <= 10000 else 'c2c'
        
        center_sampler = ConstrainedOverlapCenters(
                            overlap_mode=overlap_mode, **center_args
                            )

        distribution_sampler = FixedProportionMix(distributions_parsed)

        Archetype.__init__(
            self, n_clusters, dim, n_samples, name, scale,
            MaxMinCovarianceSampler(**covariance_args),
            center_sampler,
            MaxMinGroupSizeSampler(**groupsize_args),
            distribution_sampler,
            **covariance_args, **groupsize_args, **center_args,
            )

        
        